import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch


def suppress_torchcodec_warnings() -> None:
    """Silence pyannote's optional torchcodec warning when using in-memory audio."""
    warnings.filterwarnings(
        "ignore",
        message=r".*torchcodec.*",
        category=UserWarning,
    )


suppress_torchcodec_warnings()


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def load_hf_token_from_env_file(dotenv_path: Optional[Union[str, os.PathLike]] = None) -> Optional[str]:
    """Load the Hugging Face token from a local .env file if present."""
    candidate_paths = []
    if dotenv_path is not None:
        candidate_paths.append(Path(dotenv_path))

    cwd_path = Path.cwd()
    candidate_paths.extend([
        cwd_path / ".env",
        Path(__file__).resolve().parent / ".env",
    ])

    for path in candidate_paths:
        if not path:
            continue
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[len("export "):].strip()
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == "HF_TOKEN" and value:
                    return value
        except OSError:
            continue
    return None


@dataclass
class SpeakerRegistry:
    """Keep stable speaker IDs and colors for a live transcript session."""

    speakers: Dict[str, Dict[str, object]] = field(default_factory=dict)
    color_palette: List[str] = field(default_factory=lambda: [
        "#ff6b6b",
        "#4ecdc4",
        "#ffe66d",
        "#7bdff2",
        "#c77dff",
        "#8dd3c7",
        "#fb8500",
    ])

    def _normalize_label(self, label: str) -> str:
        if not label:
            return "unknown"
        normalized = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
        return normalized or "unknown"

    def register_label(self, label: Optional[str]) -> Tuple[str, str]:
        normalized = self._normalize_label(label or "unknown")
        if normalized in self.speakers:
            speaker = self.speakers[normalized]
            return str(speaker["id"]), str(speaker["color"])

        speaker_id = normalized
        color = self.color_palette[len(self.speakers) % len(self.color_palette)]
        self.speakers[normalized] = {"id": speaker_id, "color": color}
        return speaker_id, color


class DiarizationContext:
    """Wrap the local pyannote diarization pipeline for live-stream use."""

    def __init__(
        self,
        enabled: bool = False,
        hf_token: Optional[str] = None,
        model_id: str = "pyannote/speaker-diarization-3.1",
        segmentation_model_id: str = "pyannote/segmentation-3.0",
    ):
        self.enabled = enabled
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or load_hf_token_from_env_file()
        self.model_id = model_id
        self.segmentation_model_id = segmentation_model_id
        self.registry = SpeakerRegistry()
        self.pipeline = None
        self.pipeline_error: Optional[str] = None

    def is_ready(self) -> bool:
        return self.enabled and self.pipeline is not None and not self.pipeline_error

    def disable(self, reason: str) -> None:
        """Disable diarization without interrupting the transcription pipeline."""
        self.enabled = False
        self.pipeline_error = reason

    def initialize(self):
        if self.pipeline is not None or not self.enabled:
            return self.pipeline
        if not self.hf_token:
            self.pipeline_error = "HF_TOKEN is not set; diarization remains disabled"
            return None

        suppress_torchcodec_warnings()
        try:
            from pyannote.audio import Pipeline
        except Exception as exc:  # pragma: no cover - environment specific
            self.pipeline_error = f"pyannote.audio import failed: {exc}"
            return None

        try:
            self.pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=self.hf_token)
        except TypeError:
            self.pipeline = Pipeline.from_pretrained(self.model_id, token=self.hf_token)
        except Exception as exc:  # pragma: no cover - environment specific
            self.pipeline_error = f"Diarization model load failed: {exc}"
            return None

        self.pipeline_error = None
        return self.pipeline

    def _pcm_to_waveform(self, pcm_bytes: bytes, sample_rate: int) -> torch.Tensor:
        if not pcm_bytes:
            return torch.zeros(1, 1)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        pcm = pcm / 32768.0
        return torch.from_numpy(pcm.reshape(1, -1))

    def diarize_pcm(self, pcm_bytes: bytes, sample_rate: int, min_speakers: int = 1, max_speakers: int = 8):
        if not self.enabled:
            raise RuntimeError("Diarization disabled")

        pipeline = self.initialize()
        if pipeline is None:
            raise RuntimeError(self.pipeline_error or "Diarization pipeline unavailable")

        waveform = self._pcm_to_waveform(pcm_bytes, sample_rate)
        try:
            result = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(f"Diarization run failed: {exc}") from exc

        segments = []

        if hasattr(result, "itertracks"):
            try:
                for turn, _, speaker in result.itertracks(yield_label=True):
                    segments.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": str(speaker),
                    })
                return segments
            except AttributeError:
                pass

        try:
            annotation = getattr(result, "speaker_diarization", None)
            if annotation is None:
                annotation = result
            for turn, track, label in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(label),
                })
        except Exception:
            segments = []

        return segments

    def select_speaker_for_time(self, segments, elapsed_seconds: float) -> Optional[str]:
        if not segments:
            return None
        for segment in segments:
            if segment.get("start", 0.0) <= elapsed_seconds < segment.get("end", float("inf")):
                return segment.get("speaker")
        return segments[-1].get("speaker")
