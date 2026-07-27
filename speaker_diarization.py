import os
import re
import threading
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
        message=r"(?s).*torchcodec.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*degrees of freedom is <= 0.*",
        category=UserWarning,
    )
    # Some pyannote pooling paths briefly receive an empty mask at a rolling
    # window edge. The pipeline safely ignores that frame; do not flood the
    # operator console with NumPy implementation warnings.
    warnings.filterwarnings(
        "ignore",
        message=r"Mean of empty slice",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"invalid value encountered in divide",
        category=RuntimeWarning,
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
        "#FF5D73",
        "#00D4C8",
        "#FFD166",
        "#4D96FF",
        "#C77DFF",
        "#2ECC71",
        "#FF8C00",
        "#F72585",
        "#A3FF12",
        "#B8C0FF",
    ])

    def _normalize_label(self, label: str) -> str:
        if not label:
            return "unknown"
        normalized = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
        return normalized or "unknown"

    def register_label(self, label: Optional[str]) -> Tuple[str, str]:
        normalized = self._normalize_label(label or "unknown")
        if normalized == "unknown":
            return "Identifying speaker", "#9CA3AF"
        if normalized in self.speakers:
            speaker = self.speakers[normalized]
            return str(speaker["id"]), str(speaker["color"])

        speaker_number = len(self.speakers) + 1
        speaker_id = f"Speaker {speaker_number}"
        if speaker_number <= len(self.color_palette):
            color = self.color_palette[speaker_number - 1]
        else:
            # Open-ended news streams may introduce dozens of voices. Golden
            # angle spacing avoids repeating the ten presentation colors.
            hue = round((speaker_number - 1) * 137.508) % 360
            lightness = 64 if speaker_number % 2 else 72
            color = f"hsl({hue} 82% {lightness}%)"
        self.speakers[normalized] = {"id": speaker_id, "color": color}
        return speaker_id, color


@dataclass
class LiveSpeakerTracker:
    """Reconcile window-local diarization labels into stable live identities."""

    previous_segments: List[Dict[str, object]] = field(default_factory=list)
    speaker_profiles: Dict[str, np.ndarray] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    next_speaker_number: int = 1
    # News mode prioritizes detecting a turn over reusing an uncertain
    # identity. Only strong voice similarity may suppress a color change.
    # A small tolerance below the Community-1 default lets a returning anchor
    # reclaim the original color from a short live-edge voiceprint. Keep this
    # above the ambiguous 0.60 case so distinct speakers are not collapsed.
    voice_match_threshold: float = 0.62
    max_session_speakers: Optional[int] = None
    single_cluster_change_count: int = 0
    pending_change_embedding: Optional[np.ndarray] = None

    def canonical_label(self, label: str) -> str:
        while label in self.aliases:
            label = self.aliases[label]
        return label

    @staticmethod
    def _overlap_seconds(left: Mapping[str, object], right: Mapping[str, object]) -> float:
        return max(
            0.0,
            min(float(left["end"]), float(right["end"]))
            - max(float(left["start"]), float(right["start"])),
        )

    def update(
        self,
        window_segments: List[Dict[str, object]],
        window_start_seconds: float,
        window_embeddings: Optional[Mapping[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        if not window_segments:
            # Brief silence or an inconclusive pass must not erase the last
            # confirmed identity and force the UI back to an unknown color.
            return list(self.previous_segments)

        absolute_segments = [
            {
                "start": float(segment["start"]) + window_start_seconds,
                "end": float(segment["end"]) + window_start_seconds,
                "speaker": str(segment["speaker"]),
            }
            for segment in window_segments
        ]

        local_labels = list(dict.fromkeys(str(s["speaker"]) for s in absolute_segments))
        stable_labels = list(dict.fromkeys(
            [self.canonical_label(label) for label in self.speaker_profiles]
            + [str(s["speaker"]) for s in self.previous_segments]
        ))
        label_map: Dict[str, str] = {}
        used_stable_labels = set()
        active_previous_speaker = (
            self.canonical_label(str(self.previous_segments[-1]["speaker"]))
            if self.previous_segments
            else None
        )
        # A single rolling cluster can produce a noisy embedding for one pass
        # (music, telephone audio, emphasis). Require three consecutive
        # low-similarity passes before abandoning the active identity. At a
        # 0.75s stride this confirms a real voice change quickly without the
        # Speaker 1 -> 2 -> 1 color oscillation seen at session startup.
        if (
            len(local_labels) == 1
            and active_previous_speaker is not None
            and window_embeddings
            and local_labels[0] in window_embeddings
            and active_previous_speaker in self.speaker_profiles
        ):
            active_similarity = float(
                np.dot(
                    window_embeddings[local_labels[0]],
                    self.speaker_profiles[active_previous_speaker],
                )
            )
            if active_similarity < 0.50:
                current_embedding = window_embeddings[local_labels[0]]
                pending_similarity = (
                    float(np.dot(current_embedding, self.pending_change_embedding))
                    if self.pending_change_embedding is not None
                    else -1.0
                )
                if pending_similarity >= 0.80:
                    self.single_cluster_change_count += 1
                else:
                    self.single_cluster_change_count = 1
                self.pending_change_embedding = current_embedding.copy()
            else:
                self.single_cluster_change_count = 0
                self.pending_change_embedding = None
        elif len(local_labels) != 1:
            self.single_cluster_change_count = 0
            self.pending_change_embedding = None
        if window_embeddings:
            # Community-1 embeddings are authoritative. Match all local
            # clusters against profiles that existed before this pass, then
            # create every unmatched identity together. Never let timeline
            # overlap force a genuinely different voice into the old speaker.
            voice_candidates = []
            for local_label in local_labels:
                embedding = window_embeddings.get(local_label)
                if embedding is None:
                    continue
                for stable_label, profile in self.speaker_profiles.items():
                    canonical = self.canonical_label(stable_label)
                    similarity = float(np.dot(embedding, profile))
                    if similarity >= self.voice_match_threshold:
                        voice_candidates.append(
                            (similarity, local_label, canonical)
                        )
            for _, local_label, stable_label in sorted(
                voice_candidates,
                reverse=True,
            ):
                if local_label in label_map or stable_label in used_stable_labels:
                    continue
                label_map[local_label] = stable_label
                used_stable_labels.add(stable_label)

        # Rolling-window labels are temporary. Even when Community-1 exposes
        # embeddings, a short/noisy pass can produce a weak voiceprint for the
        # same continuing speaker. Reconcile any still-unmatched cluster by
        # shared broadcast time. A true change is normally represented by two
        # local clusters in the boundary window, so the new cluster remains
        # free to receive a new session identity.
        candidates = []
        for local_label in local_labels:
            if local_label in label_map:
                continue
            local_embedding = (
                window_embeddings.get(local_label)
                if window_embeddings
                else None
            )
            local_parts = [
                s for s in absolute_segments if s["speaker"] == local_label
            ]
            local_duration = sum(
                max(0.0, float(s["end"]) - float(s["start"]))
                for s in local_parts
            )
            for stable_label in stable_labels:
                if stable_label in used_stable_labels:
                    continue
                stable_profile = self.speaker_profiles.get(stable_label)
                # With one local cluster, overlapping broadcast time is
                # stronger evidence of identity continuity than one noisy
                # rolling embedding (voice-over, codec changes, and background
                # speech can move that embedding substantially). A genuine
                # hand-off is represented by two local clusters while both
                # voices are still inside the ten-second rolling window.
                if local_embedding is not None and stable_profile is not None:
                    continuity_similarity = float(
                        np.dot(local_embedding, stable_profile)
                    )
                    single_cluster_continuity = (
                        len(local_labels) == 1
                        and stable_label == active_previous_speaker
                        and self.single_cluster_change_count < 2
                    )
                    if (
                        continuity_similarity < 0.50
                        and not single_cluster_continuity
                    ):
                        continue
                stable_parts = [
                    s
                    for s in self.previous_segments
                    if s["speaker"] == stable_label
                ]
                overlap = sum(
                    self._overlap_seconds(local_part, stable_part)
                    for local_part in local_parts
                    for stable_part in stable_parts
                )
                # Require meaningful continuity instead of allowing a tiny
                # boundary overlap to swallow a genuine new voice.
                if overlap >= min(1.0, max(0.35, local_duration * 0.20)):
                    candidates.append((overlap, local_label, stable_label))
        for _, local_label, stable_label in sorted(candidates, reverse=True):
            if local_label in label_map or stable_label in used_stable_labels:
                continue
            label_map[local_label] = stable_label
            used_stable_labels.add(stable_label)

        for local_label in local_labels:
            if local_label not in label_map:
                if (
                    self.max_session_speakers is None
                    or self.next_speaker_number <= self.max_session_speakers
                ):
                    label_map[local_label] = f"live-speaker-{self.next_speaker_number}"
                    self.next_speaker_number += 1
                elif stable_labels:
                    # Once the configured presentation palette is full, keep
                    # the most recently active identity instead of inventing
                    # unbounded speakers and cycling colors.
                    label_map[local_label] = self.canonical_label(stable_labels[-1])
                else:
                    label_map[local_label] = "live-speaker-1"

        reconciled = [
            {
                **segment,
                "speaker": self.canonical_label(label_map[str(segment["speaker"])]),
            }
            for segment in absolute_segments
        ]
        self.previous_segments = reconciled
        if (
            active_previous_speaker is not None
            and reconciled
            and str(reconciled[-1]["speaker"]) != active_previous_speaker
        ):
            self.single_cluster_change_count = 0
            self.pending_change_embedding = None
        if window_embeddings:
            self.apply_embeddings(window_embeddings, label_map)
        return reconciled

    def apply_embeddings(
        self,
        window_embeddings: Mapping[str, np.ndarray],
        label_map: Mapping[str, str],
    ) -> None:
        """Update assigned centroids without merging clusters from this pass."""
        for local_label, raw_assigned_label in label_map.items():
            embedding = window_embeddings.get(local_label)
            if embedding is None:
                continue
            assigned_label = self.canonical_label(raw_assigned_label)
            previous = self.speaker_profiles.get(assigned_label)
            # Rolling windows overlap heavily, so the same broadcast region is
            # observed repeatedly. Give an established session voiceprint only
            # a small update from each pass; a noisy cross-talk or transition
            # window must not gradually drag an anchor/reporter profile toward
            # another speaker over a long live programme.
            profile = (
                embedding
                if previous is None
                else (0.95 * previous + 0.05 * embedding)
            )
            norm = float(np.linalg.norm(profile))
            if norm > 0:
                self.speaker_profiles[assigned_label] = profile / norm

        self.previous_segments = [
            {**segment, "speaker": self.canonical_label(str(segment["speaker"]))}
            for segment in self.previous_segments
        ]

    def speaker_at(
        self,
        elapsed_seconds: float,
        *,
        fallback_to_latest: bool = True,
    ) -> Optional[str]:
        if not self.previous_segments:
            return None
        for segment in self.previous_segments:
            if float(segment["start"]) <= elapsed_seconds < float(segment["end"]):
                return str(segment["speaker"])
        if not fallback_to_latest:
            return None
        # The diarizer runs asynchronously. If transcription is ahead of the
        # newest completed analysis, retain its latest confirmed speaker
        # instead of reverting to an "unknown" color.
        return str(self.previous_segments[-1]["speaker"])


@dataclass
class LiveTurnCommitter:
    """Freeze settled speaker turns while leaving the live edge revisable."""

    commit_lag: float = 1.0
    merge_gap: float = 0.4
    committed_until: float = 0.0
    turns: List[Dict[str, object]] = field(default_factory=list)

    def commit(
        self,
        window_segments: List[Dict[str, object]],
        stream_end: float,
    ) -> List[Dict[str, object]]:
        commit_edge = max(0.0, float(stream_end) - self.commit_lag)
        if commit_edge <= self.committed_until + 1e-3:
            return []

        new_turns: List[Dict[str, object]] = []
        for segment in sorted(window_segments, key=lambda item: float(item["start"])):
            start = max(float(segment["start"]), self.committed_until)
            end = min(float(segment["end"]), commit_edge)
            if end <= start:
                continue
            turn = {
                "start": start,
                "end": end,
                "speaker": str(segment["speaker"]),
            }
            new_turns.append(turn)
            if (
                self.turns
                and str(self.turns[-1]["speaker"]) == str(turn["speaker"])
                and start - float(self.turns[-1]["end"]) <= self.merge_gap
            ):
                self.turns[-1]["end"] = max(float(self.turns[-1]["end"]), end)
            else:
                self.turns.append(dict(turn))

        # The edge is intentionally advanced even across silence: future
        # rolling windows must not rewrite already settled broadcast time.
        self.committed_until = commit_edge
        return new_turns

    def speaker_at(self, elapsed_seconds: float) -> Optional[str]:
        for segment in reversed(self.turns):
            if float(segment["start"]) <= elapsed_seconds < float(segment["end"]):
                return str(segment["speaker"])
        return None

    def speaker_for_interval(
        self,
        start_seconds: float,
        end_seconds: float,
    ) -> Optional[str]:
        """Return the speaker with the greatest committed overlap."""
        scores: Dict[str, float] = {}
        latest_end: Dict[str, float] = {}
        for segment in self.turns:
            overlap = max(
                0.0,
                min(float(segment["end"]), end_seconds)
                - max(float(segment["start"]), start_seconds),
            )
            if overlap <= 0:
                continue
            speaker = str(segment["speaker"])
            scores[speaker] = scores.get(speaker, 0.0) + overlap
            latest_end[speaker] = max(
                latest_end.get(speaker, 0.0),
                float(segment["end"]),
            )
        if not scores:
            return None
        return max(scores, key=lambda speaker: (scores[speaker], latest_end[speaker]))


class DiarizationContext:
    """Wrap the local pyannote diarization pipeline for live-stream use."""

    def __init__(
        self,
        enabled: bool = False,
        hf_token: Optional[str] = None,
        model_id: str = "pyannote/speaker-diarization-community-1",
        segmentation_model_id: str = "pyannote/segmentation-3.0",
    ):
        self.enabled = enabled
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or load_hf_token_from_env_file()
        self.model_id = model_id
        self.segmentation_model_id = segmentation_model_id
        self.registry = SpeakerRegistry()
        self.pipeline = None
        self.pipeline_error: Optional[str] = None
        self.last_window_embeddings: Dict[str, np.ndarray] = {}
        self._initialize_lock = threading.Lock()

    def is_ready(self) -> bool:
        return self.enabled and self.pipeline is not None and not self.pipeline_error

    def disable(self, reason: str) -> None:
        """Disable diarization without interrupting the transcription pipeline."""
        self.enabled = False
        self.pipeline_error = reason

    def initialize(self):
        with self._initialize_lock:
            return self._initialize_unlocked()

    def _initialize_unlocked(self):
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
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_id,
                    token=self.hf_token,
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_id,
                    use_auth_token=self.hf_token,
                )
        except Exception as primary_exc:  # pragma: no cover - environment specific
            # Community-1 provides exclusive turns and pass-native embeddings.
            # Retain 3.1 as a compatibility fallback for installations whose
            # token has not yet accepted the Community-1 model terms.
            fallback_id = "pyannote/speaker-diarization-3.1"
            try:
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        fallback_id,
                        token=self.hf_token,
                    )
                except TypeError:
                    self.pipeline = Pipeline.from_pretrained(
                        fallback_id,
                        use_auth_token=self.hf_token,
                    )
                self.model_id = fallback_id
                print(
                    "[diarization] Community-1 unavailable; using 3.1 fallback: "
                    f"{primary_exc}"
                )
            except Exception as fallback_exc:
                self.pipeline_error = (
                    "Diarization model load failed: "
                    f"community-1={primary_exc}; 3.1={fallback_exc}"
                )
                return None

        self.pipeline_error = None
        if self.model_id == "pyannote/speaker-diarization-community-1":
            try:
                self.pipeline.instantiate({
                    "clustering": {
                        "Fa": 0.07,
                        "Fb": 0.8,
                        # Community-1 is calibrated at 0.60. Lowering this to
                        # 0.47 collapsed distinct anchor/reporter voices into
                        # one cluster for entire live sessions. Cross-window
                        # identity stability is handled by LiveSpeakerTracker,
                        # so retain the model's native separation threshold.
                        "threshold": 0.60,
                    },
                    "segmentation": {"min_duration_off": 0.1},
                })
            except Exception as exc:
                # Model defaults remain valid if this pyannote release uses a
                # different parameter schema.
                print(f"[diarization] Using Community-1 default parameters: {exc}")
        return self.pipeline

    def _pcm_to_waveform(self, pcm_bytes: bytes, sample_rate: int) -> torch.Tensor:
        if not pcm_bytes:
            return torch.zeros(1, 1)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        pcm = pcm / 32768.0
        return torch.from_numpy(pcm.reshape(1, -1))

    @staticmethod
    def _stabilize_micro_turns(
        segments: List[Dict[str, object]],
        minimum_turn_seconds: float = 0.25,
    ) -> List[Dict[str, object]]:
        """Remove sub-second clustering chatter without hiding real turns."""
        ordered = sorted(segments, key=lambda item: float(item["start"]))
        stabilized: List[Dict[str, object]] = []
        index = 0
        while index < len(ordered):
            current = dict(ordered[index])
            duration = float(current["end"]) - float(current["start"])
            previous = stabilized[-1] if stabilized else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None

            if duration < minimum_turn_seconds and previous is not None:
                # Only absorb a genuinely microscopic A-B-A glitch. The old
                # 0.8s threshold hid short questions and interjections.
                if (
                    following is not None
                    and str(previous["speaker"]) == str(following["speaker"])
                ):
                    previous["end"] = max(
                        float(previous["end"]),
                        float(following["end"]),
                    )
                    index += 2
                    continue
            if (
                previous is not None
                and str(previous["speaker"]) == str(current["speaker"])
                and float(current["start"]) - float(previous["end"]) <= 0.15
            ):
                previous["end"] = max(
                    float(previous["end"]),
                    float(current["end"]),
                )
            else:
                stabilized.append(current)
            index += 1
        return stabilized

    def diarize_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        if not self.enabled:
            raise RuntimeError("Diarization disabled")

        pipeline = self.initialize()
        if pipeline is None:
            raise RuntimeError(self.pipeline_error or "Diarization pipeline unavailable")

        waveform = self._pcm_to_waveform(pcm_bytes, sample_rate)
        try:
            speaker_constraints = {}
            if min_speakers is not None:
                speaker_constraints["min_speakers"] = min_speakers
            if max_speakers is not None:
                speaker_constraints["max_speakers"] = max_speakers
            result = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **speaker_constraints,
            )
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(f"Diarization run failed: {exc}") from exc

        segments = []
        self.last_window_embeddings = {}

        annotation = getattr(result, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(result, "speaker_diarization", None)
        if annotation is None:
            annotation = result

        try:
            for turn, _, label in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(label),
                })
        except Exception:
            return []
        segments = self._stabilize_micro_turns(segments)

        # Community-1 exposes embeddings aligned with annotation.labels().
        # Matching identities from the same pass avoids a second asynchronous
        # embedding job changing colors after turns were already committed.
        raw_embeddings = getattr(result, "speaker_embeddings", None)
        if raw_embeddings is not None and hasattr(annotation, "labels"):
            try:
                labels = list(annotation.labels())
                for index, label in enumerate(labels):
                    if isinstance(raw_embeddings, Mapping):
                        raw_embedding = raw_embeddings.get(label)
                        if raw_embedding is None:
                            raw_embedding = raw_embeddings.get(str(label))
                        if raw_embedding is None:
                            continue
                    else:
                        if index >= len(raw_embeddings):
                            break
                        raw_embedding = raw_embeddings[index]
                    embedding = np.asarray(
                        raw_embedding,
                        dtype=np.float32,
                    ).reshape(-1)
                    norm = float(np.linalg.norm(embedding))
                    if norm > 0 and not np.isnan(embedding).any():
                        self.last_window_embeddings[str(label)] = embedding / norm
            except Exception:
                self.last_window_embeddings = {}

        return segments

    def embed_speakers_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        segments: List[Dict[str, object]],
    ) -> Dict[str, np.ndarray]:
        """Create one normalized voiceprint per window-local speaker label."""
        pipeline = self.initialize()
        embedder = getattr(pipeline, "_embedding", None) if pipeline is not None else None
        if embedder is None or not segments:
            return {}

        waveform = self._pcm_to_waveform(pcm_bytes, sample_rate)
        embeddings: Dict[str, np.ndarray] = {}
        labels = list(dict.fromkeys(str(segment["speaker"]) for segment in segments))
        for label in labels:
            parts = []
            for segment in segments:
                if str(segment["speaker"]) != label:
                    continue
                start = max(0, int(float(segment["start"]) * sample_rate))
                end = min(waveform.shape[-1], int(float(segment["end"]) * sample_rate))
                if end > start:
                    parts.append(waveform[:, start:end])
            if not parts:
                continue
            voice = torch.cat(parts, dim=-1)[:, : sample_rate * 8]
            if voice.shape[-1] < int(sample_rate * 2.0):
                continue
            if float(torch.sqrt(torch.mean(voice * voice))) < 0.005:
                continue
            with torch.inference_mode():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Mean of empty slice")
                    warnings.filterwarnings("ignore", message=r"invalid value encountered in divide")
                    raw = np.asarray(embedder(voice.unsqueeze(0))).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(raw))
            if norm > 0 and np.isfinite(norm):
                embeddings[label] = raw / norm
        return embeddings

    def select_speaker_for_time(self, segments, elapsed_seconds: float) -> Optional[str]:
        if not segments:
            return None
        for segment in segments:
            if segment.get("start", 0.0) <= elapsed_seconds < segment.get("end", float("inf")):
                return segment.get("speaker")
        return segments[-1].get("speaker")
