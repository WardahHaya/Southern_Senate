"""Controlled native-streaming smoke test; not used by the production UI."""

from __future__ import annotations

import threading
import time
import sys
import subprocess
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from YouTubeLiveTranscribe import DEFAULT_CACHE_DIR, load_voxtral_model
from voxtral_live_stream import stream_voxtral_text


def main() -> None:
    model, processor, device = load_voxtral_model(
        DEFAULT_CACHE_DIR,
        model_id="mistralai/Voxtral-Mini-3B-Realtime-2602",
    )
    audio_path = hf_hub_download(
        repo_id="patrickvonplaten/audio_samples",
        filename="bcn_weather.mp3",
        repo_type="dataset",
    )
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            audio_path,
            "-ar",
            str(processor.feature_extractor.sampling_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = (
        np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
    )
    right_pad = int(
        processor.num_right_pad_tokens() * processor.raw_audio_length_per_tok
    )
    samples = np.pad(samples, (0, right_pad))
    pcm = (
        np.clip(samples, -1.0, 1.0) * 32767.0
    ).astype(np.int16).tobytes()
    block_bytes = int(processor.feature_extractor.sampling_rate * 0.25 * 2)
    chunks = [
        pcm[offset : offset + block_bytes]
        for offset in range(0, len(pcm), block_bytes)
    ]

    started = time.perf_counter()
    pieces = []
    for piece, audio_cursor in stream_voxtral_text(
        chunks,
        model,
        processor,
        device,
        threading.Event(),
    ):
        pieces.append(piece)
        print(f"[{audio_cursor:6.2f}s] {piece}", end="", flush=True)
    elapsed = time.perf_counter() - started
    print()
    print(f"SMOKE_OK elapsed={elapsed:.2f}s text={''.join(pieces).strip()!r}")


if __name__ == "__main__":
    main()
