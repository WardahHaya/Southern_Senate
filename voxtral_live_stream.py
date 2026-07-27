"""Stateful PCM windowing for Voxtral Realtime's native causal decoder."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np


@dataclass
class VoxtralPCMWindowBuffer:
    """Convert arbitrary PCM arrivals into Voxtral's overlapping live windows."""

    first_samples: int
    chunk_samples: int
    first_mel_frames: int
    audio_length_per_token: int
    hop_length: int
    win_length: int
    max_buffered_samples: int | None = None
    _samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        init=False,
    )
    _base_sample: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
    )

    def append_pcm16(
        self,
        pcm_bytes: bytes,
        stop_event: threading.Event | None = None,
    ) -> bool:
        if not pcm_bytes:
            return True
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        with self._condition:
            while (
                self.max_buffered_samples is not None
                and len(self._samples) + len(samples) > self.max_buffered_samples
                and not self._closed
                and not (stop_event and stop_event.is_set())
            ):
                # Backpressure is essential: without it the feeder can ingest
                # many minutes while CUDA inference is behind, hiding the real
                # backlog and growing process memory until Windows kills it.
                self._condition.wait(timeout=0.1)
            if self._closed:
                raise RuntimeError("cannot append to a closed Voxtral PCM buffer")
            if stop_event and stop_event.is_set():
                return False
            self._samples = np.concatenate((self._samples, samples))
            self._condition.notify_all()
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _wait_for_window(
        self,
        absolute_start: int,
        sample_count: int,
        stop_event: threading.Event,
    ) -> np.ndarray | None:
        absolute_end = absolute_start + sample_count
        with self._condition:
            while not stop_event.is_set():
                available_end = self._base_sample + len(self._samples)
                if available_end >= absolute_end:
                    local_start = absolute_start - self._base_sample
                    local_end = local_start + sample_count
                    return self._samples[local_start:local_end].copy()
                if self._closed:
                    return None
                self._condition.wait(timeout=0.1)
        return None

    def _discard_before(self, absolute_sample: int) -> None:
        with self._condition:
            discard = min(
                max(0, absolute_sample - self._base_sample),
                len(self._samples),
            )
            if discard:
                self._samples = self._samples[discard:].copy()
                self._base_sample += discard
                self._condition.notify_all()

    def windows(self, stop_event: threading.Event) -> Iterator[tuple[np.ndarray, bool]]:
        first = self._wait_for_window(0, self.first_samples, stop_event)
        if first is None:
            return
        yield first, True

        mel_frame_index = self.first_mel_frames
        next_start = mel_frame_index * self.hop_length - self.win_length // 2
        while not stop_event.is_set():
            chunk = self._wait_for_window(
                next_start,
                self.chunk_samples,
                stop_event,
            )
            if chunk is None:
                return
            yield chunk, False
            mel_frame_index += self.audio_length_per_token
            following_start = (
                mel_frame_index * self.hop_length - self.win_length // 2
            )
            self._discard_before(following_start)
            next_start = following_start


def make_voxtral_pcm_buffer(processor) -> VoxtralPCMWindowBuffer:
    sample_rate = int(processor.feature_extractor.sampling_rate)
    return VoxtralPCMWindowBuffer(
        first_samples=int(processor.num_samples_first_audio_chunk),
        chunk_samples=int(processor.num_samples_per_audio_chunk),
        first_mel_frames=int(processor.num_mel_frames_first_audio_chunk),
        audio_length_per_token=int(processor.audio_length_per_tok),
        hop_length=int(processor.feature_extractor.hop_length),
        win_length=int(processor.feature_extractor.win_length),
        # Keep only a modest live edge in RAM. Remaining lossless audio stays
        # in AudioSpool, where backlog is visible and recoverable.
        max_buffered_samples=sample_rate * 8,
    )


def stream_voxtral_text(
    pcm_chunks: Iterable[bytes],
    model,
    processor,
    device: str,
    stop_event: threading.Event,
    *,
    on_audio_progress: Callable[[float], None] | None = None,
    max_audio_seconds: float | None = None,
) -> Iterator[tuple[str, float]]:
    """Yield native Voxtral text pieces and their causal audio cursor."""
    import torch
    from transformers import TextIteratorStreamer

    sample_rate = int(processor.feature_extractor.sampling_rate)
    pcm_buffer = make_voxtral_pcm_buffer(processor)
    feeder_error: list[BaseException] = []
    feeder_shutdown = threading.Event()

    def feed_pcm() -> None:
        try:
            for pcm_chunk in pcm_chunks:
                if stop_event.is_set():
                    break
                if not pcm_buffer.append_pcm16(pcm_chunk, stop_event):
                    break
        except BaseException as exc:  # surfaced on the consumer thread
            if not feeder_shutdown.is_set():
                feeder_error.append(exc)
        finally:
            pcm_buffer.close()

    feeder = threading.Thread(
        target=feed_pcm,
        daemon=True,
        name="VoxtralPCMFeeder",
    )
    feeder.start()

    windows = pcm_buffer.windows(stop_event)
    try:
        first_audio, _ = next(windows)
    except StopIteration:
        return

    first_inputs = processor(
        first_audio,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )
    first_inputs = first_inputs.to(device, dtype=model.dtype)
    progress_lock = threading.Lock()
    progress_seconds = len(first_audio) / sample_rate
    generation_error: list[BaseException] = []

    def input_features_generator():
        nonlocal progress_seconds
        yield first_inputs.input_features
        if on_audio_progress is not None:
            on_audio_progress(progress_seconds)
        advance_seconds = (
            int(processor.audio_length_per_tok)
            * int(processor.feature_extractor.hop_length)
            / sample_rate
        )
        for audio_window, _ in windows:
            with progress_lock:
                if (
                    max_audio_seconds is not None
                    and progress_seconds >= max_audio_seconds
                ):
                    return
            inputs = processor(
                audio_window,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            inputs = inputs.to(device, dtype=model.dtype)
            with progress_lock:
                progress_seconds += advance_seconds
                current_progress = progress_seconds
            if on_audio_progress is not None:
                on_audio_progress(current_progress)
            yield inputs.input_features

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(
                    input_ids=first_inputs.input_ids,
                    input_features=input_features_generator(),
                    num_delay_tokens=first_inputs.num_delay_tokens,
                    streamer=streamer,
                )
        except BaseException as exc:
            generation_error.append(exc)
            streamer.end()

    generation_thread = threading.Thread(
        target=generate,
        daemon=True,
        name="VoxtralNativeGenerator",
    )
    generation_thread.start()

    for text_piece in streamer:
        with progress_lock:
            audio_cursor = progress_seconds
        if text_piece:
            yield text_piece, audio_cursor

    # Generation may end because the bounded live segment is complete while
    # the feeder is waiting on backpressure. Wake it without treating the
    # intentional segment rollover as a capture failure.
    feeder_shutdown.set()
    pcm_buffer.close()
    generation_thread.join(timeout=5)
    feeder.join(timeout=1)
    if generation_error:
        raise RuntimeError(
            f"native Voxtral streaming failed: {generation_error[0]}"
        ) from generation_error[0]
    if feeder_error:
        raise RuntimeError(
            f"native Voxtral PCM feed failed: {feeder_error[0]}"
        ) from feeder_error[0]
