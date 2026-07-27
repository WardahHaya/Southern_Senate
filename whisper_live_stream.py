"""Low-latency faster-whisper transcription with word-level speaker fusion.

The ASR model and pyannote diarization run independently over the same audio
timeline.  ``transcribe_whisper_window`` joins them by assigning every Whisper
word to the pyannote speaker with the greatest interval overlap (provided by
``resolve_speaker``), then groups adjacent words without crossing speakers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import numpy as np


SpeakerResolver = Callable[[float, float], str | None]


@dataclass(frozen=True)
class WhisperTimedWord:
    text: str
    start: float
    end: float
    speaker: str | None


@dataclass(frozen=True)
class WhisperSpeakerUnit:
    """A contiguous transcript unit belonging to one diarized speaker."""

    text: str
    start: float
    end: float
    speaker: str | None
    words: tuple[WhisperTimedWord, ...] = ()


class WhisperCaptionBuffer:
    """Turn word-timestamp updates into readable live caption phrases."""

    def __init__(
        self,
        *,
        max_words: int = 24,
        max_duration: float | None = 5.0,
        min_sentence_words: int = 4,
        min_clause_words: int = 10,
    ):
        self.max_words = max_words
        self.max_duration = max_duration
        self.min_sentence_words = min_sentence_words
        self.min_clause_words = min_clause_words
        self._pending: WhisperSpeakerUnit | None = None

    def _ready(self) -> bool:
        if self._pending is None:
            return False
        word_count = len(self._pending.text.split())
        text = self._pending.text.rstrip()
        sentence_end = text.endswith((".", "?", "!"))
        clause_end = text.endswith((",", ";", ":"))
        duration_ready = (
            self.max_duration is not None
            and self._pending.end - self._pending.start >= self.max_duration
        )
        return (
            word_count >= self.max_words
            or duration_ready
            or (sentence_end and word_count >= self.min_sentence_words)
            or (clause_end and word_count >= self.min_clause_words)
        )

    @staticmethod
    def _word_units(unit: WhisperSpeakerUnit) -> list[WhisperSpeakerUnit]:
        """Expand an ASR update so punctuation and turns can flush mid-unit."""
        if unit.words:
            return [
                WhisperSpeakerUnit(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=word.speaker or unit.speaker,
                    words=(word,),
                )
                for word in unit.words
            ]
        words = unit.text.split()
        if not words:
            return []
        step = max(0.03, unit.end - unit.start) / len(words)
        return [
            WhisperSpeakerUnit(
                text=text,
                start=unit.start + index * step,
                end=unit.start + (index + 1) * step,
                speaker=unit.speaker,
            )
            for index, text in enumerate(words)
        ]

    def push(
        self,
        units: list[WhisperSpeakerUnit],
    ) -> list[WhisperSpeakerUnit]:
        committed: list[WhisperSpeakerUnit] = []
        atomic_units = [
            word_unit
            for unit in units
            for word_unit in self._word_units(unit)
        ]
        for unit in atomic_units:
            if self._pending is None:
                self._pending = unit
            elif self._pending.speaker != unit.speaker:
                # A label may be temporarily unknown at startup. If pyannote
                # identifies the continuing voice on the next update, attach
                # the pending words instead of manufacturing a speaker turn.
                if self._pending.speaker is None and unit.speaker is not None:
                    self._pending = WhisperSpeakerUnit(
                        text=self._pending.text,
                        start=self._pending.start,
                        end=self._pending.end,
                        speaker=unit.speaker,
                        words=tuple(
                            WhisperTimedWord(
                                word.text,
                                word.start,
                                word.end,
                                unit.speaker,
                            )
                            for word in self._pending.words
                        ),
                    )
                else:
                    committed.append(self._pending)
                    self._pending = unit

            if (
                self._pending is not unit
                and self._pending is not None
                and self._pending.speaker == unit.speaker
            ):
                # The pending label may have been upgraded from unknown above.
                self._pending = WhisperSpeakerUnit(
                    text=f"{self._pending.text} {unit.text}".strip(),
                    start=self._pending.start,
                    end=unit.end,
                    speaker=self._pending.speaker,
                    words=self._pending.words + unit.words,
                )

            if self._ready():
                committed.append(self._pending)
                self._pending = None
        return committed

    def flush(self) -> list[WhisperSpeakerUnit]:
        if self._pending is None:
            return []
        unit = self._pending
        self._pending = None
        return [unit]


def split_unit_at_speaker_boundaries(
    unit: WhisperSpeakerUnit,
    resolve_speaker: SpeakerResolver,
) -> list[WhisperSpeakerUnit]:
    """Re-split a buffered phrase using the latest settled speaker timeline.

    Caption buffering intentionally waits for readable context. During that
    wait pyannote may refine a turn boundary inside the phrase. Resolve each
    exact Whisper word interval against the settled timeline and never let the
    next speaker's words remain attached to the previous speaker's caption.
    Uniform timing is retained only as a compatibility fallback for callers
    that construct a unit without Whisper word timestamps.
    """

    timed_words = list(unit.words)
    if not timed_words:
        raw_words = unit.text.split()
        if not raw_words:
            return []
        duration = max(0.03, unit.end - unit.start)
        word_duration = duration / len(raw_words)
        timed_words = [
            WhisperTimedWord(
                text=word,
                start=unit.start + index * word_duration,
                end=unit.start + (index + 1) * word_duration,
                speaker=unit.speaker,
            )
            for index, word in enumerate(raw_words)
        ]
    if not timed_words:
        return []
    resolved: list[tuple[str, float, float, str | None]] = []
    for word in timed_words:
        speaker = resolve_speaker(word.start, word.end) or word.speaker
        resolved.append((word.text, word.start, word.end, speaker))

    split_units: list[WhisperSpeakerUnit] = []
    for word, start, end, speaker in resolved:
        if split_units and split_units[-1].speaker == speaker:
            previous = split_units[-1]
            split_units[-1] = WhisperSpeakerUnit(
                text=f"{previous.text} {word}",
                start=previous.start,
                end=end,
                speaker=speaker,
                words=previous.words + (
                    WhisperTimedWord(word, start, end, speaker),
                ),
            )
        else:
            split_units.append(
                WhisperSpeakerUnit(
                    text=word,
                    start=start,
                    end=end,
                    speaker=speaker,
                    words=(WhisperTimedWord(word, start, end, speaker),),
                )
            )
    return split_units


def _normalized_token(token: str) -> str:
    return re.sub(r"[^\w']+", "", token, flags=re.UNICODE).casefold()


def remove_repeated_window_prefix(
    units: list[WhisperSpeakerUnit],
    recent_tokens: list[str],
    *,
    max_overlap_words: int = 24,
    last_emitted_time: float | None = None,
    timestamp_tolerance: float = 0.4,
) -> tuple[list[WhisperSpeakerUnit], list[str]]:
    """Remove rolling-window words already emitted despite timestamp drift."""

    flattened: list[WhisperTimedWord] = []
    for unit in units:
        if unit.words:
            flattened.extend(unit.words)
            continue
        raw_words = unit.text.split()
        duration = max(0.03, unit.end - unit.start)
        step = duration / max(1, len(raw_words))
        flattened.extend(
            WhisperTimedWord(
                word,
                unit.start + index * step,
                unit.start + (index + 1) * step,
                unit.speaker,
            )
            for index, word in enumerate(raw_words)
        )
    if not flattened:
        return [], recent_tokens[-max_overlap_words:]

    previous = [_normalized_token(word) for word in recent_tokens]
    current = [_normalized_token(word.text) for word in flattened]
    overlap = 0
    limit = min(len(previous), len(current), max_overlap_words)
    for size in range(limit, 0, -1):
        timestamp_is_overlap = (
            last_emitted_time is None
            or all(
                word.end <= last_emitted_time + timestamp_tolerance
                for word in flattened[:size]
            )
        )
        if timestamp_is_overlap and previous[-size:] == current[:size]:
            overlap = size
            break

    remaining = flattened[overlap:]
    rebuilt: list[WhisperSpeakerUnit] = []
    for word in remaining:
        if rebuilt and rebuilt[-1].speaker == word.speaker:
            prior = rebuilt[-1]
            rebuilt[-1] = WhisperSpeakerUnit(
                text=f"{prior.text} {word.text}",
                start=prior.start,
                end=word.end,
                speaker=prior.speaker,
                words=prior.words + (word,),
            )
        else:
            rebuilt.append(
                WhisperSpeakerUnit(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=word.speaker,
                    words=(word,),
                )
            )

    accepted_tokens = [word.text for word in remaining]
    updated_history = (recent_tokens + accepted_tokens)[-max_overlap_words:]
    return rebuilt, updated_history


def remove_timestamp_covered_words(
    units: list[WhisperSpeakerUnit],
    covered_intervals: list[tuple[float, float]],
    *,
    center_tolerance: float = 0.22,
) -> tuple[list[WhisperSpeakerUnit], list[tuple[float, float]]]:
    """Reject revised rolling-window words while preserving uncovered gaps.

    Token-only deduplication fails when Whisper changes the spelling of an
    already emitted phrase. Absolute word timestamps are stable enough to
    identify that revision. A word omitted by an earlier pass has no covered
    interval, so a later pass can still recover it.
    """
    kept: list[WhisperTimedWord] = []
    # Compare this ASR pass only with coverage from earlier passes. Consecutive
    # words in fast speech can have overlapping timestamps or centers less
    # than 220 ms apart; adding each new word immediately made its neighbor
    # look like a duplicate and silently dropped real content.
    prior_coverage = list(covered_intervals)
    new_coverage: list[tuple[float, float]] = []
    for unit in units:
        words = unit.words
        if not words:
            raw = unit.text.split()
            step = max(0.03, unit.end - unit.start) / max(1, len(raw))
            words = tuple(
                WhisperTimedWord(
                    text,
                    unit.start + index * step,
                    unit.start + (index + 1) * step,
                    unit.speaker,
                )
                for index, text in enumerate(raw)
            )
        for word in words:
            duration = max(0.03, word.end - word.start)
            center = (word.start + word.end) / 2.0
            duplicate = False
            for old_start, old_end in prior_coverage:
                overlap = max(
                    0.0,
                    min(word.end, old_end) - max(word.start, old_start),
                )
                old_center = (old_start + old_end) / 2.0
                if overlap >= duration * 0.55 or abs(center - old_center) <= center_tolerance:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(word)
            new_coverage.append((word.start, word.end))

    rebuilt: list[WhisperSpeakerUnit] = []
    for word in kept:
        if rebuilt and rebuilt[-1].speaker == word.speaker:
            previous = rebuilt[-1]
            rebuilt[-1] = WhisperSpeakerUnit(
                text=f"{previous.text} {word.text}",
                start=previous.start,
                end=word.end,
                speaker=word.speaker,
                words=previous.words + (word,),
            )
        else:
            rebuilt.append(
                WhisperSpeakerUnit(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=word.speaker,
                    words=(word,),
                )
            )
    coverage = prior_coverage + new_coverage
    newest = max((end for _, end in coverage), default=0.0)
    coverage = [
        interval for interval in coverage
        if interval[1] >= newest - 30.0
    ]
    return rebuilt, coverage


def transcribe_whisper_window(
    model,
    pcm16: bytes,
    *,
    context_start: float,
    emitted_audio_end: float,
    late_recovery_seconds: float = 0.0,
    commit_before: float | None = None,
    language: str,
    resolve_speaker: SpeakerResolver,
    merge_gap: float = 0.8,
) -> list[WhisperSpeakerUnit]:
    """Transcribe one overlapped PCM window and fuse words with pyannote.

    ``emitted_audio_end`` removes words repeated by the acoustic overlap. The
    resolver is deliberately injected so this module stays compatible with the
    project's rolling pyannote tracker and stable cross-window speaker IDs.
    """

    audio = (
        np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    )
    segments, _ = model.transcribe(
        audio,
        language=language or "en",
        # The verified GPU has ample headroom. A small beam substantially
        # reduces dropped/repeated broadcast words while remaining realtime.
        beam_size=5,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        temperature=0.0,
        hallucination_silence_threshold=1.0,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    words: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            start = context_start + float(word.start)
            end = max(start + 0.03, context_start + float(word.end))
            # A later rolling pass can recover a word that an earlier pass
            # omitted between two already emitted words. Keep a short history
            # revisable instead of permanently discarding every timestamp
            # behind the newest emitted word. Token overlap removal suppresses
            # the repeated context after this recovery step.
            recovery_cutoff = emitted_audio_end - max(
                0.0,
                late_recovery_seconds,
            )
            if end <= recovery_cutoff + 0.03:
                continue
            # The decoder frequently revises the final partial word on the
            # following one-second stride. Hold that unstable edge briefly so
            # both the provisional and corrected spelling are not displayed.
            if commit_before is not None and end > commit_before:
                continue
            text = str(word.word).strip()
            if not text:
                continue
            speaker = resolve_speaker(start, end)
            words.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "timed_word": WhisperTimedWord(text, start, end, speaker),
                }
            )

    units: list[dict] = []
    for word in words:
        if (
            units
            and units[-1]["speaker"] == word["speaker"]
            and word["start"] - units[-1]["end"] <= merge_gap
        ):
            units[-1]["words"].append(word["text"])
            units[-1]["timed_words"].append(word["timed_word"])
            units[-1]["end"] = word["end"]
        else:
            units.append(
                {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word["text"]],
                    "timed_words": [word["timed_word"]],
                }
            )

    return [
        WhisperSpeakerUnit(
            text=" ".join(unit["words"]),
            start=unit["start"],
            end=unit["end"],
            speaker=unit["speaker"],
            words=tuple(unit["timed_words"]),
        )
        for unit in units
    ]
