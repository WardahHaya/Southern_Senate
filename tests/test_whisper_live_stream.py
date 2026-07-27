import unittest

from whisper_live_stream import (
    WhisperCaptionBuffer,
    WhisperSpeakerUnit,
    WhisperTimedWord,
    remove_repeated_window_prefix,
    remove_timestamp_covered_words,
    split_unit_at_speaker_boundaries,
    transcribe_whisper_window,
)


class _Word:
    def __init__(self, text, start, end):
        self.word = text
        self.start = start
        self.end = end


class _Segment:
    def __init__(self, words):
        self.words = words


class _Model:
    def transcribe(self, _audio, **kwargs):
        self.kwargs = kwargs
        return [
            _Segment(
                [
                    _Word("hello", 0.1, 0.4),
                    _Word("there", 0.4, 0.8),
                    _Word("senator", 0.8, 1.2),
                ]
            )
        ], None


class WhisperLiveStreamTests(unittest.TestCase):
    def test_timestamp_drift_does_not_repeat_window_edge_words(self):
        units = [
            WhisperSpeakerUnit(
                text="spreading. Reinforcements including emergency personnel",
                start=4.9,
                end=6.0,
                speaker="speaker-a",
            )
        ]
        cleaned, history = remove_repeated_window_prefix(
            units,
            ["stop", "the", "flames", "from", "spreading."],
        )

        self.assertEqual(
            [unit.text for unit in cleaned],
            ["Reinforcements including emergency personnel"],
        )
        self.assertEqual(history[-3:], ["including", "emergency", "personnel"])

    def test_intentional_repeated_word_after_live_edge_is_preserved(self):
        units = [
            WhisperSpeakerUnit(
                text="again",
                start=2.0,
                end=2.3,
                speaker="speaker-a",
                words=(
                    WhisperTimedWord("again", 2.0, 2.3, "speaker-a"),
                ),
            )
        ]
        cleaned, _history = remove_repeated_window_prefix(
            units,
            ["say", "it", "again"],
            last_emitted_time=1.0,
        )

        self.assertEqual([unit.text for unit in cleaned], ["again"])

    def test_timestamp_coverage_rejects_reworded_duplicate_but_keeps_gap(self):
        units = [
            WhisperSpeakerUnit(
                "corrected missing",
                1.0,
                2.0,
                "speaker-a",
                words=(
                    WhisperTimedWord("corrected", 1.0, 1.4, "speaker-a"),
                    WhisperTimedWord("missing", 1.55, 1.8, "speaker-a"),
                ),
            )
        ]
        cleaned, coverage = remove_timestamp_covered_words(
            units,
            [(0.98, 1.42)],
        )
        self.assertEqual([unit.text for unit in cleaned], ["missing"])
        self.assertIn((1.55, 1.8), coverage)

    def test_timestamp_coverage_keeps_fast_neighboring_words_in_same_pass(self):
        words = (
            WhisperTimedWord("the", 2.00, 2.18, "speaker-a"),
            WhisperTimedWord("speaker", 2.16, 2.36, "speaker-a"),
            WhisperTimedWord("continues", 2.34, 2.58, "speaker-a"),
        )
        cleaned, _coverage = remove_timestamp_covered_words(
            [
                WhisperSpeakerUnit(
                    "the speaker continues",
                    2.00,
                    2.58,
                    "speaker-a",
                    words=words,
                )
            ],
            [],
        )
        self.assertEqual(
            [unit.text for unit in cleaned],
            ["the speaker continues"],
        )

    def test_caption_buffer_commits_readable_phrases(self):
        buffer = WhisperCaptionBuffer(max_words=8, max_duration=2.0)
        first = WhisperSpeakerUnit("As smoke rises", 0.0, 0.8, "speaker-a")
        second = WhisperSpeakerUnit(
            "over vast tracts of forest.",
            0.8,
            1.8,
            "speaker-a",
        )

        self.assertEqual(buffer.push([first]), [])
        committed = buffer.push([second])
        self.assertEqual(
            [unit.text for unit in committed],
            ["As smoke rises over vast tracts of forest."],
        )

    def test_caption_buffer_flushes_on_confirmed_speaker_change(self):
        buffer = WhisperCaptionBuffer(max_words=12, max_duration=3.0)
        buffer.push(
            [WhisperSpeakerUnit("Anchor introduction", 0.0, 0.8, "speaker-a")]
        )
        committed = buffer.push(
            [WhisperSpeakerUnit("Reporter begins", 0.8, 1.4, "speaker-b")]
        )

        self.assertEqual(len(committed), 1)
        self.assertEqual(committed[0].speaker, "speaker-a")
        self.assertEqual(buffer.flush()[0].speaker, "speaker-b")

    def test_caption_length_varies_at_internal_sentence_boundary(self):
        buffer = WhisperCaptionBuffer(
            max_words=20,
            max_duration=4.0,
            min_sentence_words=4,
        )
        words = (
            WhisperTimedWord("This", 0.0, 0.2, "speaker-a"),
            WhisperTimedWord("is", 0.2, 0.4, "speaker-a"),
            WhisperTimedWord("the", 0.4, 0.6, "speaker-a"),
            WhisperTimedWord("first.", 0.6, 0.8, "speaker-a"),
            WhisperTimedWord("Next", 0.8, 1.0, "speaker-a"),
            WhisperTimedWord("thought", 1.0, 1.2, "speaker-a"),
        )
        committed = buffer.push([
            WhisperSpeakerUnit(
                "This is the first. Next thought",
                0.0,
                1.2,
                "speaker-a",
                words=words,
            )
        ])
        self.assertEqual([unit.text for unit in committed], ["This is the first."])
        self.assertEqual(buffer.flush()[0].text, "Next thought")

    def test_buffered_phrase_is_resplit_at_settled_speaker_boundary(self):
        words = "one two three four five six seven eight nine ten eleven twelve next speaker starts now"
        unit = WhisperSpeakerUnit(words, 0.0, 16.0, "speaker-a")

        parts = split_unit_at_speaker_boundaries(
            unit,
            lambda start, _end: "speaker-a" if start < 12.0 else "speaker-b",
        )

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].speaker, "speaker-a")
        self.assertEqual(len(parts[0].text.split()), 12)
        self.assertEqual(parts[1].speaker, "speaker-b")
        self.assertEqual(parts[1].text, "next speaker starts now")

    def test_exact_word_timestamps_drive_dynamic_speaker_boundary(self):
        timed_words = (
            WhisperTimedWord("first", 0.0, 0.4, "speaker-a"),
            WhisperTimedWord("speaker", 0.4, 0.9, "speaker-a"),
            # Uneven timing proves this is not a fixed word-count split.
            WhisperTimedWord("second", 2.7, 3.0, "speaker-a"),
            WhisperTimedWord("speaker", 3.0, 3.6, "speaker-a"),
        )
        unit = WhisperSpeakerUnit(
            "first speaker second speaker",
            0.0,
            3.6,
            "speaker-a",
            timed_words,
        )

        parts = split_unit_at_speaker_boundaries(
            unit,
            lambda start, _end: "speaker-a" if start < 2.5 else "speaker-b",
        )

        self.assertEqual(
            [(part.speaker, part.text) for part in parts],
            [
                ("speaker-a", "first speaker"),
                ("speaker-b", "second speaker"),
            ],
        )

    def test_words_are_split_at_pyannote_speaker_change(self):
        model = _Model()

        def resolve_speaker(start, _end):
            return "speaker-a" if start < 0.8 else "speaker-b"

        units = transcribe_whisper_window(
            model,
            b"\0\0" * 32000,
            context_start=0.0,
            emitted_audio_end=0.0,
            language="en",
            resolve_speaker=resolve_speaker,
        )

        self.assertEqual([unit.speaker for unit in units], ["speaker-a", "speaker-b"])
        self.assertEqual([unit.text for unit in units], ["hello there", "senator"])
        self.assertTrue(model.kwargs["word_timestamps"])
        self.assertEqual(model.kwargs["beam_size"], 5)
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 3)
        self.assertEqual(model.kwargs["repetition_penalty"], 1.1)
        self.assertEqual(model.kwargs["hallucination_silence_threshold"], 1.0)

    def test_overlap_words_are_not_emitted_twice(self):
        units = transcribe_whisper_window(
            _Model(),
            b"\0\0" * 32000,
            context_start=0.0,
            emitted_audio_end=0.8,
            language="en",
            resolve_speaker=lambda _start, _end: "speaker-a",
        )

        self.assertEqual([unit.text for unit in units], ["senator"])

    def test_unstable_live_edge_is_held_for_next_window(self):
        units = transcribe_whisper_window(
            _Model(),
            b"\0\0" * 32000,
            context_start=0.0,
            emitted_audio_end=0.0,
            commit_before=0.9,
            language="en",
            resolve_speaker=lambda _start, _end: "speaker-a",
        )

        self.assertEqual([unit.text for unit in units], ["hello there"])

    def test_late_recovery_keeps_previously_omitted_word(self):
        units = transcribe_whisper_window(
            _Model(),
            b"\0\0" * 32000,
            context_start=0.0,
            emitted_audio_end=1.2,
            late_recovery_seconds=1.0,
            language="en",
            resolve_speaker=lambda _start, _end: "speaker-a",
        )
        cleaned, _history = remove_repeated_window_prefix(
            units,
            ["hello", "there"],
        )

        self.assertEqual([unit.text for unit in cleaned], ["senator"])


if __name__ == "__main__":
    unittest.main()
