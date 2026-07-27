import tempfile
import unittest
import warnings
import numpy as np
from unittest.mock import patch
from pathlib import Path

from speaker_diarization import (
    DiarizationContext,
    LiveSpeakerTracker,
    LiveTurnCommitter,
    SpeakerRegistry,
    env_flag,
    load_hf_token_from_env_file,
    suppress_torchcodec_warnings,
)


class SpeakerRegistryTests(unittest.TestCase):
    def test_new_labels_get_stable_ids_and_colors(self):
        registry = SpeakerRegistry()

        speaker_a, color_a = registry.register_label("SPEAKER_00")
        speaker_b, color_b = registry.register_label("SPEAKER_00")
        speaker_c, color_c = registry.register_label("SPEAKER_01")

        self.assertEqual(speaker_a, speaker_b)
        self.assertEqual(color_a, color_b)
        self.assertNotEqual(speaker_a, speaker_c)
        self.assertNotEqual(color_a, color_c)

    def test_unknown_label_is_normalized(self):
        registry = SpeakerRegistry()
        speaker_id, _ = registry.register_label(" speaker-1 ")
        self.assertEqual(speaker_id, "Speaker 1")

    def test_unknown_speaker_has_pending_label(self):
        registry = SpeakerRegistry()
        speaker_id, _ = registry.register_label(None)
        self.assertEqual(speaker_id, "Identifying speaker")

    def test_select_speaker_for_time_uses_segment_containing_target(self):
        context = DiarizationContext(enabled=False)
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        self.assertEqual(context.select_speaker_for_time(segments, 2.5), "SPEAKER_01")

    def test_exact_live_lookup_does_not_guess_outside_known_segments(self):
        tracker = LiveSpeakerTracker()
        tracker.previous_segments = [
            {"start": 1.0, "end": 2.0, "speaker": "live-speaker-1"},
        ]
        self.assertIsNone(tracker.speaker_at(3.0, fallback_to_latest=False))
        self.assertEqual(tracker.speaker_at(3.0), "live-speaker-1")

    def test_live_turn_committer_freezes_only_settled_region(self):
        committer = LiveTurnCommitter(commit_lag=1.0)
        first = committer.commit([
            {"start": 0.0, "end": 2.0, "speaker": "live-speaker-1"},
            {"start": 2.0, "end": 4.0, "speaker": "live-speaker-2"},
        ], stream_end=4.0)
        self.assertEqual(first[-1]["end"], 3.0)
        self.assertEqual(committer.speaker_at(2.5), "live-speaker-2")

        # A later window cannot rewrite time that was already committed.
        committer.commit([
            {"start": 1.0, "end": 3.0, "speaker": "live-speaker-9"},
            {"start": 3.0, "end": 5.0, "speaker": "live-speaker-2"},
        ], stream_end=5.0)
        self.assertEqual(committer.speaker_at(2.5), "live-speaker-2")
        self.assertEqual(committer.committed_until, 4.0)

    def test_interval_assignment_uses_greatest_speaker_overlap(self):
        committer = LiveTurnCommitter(commit_lag=0.0)
        committer.commit([
            {"start": 0.0, "end": 0.4, "speaker": "live-speaker-1"},
            {"start": 0.4, "end": 2.0, "speaker": "live-speaker-2"},
        ], stream_end=2.0)
        self.assertEqual(
            committer.speaker_for_interval(0.0, 2.0),
            "live-speaker-2",
        )

    def test_load_hf_token_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dotenv_path = Path(tmp_dir) / ".env"
            dotenv_path.write_text("HF_TOKEN=abc123\n", encoding="utf-8")
            self.assertEqual(load_hf_token_from_env_file(dotenv_path), "abc123")

    def test_diarization_can_be_explicitly_disabled(self):
        context = DiarizationContext(enabled=True, hf_token="unused")
        context.disable("decoder unavailable")

        self.assertFalse(context.enabled)
        self.assertFalse(context.is_ready())
        self.assertEqual(context.pipeline_error, "decoder unavailable")

    def test_environment_flag_accepts_false_values(self):
        with patch.dict("os.environ", {"DIARIZATION_ENABLED": "off"}):
            self.assertFalse(env_flag("DIARIZATION_ENABLED", default=True))

    def test_torchcodec_warnings_are_suppressed(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("default")
            suppress_torchcodec_warnings()
            warnings.warn(
                "torchcodec is not installed correctly so built-in audio decoding will fail.\n"
                "Solutions are: use audio preloaded in-memory"
            )
            self.assertEqual(captured, [])

    def test_live_tracker_keeps_identity_when_window_labels_swap(self):
        tracker = LiveSpeakerTracker()
        first = tracker.update([
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ], window_start_seconds=0.0)
        second = tracker.update([
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_99"},
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_42"},
        ], window_start_seconds=1.0)

        self.assertEqual(first[0]["speaker"], second[0]["speaker"])
        self.assertEqual(first[1]["speaker"], second[1]["speaker"])
        self.assertNotEqual(second[0]["speaker"], second[1]["speaker"])

    def test_live_tracker_uses_absolute_timeline(self):
        tracker = LiveSpeakerTracker()
        tracker.update([
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ], window_start_seconds=10.0)

        self.assertEqual(tracker.speaker_at(10.5), "live-speaker-1")
        self.assertEqual(tracker.speaker_at(12.5), "live-speaker-2")

    def test_empty_diarization_pass_keeps_last_confirmed_speaker(self):
        tracker = LiveSpeakerTracker()
        confirmed = tracker.update([
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        ], window_start_seconds=0.0)

        self.assertEqual(tracker.update([], 3.0), confirmed)
        self.assertEqual(tracker.speaker_at(3.2), "live-speaker-1")

    def test_returning_questioner_reuses_identity_and_color_key(self):
        tracker = LiveSpeakerTracker()
        voice_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        voice_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        first = tracker.update(
            [{"start": 0.0, "end": 2.0, "speaker": "LOCAL_A"}],
            0.0,
            {"LOCAL_A": voice_a},
        )
        second = tracker.update(
            [{"start": 0.0, "end": 2.0, "speaker": "LOCAL_B"}],
            10.0,
            {"LOCAL_B": voice_b},
        )
        returning = tracker.update(
            [{"start": 0.0, "end": 2.0, "speaker": "LOCAL_C"}],
            20.0,
            {"LOCAL_C": voice_a},
        )

        self.assertEqual(first[0]["speaker"], returning[0]["speaker"])
        self.assertNotEqual(first[0]["speaker"], second[0]["speaker"])

    def test_session_supports_ten_stable_speakers(self):
        tracker = LiveSpeakerTracker()
        labels = []
        for index in range(10):
            result = tracker.update(
                [{"start": 0.0, "end": 1.0, "speaker": f"LOCAL_{index}"}],
                float(index * 10),
            )
            labels.append(result[0]["speaker"])

        self.assertEqual(len(set(labels)), 10)

    def test_optional_explicit_cap_stops_at_ten_speakers(self):
        tracker = LiveSpeakerTracker(max_session_speakers=10)
        labels = []
        for index in range(11):
            result = tracker.update(
                [{"start": 0.0, "end": 1.0, "speaker": f"LOCAL_{index}"}],
                float(index * 10),
            )
            labels.append(result[0]["speaker"])

        self.assertEqual(len(set(labels)), 10)
        self.assertEqual(labels[-1], labels[-2])

    def test_open_world_session_supports_twenty_speakers(self):
        tracker = LiveSpeakerTracker()
        labels = []
        for index in range(20):
            result = tracker.update(
                [{"start": 0.0, "end": 1.0, "speaker": f"LOCAL_{index}"}],
                float(index * 10),
            )
            labels.append(result[0]["speaker"])
        self.assertEqual(len(set(labels)), 20)

    def test_three_speakers_receive_three_distinct_colors(self):
        registry = SpeakerRegistry()
        assignments = [
            registry.register_label(f"live-speaker-{index}")
            for index in range(1, 4)
        ]
        self.assertEqual(len({speaker_id for speaker_id, _ in assignments}), 3)
        self.assertEqual(len({color for _, color in assignments}), 3)

    def test_ten_speakers_receive_ten_distinct_colors(self):
        registry = SpeakerRegistry()
        colors = {
            registry.register_label(f"live-speaker-{index}")[1]
            for index in range(1, 11)
        }
        self.assertEqual(len(colors), 10)

    def test_twenty_speakers_receive_twenty_distinct_colors(self):
        registry = SpeakerRegistry()
        colors = {
            registry.register_label(f"live-speaker-{index}")[1]
            for index in range(1, 21)
        }
        self.assertEqual(len(colors), 20)

    def test_authoritative_embedding_match_reuses_returning_identity(self):
        tracker = LiveSpeakerTracker()
        voice_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        voice_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        tracker.update(
            [{"start": 0, "end": 2, "speaker": "A"}],
            0.0,
            {"A": voice_a},
        )
        tracker.update(
            [{"start": 0, "end": 2, "speaker": "B"}],
            10.0,
            {"B": voice_b},
        )
        tracker.update(
            [{"start": 0, "end": 2, "speaker": "A2"}],
            20.0,
            {"A2": voice_a},
        )

        self.assertEqual(tracker.speaker_at(20.5), "live-speaker-1")

    def test_different_voices_in_same_window_are_never_merged(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [
                {"start": 0, "end": 2, "speaker": "A"},
                {"start": 2, "end": 4, "speaker": "B"},
            ],
            0.0,
            {
                "A": np.array([1.0, 0.0], dtype=np.float32),
                "B": np.array([0.6, 0.8], dtype=np.float32),
            },
        )
        self.assertEqual(tracker.speaker_at(1.0), "live-speaker-1")
        self.assertEqual(tracker.speaker_at(3.0), "live-speaker-2")

    def test_uncertain_voice_similarity_prefers_a_visible_change(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [{"start": 0, "end": 2, "speaker": "ANCHOR"}],
            0.0,
            {"ANCHOR": np.array([1.0, 0.0], dtype=np.float32)},
        )
        result = tracker.update(
            [{"start": 0, "end": 2, "speaker": "REPORTER"}],
            10.0,
            {"REPORTER": np.array([0.6, 0.8], dtype=np.float32)},
        )
        self.assertEqual(result[0]["speaker"], "live-speaker-2")

    def test_overlapping_window_reuses_continuing_voice_when_embedding_is_weak(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [{"start": 0, "end": 8, "speaker": "LOCAL_A"}],
            0.0,
            {"LOCAL_A": np.array([1.0, 0.0], dtype=np.float32)},
        )
        result = tracker.update(
            [{"start": 0, "end": 8, "speaker": "RENAMED_A"}],
            2.0,
            {"RENAMED_A": np.array([0.5, 0.866], dtype=np.float32)},
        )
        self.assertEqual(result[0]["speaker"], "live-speaker-1")

    def test_one_noisy_single_cluster_embedding_keeps_identity(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [{"start": 0, "end": 8, "speaker": "ANCHOR"}],
            0.0,
            {"ANCHOR": np.array([1.0, 0.0], dtype=np.float32)},
        )
        result = tracker.update(
            [{"start": 0, "end": 6, "speaker": "DRIFTED"}],
            2.0,
            {"DRIFTED": np.array([0.42, 0.907], dtype=np.float32)},
        )
        self.assertEqual(result[0]["speaker"], "live-speaker-1")

    def test_consistent_single_cluster_voice_change_is_detected(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [{"start": 0, "end": 8, "speaker": "ANCHOR"}],
            0.0,
            {"ANCHOR": np.array([1.0, 0.0], dtype=np.float32)},
        )
        first = tracker.update(
            [{"start": 0, "end": 6, "speaker": "GUEST_A"}],
            2.0,
            {"GUEST_A": np.array([0.0, 1.0], dtype=np.float32)},
        )
        second = tracker.update(
            [{"start": 0, "end": 6, "speaker": "GUEST_B"}],
            2.75,
            {"GUEST_B": np.array([0.02, 0.9998], dtype=np.float32)},
        )
        self.assertEqual(first[0]["speaker"], "live-speaker-1")
        self.assertEqual(second[0]["speaker"], "live-speaker-2")

    def test_detected_two_cluster_boundary_creates_new_identity(self):
        tracker = LiveSpeakerTracker()
        tracker.update(
            [{"start": 0, "end": 8, "speaker": "ANCHOR"}],
            0.0,
            {"ANCHOR": np.array([1.0, 0.0], dtype=np.float32)},
        )
        boundary = tracker.update(
            [
                {"start": 0, "end": 3, "speaker": "ANCHOR_LOCAL"},
                {"start": 3, "end": 6, "speaker": "GUEST_LOCAL"},
            ],
            2.0,
            {
                "ANCHOR_LOCAL": np.array([1.0, 0.0], dtype=np.float32),
                "GUEST_LOCAL": np.array([0.0, 1.0], dtype=np.float32),
            },
        )
        self.assertEqual(boundary[0]["speaker"], "live-speaker-1")
        self.assertEqual(boundary[1]["speaker"], "live-speaker-2")

    def test_micro_turn_between_same_speaker_is_absorbed(self):
        stabilized = DiarizationContext._stabilize_micro_turns([
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.0, "end": 2.15, "speaker": "B"},
            {"start": 2.15, "end": 4.0, "speaker": "A"},
        ])
        self.assertEqual(stabilized, [
            {"start": 0.0, "end": 4.0, "speaker": "A"},
        ])

    def test_real_short_turn_is_retained(self):
        stabilized = DiarizationContext._stabilize_micro_turns([
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.0, "end": 2.9, "speaker": "B"},
            {"start": 2.9, "end": 4.0, "speaker": "A"},
        ])
        self.assertEqual([item["speaker"] for item in stabilized], ["A", "B", "A"])


if __name__ == "__main__":
    unittest.main()
