import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

from speaker_diarization import (
    DiarizationContext,
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
        self.assertEqual(speaker_id, "speaker-1")

    def test_select_speaker_for_time_uses_segment_containing_target(self):
        context = DiarizationContext(enabled=False)
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        self.assertEqual(context.select_speaker_for_time(segments, 2.5), "SPEAKER_01")

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


if __name__ == "__main__":
    unittest.main()
