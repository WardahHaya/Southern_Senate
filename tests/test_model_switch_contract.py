import unittest
from pathlib import Path


class ModelSwitchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(__file__).resolve().parents[1]
        cls.source = (project / "YouTubeLiveTranscribeUI.py").read_text(
            encoding="utf-8"
        )
        cls.whisper_source = (project / "whisper_live_stream.py").read_text(
            encoding="utf-8"
        )

    def test_all_switch_paths_use_memory_safe_loader(self):
        self.assertIn("def _release_loaded_model():", self.source)
        self.assertIn("torch.cuda.empty_cache()", self.source)
        self.assertGreaterEqual(self.source.count("_begin_model_load("), 3)
        self.assertIn("unload_ctranslate()", self.source)

    def test_unsupported_24b_model_is_not_offered(self):
        catalog = self.source.split('@app.route("/models")', 1)[1].split(
            '@app.route("/load_model"', 1
        )[0]
        self.assertNotIn("Voxtral-Small-24B", catalog)

    def test_faster_whisper_is_a_separate_stable_backend(self):
        self.assertIn(
            'FASTER_WHISPER_MODEL_ID = "faster-whisper/large-v3-turbo"',
            self.source,
        )
        self.assertIn("WhisperModel(", self.source)
        self.assertIn("compute_type=(", self.source)
        self.assertIn("from whisper_live_stream import (", self.source)
        self.assertIn("transcribe_whisper_window,", self.source)
        self.assertIn("remove_repeated_window_prefix,", self.source)
        self.assertIn("word_timestamps=True", self.whisper_source)
        self.assertIn("resolve_speaker(start, end)", self.whisper_source)
        self.assertIn('"speaker_alignment_delay": 0.0', self.source)
        self.assertIn(
            "if _is_faster_whisper_model(_current_model_id):",
            self.source,
        )

    def test_model_can_be_explicitly_unloaded(self):
        self.assertIn('@app.route("/unload_model"', self.source)
        self.assertIn("_current_model_id = None", self.source)
        self.assertIn("freed_vram_gb", self.source)

    def test_duplicate_model_load_requests_are_coalesced(self):
        self.assertIn(
            "_current_model_id == model_id and not _model_ready.is_set()",
            self.source,
        )

    def test_stale_loader_cannot_overwrite_selected_backend(self):
        self.assertGreaterEqual(
            self.source.count("requested_model_id != _current_model_id"),
            3,
        )

    def test_duplicate_app_launch_is_rejected_before_cuda_load(self):
        main_block = self.source.split('if __name__ == "__main__":', 1)[1]
        self.assertLess(
            main_block.index("_port_is_available("),
            main_block.index("_start_background_services()"),
        )
        self.assertIn("refusing a duplicate", main_block)

    def test_live_pipeline_has_bounded_catchup_and_committed_diarization_window(self):
        self.assertIn("8 if queued_items >= 7", self.source)
        self.assertIn("else 6 if queued_items >= 5", self.source)
        self.assertIn("else 2 if queued_items >= 1", self.source)
        self.assertIn("else 1", self.source)
        self.assertIn("diarization_window_seconds = 10.0", self.source)
        self.assertIn("diarization_run_interval = 0.75", self.source)
        self.assertIn("embed_speakers_pcm(", self.source)
        self.assertIn("diarization_warmup_seconds = 2.0", self.source)
        self.assertIn("LiveTurnCommitter(commit_lag=0.5", self.source)
        self.assertIn("speaker_alignment_delay_seconds = 0.75", self.source)
        self.assertIn("max_audio_seconds=90.0", self.source)
        self.assertIn("Rolling native context at 90s", self.source)
        self.assertIn("max_speakers=4", self.source)
        self.assertIn('"provisional": True', self.source)
        self.assertIn("for unit_index, audio_item in enumerate(audio_items)", self.source)
        self.assertIn("word_start = round(", self.source)
        self.assertIn('len(attribution_units[-1]["words"]) + len(unit_words) <= 16', self.source)
        self.assertIn("chunk_seconds = 1.0", self.source)
        self.assertIn("tail_seconds = max(", self.source)
        self.assertIn("inference_max_new_tokens = min(", self.source)
        self.assertIn("128,", self.source)
        self.assertIn("effective_chunk_seconds * 12.0", self.source)
        self.assertIn("diarization_jobs.get_nowait()", self.source)
        proxy = self.source.split('def proxy_video():', 1)[1].split(
            '@app.route("/start"', 1
        )[0]
        self.assertNotIn('"-an"', proxy)
        self.assertIn('"-c:a", "aac"', proxy)


if __name__ == "__main__":
    unittest.main()
