import unittest
from pathlib import Path


class UiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (
            Path(__file__).resolve().parents[1] / "templates" / "index.html"
        ).read_text(encoding="utf-8")

    def test_backend_integration_hooks_are_preserved(self):
        for element_id in (
            "mainBtn",
            "urlInput",
            "transcript",
            "sourceStatus",
            "transcriptionStatus",
            "backlogStatus",
            "rtfStatus",
            "consoleBox",
        ):
            self.assertIn(f'id="{element_id}"', self.html)

        for endpoint in (
            "/status", "/models", "/load_model", "/unload_model",
            "/start", "/stop", "/stream", "/metrics",
        ):
            self.assertIn(endpoint, self.html)

    def test_operator_experience_controls_exist(self):
        self.assertIn('id="videoBox"', self.html)
        self.assertIn("function updateVideoBox(", self.html)
        self.assertIn('id="consoleBox"', self.html)
        self.assertIn("function startConsole()", self.html)
        self.assertIn('id="unloadModelBtn"', self.html)
        self.assertIn("function unloadModel()", self.html)

    def test_live_processing_telemetry_is_visible(self):
        for status_id in ("sourceStatus", "transcriptionStatus", "backlogStatus", "rtfStatus"):
            self.assertIn(f'id="{status_id}"', self.html)
        self.assertIn("live.source_idle_seconds", self.html)
        self.assertIn("live.realtime_factor", self.html)

    def test_authenticated_video_and_retroactive_speaker_updates_exist(self):
        self.assertIn("/proxy_video?url=", self.html)
        self.assertIn("cookie_browser=", self.html)
        self.assertIn("msg.type === 'speaker_update'", self.html)
        self.assertIn(
            "updateActiveSpeaker(msg.speaker_id, msg.speaker_color);",
            self.html,
        )
        self.assertIn("function updateEntrySpeaker(", self.html)
        self.assertIn('id="liveVideo"', self.html)
        self.assertIn("autoplay muted controls playsinline", self.html)
        self.assertIn("object-fit:contain", self.html)
        self.assertIn('id="currentSpeakerIndicator"', self.html)
        self.assertIn('id="videoSpeakerOverlay"', self.html)
        self.assertIn("function updateActiveSpeaker(", self.html)
        self.assertIn("msg.type === 'speaker_turn'", self.html)
        script = self.html.split("<script>", 1)[1].split("</script>", 1)[0]
        self.assertNotIn("@keyframes", script)
        self.assertIn("function jumpPreviewToLive()", self.html)


if __name__ == "__main__":
    unittest.main()
