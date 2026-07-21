import unittest
from unittest.mock import patch

from YouTubeLiveTranscribe import validate_runtime_environment


class WorkflowIntegrationTests(unittest.TestCase):
    def test_runtime_environment_reports_missing_ffmpeg(self):
        with patch("YouTubeLiveTranscribe.shutil.which", side_effect=lambda name: None if name == "ffmpeg" else "/tmp/yt-dlp"):
            status = validate_runtime_environment()

        self.assertFalse(status["ffmpeg_available"])
        self.assertIn("ffmpeg", " ".join(status["issues"]).lower())


if __name__ == "__main__":
    unittest.main()
