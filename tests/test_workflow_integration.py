import unittest
from types import SimpleNamespace
from unittest.mock import patch

from YouTubeLiveTranscribe import (
    get_audio_stream_url,
    get_video_stream_url,
    validate_runtime_environment,
)


class WorkflowIntegrationTests(unittest.TestCase):
    def test_runtime_environment_reports_missing_ffmpeg(self):
        with patch("YouTubeLiveTranscribe.shutil.which", side_effect=lambda name: None if name == "ffmpeg" else "/tmp/yt-dlp"):
            status = validate_runtime_environment()

        self.assertFalse(status["ffmpeg_available"])
        self.assertIn("ffmpeg", " ".join(status["issues"]).lower())

    def test_runtime_environment_accepts_python_yt_dlp_module(self):
        with (
            patch("YouTubeLiveTranscribe.shutil.which", return_value="/tmp/ffmpeg"),
            patch("YouTubeLiveTranscribe.importlib.util.find_spec", return_value=object()),
        ):
            status = validate_runtime_environment()

        self.assertTrue(status["yt_dlp_available"])
        self.assertEqual(status["issues"], [])

    def test_stream_resolution_uses_only_explicit_browser_cookies(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="https://example.test/audio\n",
            stderr="",
        )
        with (
            patch("YouTubeLiveTranscribe._yt_dlp_base_cmd", return_value=["yt-dlp"]),
            patch("YouTubeLiveTranscribe._find_deno", return_value=r"C:\tools\deno.exe"),
            patch("YouTubeLiveTranscribe.subprocess.run", return_value=completed) as run,
        ):
            url = get_audio_stream_url(
                "https://youtube.com/watch?v=test",
                cookie_browser="edge",
            )

        self.assertEqual(url, "https://example.test/audio")
        command = run.call_args.args[0]
        self.assertIn("--cookies-from-browser", command)
        self.assertEqual(command[command.index("--cookies-from-browser") + 1], "edge")
        self.assertEqual(
            command[command.index("--js-runtimes") + 1],
            r"deno:C:\tools\deno.exe",
        )

    def test_video_resolution_uses_muxed_720p_format(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="https://example.test/video\n",
            stderr="",
        )
        with (
            patch("YouTubeLiveTranscribe._yt_dlp_base_cmd", return_value=["yt-dlp"]),
            patch("YouTubeLiveTranscribe._find_deno", return_value=r"C:\tools\deno.exe"),
            patch("YouTubeLiveTranscribe.subprocess.run", return_value=completed) as run,
        ):
            url = get_video_stream_url(
                "https://youtube.com/watch?v=test",
                cookie_browser="firefox",
            )

        self.assertEqual(url, "https://example.test/video")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-f") + 1], "best[height<=720]/best")
        self.assertEqual(
            command[command.index("--cookies-from-browser") + 1],
            "firefox",
        )

    def test_bot_gate_explains_how_to_retry(self):
        completed = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Sign in to confirm you’re not a bot",
        )
        with (
            patch("YouTubeLiveTranscribe._yt_dlp_base_cmd", return_value=["yt-dlp"]),
            patch("YouTubeLiveTranscribe.subprocess.run", return_value=completed),
        ):
            with self.assertRaisesRegex(RuntimeError, "Select your signed-in browser"):
                get_audio_stream_url("https://youtube.com/watch?v=test")

    def test_locked_chrome_cookie_database_has_actionable_error(self):
        completed = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ERROR: Could not copy Chrome cookie database.",
        )
        with (
            patch("YouTubeLiveTranscribe._yt_dlp_base_cmd", return_value=["yt-dlp"]),
            patch("YouTubeLiveTranscribe.subprocess.run", return_value=completed),
        ):
            with self.assertRaisesRegex(RuntimeError, "Fully exit Chrome"):
                get_audio_stream_url(
                    "https://youtube.com/watch?v=test",
                    cookie_browser="chrome",
                )


if __name__ == "__main__":
    unittest.main()
