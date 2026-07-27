import queue
import tempfile
import unittest
from pathlib import Path

from audio_spool import AudioSpool


class AudioSpoolTests(unittest.TestCase):
    def test_chunks_are_fifo_and_persist_until_acknowledged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spool = AudioSpool(Path(tmp_dir) / "session")
            spool.put(b"first")
            spool.put(b"second")

            first = spool.get_nowait()
            second = spool.get_nowait()

            self.assertEqual(first.data, b"first")
            self.assertEqual(second.data, b"second")
            self.assertTrue(first.path.exists())
            self.assertTrue(second.path.exists())

            spool.ack([first, second])
            self.assertFalse(first.path.exists())
            self.assertFalse(second.path.exists())
            spool.close()
            self.assertFalse(spool.directory.exists())

    def test_empty_spool_raises_queue_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spool = AudioSpool(Path(tmp_dir) / "session")
            with self.assertRaises(queue.Empty):
                spool.get_nowait()

    def test_late_capture_write_after_close_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spool = AudioSpool(Path(tmp_dir) / "session")
            spool.close()
            spool.put(b"late audio")
            self.assertFalse(spool.directory.exists())
            self.assertTrue(spool.empty())


if __name__ == "__main__":
    unittest.main()
