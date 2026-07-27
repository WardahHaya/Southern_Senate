import threading
import unittest

import numpy as np

from voxtral_live_stream import VoxtralPCMWindowBuffer


def pcm16(values):
    return np.asarray(values, dtype=np.int16).tobytes()


class VoxtralPCMWindowBufferTests(unittest.TestCase):
    def make_buffer(self):
        return VoxtralPCMWindowBuffer(
            first_samples=9,
            chunk_samples=5,
            first_mel_frames=5,
            audio_length_per_token=2,
            hop_length=2,
            win_length=2,
        )

    def test_arbitrary_arrivals_produce_exact_overlapping_windows(self):
        buffer = self.make_buffer()
        stop = threading.Event()
        buffer.append_pcm16(pcm16(range(7)))
        buffer.append_pcm16(pcm16(range(7, 16)))
        buffer.close()

        windows = list(buffer.windows(stop))
        self.assertEqual([is_first for _, is_first in windows], [True, False])
        np.testing.assert_allclose(
            windows[0][0] * 32768.0,
            np.arange(0, 9),
        )
        # first_mel_frames * hop - win/2 = 5*2-1 = sample 9
        np.testing.assert_allclose(
            windows[1][0] * 32768.0,
            np.arange(9, 14),
        )

    def test_close_without_enough_audio_ends_cleanly(self):
        buffer = self.make_buffer()
        stop = threading.Event()
        buffer.append_pcm16(pcm16(range(4)))
        buffer.close()
        self.assertEqual(list(buffer.windows(stop)), [])

    def test_append_after_close_is_rejected(self):
        buffer = self.make_buffer()
        buffer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            buffer.append_pcm16(pcm16([1, 2]))

    def test_bounded_buffer_stops_accepting_audio_after_stop(self):
        buffer = self.make_buffer()
        buffer.max_buffered_samples = 4
        stop = threading.Event()
        buffer.append_pcm16(pcm16(range(4)), stop)
        stop.set()
        self.assertFalse(buffer.append_pcm16(pcm16([5]), stop))


if __name__ == "__main__":
    unittest.main()
