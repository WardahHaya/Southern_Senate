"""Durable, FIFO storage for captured live-audio chunks."""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpoolChunk:
    sequence: int
    path: Path
    data: bytes


class AudioSpool:
    """Queue-compatible audio buffer whose payloads survive process failures."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[int, Path]] = queue.Queue()
        self._sequence = 0
        self._write_lock = threading.Lock()
        self._closed = False

    def put(self, data: bytes, timeout=None) -> None:
        del timeout  # queue-compatible signature; disk capacity is the bound.
        with self._write_lock:
            # Capture can finish a read at the same instant a session stops.
            # Dropping that final post-stop write is correct and prevents the
            # writer from recreating or addressing a closed session directory.
            if self._closed:
                return
            sequence = self._sequence
            self._sequence += 1
            final_path = self.directory / f"{sequence:012d}.pcm"
            temp_path = self.directory / f"{sequence:012d}.tmp"
            with temp_path.open("wb") as handle:
                handle.write(data)
                handle.flush()
            os.replace(temp_path, final_path)
            self._queue.put((sequence, final_path))

    def _read(self, item: tuple[int, Path]) -> SpoolChunk:
        sequence, path = item
        return SpoolChunk(sequence=sequence, path=path, data=path.read_bytes())

    def get(self, timeout=None) -> SpoolChunk:
        return self._read(self._queue.get(timeout=timeout))

    def get_nowait(self) -> SpoolChunk:
        return self._read(self._queue.get_nowait())

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def ack(self, chunks: Iterable[SpoolChunk]) -> None:
        for chunk in chunks:
            try:
                chunk.path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        """Remove only an empty session directory; retain unacknowledged audio."""
        with self._write_lock:
            self._closed = True
            try:
                self.directory.rmdir()
            except OSError:
                pass
