"""
FencerAI Frame Buffer Module
============================
Version: 2.0 | Last Updated: 2026-04-02

Thread-safe circular buffer for live video frames.
Provides smooth frame delivery even with variable capture rates.

Features:
- Lock-free design for minimal latency
- Configurable buffer size
- Frame dropping when processing can't keep up
- Timestamp tracking for each frame
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TimestampedFrame:
    """Frame with timestamp for latency tracking."""
    frame: np.ndarray
    timestamp: float  # Monotonic time when captured
    frame_id: int
    dropped: bool = False  # True if this frame replaced an unprocessed frame


class FrameBuffer:
    """
    Thread-safe circular buffer for video frames.

    Features:
    - Stores up to max_size frames
    - Automatically drops oldest frames when full
    - Thread-safe get/put operations
    - Tracks dropped frames for monitoring

    Example:
        buffer = FrameBuffer(max_size=10)

        # Producer thread (capture)
        buffer.put(frame)

        # Consumer thread (processing)
        frame = buffer.get()
        if frame is not None:
            process(frame)
    """

    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self._buffer: deque[TimestampedFrame] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._frame_id_counter = 0
        self._total_dropped = 0
        self._total_processed = 0

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> TimestampedFrame:
        """
        Add a frame to the buffer.

        If the buffer is full, the oldest unprocessed frame is dropped.

        Args:
            frame: Video frame (H, W, 3) in BGR format
            timestamp: Optional timestamp (defaults to monotonic time)

        Returns:
            The TimestampedFrame that was added
        """
        if timestamp is None:
            timestamp = time.monotonic()

        with self._lock:
            # Check if we're about to drop a frame
            dropped = False
            if len(self._buffer) >= self.max_size:
                dropped = True
                self._total_dropped += 1

            self._frame_id_counter += 1
            ts_frame = TimestampedFrame(
                frame=frame.copy(),  # Copy to avoid reference issues
                timestamp=timestamp,
                frame_id=self._frame_id_counter,
                dropped=dropped,
            )
            self._buffer.append(ts_frame)
            return ts_frame

    def get(self) -> Optional[TimestampedFrame]:
        """
        Get the oldest frame from the buffer.

        Returns:
            The oldest TimestampedFrame, or None if buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None

            self._total_processed += 1
            return self._buffer.popleft()

    def peek(self) -> Optional[TimestampedFrame]:
        """
        Look at the oldest frame without removing it.

        Returns:
            The oldest TimestampedFrame, or None if empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[0]

    def get_latest(self) -> Optional[TimestampedFrame]:
        """
        Get the newest frame from the buffer.

        Useful when you only care about the latest frame.

        Returns:
            The newest TimestampedFrame, or None if empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        """Get current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.max_size

    @property
    def stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.max_size,
                "total_processed": self._total_processed,
                "total_dropped": self._total_dropped,
                "drop_rate": (
                    self._total_dropped / self._total_processed
                    if self._total_processed > 0
                    else 0.0
                ),
            }

    def reset_stats(self) -> None:
        """Reset dropped/processed counters."""
        with self._lock:
            self._total_dropped = 0
            self._total_processed = 0

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(size={self.size}/{self.max_size}, "
            f"dropped={self._total_dropped}, "
            f"processed={self._total_processed})"
        )


class SyncedFrameBuffer(FrameBuffer):
    """
    Frame buffer with audio sync capability.

    Extends FrameBuffer with audio timestamp synchronization
    for blade touch detection alignment.
    """

    def __init__(self, max_size: int = 30, audio_offset_ms: float = 0.0):
        """
        Initialize synced frame buffer.

        Args:
            max_size: Maximum buffer size
            audio_offset_ms: Audio delay offset in milliseconds
        """
        super().__init__(max_size=max_size)
        self._audio_offset_s = audio_offset_ms / 1000.0

    def sync_to_audio(self, audio_timestamp: float) -> Optional[TimestampedFrame]:
        """
        Get frame synced to audio timestamp.

        Args:
            audio_timestamp: Audio event timestamp

        Returns:
            Frame closest to audio timestamp, or None
        """
        target_time = audio_timestamp - self._audio_offset_s

        with self._lock:
            if len(self._buffer) == 0:
                return None

            # Find closest frame
            closest = None
            min_diff = float('inf')
            for ts_frame in self._buffer:
                diff = abs(ts_frame.timestamp - target_time)
                if diff < min_diff:
                    min_diff = diff
                    closest = ts_frame

            return closest

    def set_audio_offset(self, offset_ms: float) -> None:
        """
        Set audio offset in milliseconds.

        Args:
            offset_ms: Audio delay offset
        """
        self._audio_offset_s = offset_ms / 1000.0
