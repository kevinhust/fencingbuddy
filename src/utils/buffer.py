"""
FencerAI Timestamped Buffer
============================
Version: 1.0 | Last Updated: 2026-03-27

Thread-safe circular buffer for storing timestamped frame data and audio samples.
Supports frame dropping when buffer exceeds max_size and audio-video sync detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Iterator
from collections import deque
import threading
import numpy as np


# =============================================================================
# Buffer Entry
# =============================================================================

@dataclass
class BufferEntry:
    """Single entry in the timestamped buffer."""
    frame_id: int
    timestamp: float
    frame_data: Any  # Can be dict, FrameData, or any frame representation
    audio_sample: Optional[np.ndarray] = None


# =============================================================================
# TimestampedBuffer
# =============================================================================

class TimestampedBuffer:
    """
    Thread-safe circular buffer for timestamped frame data and audio samples.

    Features:
    - Thread-safe with proper locking
    - Frame dropping when buffer exceeds max_size (circular buffer behavior)
    - Timestamp-based frame retrieval
    - Audio-video sync detection via cross-correlation

    Args:
        max_size: Maximum number of frames to store (default: 1000)
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque[BufferEntry] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Return current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[BufferEntry]:
        """Iterate over buffer entries."""
        with self._lock:
            return iter(list(self._buffer))

    def append(
        self,
        frame_id: int,
        timestamp: float,
        frame_data: Any,
        audio_sample: Optional[np.ndarray] = None
    ) -> None:
        """
        Append a frame to the buffer.

        If buffer is full, oldest frame is dropped (circular buffer behavior).

        Args:
            frame_id: Sequential frame index
            timestamp: Time in seconds from video start
            frame_data: Frame data (dict, FrameData, etc.)
            audio_sample: Optional audio sample array
        """
        entry = BufferEntry(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_data=frame_data,
            audio_sample=audio_sample
        )
        with self._lock:
            self._buffer.append(entry)

    def get_frame_range(
        self,
        start_ts: float,
        end_ts: float
    ) -> List[BufferEntry]:
        """
        Get frames within a timestamp range.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            List of BufferEntry objects within the range
        """
        with self._lock:
            return [
                entry for entry in self._buffer
                if start_ts <= entry.timestamp <= end_ts
            ]

    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._buffer.clear()

    def detect_sync(self) -> Optional[dict]:
        """
        Detect audio-video sync using cross-correlation.

        Analyzes audio samples across frames to detect synchronization issues.
        Requires at least 3 frames with audio samples.

        Returns:
            Dictionary with sync info (delay_ms, confidence) or None if insufficient data
        """
        with self._lock:
            # Collect frames with audio samples
            frames_with_audio = [
                (entry.timestamp, entry.audio_sample)
                for entry in self._buffer
                if entry.audio_sample is not None
            ]

            if len(frames_with_audio) < 3:
                return None

            try:
                # Extract audio samples
                timestamps = np.array([ts for ts, _ in frames_with_audio])
                audio_samples = [audio for _, audio in frames_with_audio]

                # Simple sync detection: check if audio amplitude correlates across frames
                # This is a simplified version - real implementation would use
                # cross-correlation between consecutive frames
                if len(audio_samples) < 2:
                    return None

                # Compute correlation between consecutive audio samples
                correlations = []
                for i in range(len(audio_samples) - 1):
                    audio1 = audio_samples[i].flatten()
                    audio2 = audio_samples[i + 1].flatten()

                    # Ensure same length
                    min_len = min(len(audio1), len(audio2))
                    if min_len < 10:  # Too short for correlation
                        continue

                    audio1 = audio1[:min_len]
                    audio2 = audio2[:min_len]

                    # Compute normalized cross-correlation
                    corr = np.corrcoef(audio1, audio2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

                if not correlations:
                    return None

                avg_correlation = np.mean(correlations)

                # Compute expected vs actual time difference
                time_diffs = np.diff(timestamps)
                avg_time_diff = np.mean(time_diffs)

                # Estimate delay based on correlation
                # Lower correlation suggests audio-video mismatch
                confidence = max(0.0, min(1.0, (avg_correlation + 1) / 2))

                return {
                    "delay_ms": 0.0,  # Placeholder - real implementation would compute this
                    "confidence": confidence,
                    "avg_correlation": float(avg_correlation),
                    "num_samples": len(correlations)
                }

            except Exception:
                return None
