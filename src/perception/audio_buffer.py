"""
FencerAI Audio Buffer
====================
Version: 1.0 | Last Updated: 2026-03-27

Thread-safe circular buffer for storing audio samples.
Used for audio event detection with blade sounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import threading
from collections import deque


# =============================================================================
# Audio Buffer Entry
# =============================================================================

@dataclass
class AudioBufferEntry:
    """Single entry in the audio buffer."""
    timestamp: float  # Time in seconds from video start
    sample_rate: int  # Sample rate in Hz
    samples: np.ndarray  # Audio samples
    channels: int  # Number of channels (1=mono, 2=stereo)


# =============================================================================
# AudioBuffer
# =============================================================================

class AudioBuffer:
    """
    Thread-safe circular buffer for audio samples.

    Used to store audio samples for later processing and event detection.
    Supports frame-aligned audio extraction for sync with video frames.

    Args:
        max_size: Maximum number of audio entries to store
        sample_rate: Expected sample rate in Hz (default 44100)
    """

    def __init__(self, max_size: int = 1000, sample_rate: int = 44100) -> None:
        self.max_size = max_size
        self.sample_rate = sample_rate
        self._buffer: deque[AudioBufferEntry] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Return current number of entries in buffer."""
        with self._lock:
            return len(self._buffer)

    def append(self, timestamp: float, samples: np.ndarray, channels: int = 1) -> None:
        """
        Append audio samples to the buffer.

        Args:
            timestamp: Time in seconds from video start
            samples: Audio samples as numpy array
            channels: Number of channels (1=mono, 2=stereo)
        """
        entry = AudioBufferEntry(
            timestamp=timestamp,
            sample_rate=self.sample_rate,
            samples=samples.copy(),
            channels=channels,
        )
        with self._lock:
            self._buffer.append(entry)

    def get_samples_in_range(
        self,
        start_ts: float,
        end_ts: float,
    ) -> List[np.ndarray]:
        """
        Get all audio samples within a timestamp range.

        Args:
            start_ts: Start timestamp in seconds
            end_ts: End timestamp in seconds

        Returns:
            List of sample arrays within the range
        """
        with self._lock:
            return [
                entry.samples
                for entry in self._buffer
                if start_ts <= entry.timestamp <= end_ts
            ]

    def get_latest(self, num_samples: int = 1024) -> Optional[np.ndarray]:
        """
        Get the latest N samples from the buffer.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Latest samples as concatenated array, or None if buffer empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None

            # Collect samples from the end
            samples_list = []
            total_samples = 0
            for entry in reversed(self._buffer):
                samples_list.append(entry.samples)
                total_samples += len(entry.samples)
                if total_samples >= num_samples:
                    break

            # Concatenate and return last num_samples
            if samples_list:
                combined = np.concatenate(list(reversed(samples_list)))
                return combined[-num_samples:] if len(combined) >= num_samples else combined
            return None

    def compute_rms(self, num_samples: int = 1024) -> Optional[float]:
        """
        Compute RMS (Root Mean Square) amplitude of latest samples.

        Args:
            num_samples: Number of samples to analyze

        Returns:
            RMS amplitude, or None if buffer empty
        """
        samples = self.get_latest(num_samples)
        if samples is None:
            return None

        # Compute RMS
        return float(np.sqrt(np.mean(samples ** 2)))

    def compute_energy(self, num_samples: int = 1024) -> Optional[float]:
        """
        Compute energy of latest samples.

        Args:
            num_samples: Number of samples to analyze

        Returns:
            Total energy, or None if buffer empty
        """
        samples = self.get_latest(num_samples)
        if samples is None:
            return None

        return float(np.sum(samples ** 2))

    def clear(self) -> None:
        """Clear all samples from buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def total_samples(self) -> int:
        """Return total number of samples in buffer."""
        with self._lock:
            return sum(len(entry.samples) for entry in self._buffer)
