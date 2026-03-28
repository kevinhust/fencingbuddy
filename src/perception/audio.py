"""
FencerAI Audio Event Detector
=============================
Version: 1.0 | Last Updated: 2026-03-27

Audio-based event detection for fencing actions.
Detects blade touches and parries using audio amplitude analysis.

Architecture (per AD7):
    - Audio touch flag in feature vector index 100
    - Uses amplitude threshold for blade contact detection
    - Short-term energy analysis for transient detection
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np

from src.perception.audio_buffer import AudioBuffer, AudioBufferEntry
from src.utils.schemas import AudioEvent
from src.utils.constants import (
    BLADE_TOUCH,
    PARRY_BEAT,
    FOOTSTEP,
)


# =============================================================================
# Audio Event Detector
# =============================================================================

class AudioDetector:
    """
    Audio event detector for fencing actions.

    Uses amplitude and energy analysis to detect:
    - Blade touches (high-frequency transient)
    - Parry beats (rhythmic patterns)
    - Footsteps (low-frequency patterns)

    Args:
        sample_rate: Audio sample rate in Hz (default 44100)
        energy_threshold: Threshold for energy-based detection (default 0.1)
        touch_duration_ms: Minimum duration for touch event in ms (default 50)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        energy_threshold: float = 0.1,
        touch_duration_ms: int = 50,
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.touch_duration_ms = touch_duration_ms
        self.touch_duration_samples = int(sample_rate * touch_duration_ms / 1000)

        # Detection state
        self._is_touch_active = False
        self._touch_start_time: Optional[float] = None
        self._last_event_time: Optional[float] = None
        self._min_event_interval = 0.1  # 100ms minimum between events

    def detect_events(
        self,
        audio_buffer: AudioBuffer,
        current_time: float,
    ) -> List[AudioEvent]:
        """
        Detect audio events from audio buffer.

        Args:
            audio_buffer: AudioBuffer containing audio samples
            current_time: Current video timestamp in seconds

        Returns:
            List of detected AudioEvent objects
        """
        events: List[AudioEvent] = []

        # Get latest samples for analysis
        num_samples = max(1024, self.touch_duration_samples)
        samples = audio_buffer.get_latest(num_samples)

        if samples is None:
            return events

        # Compute energy
        energy = audio_buffer.compute_energy(num_samples)
        if energy is None:
            return events

        # Normalize energy
        normalized_energy = energy / num_samples

        # Check for touch event
        if normalized_energy > self.energy_threshold:
            if not self._is_touch_active:
                # Check minimum interval
                if (self._last_event_time is None or
                    current_time - self._last_event_time >= self._min_event_interval):
                    self._is_touch_active = True
                    self._touch_start_time = current_time

        # End of touch
        elif self._is_touch_active:
            self._is_touch_active = False
            touch_duration = current_time - (self._touch_start_time or 0)

            if touch_duration >= self.touch_duration_ms / 1000:
                # Classify event type based on energy characteristics
                event_type = self._classify_event(normalized_energy)

                events.append(AudioEvent(
                    timestamp=self._touch_start_time or current_time,
                    event_type=event_type,
                    confidence=min(1.0, normalized_energy / self.energy_threshold),
                ))
                self._last_event_time = current_time

        return events

    def _classify_event(self, energy: float) -> str:
        """
        Classify event type based on energy characteristics.

        Args:
            energy: Normalized energy value

        Returns:
            Event type string
        """
        # Higher energy = blade touch, lower = footstep
        if energy > self.energy_threshold * 3:
            return BLADE_TOUCH
        elif energy > self.energy_threshold * 2:
            return PARRY_BEAT
        else:
            return FOOTSTEP

    def detect_touch_simple(
        self,
        audio_buffer: AudioBuffer,
        current_time: float,
    ) -> Optional[AudioEvent]:
        """
        Simple touch detection for feature extraction.

        Returns a single touch event if detected, None otherwise.
        This is a simplified version for use in the feature pipeline.

        Args:
            audio_buffer: AudioBuffer containing audio samples
            current_time: Current video timestamp in seconds

        Returns:
            AudioEvent if touch detected, None otherwise
        """
        events = self.detect_events(audio_buffer, current_time)

        # Return first blade touch event
        for event in events:
            if event.event_type == BLADE_TOUCH:
                return event

        return None

    def reset(self) -> None:
        """Reset detection state."""
        self._is_touch_active = False
        self._touch_start_time = None
        self._last_event_time = None

    def __repr__(self) -> str:
        return (
            f"AudioDetector("
            f"sample_rate={self.sample_rate}, "
            f"energy_threshold={self.energy_threshold}, "
            f"touch_duration_ms={self.touch_duration_ms})"
        )
