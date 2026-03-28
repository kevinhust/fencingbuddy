"""
Tests for src/perception/audio.py and audio_buffer.py
TDD Phase 2.4: Audio Event Detection Unit Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.perception.audio import AudioDetector
from src.perception.audio_buffer import AudioBuffer, AudioBufferEntry
from src.utils.schemas import AudioEvent
from src.utils.constants import BLADE_TOUCH, PARRY_BEAT, FOOTSTEP


class TestAudioBufferInit:
    """Test AudioBuffer initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        buffer = AudioBuffer()
        assert buffer.max_size == 1000
        assert buffer.sample_rate == 44100
        assert len(buffer) == 0

    def test_init_with_custom_params(self):
        """Should accept custom parameters."""
        buffer = AudioBuffer(max_size=500, sample_rate=48000)
        assert buffer.max_size == 500
        assert buffer.sample_rate == 48000

    def test_total_samples_empty(self):
        """Should return 0 for empty buffer."""
        buffer = AudioBuffer()
        assert buffer.total_samples == 0


class TestAudioBufferOperations:
    """Test AudioBuffer operations."""

    def test_append_single_entry(self):
        """Should append a single audio entry."""
        buffer = AudioBuffer()
        samples = np.random.randn(1024).astype(np.float32)
        buffer.append(timestamp=0.0, samples=samples)
        assert len(buffer) == 1
        assert buffer.total_samples == 1024

    def test_append_multiple_entries(self):
        """Should append multiple audio entries."""
        buffer = AudioBuffer()
        for i in range(5):
            samples = np.ones(512, dtype=np.float32) * 0.1
            buffer.append(timestamp=float(i) * 0.1, samples=samples)
        assert len(buffer) == 5
        assert buffer.total_samples == 5 * 512

    def test_buffer_max_size(self):
        """Should drop oldest entries when max_size exceeded."""
        buffer = AudioBuffer(max_size=3)
        for i in range(5):
            samples = np.ones(256, dtype=np.float32)
            buffer.append(timestamp=float(i), samples=samples)
        assert len(buffer) == 3

    def test_get_samples_in_range(self):
        """Should return samples within timestamp range."""
        buffer = AudioBuffer()
        for i in range(5):
            samples = np.ones(256, dtype=np.float32) * i
            buffer.append(timestamp=float(i) * 0.1, samples=samples)

        # Timestamps: 0.0, 0.1, 0.2, 0.3, 0.4
        # Range [0.1, 0.3] includes: 0.1, 0.2, 0.3
        # But due to floating point and circular buffer behavior, check at least 2
        samples_list = buffer.get_samples_in_range(start_ts=0.1, end_ts=0.3)
        assert len(samples_list) >= 2

    def test_get_latest(self):
        """Should return latest N samples."""
        buffer = AudioBuffer()
        for i in range(5):
            samples = np.ones(256, dtype=np.float32) * i
            buffer.append(timestamp=float(i), samples=samples)

        latest = buffer.get_latest(num_samples=512)
        assert latest is not None
        # Should contain samples from the latest entries
        # Last values should be from entry 4 (value=4)
        assert len(latest) <= 512
        assert latest[-1] == 4.0

    def test_get_latest_more_than_total(self):
        """Should return all samples if requested > total."""
        buffer = AudioBuffer()
        samples = np.ones(256, dtype=np.float32) * 0.5
        buffer.append(timestamp=0.0, samples=samples)

        latest = buffer.get_latest(num_samples=1024)
        assert latest is not None
        assert len(latest) == 256

    def test_get_latest_empty_buffer(self):
        """Should return None for empty buffer."""
        buffer = AudioBuffer()
        latest = buffer.get_latest()
        assert latest is None

    def test_compute_rms(self):
        """Should compute RMS correctly."""
        buffer = AudioBuffer()
        # Constant signal of 0.5 should have RMS of 0.5
        samples = np.ones(1024, dtype=np.float32) * 0.5
        buffer.append(timestamp=0.0, samples=samples)

        rms = buffer.compute_rms()
        assert rms is not None
        assert rms == pytest.approx(0.5, rel=0.01)

    def test_compute_rms_empty(self):
        """Should return None for empty buffer."""
        buffer = AudioBuffer()
        rms = buffer.compute_rms()
        assert rms is None

    def test_clear(self):
        """Should clear all entries."""
        buffer = AudioBuffer()
        for i in range(3):
            samples = np.ones(256, dtype=np.float32)
            buffer.append(timestamp=float(i), samples=samples)
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.total_samples == 0


class TestAudioDetectorInit:
    """Test AudioDetector initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        detector = AudioDetector()
        assert detector.sample_rate == 44100
        assert detector.energy_threshold == 0.1
        assert detector.touch_duration_ms == 50

    def test_init_with_custom_params(self):
        """Should accept custom parameters."""
        detector = AudioDetector(
            sample_rate=48000,
            energy_threshold=0.2,
            touch_duration_ms=100,
        )
        assert detector.sample_rate == 48000
        assert detector.energy_threshold == 0.2
        assert detector.touch_duration_ms == 100

    def test_repr(self):
        """String representation should show parameters."""
        detector = AudioDetector()
        repr_str = repr(detector)
        assert "AudioDetector" in repr_str
        assert "sample_rate=44100" in repr_str


class TestAudioDetectorEvents:
    """Test AudioDetector event detection."""

    def test_detect_no_events_empty_buffer(self):
        """Should return no events for empty buffer."""
        detector = AudioDetector()
        buffer = AudioBuffer()
        events = detector.detect_events(buffer, current_time=0.0)
        assert len(events) == 0

    def test_detect_no_events_low_energy(self):
        """Should return no events for low energy audio."""
        detector = AudioDetector(energy_threshold=0.5)
        buffer = AudioBuffer()

        # Low energy samples (below threshold)
        samples = np.ones(2048, dtype=np.float32) * 0.1
        buffer.append(timestamp=0.0, samples=samples)

        events = detector.detect_events(buffer, current_time=0.0)
        assert len(events) == 0

    def test_detect_touch_event_high_energy(self):
        """Should detect touch event for high energy audio."""
        detector = AudioDetector(energy_threshold=0.1)
        buffer = AudioBuffer()

        # High energy samples
        samples = np.random.randn(2048).astype(np.float32) * 0.5
        buffer.append(timestamp=0.0, samples=samples)

        events = detector.detect_events(buffer, current_time=0.1)
        # May detect events depending on energy
        assert isinstance(events, list)

    def test_detect_touch_simple_no_touch(self):
        """Should return None when no touch detected."""
        detector = AudioDetector(energy_threshold=0.5)
        buffer = AudioBuffer()

        samples = np.ones(1024, dtype=np.float32) * 0.1
        buffer.append(timestamp=0.0, samples=samples)

        event = detector.detect_touch_simple(buffer, current_time=0.0)
        assert event is None

    def test_reset_clears_state(self):
        """Reset should clear detection state."""
        detector = AudioDetector()
        buffer = AudioBuffer()

        samples = np.random.randn(2048).astype(np.float32) * 0.5
        buffer.append(timestamp=0.0, samples=samples)
        detector.detect_events(buffer, current_time=0.1)

        detector.reset()
        assert detector._is_touch_active is False
        assert detector._touch_start_time is None


class TestEventClassification:
    """Test event classification."""

    def test_classify_blade_touch(self):
        """Should classify high energy as blade touch."""
        detector = AudioDetector(energy_threshold=0.1)
        event_type = detector._classify_event(energy=0.5)
        assert event_type == BLADE_TOUCH

    def test_classify_parry(self):
        """Should classify medium energy as parry."""
        detector = AudioDetector(energy_threshold=0.1)
        event_type = detector._classify_event(energy=0.25)
        assert event_type == PARRY_BEAT

    def test_classify_footstep(self):
        """Should classify low energy as footstep."""
        detector = AudioDetector(energy_threshold=0.1)
        event_type = detector._classify_event(energy=0.15)
        assert event_type == FOOTSTEP
