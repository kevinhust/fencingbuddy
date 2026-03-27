"""
Tests for src/utils/buffer.py
TDD Phase 1.3: TimestampedBuffer
"""

import pytest
import numpy as np
import threading
import time
from typing import List, Tuple


class TestTimestampedBuffer:
    """Test TimestampedBuffer functionality."""

    def test_buffer_initialization(self):
        """Buffer should initialize with default max_size."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        assert buf.max_size == 1000
        assert len(buf) == 0

    def test_buffer_custom_max_size(self):
        """Buffer should initialize with custom max_size."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer(max_size=100)
        assert buf.max_size == 100

    def test_append_single_frame(self):
        """Should append single frame to buffer."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        frame_data = {"poses": []}
        buf.append(frame_id=1, timestamp=0.033, frame_data=frame_data, audio_sample=None)
        assert len(buf) == 1

    def test_append_multiple_frames(self):
        """Should append multiple frames to buffer."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(10):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.033, frame_data=frame_data, audio_sample=None)
        assert len(buf) == 10

    def test_buffer_frame_dropping(self):
        """Should drop oldest frames when exceeding max_size."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer(max_size=5)
        for i in range(10):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.033, frame_data=frame_data, audio_sample=None)
        # Should only have 5 newest frames
        assert len(buf) == 5
        # Oldest frame (id=0) should be dropped
        frames = list(buf._buffer)
        assert frames[0].frame_id == 5  # First remaining should be id=5
        assert frames[-1].frame_id == 9  # Last should be id=9

    def test_get_frame_range(self):
        """Should retrieve frames within timestamp range."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(10):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.1, frame_data=frame_data, audio_sample=None)
        # Get frames from timestamp 0.2 to 0.5 (should be frames 2, 3, 4, 5)
        result = buf.get_frame_range(start_ts=0.2, end_ts=0.5)
        assert len(result) == 4
        assert result[0].frame_id == 2
        assert result[-1].frame_id == 5

    def test_get_frame_range_partial(self):
        """Should get frames when only start_ts matches."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(5):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.1, frame_data=frame_data, audio_sample=None)
        # Timestamps: 0.0, 0.1, 0.2, 0.3, 0.4
        # Range 0.15 to 10.0 includes: 0.2, 0.3, 0.4 (frames 2, 3, 4)
        result = buf.get_frame_range(start_ts=0.15, end_ts=10.0)
        assert len(result) == 3
        assert result[0].frame_id == 2
        assert result[-1].frame_id == 4

    def test_get_frame_range_no_match(self):
        """Should return empty list when no frames in range."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(5):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.1, frame_data=frame_data, audio_sample=None)
        result = buf.get_frame_range(start_ts=10.0, end_ts=20.0)
        assert len(result) == 0

    def test_get_frame_range_with_audio(self):
        """Should retrieve frames with audio samples."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        audio = np.array([0.1, 0.2, 0.3])
        for i in range(3):
            frame_data = {"poses": [i]}
            buf.append(frame_id=i, timestamp=i*0.1, frame_data=frame_data, audio_sample=audio)
        result = buf.get_frame_range(start_ts=0.0, end_ts=0.3)
        assert len(result) == 3
        for frame in result:
            assert frame.audio_sample is not None

    def test_len_after_drop(self):
        """Length should be correct after dropping frames."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer(max_size=3)
        for i in range(5):
            buf.append(frame_id=i, timestamp=i*0.1, frame_data={"poses": []}, audio_sample=None)
        assert len(buf) == 3

    def test_clear_buffer(self):
        """Should clear all frames from buffer."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(5):
            buf.append(frame_id=i, timestamp=i*0.1, frame_data={"poses": []}, audio_sample=None)
        buf.clear()
        assert len(buf) == 0


class TestThreadSafety:
    """Test thread-safety of TimestampedBuffer."""

    def test_concurrent_append(self):
        """Should handle concurrent appends from multiple threads."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer(max_size=1000)
        num_threads = 5
        frames_per_thread = 100

        def append_frames(start_idx):
            for i in range(frames_per_thread):
                buf.append(
                    frame_id=start_idx + i,
                    timestamp=(start_idx + i) * 0.033,
                    frame_data={"poses": []},
                    audio_sample=None
                )

        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=append_frames, args=(t * frames_per_thread,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All frames should be appended
        assert len(buf) == num_threads * frames_per_thread

    def test_concurrent_append_and_read(self):
        """Should handle concurrent appends and reads."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer(max_size=500)
        num_operations = 200

        def append_frames():
            for i in range(num_operations):
                buf.append(
                    frame_id=i,
                    timestamp=i * 0.033,
                    frame_data={"poses": []},
                    audio_sample=None
                )
                time.sleep(0.001)

        def read_frames():
            for _ in range(num_operations // 2):
                _ = buf.get_frame_range(start_ts=0.0, end_ts=10.0)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=append_frames),
            threading.Thread(target=read_frames),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Buffer should have some frames (some may have been dropped)
        assert len(buf) > 0


class TestAudioVideoSync:
    """Test audio-video sync detection."""

    def test_sync_detection_basic(self):
        """Should detect audio-video sync using cross-correlation."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        # Create audio samples with known pattern
        audio_template = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1000))
        for i in range(10):
            buf.append(
                frame_id=i,
                timestamp=i * 0.033,
                frame_data={"poses": []},
                audio_sample=audio_template * (i + 1)  # Scale by frame
            )
        sync_info = buf.detect_sync()
        assert sync_info is not None

    def test_sync_detection_no_audio(self):
        """Should return None when no audio samples."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(5):
            buf.append(
                frame_id=i,
                timestamp=i * 0.033,
                frame_data={"poses": []},
                audio_sample=None
            )
        sync_info = buf.detect_sync()
        assert sync_info is None

    def test_sync_detection_insufficient_samples(self):
        """Should return None with insufficient audio samples."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        audio = np.array([0.1, 0.2, 0.3])
        # Only 2 samples - not enough for meaningful correlation
        for i in range(2):
            buf.append(
                frame_id=i,
                timestamp=i * 0.033,
                frame_data={"poses": []},
                audio_sample=audio
            )
        sync_info = buf.detect_sync()
        assert sync_info is None


class TestEdgeCases:
    """Test edge cases."""

    def test_get_frame_range_empty_buffer(self):
        """Should return empty list for empty buffer."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        result = buf.get_frame_range(start_ts=0.0, end_ts=1.0)
        assert result == []

    def test_negative_timestamp_range(self):
        """Should handle negative timestamp range correctly."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        for i in range(5):
            buf.append(
                frame_id=i,
                timestamp=i * 0.1 - 0.5,  # Timestamps: -0.5, -0.4, -0.3, -0.2, -0.1
                frame_data={"poses": []},
                audio_sample=None
            )
        # Range -0.5 to 0.0 includes all 5 frames (no frame at exactly 0.0)
        result = buf.get_frame_range(start_ts=-0.5, end_ts=0.0)
        assert len(result) == 5

    def test_large_audio_sample(self):
        """Should handle large audio samples."""
        from src.utils.buffer import TimestampedBuffer
        buf = TimestampedBuffer()
        large_audio = np.random.randn(100000)  # Large audio sample
        buf.append(
            frame_id=0,
            timestamp=0.0,
            frame_data={"poses": []},
            audio_sample=large_audio
        )
        assert len(buf) == 1
        retrieved = buf.get_frame_range(start_ts=0.0, end_ts=0.1)
        assert retrieved[0].audio_sample is not None
