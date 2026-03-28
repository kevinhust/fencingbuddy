"""
Tests for src/perception/pipeline.py
TDD Phase 2.5: Perception Pipeline Integration Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.perception.pipeline import PerceptionPipeline
from src.perception.rtmpose import RTMPoseEstimator
from src.perception.calibrator import HomographyCalibrator
from src.utils.schemas import FencerPose, Keypoint


class TestPerceptionPipelineInit:
    """Test PerceptionPipeline initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        pipeline = PerceptionPipeline()
        assert pipeline.pose_estimator is not None
        assert pipeline.tracker is not None
        assert pipeline.enable_audio is False
        assert pipeline.frame_count == 0

    def test_init_with_pose_estimator(self):
        """Should accept custom pose estimator."""
        estimator = RTMPoseEstimator(conf_threshold=0.5)
        pipeline = PerceptionPipeline(pose_estimator=estimator)
        assert pipeline.pose_estimator.conf_threshold == 0.5

    def test_init_with_audio(self):
        """Should enable audio when requested."""
        pipeline = PerceptionPipeline(enable_audio=True)
        assert pipeline.enable_audio is True
        assert pipeline.audio_buffer is not None
        assert pipeline.audio_detector is not None

    def test_init_without_audio(self):
        """Should not create audio components when disabled."""
        pipeline = PerceptionPipeline(enable_audio=False)
        assert pipeline.audio_buffer is None
        assert pipeline.audio_detector is None

    def test_repr(self):
        """String representation should show parameters."""
        pipeline = PerceptionPipeline()
        repr_str = repr(pipeline)
        assert "PerceptionPipeline" in repr_str
        assert "enable_audio=False" in repr_str


class TestPerceptionPipelineProcessFrame:
    """Test frame processing through the pipeline."""

    def test_process_frame_returns_framedata(self):
        """Should return FrameData object."""
        pipeline = PerceptionPipeline()

        # Create a synthetic frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        frame_data = pipeline.process_frame(frame, timestamp=0.0, frame_id=0)

        assert frame_data is not None
        assert hasattr(frame_data, 'frame_id')
        assert hasattr(frame_data, 'timestamp')
        assert hasattr(frame_data, 'poses')

    def test_process_frame_increments_count(self):
        """Should increment frame count."""
        pipeline = PerceptionPipeline()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        pipeline.process_frame(frame, timestamp=0.0, frame_id=0)
        assert pipeline.frame_count == 1

        pipeline.process_frame(frame, timestamp=0.03, frame_id=1)
        assert pipeline.frame_count == 2

    def test_process_frame_with_audio(self):
        """Should process audio samples when enabled."""
        pipeline = PerceptionPipeline(enable_audio=True)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        audio_samples = np.random.randn(1024).astype(np.float32) * 0.1

        frame_data = pipeline.process_frame(
            frame, timestamp=0.0, frame_id=0, audio_samples=audio_samples
        )

        # Audio event may or may not be detected
        assert frame_data is not None
        assert frame_data.frame_id == 0


class TestPerceptionPipelineCalibration:
    """Test calibration integration."""

    def test_set_calibrator(self):
        """Should accept a calibrator."""
        pipeline = PerceptionPipeline()
        calibrator = HomographyCalibrator()
        pipeline.set_calibrator(calibrator)
        assert pipeline.calibrator is calibrator

    def test_is_calibrated_false_initially(self):
        """Should not be calibrated initially."""
        pipeline = PerceptionPipeline()
        assert pipeline.is_calibrated() is False

    def test_is_calibrated_with_calibrator(self):
        """Should reflect calibrator state."""
        pipeline = PerceptionPipeline()
        calibrator = HomographyCalibrator()
        pipeline.set_calibrator(calibrator)
        assert pipeline.is_calibrated() is False

        # Calibrate
        calibrator.add_point(0, 0, 0.0, 0.0)
        calibrator.add_point(640, 0, 14.0, 0.0)
        calibrator.add_point(0, 480, 0.0, 1.8)
        calibrator.add_point(640, 480, 14.0, 1.8)
        calibrator.calibrate()

        assert pipeline.is_calibrated() is True


class TestPerceptionPipelineReset:
    """Test pipeline reset functionality."""

    def test_reset_clears_state(self):
        """Should reset all state."""
        pipeline = PerceptionPipeline()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Process some frames
        pipeline.process_frame(frame, timestamp=0.0, frame_id=0)
        pipeline.process_frame(frame, timestamp=0.03, frame_id=1)

        assert pipeline.frame_count == 2

        # Reset
        pipeline.reset()

        assert pipeline.frame_count == 0
        assert pipeline._is_initialized is False

    def test_reset_with_audio(self):
        """Should reset audio components when enabled."""
        pipeline = PerceptionPipeline(enable_audio=True)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        audio_samples = np.random.randn(1024).astype(np.float32)

        # Process a frame with audio
        pipeline.process_frame(
            frame, timestamp=0.0, frame_id=0, audio_samples=audio_samples
        )

        # Reset
        pipeline.reset()

        # Audio buffer should be cleared
        assert pipeline.audio_buffer.total_samples == 0


class TestPerceptionPipelineIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline_workflow(self):
        """Test complete processing workflow."""
        pipeline = PerceptionPipeline()

        # Create synthetic frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Process first frame (initialization)
        frame_data_0 = pipeline.process_frame(frame, timestamp=0.0, frame_id=0)

        # Process subsequent frames
        frame_data_1 = pipeline.process_frame(frame, timestamp=0.03, frame_id=1)
        frame_data_2 = pipeline.process_frame(frame, timestamp=0.06, frame_id=2)

        assert frame_data_0.frame_id == 0
        assert frame_data_1.frame_id == 1
        assert frame_data_2.frame_id == 2
        assert pipeline.frame_count == 3

    def test_pipeline_with_calibrator(self):
        """Test pipeline with calibration."""
        pipeline = PerceptionPipeline()

        # Set up calibrator
        calibrator = HomographyCalibrator()
        calibrator.add_point(0, 0, 0.0, 0.0)
        calibrator.add_point(640, 0, 14.0, 0.0)
        calibrator.add_point(0, 480, 0.0, 1.8)
        calibrator.add_point(640, 480, 14.0, 1.8)
        calibrator.calibrate()
        pipeline.set_calibrator(calibrator)

        # Verify calibrator is set
        assert pipeline.is_calibrated() is True

        # Process frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        frame_data = pipeline.process_frame(frame, timestamp=0.0, frame_id=0)

        # Frame data should have homography matrix
        assert frame_data.homography_matrix is not None
