"""
Tests for src/recognition/feature_extractor.py
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.recognition.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorState,
    extract_single_pose_features,
)
from src.utils.schemas import FencerPose, Keypoint, FrameData, FeatureMatrix
from src.utils.constants import FEATURE_DIM


class TestFeatureExtractorState:
    """Test FeatureExtractorState."""

    def test_state_initialization(self):
        """State should initialize with correct defaults."""
        state = FeatureExtractorState(alpha=0.8)

        assert state.alpha == 0.8
        assert state.geometry_history[0] is None
        assert state.geometry_history[1] is None
        assert state.com_history[0] is None
        assert state.com_history[1] is None
        assert state.velocity_history[0] is None
        assert state.velocity_history[1] is None
        assert state.previous_timestamp is None

    def test_state_reset(self):
        """Reset should clear all history."""
        state = FeatureExtractorState()
        state.geometry_history[0] = np.ones(24)
        state.com_history[0] = (100.0, 200.0)
        state.velocity_history[0] = np.ones(24) * 5.0
        state.previous_timestamp = 1.0

        state.reset()

        assert state.geometry_history[0] is None
        assert state.com_history[0] is None
        assert state.velocity_history[0] is None
        assert state.previous_timestamp is None


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def create_fencer_pose(self, fencer_id: int) -> FencerPose:
        """Helper to create a valid FencerPose."""
        return FencerPose(
            fencer_id=fencer_id,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),   # nose
                Keypoint(x=120, y=40, conf=0.9),  # left_eye
                Keypoint(x=130, y=40, conf=0.9),  # right_eye
                Keypoint(x=110, y=45, conf=0.9),  # left_ear
                Keypoint(x=140, y=45, conf=0.9), # right_ear
                Keypoint(x=150, y=100, conf=0.9), # left_shoulder
                Keypoint(x=250, y=100, conf=0.9), # right_shoulder
                Keypoint(x=140, y=150, conf=0.9), # left_elbow
                Keypoint(x=260, y=150, conf=0.9), # right_elbow
                Keypoint(x=130, y=200, conf=0.9), # left_wrist
                Keypoint(x=270, y=200, conf=0.9), # right_wrist
                Keypoint(x=170, y=250, conf=0.9), # left_hip
                Keypoint(x=230, y=250, conf=0.9), # right_hip
                Keypoint(x=165, y=350, conf=0.9), # left_knee
                Keypoint(x=235, y=350, conf=0.9), # right_knee
                Keypoint(x=160, y=450, conf=0.9), # left_ankle
                Keypoint(x=240, y=450, conf=0.9), # right_ankle
            ],
            is_canonical_flipped=False,
        )

    def test_extractor_initialization(self):
        """Extractor should initialize with default alphas."""
        extractor = FeatureExtractor()

        assert extractor.velocity_alpha == 0.7
        assert extractor.acceleration_alpha == 0.7
        assert extractor._state is not None

    def test_extract_frame_features_empty_frame(self):
        """Empty frame should return zeros."""
        extractor = FeatureExtractor()
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[],
        )

        features = extractor.extract_frame_features(frame)

        assert features.shape == (2, FEATURE_DIM)
        assert np.all(features == 0.0)

    def test_extract_frame_features_single_fencer(self):
        """Single fencer should populate correct row."""
        extractor = FeatureExtractor()
        pose = self.create_fencer_pose(fencer_id=0)
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
        )

        features = extractor.extract_frame_features(frame)

        # Left fencer (id=0) should have features
        assert not np.all(features[0] == 0.0)
        # Right fencer (id=1) should be zeros
        assert np.all(features[1] == 0.0)

    def test_extract_frame_features_two_fencers(self):
        """Two fencers should populate both rows."""
        extractor = FeatureExtractor()
        left_pose = self.create_fencer_pose(fencer_id=0)
        right_pose = self.create_fencer_pose(fencer_id=1)
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[left_pose, right_pose],
        )

        features = extractor.extract_frame_features(frame)

        # Both should have features
        assert not np.all(features[0] == 0.0)
        assert not np.all(features[1] == 0.0)

    def test_extract_frame_features_updates_state(self):
        """Extract should update internal state."""
        extractor = FeatureExtractor()
        pose = self.create_fencer_pose(fencer_id=0)
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
        )

        extractor.extract_frame_features(frame)

        assert extractor._state.geometry_history[0] is not None
        assert extractor._state.com_history[0] is not None

    def test_extract_frame_features_velocity_with_dt(self):
        """Velocity should be computed when dt > 0."""
        extractor = FeatureExtractor()

        # First frame
        pose1 = self.create_fencer_pose(fencer_id=0)
        frame1 = FrameData(frame_id=0, timestamp=0.0, poses=[pose1])
        features1 = extractor.extract_frame_features(frame1)

        # Second frame (dt > 0)
        pose2 = self.create_fencer_pose(fencer_id=0)
        frame2 = FrameData(frame_id=1, timestamp=0.033, poses=[pose2])
        features2 = extractor.extract_frame_features(frame2)

        # Velocity features (49:73) should be present
        velocity1 = features1[0, 49:73]
        velocity2 = features2[0, 49:73]
        # First frame velocity is zero (no history)
        assert np.allclose(velocity1, 0.0)

    def test_extract_sequence_features(self):
        """Sequence extraction should build proper FeatureMatrix."""
        extractor = FeatureExtractor()

        frames = []
        for i in range(5):
            pose = self.create_fencer_pose(fencer_id=0)
            frame = FrameData(
                frame_id=i,
                timestamp=i * 0.033,
                poses=[pose],
            )
            frames.append(frame)

        result = extractor.extract_sequence_features(frames)

        assert isinstance(result, FeatureMatrix)
        assert result.features.shape == (5, 2, FEATURE_DIM)
        assert len(result.timestamps) == 5
        assert len(result.frame_ids) == 5

    def test_extract_sequence_features_empty_raises(self):
        """Empty sequence should raise ValueError."""
        extractor = FeatureExtractor()

        with pytest.raises(ValueError, match="Cannot extract features from empty"):
            extractor.extract_sequence_features([])

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        extractor = FeatureExtractor()

        pose = self.create_fencer_pose(fencer_id=0)
        frame = FrameData(frame_id=0, timestamp=0.0, poses=[pose])
        extractor.extract_frame_features(frame)

        extractor.reset()

        assert extractor._state.geometry_history[0] is None
        assert extractor._state.com_history[0] is None
        assert extractor._state.velocity_history[0] is None


class TestExtractSinglePoseFeatures:
    """Test convenience function."""

    def create_pose(self) -> FencerPose:
        """Create a valid FencerPose."""
        return FencerPose(
            fencer_id=0,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )

    def test_extract_single_pose_features_shape(self):
        """Should return array of shape (101,)."""
        pose = self.create_pose()
        result = extract_single_pose_features(pose)

        assert result.shape == (101,)
        assert result.dtype == np.float32

    def test_extract_single_pose_features_with_audio_flag(self):
        """Should handle audio_flag parameter."""
        pose = self.create_pose()
        result = extract_single_pose_features(pose, audio_flag=1.0)

        # Audio flag should be 1.0 at index 100
        assert result[100] == 1.0

    def test_extract_single_pose_features_right_fencer_canonical(self):
        """Right fencer should be treated as canonical."""
        pose = FencerPose(
            fencer_id=1,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )

        result = extract_single_pose_features(pose, audio_flag=0.0)

        assert result.shape == (101,)
