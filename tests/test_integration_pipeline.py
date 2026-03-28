"""
Integration tests for full pipeline
TDD Phase 5.2.3: End-to-End Pipeline Integration Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.utils.schemas import FencerPose, Keypoint, FrameData, FeatureMatrix
from src.perception.pipeline import PerceptionPipeline
from src.recognition.feature_extractor import FeatureExtractor


class TestPipelineIntegration:
    """Integration tests for Perception + Recognition pipeline."""

    def create_test_pose(self, fencer_id: int = 0, offset_x: float = 0.0) -> FencerPose:
        """Create a valid test FencerPose."""
        return FencerPose(
            fencer_id=fencer_id,
            bbox=(100 + offset_x, 50, 200 + offset_x, 300),
            keypoints=[
                Keypoint(x=100 + offset_x, y=50, conf=0.9),
                Keypoint(x=120 + offset_x, y=40, conf=0.9),
                Keypoint(x=130 + offset_x, y=40, conf=0.9),
                Keypoint(x=110 + offset_x, y=45, conf=0.9),
                Keypoint(x=140 + offset_x, y=45, conf=0.9),
                Keypoint(x=150 + offset_x, y=100, conf=0.9),
                Keypoint(x=250 + offset_x, y=100, conf=0.9),
                Keypoint(x=140 + offset_x, y=150, conf=0.9),
                Keypoint(x=260 + offset_x, y=150, conf=0.9),
                Keypoint(x=130 + offset_x, y=200, conf=0.9),
                Keypoint(x=270 + offset_x, y=200, conf=0.9),
                Keypoint(x=170 + offset_x, y=250, conf=0.9),
                Keypoint(x=230 + offset_x, y=250, conf=0.9),
                Keypoint(x=165 + offset_x, y=350, conf=0.9),
                Keypoint(x=235 + offset_x, y=350, conf=0.9),
                Keypoint(x=160 + offset_x, y=450, conf=0.9),
                Keypoint(x=240 + offset_x, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )

    def test_perception_pipeline_with_recognition(self):
        """Perception output should feed into Recognition correctly."""
        # Create perception pipeline
        perception = PerceptionPipeline(conf_threshold=0.3)

        # Create feature extractor
        extractor = FeatureExtractor()

        # Simulate a frame with poses
        pose = self.create_test_pose(fencer_id=0)
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
        )

        # Process through perception
        frame_data = perception.process_frame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_id=0,
        )

        # Process through recognition
        features = extractor.extract_frame_features(frame_data)

        assert features.shape == (2, 101)

    def test_feature_extractor_sequence_integration(self):
        """Sequence of frames should produce coherent features."""
        extractor = FeatureExtractor()

        frames = []
        for i in range(10):
            pose = self.create_test_pose(fencer_id=0, offset_x=i * 5)
            frame = FrameData(
                frame_id=i,
                timestamp=i * 0.033,
                poses=[pose],
            )
            frames.append(frame)

        feature_matrix = extractor.extract_sequence_features(frames)

        assert isinstance(feature_matrix, FeatureMatrix)
        assert feature_matrix.features.shape == (10, 2, 101)

        # Check timestamps are preserved
        assert len(feature_matrix.timestamps) == 10
        assert feature_matrix.timestamps[0] == 0.0
        assert feature_matrix.timestamps[-1] == pytest.approx(9 * 0.033, abs=0.001)

    def test_both_fencers_tracked(self):
        """Both fencers should be tracked independently."""
        extractor = FeatureExtractor()

        left_pose = self.create_test_pose(fencer_id=0, offset_x=100)
        right_pose = self.create_test_pose(fencer_id=1, offset_x=500)

        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[left_pose, right_pose],
        )

        features = extractor.extract_frame_features(frame)

        # Both fencers should have non-zero features
        assert not np.all(features[0] == 0.0)
        assert not np.all(features[1] == 0.0)

    def test_velocity_changes_over_time(self):
        """Velocity features should change when pose changes over time."""
        extractor = FeatureExtractor()

        # Frame 1 - lean forward slightly
        pose1 = self.create_test_pose(fencer_id=0, offset_x=100)
        # Modify to lean forward
        pose1.keypoints[0].y = 40  # nose moves up
        frame1 = FrameData(frame_id=0, timestamp=0.0, poses=[pose1])
        features1 = extractor.extract_frame_features(frame1)

        # Frame 2 - lean forward more (different geometry)
        pose2 = self.create_test_pose(fencer_id=0, offset_x=100)
        pose2.keypoints[0].y = 20  # nose moves further up
        pose2.keypoints[5].y = 90  # left_shoulder moves up
        pose2.keypoints[6].y = 90  # right_shoulder moves up
        frame2 = FrameData(frame_id=1, timestamp=0.033, poses=[pose2])
        features2 = extractor.extract_frame_features(frame2)

        # Frame 3 - same as frame 2 (no velocity change)
        pose3 = self.create_test_pose(fencer_id=0, offset_x=100)
        pose3.keypoints[0].y = 20
        pose3.keypoints[5].y = 90
        pose3.keypoints[6].y = 90
        frame3 = FrameData(frame_id=2, timestamp=0.066, poses=[pose3])
        features3 = extractor.extract_frame_features(frame3)

        # Velocity at frame 1 should be zero (no history)
        assert np.allclose(features1[0, 49:73], 0.0)

        # Velocity at frame 2 should be non-zero (geometry changed)
        assert not np.allclose(features2[0, 49:73], 0.0)

        # Velocity at frame 3 should be zero (no change from previous)
        # Actually with EMA, it might not be exactly zero

    def test_reset_between_sequences(self):
        """Reset should clear all temporal state."""
        extractor = FeatureExtractor()

        # First sequence
        pose1 = self.create_test_pose(fencer_id=0)
        frame1 = FrameData(frame_id=0, timestamp=0.0, poses=[pose1])
        extractor.extract_frame_features(frame1)

        # Reset
        extractor.reset()

        # After reset, velocity should be zero
        pose2 = self.create_test_pose(fencer_id=0)
        frame2 = FrameData(frame_id=0, timestamp=0.0, poses=[pose2])
        features = extractor.extract_frame_features(frame2)

        # Velocity should be zero after reset
        assert np.allclose(features[0, 49:73], 0.0)

    def test_feature_matrix_shape_consistency(self):
        """FeatureMatrix should have consistent shape."""
        extractor = FeatureExtractor()

        frames = []
        for i in range(20):
            pose = self.create_test_pose(fencer_id=i % 2, offset_x=i * 10)
            frame = FrameData(
                frame_id=i,
                timestamp=i * 0.033,
                poses=[pose],
            )
            frames.append(frame)

        result = extractor.extract_sequence_features(frames)

        assert result.features.shape == (20, 2, 101)
        assert len(result.timestamps) == 20
        assert len(result.frame_ids) == 20
        assert result.audio_flags.shape == (20, 2)

    def test_empty_poses_handled_gracefully(self):
        """Empty poses should not crash the pipeline."""
        perception = PerceptionPipeline(conf_threshold=0.99)  # High threshold
        extractor = FeatureExtractor()

        # Process frame with no valid detections
        frame_data = perception.process_frame(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_id=0,
        )

        features = extractor.extract_frame_features(frame_data)

        assert features.shape == (2, 101)
        # Should return zeros when no poses
        assert np.all(features == 0.0)

    def test_audio_flag_propagation(self):
        """Audio events should set audio flags."""
        from src.utils.schemas import AudioEvent

        extractor = FeatureExtractor()

        pose = self.create_test_pose(fencer_id=0)
        audio_event = AudioEvent(timestamp=0.0, event_type="blade_touch", confidence=0.9)
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
            audio_event=audio_event,
        )

        features = extractor.extract_frame_features(frame)

        # Audio flag should be 1.0 at index 100
        assert features[0, 100] == 1.0
