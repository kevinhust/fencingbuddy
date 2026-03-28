"""
Tests for src/utils/schemas.py
TDD Phase 1.2: Pydantic Schemas Unit Tests
"""

import pytest
import numpy as np
from typing import List, Tuple, Optional


class TestKeypoint:
    """Test Keypoint validation."""

    def test_valid_keypoint(self):
        """Valid keypoint should create successfully."""
        from src.utils.schemas import Keypoint
        kp = Keypoint(x=100.0, y=200.0, conf=0.95)
        assert kp.x == 100.0
        assert kp.y == 200.0
        assert kp.conf == 0.95

    def test_keypoint_x_negative_fails(self):
        """Keypoint x < 0 should fail validation."""
        from src.utils.schemas import Keypoint
        with pytest.raises(ValueError):
            Keypoint(x=-1.0, y=100.0, conf=0.5)

    def test_keypoint_y_negative_fails(self):
        """Keypoint y < 0 should fail validation."""
        from src.utils.schemas import Keypoint
        with pytest.raises(ValueError):
            Keypoint(x=100.0, y=-1.0, conf=0.5)

    def test_keypoint_conf_below_zero_fails(self):
        """Keypoint conf < 0 should fail validation."""
        from src.utils.schemas import Keypoint
        with pytest.raises(ValueError):
            Keypoint(x=100.0, y=200.0, conf=-0.1)

    def test_keypoint_conf_above_one_fails(self):
        """Keypoint conf > 1 should fail validation."""
        from src.utils.schemas import Keypoint
        with pytest.raises(ValueError):
            Keypoint(x=100.0, y=200.0, conf=1.5)

    def test_keypoint_to_numpy(self):
        """Should convert to numpy array [x, y, conf]."""
        from src.utils.schemas import Keypoint
        kp = Keypoint(x=100.0, y=200.0, conf=0.95)
        arr = kp.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr, [100.0, 200.0, 0.95])

    def test_keypoint_from_numpy(self):
        """Should create from numpy array [x, y, conf]."""
        from src.utils.schemas import Keypoint
        arr = np.array([100.0, 200.0, 0.95], dtype=np.float32)
        kp = Keypoint.from_numpy(arr)
        assert kp.x == 100.0
        assert kp.y == 200.0
        assert kp.conf == pytest.approx(0.95, rel=1e-6)


class TestFencerPose:
    """Test FencerPose validation."""

    def test_valid_pose_left_fencer(self):
        """Valid pose for left fencer (id=0) should create successfully."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        assert pose.fencer_id == 0
        assert pose.bbox == (10, 20, 100, 200)
        assert len(pose.keypoints) == 12
        assert pose.is_canonical_flipped is False

    def test_valid_pose_right_fencer(self):
        """Valid pose for right fencer (id=1) should create successfully."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=1, bbox=(10, 20, 100, 200), keypoints=keypoints)
        assert pose.fencer_id == 1

    def test_pose_fencer_id_invalid(self):
        """Fencer id not 0 or 1 should fail."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        with pytest.raises(ValueError):
            FencerPose(fencer_id=2, bbox=(10, 20, 100, 200), keypoints=keypoints)

    def test_pose_bbox_x2_less_than_x1_fails(self):
        """bbox x2 <= x1 should fail validation."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        with pytest.raises(ValueError):
            FencerPose(fencer_id=0, bbox=(100, 20, 50, 200), keypoints=keypoints)

    def test_pose_bbox_y2_less_than_y1_fails(self):
        """bbox y2 <= y1 should fail validation."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        with pytest.raises(ValueError):
            FencerPose(fencer_id=0, bbox=(10, 200, 100, 50), keypoints=keypoints)

    def test_pose_too_few_keypoints_fails(self):
        """Pose with < 12 keypoints should fail."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(11)]
        with pytest.raises(ValueError):
            FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)

    def test_pose_too_many_keypoints_fails(self):
        """Pose with > 33 keypoints should fail."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(34)]
        with pytest.raises(ValueError):
            FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)

    def test_pose_is_canonical_flipped_default(self):
        """is_canonical_flipped should default to False."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        assert pose.is_canonical_flipped is False

    def test_pose_get_keypoint_array(self):
        """Should convert keypoints to (N, 3) numpy array."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=float(i), y=float(i*2), conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        arr = pose.get_keypoint_array()
        assert arr.shape == (12, 3)
        assert arr.dtype == np.float32

    def test_pose_shoulder_width(self):
        """Should calculate shoulder width from keypoints."""
        from src.utils.schemas import FencerPose, Keypoint
        # Create keypoints with known shoulder positions (indices 5=L, 2=R for COCO)
        keypoints = [Keypoint(x=float(i), y=float(i), conf=0.9) for i in range(17)]
        # L_shoulder at index 5: (5, 5)
        # R_shoulder at index 2: (2, 2)
        # Distance should be sqrt((5-2)^2 + (5-2)^2) = sqrt(18) = ~4.24
        pose = FencerPose(fencer_id=0, bbox=(0, 0, 10, 10), keypoints=keypoints)
        width = pose.shoulder_width()
        assert width > 0


class TestAudioEvent:
    """Test AudioEvent validation."""

    def test_valid_audio_event(self):
        """Valid audio event should create successfully."""
        from src.utils.schemas import AudioEvent
        event = AudioEvent(timestamp=1.5, event_type="blade_touch", confidence=0.9)
        assert event.timestamp == 1.5
        assert event.event_type == "blade_touch"
        assert event.confidence == 0.9

    def test_audio_event_negative_timestamp_fails(self):
        """Audio event with negative timestamp should fail."""
        from src.utils.schemas import AudioEvent
        with pytest.raises(ValueError):
            AudioEvent(timestamp=-0.1, event_type="blade_touch", confidence=0.9)

    def test_audio_event_confidence_below_zero_fails(self):
        """Audio event confidence < 0 should fail."""
        from src.utils.schemas import AudioEvent
        with pytest.raises(ValueError):
            AudioEvent(timestamp=1.5, event_type="blade_touch", confidence=-0.1)

    def test_audio_event_confidence_above_one_fails(self):
        """Audio event confidence > 1 should fail."""
        from src.utils.schemas import AudioEvent
        with pytest.raises(ValueError):
            AudioEvent(timestamp=1.5, event_type="blade_touch", confidence=1.5)

    def test_audio_event_default_type(self):
        """Default event type should be 'blade_touch'."""
        from src.utils.schemas import AudioEvent
        event = AudioEvent(timestamp=1.5, confidence=0.9)
        assert event.event_type == "blade_touch"


class TestFrameData:
    """Test FrameData validation."""

    def test_valid_frame_data(self):
        """Valid frame data should create successfully."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose])
        assert frame.frame_id == 1
        assert frame.timestamp == 0.033
        assert len(frame.poses) == 1

    def test_frame_data_two_poses(self):
        """Frame with 2 poses should create successfully."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose0 = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        pose1 = FencerPose(fencer_id=1, bbox=(110, 20, 200, 200), keypoints=keypoints)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose0, pose1])
        assert len(frame.poses) == 2

    def test_frame_data_no_poses_valid(self):
        """Frame with 0 poses is now valid (allows empty for no detections)."""
        from src.utils.schemas import FrameData
        # Empty poses are now allowed
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[])
        assert len(frame.poses) == 0

    def test_frame_data_three_poses_fails(self):
        """Frame with 3 poses should fail (max_length=2)."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        # Use valid fencer_ids (0 and 1) for 3 poses - fencer_id uniqueness not enforced
        poses = [
            FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints),
            FencerPose(fencer_id=0, bbox=(60, 20, 150, 200), keypoints=keypoints),  # duplicate id
            FencerPose(fencer_id=1, bbox=(110, 20, 200, 200), keypoints=keypoints),
        ]
        with pytest.raises(ValueError):
            FrameData(frame_id=1, timestamp=0.033, poses=poses)

    def test_frame_data_negative_frame_id_fails(self):
        """Frame with negative frame_id should fail."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        with pytest.raises(ValueError):
            FrameData(frame_id=-1, timestamp=0.033, poses=[pose])

    def test_frame_data_negative_timestamp_fails(self):
        """Frame with negative timestamp should fail."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        with pytest.raises(ValueError):
            FrameData(frame_id=1, timestamp=-0.033, poses=[pose])

    def test_frame_data_with_audio_event(self):
        """Frame with audio event should create successfully."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint, AudioEvent
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        audio = AudioEvent(timestamp=0.033, event_type="blade_touch", confidence=0.9)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose], audio_event=audio)
        assert frame.audio_event is not None
        assert frame.audio_event.event_type == "blade_touch"

    def test_frame_data_with_homography(self):
        """Frame with homography matrix should create successfully."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        homography = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose], homography_matrix=homography)
        assert frame.homography_matrix is not None

    def test_frame_data_invalid_homography_size_fails(self):
        """Homography matrix not 3x3 should fail."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        with pytest.raises(ValueError):
            FrameData(frame_id=1, timestamp=0.033, poses=[pose], homography_matrix=[[1, 0], [0, 1]])

    def test_frame_data_get_pose_by_id(self):
        """Should get pose by fencer_id."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose0 = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        pose1 = FencerPose(fencer_id=1, bbox=(110, 20, 200, 200), keypoints=keypoints)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose0, pose1])
        found = frame.get_pose_by_id(1)
        assert found is not None
        assert found.fencer_id == 1

    def test_frame_data_get_pose_by_id_not_found(self):
        """Should return None if fencer_id not found."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=i*10.0, y=i*20.0, conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose])
        found = frame.get_pose_by_id(1)
        assert found is None


class TestFeatureMatrix:
    """Test FeatureMatrix validation."""

    def test_valid_feature_matrix(self):
        """Valid feature matrix should create successfully."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float32)
        fm = FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=list(range(10)))
        assert fm.features.shape == (10, 2, 101)

    def test_feature_matrix_wrong_first_dim_fails(self):
        """Feature matrix with wrong first dimension should fail."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(5, 101).astype(np.float32)
        with pytest.raises(ValueError):
            FeatureMatrix(features=features, timestamps=[0.0]*5, frame_ids=list(range(5)))

    def test_feature_matrix_wrong_second_dim_fails(self):
        """Feature matrix with wrong second dimension should fail."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 3, 101).astype(np.float32)
        with pytest.raises(ValueError):
            FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=list(range(10)))

    def test_feature_matrix_wrong_third_dim_fails(self):
        """Feature matrix with wrong third dimension should fail."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 50).astype(np.float32)
        with pytest.raises(ValueError):
            FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=list(range(10)))

    def test_feature_matrix_converts_to_float32(self):
        """Feature matrix should convert to float32."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float64)
        fm = FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=list(range(10)))
        assert fm.features.dtype == np.float32

    def test_feature_matrix_empty_timestamps_fails(self):
        """Feature matrix with empty timestamps should fail."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float32)
        with pytest.raises(ValueError):
            FeatureMatrix(features=features, timestamps=[], frame_ids=[])

    def test_feature_matrix_empty_frame_ids_fails(self):
        """Feature matrix with empty frame_ids should fail."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float32)
        with pytest.raises(ValueError):
            FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=[])

    def test_feature_matrix_with_audio_flags(self):
        """Feature matrix with audio flags should create successfully."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float32)
        audio_flags = np.zeros((10, 2), dtype=np.float32)
        audio_flags[5, 0] = 1.0  # Touch detected at frame 5
        fm = FeatureMatrix(
            features=features,
            timestamps=[0.0]*10,
            frame_ids=list(range(10)),
            audio_flags=audio_flags
        )
        assert fm.audio_flags is not None
        assert fm.audio_flags.shape == (10, 2)

    def test_feature_matrix_save_load(self, tmp_path):
        """Should save and load feature matrix."""
        from src.utils.schemas import FeatureMatrix
        features = np.random.randn(10, 2, 101).astype(np.float32)
        fm = FeatureMatrix(features=features, timestamps=[0.0]*10, frame_ids=list(range(10)))
        save_path = tmp_path / "features.npy"
        fm.save(str(save_path))
        assert save_path.exists()


class TestSerialization:
    """Test round-trip serialization/deserialization."""

    def test_keypoint_serialization_roundtrip(self):
        """Keypoint should survive JSON serialization roundtrip."""
        from src.utils.schemas import Keypoint
        kp = Keypoint(x=100.0, y=200.0, conf=0.95)
        json_str = kp.model_dump_json()
        restored = Keypoint.model_validate_json(json_str)
        assert restored.x == kp.x
        assert restored.y == kp.y
        assert restored.conf == kp.conf

    def test_fencer_pose_serialization_roundtrip(self):
        """FencerPose should survive JSON serialization roundtrip."""
        from src.utils.schemas import FencerPose, Keypoint
        keypoints = [Keypoint(x=float(i), y=float(i*2), conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        json_str = pose.model_dump_json()
        restored = FencerPose.model_validate_json(json_str)
        assert restored.fencer_id == pose.fencer_id
        assert restored.bbox == pose.bbox
        assert len(restored.keypoints) == len(pose.keypoints)

    def test_frame_data_serialization_roundtrip(self):
        """FrameData should survive JSON serialization roundtrip."""
        from src.utils.schemas import FrameData, FencerPose, Keypoint
        keypoints = [Keypoint(x=float(i), y=float(i*2), conf=0.9) for i in range(12)]
        pose = FencerPose(fencer_id=0, bbox=(10, 20, 100, 200), keypoints=keypoints)
        frame = FrameData(frame_id=1, timestamp=0.033, poses=[pose])
        json_str = frame.model_dump_json()
        restored = FrameData.model_validate_json(json_str)
        assert restored.frame_id == frame.frame_id
        assert restored.timestamp == frame.timestamp
        assert len(restored.poses) == len(frame.poses)


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_empty_frame_succeeds(self):
        """create_empty_frame now works since poses=[] is valid."""
        from src.utils.schemas import create_empty_frame
        # Empty poses are now valid
        frame = create_empty_frame(frame_id=5, timestamp=0.5)
        assert frame.frame_id == 5
        assert frame.timestamp == 0.5
        assert len(frame.poses) == 0

    def test_create_touch_audio_event(self):
        """create_touch_audio_event should create blade_touch event."""
        from src.utils.schemas import create_touch_audio_event
        event = create_touch_audio_event(timestamp=1.5, confidence=0.85)
        assert event.timestamp == 1.5
        assert event.event_type == "blade_touch"
        assert event.confidence == 0.85
