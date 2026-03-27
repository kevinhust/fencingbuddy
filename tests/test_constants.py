"""
Tests for src/utils/constants.py and src/utils/types.py
TDD Phase 1.4: Constants & Types
"""

import pytest
import numpy as np


class TestCOCOKeypointIndices:
    """Test COCO-17 keypoint index constants."""

    def test_coco_indices_exist(self):
        """COCO indices dict should exist."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES is not None

    def test_coco_indices_has_all_keypoints(self):
        """All 17 COCO keypoints should be defined."""
        from src.utils.constants import COCO_INDICES
        # Should have nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        required = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        for kp in required:
            assert kp in COCO_INDICES, f"Missing keypoint: {kp}"

    def test_coco_l_shoulder_value(self):
        """L_Shoulder index should be 5 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['left_shoulder'] == 5

    def test_coco_r_shoulder_value(self):
        """R_Shoulder index should be 6 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['right_shoulder'] == 6

    def test_coco_l_elbow_value(self):
        """L_Elbow index should be 7 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['left_elbow'] == 7

    def test_coco_r_elbow_value(self):
        """R_Elbow index should be 8 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['right_elbow'] == 8

    def test_coco_l_wrist_value(self):
        """L_Wrist index should be 9 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['left_wrist'] == 9

    def test_coco_r_wrist_value(self):
        """R_Wrist index should be 10 per COCO-17."""
        from src.utils.constants import COCO_INDICES
        assert COCO_INDICES['right_wrist'] == 10


class TestFERAKeypointIndices:
    """Test FERA 12-keypoint subset indices."""

    def test_fera_indices_exist(self):
        """FERA indices should be defined."""
        from src.utils.constants import FERA_INDICES
        assert FERA_INDICES is not None

    def test_fera_has_12_keypoints(self):
        """FERA subset should have 12 keypoints."""
        from src.utils.constants import FERA_INDICES
        assert len(FERA_INDICES) == 12


class TestFeatureIndexConstants:
    """Test 101-dimensional feature vector index constants."""

    def test_feature_dim_equals_101(self):
        """FEATURE_DIM should be 101."""
        from src.utils.constants import FEATURE_DIM
        assert FEATURE_DIM == 101

    def test_feature_groups_defined(self):
        """Feature index groups should be defined."""
        from src.utils.constants import (
            STATIC_GEOMETRY_START, STATIC_GEOMETRY_END,
            DISTANCE_START, DISTANCE_END,
            ANGULAR_START, ANGULAR_END,
            VELOCITY_START, VELOCITY_END,
            ACCELERATION_START, ACCELERATION_END,
            AUDIO_FLAG_INDEX
        )
        assert STATIC_GEOMETRY_START == 0
        assert DISTANCE_START == 26
        assert ANGULAR_START == 37
        assert VELOCITY_START == 49
        assert ACCELERATION_START == 73
        assert AUDIO_FLAG_INDEX == 100

    def test_velocity_indices_match_derivative_of_static(self):
        """Velocity indices (49-72) should be 1st derivative of static geometry (0-23)."""
        from src.utils.constants import STATIC_GEOMETRY_END, VELOCITY_START
        # Velocity starts at index 49 per AD1 (after geometry, CoM, distance, angular, torso, arm)
        assert VELOCITY_START == 49

    def test_acceleration_indices_match_derivative_of_velocity(self):
        """Acceleration indices (73-96) should be 2nd derivative of static geometry."""
        from src.utils.constants import VELOCITY_END, ACCELERATION_START
        # Acceleration starts where velocity ends
        assert ACCELERATION_START == VELOCITY_END + 1


class TestTrackerStateEnum:
    """Test tracker state enumerations."""

    def test_tracker_state_enum_exists(self):
        """TrackerState enum should exist."""
        from src.utils.constants import TrackerState
        assert TrackerState is not None

    def test_tracker_states_defined(self):
        """TrackerState should have expected states."""
        from src.utils.constants import TrackerState
        # Should have at least: Searching, Tracking, Lost
        assert hasattr(TrackerState, 'SEARCHING')
        assert hasattr(TrackerState, 'TRACKING')
        assert hasattr(TrackerState, 'LOST')


class TestAudioEventTypeConstants:
    """Test audio event type constants."""

    def test_blade_touch_defined(self):
        """BLADE_TOUCH event type should be defined."""
        from src.utils.constants import BLADE_TOUCH
        assert BLADE_TOUCH == "blade_touch"

    def test_parry_beat_defined(self):
        """PARRY_BEAT event type should be defined."""
        from src.utils.constants import PARRY_BEAT
        assert PARRY_BEAT == "parry_beat"

    def test_referee_halt_defined(self):
        """REFEREE_HALT event type should be defined."""
        from src.utils.constants import REFEREE_HALT
        assert REFEREE_HALT == "referee_halt"


class TestTypes:
    """Test type aliases."""

    def test_keypoint_array_type(self):
        """KeypointArray should be np.ndarray shape=(N, 3)."""
        from src.utils.types import KeypointArray
        # Create a valid keypoint array
        arr = np.zeros((17, 3), dtype=np.float32)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (17, 3)

    def test_pose_keypoints_type(self):
        """PoseKeypoints should be dict with keypoint arrays."""
        from src.utils.types import PoseKeypoints
        import numpy.typing as npt
        # PoseKeypoints is Dict[str, KeypointArray]
        keypoints: PoseKeypoints = {
            "left_shoulder": np.zeros((1, 3), dtype=np.float32),
            "right_shoulder": np.zeros((1, 3), dtype=np.float32),
        }
        assert isinstance(keypoints, dict)
        assert "left_shoulder" in keypoints

    def test_homography_matrix_type(self):
        """HomographyMatrix should be np.ndarray shape=(3, 3)."""
        from src.utils.types import HomographyMatrix
        # Create a valid homography matrix
        matrix = np.eye(3, dtype=np.float32)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)


class TestPisteDimensions:
    """Test standard piste dimension constants."""

    def test_default_piste_length(self):
        """Default piste length should be 14.0 meters."""
        from src.utils.constants import DEFAULT_PISTE_LENGTH
        assert DEFAULT_PISTE_LENGTH == 14.0

    def test_default_piste_width(self):
        """Default piste width should be 1.5-2.0 meters."""
        from src.utils.constants import DEFAULT_PISTE_WIDTH
        assert 1.5 <= DEFAULT_PISTE_WIDTH <= 2.0


class TestEMAAlphas:
    """Test EMA alpha constants."""

    def test_default_velocity_alpha(self):
        """Default velocity EMA alpha should be in range 0.6-0.8."""
        from src.utils.constants import DEFAULT_VELOCITY_ALPHA
        assert 0.6 <= DEFAULT_VELOCITY_ALPHA <= 0.8

    def test_default_acceleration_alpha(self):
        """Default acceleration EMA alpha should be in range 0.6-0.8."""
        from src.utils.constants import DEFAULT_ACCELERATION_ALPHA
        assert 0.6 <= DEFAULT_ACCELERATION_ALPHA <= 0.8
