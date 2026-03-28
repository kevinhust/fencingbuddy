"""
Tests for src/recognition/feature_math.py
TDD Phase 3: Feature Math Unit Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.recognition.feature_math import (
    extract_static_geometry,
    extract_angle_features,
    extract_torso_orientation,
    extract_arm_extension_features,
    compute_velocity,
    compute_acceleration,
    extract_all_features,
    EMASmoother,
    FERA_12_INDICES,
)
from src.utils.constants import COCO_INDICES


class TestStaticGeometry:
    """Test static geometry extraction."""

    def test_extract_static_geometry_shape(self):
        """Should return array of shape (24,)."""
        # Create dummy keypoints (17, 3)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 200, 0.9]
        keypoints[COCO_INDICES["right_shoulder"]] = [300, 200, 0.9]

        result = extract_static_geometry(keypoints, normalize=True)
        assert result.shape == (24,)
        assert result.dtype == np.float32

    def test_extract_static_geometry_normalized(self):
        """Should normalize by shoulder width and center at pelvis."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Left shoulder at (100, 100)
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        # Right shoulder at (300, 100) - shoulder width = 200
        keypoints[COCO_INDICES["right_shoulder"]] = [300, 100, 0.9]
        # Hips at (150, 300) and (250, 300) - pelvis center at (200, 300)
        keypoints[COCO_INDICES["left_hip"]] = [150, 300, 0.9]
        keypoints[COCO_INDICES["right_hip"]] = [250, 300, 0.9]

        result = extract_static_geometry(keypoints, normalize=True)

        # First two values should be (shoulder - pelvis) / shoulder_width
        # l_shoulder_x - pelvis_x = 100 - 200 = -100, / 200 = -0.5
        assert result[0] == pytest.approx(-0.5, abs=0.01)

    def test_extract_static_geometry_requires_17_keypoints(self):
        """Should raise ValueError if less than 17 keypoints."""
        keypoints = np.zeros((12, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Need 17 keypoints"):
            extract_static_geometry(keypoints)


class TestAngleFeatures:
    """Test angular feature extraction."""

    def test_extract_angle_features_shape(self):
        """Should return array of shape (4,)."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Set required keypoints with valid positions
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        keypoints[COCO_INDICES["right_shoulder"]] = [300, 100, 0.9]
        keypoints[COCO_INDICES["left_hip"]] = [150, 300, 0.9]
        keypoints[COCO_INDICES["right_hip"]] = [250, 300, 0.9]
        keypoints[COCO_INDICES["left_knee"]] = [150, 400, 0.9]
        keypoints[COCO_INDICES["right_knee"]] = [250, 400, 0.9]
        keypoints[COCO_INDICES["left_ankle"]] = [150, 500, 0.9]
        keypoints[COCO_INDICES["right_ankle"]] = [250, 500, 0.9]
        keypoints[COCO_INDICES["right_elbow"]] = [400, 200, 0.9]
        keypoints[COCO_INDICES["right_wrist"]] = [500, 250, 0.9]

        result = extract_angle_features(keypoints)
        assert result.shape == (4,)
        assert result.dtype == np.float32


class TestTorsoOrientation:
    """Test torso orientation extraction."""

    def test_extract_torso_orientation_shape(self):
        """Should return array of shape (2,)."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        keypoints[COCO_INDICES["right_shoulder"]] = [300, 100, 0.9]
        keypoints[COCO_INDICES["left_hip"]] = [150, 300, 0.9]
        keypoints[COCO_INDICES["right_hip"]] = [250, 300, 0.9]

        result = extract_torso_orientation(keypoints)
        assert result.shape == (2,)


class TestArmExtension:
    """Test arm extension feature extraction."""

    def test_extract_arm_extension_shape(self):
        """Should return array of shape (6,)."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Set weapon arm keypoints
        keypoints[COCO_INDICES["right_shoulder"]] = [200, 100, 0.9]
        keypoints[COCO_INDICES["right_elbow"]] = [300, 200, 0.9]
        keypoints[COCO_INDICES["right_wrist"]] = [400, 300, 0.9]
        # Set off arm
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        keypoints[COCO_INDICES["left_elbow"]] = [50, 200, 0.9]
        keypoints[COCO_INDICES["left_wrist"]] = [0, 300, 0.9]

        result = extract_arm_extension_features(keypoints, is_canonical=False)
        assert result.shape == (6,)

    def test_extract_arm_extension_canonical(self):
        """Should work with canonical (left-handed) perspective."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        keypoints[COCO_INDICES["left_elbow"]] = [50, 200, 0.9]
        keypoints[COCO_INDICES["left_wrist"]] = [0, 300, 0.9]
        keypoints[COCO_INDICES["right_shoulder"]] = [200, 100, 0.9]
        keypoints[COCO_INDICES["right_elbow"]] = [250, 200, 0.9]
        keypoints[COCO_INDICES["right_wrist"]] = [300, 300, 0.9]

        result = extract_arm_extension_features(keypoints, is_canonical=True)
        assert result.shape == (6,)


class TestVelocityAcceleration:
    """Test velocity and acceleration computation."""

    def test_compute_velocity_shape(self):
        """Should return array of shape (24,)."""
        current = np.ones(24, dtype=np.float32) * 100
        previous = np.ones(24, dtype=np.float32) * 50
        dt = 0.033  # ~30fps

        result = compute_velocity(current, previous, dt)
        assert result.shape == (24,)

    def test_compute_velocity_dt_zero(self):
        """Should return zeros if dt is zero."""
        current = np.ones(24, dtype=np.float32) * 100
        previous = np.ones(24, dtype=np.float32) * 50

        result = compute_velocity(current, previous, dt=0.0)
        assert np.allclose(result, 0.0)

    def test_compute_acceleration_shape(self):
        """Should return array of shape (24,)."""
        current = np.ones(24, dtype=np.float32) * 100
        previous = np.ones(24, dtype=np.float32) * 50
        dt = 0.033

        result = compute_acceleration(current, previous, dt)
        assert result.shape == (24,)


class TestEMASmoother:
    """Test EMA smoother."""

    def test_ema_initial(self):
        """First value returned as-is."""
        smoother = EMASmoother(alpha=0.7)
        value = np.array([100.0, 200.0])
        result = smoother.smooth(value)
        np.testing.assert_array_equal(result, value)

    def test_ema_smoothing(self):
        """Should EMA smooth subsequent values."""
        smoother = EMASmoother(alpha=0.7)
        v1 = np.array([100.0, 200.0])
        v2 = np.array([200.0, 400.0])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        # Expected: 0.7 * 200 + 0.3 * 100 = 170
        expected = 0.7 * v2 + 0.3 * v1
        np.testing.assert_allclose(result, expected)

    def test_ema_reset(self):
        """Should reset state."""
        smoother = EMASmoother(alpha=0.7)
        smoother.smooth(np.array([100.0]))
        smoother.reset()

        value = np.array([100.0])
        result = smoother.smooth(value)
        np.testing.assert_array_equal(result, value)

    def test_ema_scalar(self):
        """Should work with scalars."""
        smoother = EMASmoother(alpha=0.7)
        result1 = smoother.smooth_scalar(100.0)
        assert result1 == 100.0  # First value

        result2 = smoother.smooth_scalar(200.0)
        expected = 0.7 * 200.0 + 0.3 * 100.0
        assert result2 == pytest.approx(expected)


class TestExtractAllFeatures:
    """Test complete feature extraction."""

    def test_extract_all_features_shape(self):
        """Should return features of shape (101,)."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Set minimal required keypoints
        for i in range(17):
            keypoints[i] = [float(i * 10), float(i * 20), 0.9]

        features, geometry, com, velocity = extract_all_features(keypoints)

        assert features.shape == (101,)
        assert geometry.shape == (24,)
        assert isinstance(com, tuple)
        assert len(com) == 2
        assert velocity.shape == (24,)

    def test_extract_all_features_with_history(self):
        """Should compute velocity when history provided."""
        # Create keypoints with specific pose
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[COCO_INDICES["left_shoulder"]] = [100, 100, 0.9]
        keypoints[COCO_INDICES["right_shoulder"]] = [300, 100, 0.9]
        keypoints[COCO_INDICES["left_hip"]] = [150, 300, 0.9]
        keypoints[COCO_INDICES["right_hip"]] = [250, 300, 0.9]
        keypoints[COCO_INDICES["left_knee"]] = [150, 400, 0.9]
        keypoints[COCO_INDICES["right_knee"]] = [250, 400, 0.9]
        keypoints[COCO_INDICES["left_ankle"]] = [150, 500, 0.9]
        keypoints[COCO_INDICES["right_ankle"]] = [250, 500, 0.9]
        keypoints[COCO_INDICES["nose"]] = [200, 50, 0.9]
        # Set remaining keypoints to reasonable values
        keypoints[COCO_INDICES["left_elbow"]] = [100, 200, 0.9]
        keypoints[COCO_INDICES["right_elbow"]] = [300, 200, 0.9]
        keypoints[COCO_INDICES["left_wrist"]] = [100, 300, 0.9]
        keypoints[COCO_INDICES["right_wrist"]] = [300, 300, 0.9]
        keypoints[COCO_INDICES["left_eye"]] = [190, 60, 0.9]
        keypoints[COCO_INDICES["right_eye"]] = [210, 60, 0.9]
        keypoints[COCO_INDICES["left_ear"]] = [180, 70, 0.9]
        keypoints[COCO_INDICES["right_ear"]] = [220, 70, 0.9]

        # First call without history
        features1, geometry1, _, velocity1 = extract_all_features(keypoints, dt=0.033)

        # Second call with history - change pose to produce different geometry
        keypoints2 = keypoints.copy()
        # Lean forward: move shoulders forward (decrease y for shoulders)
        keypoints2[COCO_INDICES["left_shoulder"]] = [100, 90, 0.9]
        keypoints2[COCO_INDICES["right_shoulder"]] = [300, 90, 0.9]
        keypoints2[COCO_INDICES["nose"]] = [200, 40, 0.9]

        features2, _, _, velocity2 = extract_all_features(
            keypoints2,
            previous_geometry=geometry1,
            previous_com=(100, 200),
            previous_velocity=velocity1,
            dt=0.033,
        )

        # Velocity features should be non-zero since geometry changed
        velocity_features = features2[49:73]
        assert np.any(velocity_features != 0.0)

        # Acceleration features should also be computed
        acceleration_features = features2[73:97]
        assert np.any(acceleration_features != 0.0)


class TestFERA12Indices:
    """Test FERA 12 keypoint indices."""

    def test_fera_12_indices_count(self):
        """Should have exactly 12 indices."""
        assert len(FERA_12_INDICES) == 12

    def test_fera_12_indices_values(self):
        """Should contain correct COCO indices."""
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        assert FERA_12_INDICES == expected
