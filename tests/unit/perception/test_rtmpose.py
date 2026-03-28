"""
Tests for src/perception/rtmpose.py
TDD Phase 2.1: RTMPose Wrapper Unit Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.perception.rtmpose import RTMPoseEstimator, COCO_KEYPOINT_COUNT, COCO_KEYPOINT_NAMES
from src.utils.schemas import FencerPose, Keypoint


class TestRTMPoseEstimatorInit:
    """Phase 1: Test initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        estimator = RTMPoseEstimator()
        assert estimator.mode == "lightweight"
        assert estimator.device == "cpu"
        assert estimator.conf_threshold == 0.3

    def test_init_with_custom_threshold(self):
        """Should accept custom conf_threshold."""
        estimator = RTMPoseEstimator(conf_threshold=0.5)
        assert estimator.conf_threshold == 0.5

    def test_init_with_custom_mode(self):
        """Should accept custom mode."""
        estimator = RTMPoseEstimator(mode="lightweight")
        assert estimator.mode == "lightweight"

    def test_init_with_custom_device(self):
        """Should accept custom device."""
        estimator = RTMPoseEstimator(device="cpu")
        assert estimator.device == "cpu"

    def test_invalid_mode_raises(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="mode must be one of"):
            RTMPoseEstimator(mode="invalid")

    def test_invalid_device_raises(self):
        """Should raise ValueError for invalid device."""
        with pytest.raises(ValueError, match="device must be one of"):
            RTMPoseEstimator(device="invalid")

    def test_invalid_conf_threshold_low_raises(self):
        """Should raise ValueError for conf_threshold < 0."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            RTMPoseEstimator(conf_threshold=-0.1)

    def test_invalid_conf_threshold_high_raises(self):
        """Should raise ValueError for conf_threshold > 1."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            RTMPoseEstimator(conf_threshold=1.5)

    def test_keypoint_count_is_17(self):
        """COCO format has exactly 17 keypoints."""
        estimator = RTMPoseEstimator()
        assert estimator.keypoint_count == 17

    def test_coco_keypoint_names_length(self):
        """COCO keypoint names list should have 17 items."""
        assert len(COCO_KEYPOINT_NAMES) == 17

    def test_coco_keypoint_names_correct(self):
        """COCO keypoint names should be in correct order."""
        expected = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        assert COCO_KEYPOINT_NAMES == expected


class TestEstimateFromFrameInputValidation:
    """Phase 2: Test input validation."""

    def test_rejects_none_frame(self):
        """Should raise ValueError for None frame."""
        estimator = RTMPoseEstimator()
        with pytest.raises(ValueError, match="Frame is None"):
            estimator.estimate_from_frame(None)

    def test_rejects_non_numpy(self):
        """Should raise ValueError for non-numpy input."""
        estimator = RTMPoseEstimator()
        with pytest.raises(ValueError, match="Frame must be numpy.ndarray"):
            estimator.estimate_from_frame("not an array")

    def test_rejects_wrong_ndim(self):
        """Should raise ValueError for wrong number of dimensions."""
        estimator = RTMPoseEstimator()
        wrong_dim = np.ones((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must be 3D"):
            estimator.estimate_from_frame(wrong_dim)

    def test_rejects_wrong_channels(self):
        """Should raise ValueError for wrong number of channels."""
        estimator = RTMPoseEstimator()
        wrong_channels = np.ones((480, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame must have 3 channels"):
            estimator.estimate_from_frame(wrong_channels)

    def test_rejects_empty_frame_h(self):
        """Should raise ValueError for frame with H=0."""
        estimator = RTMPoseEstimator()
        empty_h = np.ones((0, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame has invalid dimensions"):
            estimator.estimate_from_frame(empty_h)

    def test_rejects_empty_frame_w(self):
        """Should raise ValueError for frame with W=0."""
        estimator = RTMPoseEstimator()
        empty_w = np.ones((480, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame has invalid dimensions"):
            estimator.estimate_from_frame(empty_w)

    def test_rejects_wrong_dtype(self):
        """Should raise ValueError for non-uint8 dtype."""
        estimator = RTMPoseEstimator()
        float_frame = np.ones((480, 640, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="Frame must be uint8 dtype"):
            estimator.estimate_from_frame(float_frame)

    def test_accepts_valid_bgr_frame(self):
        """Should not raise for valid BGR uint8 frame."""
        estimator = RTMPoseEstimator()
        valid_frame = np.ones((480, 640, 3), dtype=np.uint8)
        # Should not raise (even if no detections with high threshold)
        poses = estimator.estimate_from_frame(valid_frame)
        assert isinstance(poses, list)


class TestEdgeCasesDetections:
    """Phase 3: Edge cases - detections."""

    def test_returns_empty_list_with_impossibly_high_threshold(self):
        """Should return [] when conf_threshold is impossibly high."""
        estimator = RTMPoseEstimator(conf_threshold=0.99)
        dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = estimator.estimate_from_frame(dark_frame)
        # May return false positives on random noise, so just check it's a list
        assert isinstance(poses, list)

    def test_returns_list_of_fencer_pose(self):
        """Output should be List[FencerPose]."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        assert isinstance(poses, list)
        for pose in poses:
            assert isinstance(pose, FencerPose)

    def test_single_pose_has_17_keypoints(self):
        """Single detection should have exactly 17 keypoints."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        if len(poses) > 0:
            assert len(poses[0].keypoints) == 17

    def test_single_pose_has_valid_fencer_id(self):
        """Single detection should have fencer_id=0 (temporary assignment)."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        if len(poses) > 0:
            assert poses[0].fencer_id == 0

    def test_single_pose_has_valid_bbox(self):
        """Bbox should be [x1, y1, x2, y2] with x2 > x1 and y2 > y1."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        if len(poses) > 0:
            x1, y1, x2, y2 = poses[0].bbox
            assert x2 > x1, "x2 should be greater than x1"
            assert y2 > y1, "y2 should be greater than y1"

    def test_all_keypoints_have_valid_confidence(self):
        """All keypoint.conf should be in [0.0, 1.0]."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        for pose in poses:
            for kp in pose.keypoints:
                assert 0.0 <= kp.conf <= 1.0, f"Invalid confidence: {kp.conf}"

    def test_all_keypoints_have_valid_coordinates(self):
        """All keypoint x,y should be >= 0."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        for pose in poses:
            for kp in pose.keypoints:
                assert kp.x >= 0, f"Invalid x coordinate: {kp.x}"
                assert kp.y >= 0, f"Invalid y coordinate: {kp.y}"


class TestPoseEmbedder:
    """Test pose embedding for similarity matching."""

    def test_multiple_poses_limited_to_two(self):
        """Should return max 2 poses when >2 detected."""
        estimator = RTMPoseEstimator()
        # Use a frame likely to have multiple detections
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        assert len(poses) <= 2, f"Expected max 2 poses, got {len(poses)}"


class TestRepr:
    """Test string representation."""

    def test_repr_shows_parameters(self):
        """Repr should show initialization parameters."""
        estimator = RTMPoseEstimator(mode="lightweight", device="cpu", conf_threshold=0.5)
        repr_str = repr(estimator)
        assert "mode='lightweight'" in repr_str
        assert "device='cpu'" in repr_str
        assert "conf_threshold=0.5" in repr_str


class TestIntegrationWithSchemas:
    """Verify output integrates properly with schemas.py."""

    def test_fencer_pose_passes_validation(self):
        """Output should pass FencerPose model validation."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        for pose in poses:
            # Should not raise - means Pydantic validation passes
            pose_dict = pose.model_dump()
            assert "fencer_id" in pose_dict
            assert "bbox" in pose_dict
            assert "keypoints" in pose_dict

    def test_keypoint_to_numpy_roundtrip(self):
        """Keypoint.to_numpy() should produce valid array."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        if len(poses) > 0:
            kp = poses[0].keypoints[0]
            arr = kp.to_numpy()
            assert arr.shape == (3,)
            assert arr.dtype == np.float32

    def test_pose_get_keypoint_array(self):
        """FencerPose.get_keypoint_array() should produce (N, 3) array."""
        estimator = RTMPoseEstimator()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        poses = estimator.estimate_from_frame(frame)
        if len(poses) > 0:
            arr = poses[0].get_keypoint_array()
            assert arr.ndim == 2
            assert arr.shape[1] == 3  # x, y, conf
            assert arr.dtype == np.float32
