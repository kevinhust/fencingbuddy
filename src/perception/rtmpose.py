"""
FencerAI RTMPose Wrapper
========================
Version: 1.0 | Last Updated: 2026-03-27

RTMPose-based pose estimator using rtmlib ONNX runtime.
Outputs COCO 17-keypoint format validated as FencerPose Pydantic models.

Architecture:
    - Uses rtmlib.Body for combined detection + pose estimation
    - Returns raw detections (tracking is handled by separate Norfair tracker)
    - All outputs are Pydantic-validated FencerPose objects
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.utils.schemas import Keypoint, FencerPose


# =============================================================================
# COCO 17 Keypoint Index Reference
# =============================================================================

COCO_KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

COCO_KEYPOINT_COUNT = 17


# =============================================================================
# RTMPose Estimator
# =============================================================================

class RTMPoseEstimator:
    """
    RTMPose-based pose estimator using rtmlib ONNX runtime.

    This class provides a lightweight, edge-friendly pose estimation solution
    that outputs COCO 17-keypoint format. The output is validated through
    Pydantic models from src.utils.schemas.

    Args:
        mode: Model mode - 'lightweight', 'balanced', or 'performance'
        device: Device to run inference on - 'cpu' or 'cuda'
        conf_threshold: Minimum confidence for keypoint detection (default 0.3)

    Example:
        >>> estimator = RTMPoseEstimator()
        >>> frame = cv2.imread("fencing_frame.jpg")
        >>> poses = estimator.estimate_from_frame(frame)
        >>> print(f"Detected {len(poses)} fencers")
    """

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cpu",
        conf_threshold: float = 0.3,
    ) -> None:
        """
        Initialize RTMPose estimator with rtmlib Body model.

        Args:
            mode: Model complexity - 'lightweight' (fastest), 'balanced', 'performance' (most accurate)
            device: Inference device - 'cpu' or 'cuda'
            conf_threshold: Minimum confidence threshold for keypoint detection

        Raises:
            ValueError: If mode or device values are invalid
        """
        valid_modes = {"lightweight", "balanced", "performance"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        valid_devices = {"cpu", "cuda"}
        if device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{device}'")

        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0.0, 1.0], got {conf_threshold}")

        self.mode = mode
        self.device = device
        self.conf_threshold = conf_threshold

        # Initialize rtmlib Body (handles both detection and pose estimation)
        self._model: Optional["Body"] = None

    def _get_model(self) -> "Body":
        """Lazy initialization of rtmlib model."""
        if self._model is None:
            from rtmlib import Body
            self._model = Body(mode=self.mode, device=self.device)
        return self._model

    def estimate_from_frame(self, frame: np.ndarray) -> List[FencerPose]:
        """
        Estimate poses from a single video frame.

        Args:
            frame: BGR numpy array (H, W, 3) from cv2.imread or video capture

        Returns:
            List[FencerPose] - List of 0-2 fencer poses (untracked, raw detections)

        Raises:
            ValueError: If frame is invalid (wrong shape, dtype, or empty)
        """
        # Validate input
        self._validate_frame(frame)

        # Run inference
        model = self._get_model()
        keypoints, scores = model(frame)

        # Convert to FencerPose list
        poses = self._keypoints_to_poses(keypoints, scores)

        # Handle edge cases: limit to top-2 by bbox area
        if len(poses) > 2:
            poses = self._select_top_poses(poses)

        return poses

    def estimate_from_image_file(self, filepath: str) -> List[FencerPose]:
        """
        Convenience method to estimate poses from an image file.

        Args:
            filepath: Path to image file

        Returns:
            List[FencerPose] - Detected poses

        Raises:
            ValueError: If file cannot be read
        """
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not read image file: {filepath}")
        return self.estimate_from_frame(frame)

    def _validate_frame(self, frame: np.ndarray) -> None:
        """
        Validate input frame.

        Args:
            frame: Input frame to validate

        Raises:
            ValueError: If frame is invalid
        """
        if frame is None:
            raise ValueError("Frame is None")

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be numpy.ndarray, got {type(frame)}")

        if frame.ndim != 3:
            raise ValueError(f"Frame must be 3D (H, W, 3), got {frame.ndim}D")

        if frame.shape[2] != 3:
            raise ValueError(f"Frame must have 3 channels (BGR), got {frame.shape[2]}")

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame has invalid dimensions: {frame.shape}")

        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame must be uint8 dtype, got {frame.dtype}")

    def _keypoints_to_poses(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> List[FencerPose]:
        """
        Convert rtmlib output to List[FencerPose].

        Args:
            keypoints: Array of shape (N_people, 17, 2) - x, y coordinates
            scores: Array of shape (N_people, 17) - confidence scores

        Returns:
            List[FencerPose] - Validated pose objects
        """
        if keypoints.shape[0] == 0:
            return []

        poses: List[FencerPose] = []
        num_people = keypoints.shape[0]

        for person_idx in range(num_people):
            person_keypoints = keypoints[person_idx]
            person_scores = scores[person_idx]

            # Apply confidence threshold and filter keypoints
            filtered_keypoints: List[Keypoint] = []
            for kp_idx in range(COCO_KEYPOINT_COUNT):
                x, y = person_keypoints[kp_idx]
                conf = float(person_scores[kp_idx])

                # Clip coordinates to be non-negative (keypoints can be outside frame bounds)
                x = max(0.0, float(x))
                y = max(0.0, float(y))

                # Only include keypoints above threshold
                if conf >= self.conf_threshold:
                    filtered_keypoints.append(Keypoint(x=x, y=y, conf=conf))
                else:
                    # Include with actual confidence even if below threshold
                    filtered_keypoints.append(Keypoint(x=x, y=y, conf=conf))

            # Skip if too few keypoints detected
            if len(filtered_keypoints) < 12:
                continue

            # Compute bbox from keypoints
            xs = [kp.x for kp in filtered_keypoints]
            ys = [kp.y for kp in filtered_keypoints]
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

            # Create FencerPose with temporary fencer_id (0 = left by default)
            # The actual ID assignment is done by the Norfair tracker
            pose = FencerPose(
                fencer_id=0,  # Temporary - tracker will assign correct ID
                bbox=(x1, y1, x2, y2),
                keypoints=filtered_keypoints,
                is_canonical_flipped=False,
            )
            poses.append(pose)

        return poses

    def _select_top_poses(self, poses: List[FencerPose]) -> List[FencerPose]:
        """
        Select top 2 poses by bbox area when >2 detections.

        Args:
            poses: List of poses to select from

        Returns:
            List of top 2 poses by bbox area
        """
        # Compute bbox areas
        def bbox_area(pose: FencerPose) -> float:
            x1, y1, x2, y2 = pose.bbox
            return (x2 - x1) * (y2 - y1)

        # Sort by area descending
        sorted_poses = sorted(poses, key=bbox_area, reverse=True)

        # Return top 2
        return sorted_poses[:2]

    @property
    def keypoint_count(self) -> int:
        """Return 17 for COCO format."""
        return COCO_KEYPOINT_COUNT

    @staticmethod
    def get_coco_keypoint_names() -> List[str]:
        """Return COCO 17 keypoint names in order."""
        return COCO_KEYPOINT_NAMES.copy()

    def __repr__(self) -> str:
        return (
            f"RTMPoseEstimator(mode='{self.mode}', device='{self.device}', "
            f"conf_threshold={self.conf_threshold})"
        )
