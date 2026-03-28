"""
FencerAI Feature Extractor
=========================
Version: 1.0 | Last Updated: 2026-03-27

Complete feature extraction pipeline from FrameData to 101-dimensional features.
Handles canonicalization, temporal smoothing, and feature matrix building.

Architecture:
    FrameData → Canonicalize → Extract Features → (2, 101) per frame
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import numpy as np

from src.utils.schemas import FrameData, FencerPose, FeatureMatrix
from src.recognition.feature_math import (
    extract_all_features,
    EMASmoother,
    compute_velocity,
    compute_acceleration,
)
from src.utils.constants import (
    FEATURE_DIM,
    DEFAULT_VELOCITY_ALPHA,
    DEFAULT_ACCELERATION_ALPHA,
)


# =============================================================================
# Feature Extractor State
# =============================================================================

class FeatureExtractorState:
    """Maintains temporal state for feature extraction."""

    def __init__(self, alpha: float = DEFAULT_VELOCITY_ALPHA):
        self.alpha = alpha
        self.geometry_history: Dict[int, np.ndarray] = {0: None, 1: None}
        self.com_history: Dict[int, Tuple[float, float]] = {0: None, 1: None}
        self.velocity_history: Dict[int, np.ndarray] = {0: None, 1: None}
        self.velocity_smoothers: Dict[int, EMASmoother] = {
            0: EMASmoother(alpha=alpha),
            1: EMASmoother(alpha=alpha),
        }
        self.previous_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset all state."""
        self.geometry_history = {0: None, 1: None}
        self.com_history = {0: None, 1: None}
        self.velocity_history = {0: None, 1: None}
        for smoother in self.velocity_smoothers.values():
            smoother.reset()
        self.previous_timestamp = None


# =============================================================================
# Canonicalization
# =============================================================================

def canonicalize_pose(pose: FencerPose) -> FencerPose:
    """
    Canonicalize a pose to left-fencer perspective.

    If fencer_id == 1 (Right fencer), horizontally flip the pose
    so that all features are normalized to a unified perspective.

    Args:
        pose: Input pose

    Returns:
        Canonicalized pose (may be same object if already canonical)
    """
    if pose.is_canonical_flipped:
        return pose  # Already canonical

    if pose.fencer_id == 1:
        # Right fencer - flip horizontally
        frame_width = 1920.0  # Assume standard frame width, should be passed in

        # Flip all keypoints
        flipped_keypoints = []
        for kp in pose.keypoints:
            flipped_keypoints.append(type(kp)(
                x=frame_width - kp.x,
                y=kp.y,
                conf=kp.conf,
            ))

        # Flip bbox
        x1, y1, x2, y2 = pose.bbox
        new_bbox = (frame_width - x2, y1, frame_width - x1, y2)

        return FencerPose(
            fencer_id=pose.fencer_id,
            bbox=new_bbox,
            keypoints=flipped_keypoints,
            is_canonical_flipped=True,
        )

    return pose


def canonicalize_frame(frame: FrameData, frame_width: float = 1920.0) -> FrameData:
    """
    Canonicalize all poses in a frame to left-fencer perspective.

    Args:
        frame: Input frame
        frame_width: Frame width for flipping

    Returns:
        Frame with canonicalized poses
    """
    if len(frame.poses) == 0:
        return frame

    canonicalized_poses = []
    for pose in frame.poses:
        if pose.fencer_id == 1 and not pose.is_canonical_flipped:
            # Flip horizontally
            flipped_keypoints = []
            for kp in pose.keypoints:
                # Flip x coordinate (clip to [0, frame_width] since RTMPose can return values outside bounds)
                flipped_x = max(0.0, min(frame_width, frame_width - kp.x))
                flipped_keypoints.append(type(kp)(
                    x=flipped_x,
                    y=kp.y,
                    conf=kp.conf,
                ))

            x1, y1, x2, y2 = pose.bbox
            # Flip bbox and clip to valid range
            new_x1 = max(0.0, min(frame_width, frame_width - x2))
            new_x2 = max(0.0, min(frame_width, frame_width - x1))
            new_bbox = (new_x1, y1, new_x2, y2)

            canonicalized_poses.append(FencerPose(
                fencer_id=pose.fencer_id,
                bbox=new_bbox,
                keypoints=flipped_keypoints,
                is_canonical_flipped=True,
            ))
        else:
            canonicalized_poses.append(pose)

    return FrameData(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        poses=canonicalized_poses,
        audio_event=frame.audio_event,
        homography_matrix=frame.homography_matrix,
    )


# =============================================================================
# Feature Extractor
# =============================================================================

class FeatureExtractor:
    """
    Extracts 101-dimensional feature vectors from fencing frames.

    Handles:
    - Canonicalization of right-fencer poses
    - Temporal smoothing with EMA
    - Per-fencer feature extraction
    - Building feature matrices from sequences

    Args:
        calibrator: Optional HomographyCalibrator for distance features
        velocity_alpha: EMA alpha for velocity smoothing
        acceleration_alpha: EMA alpha for acceleration smoothing

    Example:
        >>> extractor = FeatureExtractor()
        >>> frame_data = pipeline.process_frame(frame, timestamp=0.0, frame_id=0)
        >>> features = extractor.extract_frame_features(frame_data)
        >>> print(f"Feature shape: {features.shape}")  # (2, 101)
    """

    def __init__(
        self,
        calibrator=None,
        velocity_alpha: float = DEFAULT_VELOCITY_ALPHA,
        acceleration_alpha: float = DEFAULT_ACCELERATION_ALPHA,
    ) -> None:
        self.calibrator = calibrator
        self.velocity_alpha = velocity_alpha
        self.acceleration_alpha = acceleration_alpha
        self._state = FeatureExtractorState(alpha=velocity_alpha)

    def extract_frame_features(
        self,
        frame: FrameData,
        frame_width: float = 1920.0,
    ) -> np.ndarray:
        """
        Extract features from a single frame.

        Args:
            frame: FrameData from perception pipeline
            frame_width: Frame width for canonicalization

        Returns:
            Array of shape (2, 101) with features for both fencers
            Pose 0 is always left fencer (canonical), pose 1 is right
        """
        # Canonicalize frame
        canonical_frame = canonicalize_frame(frame, frame_width=frame_width)

        # Compute dt
        dt = 0.0
        if self._state.previous_timestamp is not None:
            dt = frame.timestamp - self._state.previous_timestamp
        self._state.previous_timestamp = frame.timestamp

        # Extract features for each fencer
        features = np.zeros((2, FEATURE_DIM), dtype=np.float32)

        for pose in canonical_frame.poses:
            fencer_id = pose.fencer_id
            if fencer_id not in (0, 1):
                continue

            # Get keypoint array
            keypoints = pose.get_keypoint_array()  # Shape: (N, 3)

            # Extract features
            feat_101, geometry_24, com, velocity_24 = extract_all_features(
                keypoints=keypoints,
                previous_geometry=self._state.geometry_history.get(fencer_id),
                previous_com=self._state.com_history.get(fencer_id),
                previous_velocity=self._state.velocity_history.get(fencer_id),
                dt=dt,
                calibrator=self.calibrator,
                audio_flag=1.0 if canonical_frame.audio_event else 0.0,
                is_canonical=(fencer_id == 0),
                velocity_ema_alpha=self.velocity_alpha,
                acceleration_ema_alpha=self.acceleration_alpha,
            )

            features[fencer_id] = feat_101

            # Update state
            self._state.geometry_history[fencer_id] = geometry_24
            self._state.com_history[fencer_id] = com
            self._state.velocity_history[fencer_id] = velocity_24

        return features

    def extract_sequence_features(
        self,
        frames: List[FrameData],
        frame_width: float = 1920.0,
    ) -> FeatureMatrix:
        """
        Extract features from a sequence of frames.

        Args:
            frames: List of FrameData objects
            frame_width: Frame width for canonicalization

        Returns:
            FeatureMatrix with shape (N, 2, 101)
        """
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError("Cannot extract features from empty frame list")

        features = np.zeros((n_frames, 2, FEATURE_DIM), dtype=np.float32)
        timestamps = []
        frame_ids = []
        audio_flags = np.zeros((n_frames, 2), dtype=np.float32)

        for i, frame in enumerate(frames):
            timestamps.append(frame.timestamp)
            frame_ids.append(frame.frame_id)

            # Extract frame features
            frame_features = self.extract_frame_features(frame, frame_width=frame_width)
            features[i] = frame_features

            # Set audio flags
            if frame.audio_event is not None:
                audio_flags[i, :] = 1.0

        return FeatureMatrix(
            features=features,
            timestamps=timestamps,
            frame_ids=frame_ids,
            audio_flags=audio_flags,
        )

    def reset(self) -> None:
        """Reset extractor state for new video."""
        self._state.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_single_pose_features(
    pose: FencerPose,
    calibrator=None,
    audio_flag: float = 0.0,
) -> np.ndarray:
    """
    Extract 101-dimensional features from a single pose.

    Args:
        pose: FencerPose object
        calibrator: Optional HomographyCalibrator
        audio_flag: Audio touch flag

    Returns:
        Array of shape (101,)
    """
    keypoints = pose.get_keypoint_array()
    features, _, _, _ = extract_all_features(
        keypoints=keypoints,
        previous_geometry=None,
        previous_com=None,
        previous_velocity=None,
        dt=0.0,
        calibrator=calibrator,
        audio_flag=audio_flag,
        is_canonical=(pose.fencer_id == 0),
    )
    return features
