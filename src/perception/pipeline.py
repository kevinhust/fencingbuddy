"""
FencerAI Perception Pipeline
=============================
Version: 1.0 | Last Updated: 2026-03-27

Orchestration layer combining RTMPose, FencerTracker, Calibrator, and AudioDetector
into a unified perception pipeline for fencing analysis.

Pipeline Flow:
    Video Frame → RTMPose → FencerTracker → Calibrator → FrameData

Architecture (per ARCHITECTURE.md):
    - RTMPose for pose estimation
    - FencerTracker for dual-fencer tracking with referee filter
    - HomographyCalibrator for pixel-to-meter transformation
    - AudioDetector for blade touch events
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.perception.rtmpose import RTMPoseEstimator, COCO_KEYPOINT_COUNT
from src.perception.tracker import FencerTracker
from src.perception.calibrator import HomographyCalibrator
from src.perception.audio import AudioDetector
from src.perception.audio_buffer import AudioBuffer
from src.utils.schemas import FencerPose, FrameData, AudioEvent


# =============================================================================
# Perception Pipeline
# =============================================================================

class PerceptionPipeline:
    """
    Unified perception pipeline for fencing analysis.

    Combines pose estimation, tracking, calibration, and audio detection
    into a single orchestration layer.

    Args:
        pose_estimator: RTMPoseEstimator instance
        calibrator: HomographyCalibrator instance (optional)
        enable_audio: Whether to enable audio detection
        conf_threshold: Confidence threshold for pose estimation

    Example:
        >>> pipeline = PerceptionPipeline()
        >>> frame = cv2.imread("fencing_frame.jpg")
        >>> frame_data = pipeline.process_frame(frame, timestamp=0.0, frame_id=0)
        >>> print(f"Detected {len(frame_data.poses)} fencers")
    """

    def __init__(
        self,
        pose_estimator: Optional[RTMPoseEstimator] = None,
        calibrator: Optional[HomographyCalibrator] = None,
        enable_audio: bool = False,
        conf_threshold: float = 0.3,
    ) -> None:
        # Initialize components
        self.pose_estimator = pose_estimator or RTMPoseEstimator(
            conf_threshold=conf_threshold
        )
        self.tracker = FencerTracker()
        self.calibrator = calibrator
        self.enable_audio = enable_audio

        # Audio components
        if enable_audio:
            self.audio_buffer = AudioBuffer()
            self.audio_detector = AudioDetector()
        else:
            self.audio_buffer = None
            self.audio_detector = None

        # State
        self._frame_count = 0
        self._is_initialized = False
        self._frame_height: Optional[float] = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_id: int,
        audio_samples: Optional[np.ndarray] = None,
    ) -> FrameData:
        """
        Process a single video frame through the perception pipeline.

        Args:
            frame: BGR numpy array (H, W, 3) from video
            timestamp: Timestamp in seconds from video start
            frame_id: Sequential frame index
            audio_samples: Optional audio samples for this frame

        Returns:
            FrameData with detected poses and optional audio events
        """
        # Store frame height for tracker initialization
        if self._frame_height is None:
            self._frame_height = float(frame.shape[0])

        # Step 1: Pose estimation
        raw_poses = self.pose_estimator.estimate_from_frame(frame)

        # Step 2: Convert to Norfair detections and track
        detections = self._poses_to_detections(raw_poses)

        if not self._is_initialized:
            # Initialize tracker with first frame
            if len(detections) > 0:
                tracked_poses = self.tracker.initialize(
                    detections, frame_height=self._frame_height
                )
                self._is_initialized = True
            else:
                tracked_poses = []
        else:
            # Update tracker
            tracked_poses = self.tracker.update(detections)

        # Step 3: Apply homography transformation if calibrated
        if self.calibrator and self.calibrator.is_calibrated:
            tracked_poses = self._apply_homography(tracked_poses)

        # Step 4: Process audio if enabled
        audio_event: Optional[AudioEvent] = None
        if self.enable_audio and self.audio_buffer and audio_samples is not None:
            self.audio_buffer.append(timestamp=timestamp, samples=audio_samples)
            audio_event = self.audio_detector.detect_touch_simple(
                self.audio_buffer, timestamp
            )

        # Step 5: Build FrameData
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            poses=tracked_poses,
            audio_event=audio_event,
            homography_matrix=self.calibrator.get_homography_matrix()
            if self.calibrator and self.calibrator.is_calibrated
            else None,
        )

        self._frame_count += 1
        return frame_data

    def _poses_to_detections(self, poses: List[FencerPose]):
        """
        Convert FencerPose objects to Norfair Detection objects.

        Args:
            poses: List of FencerPose objects

        Returns:
            List of Norfair Detection objects
        """
        from norfair import Detection

        detections = []
        for pose in poses:
            # Extract keypoints as (N, 2) array
            keypoint_array = pose.get_keypoint_array()
            if keypoint_array.shape[0] < 12:
                continue

            # Get first 12 keypoints (FERA subset)
            keypoints_2d = keypoint_array[:12, :2]
            scores = keypoint_array[:12, 2]

            detection = Detection(
                points=keypoints_2d,
                scores=scores,
                data={'keypoints': keypoint_array},
            )
            detections.append(detection)

        return detections

    def _apply_homography(self, poses: List[FencerPose]) -> List[FencerPose]:
        """
        Apply homography transformation to pose keypoints.

        Args:
            poses: List of FencerPose objects

        Returns:
            List of FencerPose with transformed keypoints
        """
        from src.utils.schemas import Keypoint

        if not self.calibrator or not self.calibrator.is_calibrated:
            return poses

        transformed_poses = []
        for pose in poses:
            new_keypoints = []
            for kp in pose.keypoints:
                meter_x, meter_y = self.calibrator.pixel_to_meter(kp.x, kp.y)
                new_keypoints.append(Keypoint(x=meter_x, y=meter_y, conf=kp.conf))

            # Compute new bbox
            xs = [kp.x for kp in new_keypoints]
            ys = [kp.y for kp in new_keypoints]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            transformed_poses.append(FencerPose(
                fencer_id=pose.fencer_id,
                bbox=(x1, y1, x2, y2),
                keypoints=new_keypoints,
                is_canonical_flipped=pose.is_canonical_flipped,
            ))

        return transformed_poses

    def set_calibrator(self, calibrator: HomographyCalibrator) -> None:
        """
        Set the homography calibrator.

        Args:
            calibrator: HomographyCalibrator instance
        """
        self.calibrator = calibrator

    def is_calibrated(self) -> bool:
        """Return True if calibrator is set and calibrated."""
        return self.calibrator is not None and self.calibrator.is_calibrated

    def reset(self) -> None:
        """Reset pipeline state for new video."""
        self.tracker.reset()
        self._frame_count = 0
        self._is_initialized = False
        self._frame_height = None
        if self.audio_buffer:
            self.audio_buffer.clear()
        if self.audio_detector:
            self.audio_detector.reset()

    @property
    def frame_count(self) -> int:
        """Return total number of frames processed."""
        return self._frame_count

    def __repr__(self) -> str:
        return (
            f"PerceptionPipeline("
            f"enable_audio={self.enable_audio}, "
            f"is_calibrated={self.is_calibrated()}, "
            f"frames_processed={self._frame_count})"
        )
