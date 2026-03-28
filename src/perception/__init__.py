"""
FencerAI Perception Module
=========================
RTMPose, Norfair Tracker, Calibrator, and Audio Detection.
"""

from __future__ import annotations

from src.perception.rtmpose import (
    RTMPoseEstimator,
    COCO_KEYPOINT_NAMES,
    COCO_KEYPOINT_COUNT,
)

from src.perception.tracker import (
    FencerTracker,
    PoseEmbedder,
    EMAPredictor,
    fence_detection_distance,
)

from src.perception.calibrator import (
    HomographyCalibrator,
    CalibrationPoint,
)

from src.perception.audio import (
    AudioDetector,
)

from src.perception.audio_buffer import (
    AudioBuffer,
    AudioBufferEntry,
)

from src.perception.pipeline import (
    PerceptionPipeline,
)

__all__ = [
    "RTMPoseEstimator",
    "FencerTracker",
    "PoseEmbedder",
    "EMAPredictor",
    "fence_detection_distance",
    "HomographyCalibrator",
    "CalibrationPoint",
    "AudioDetector",
    "AudioBuffer",
    "AudioBufferEntry",
    "PerceptionPipeline",
    "COCO_KEYPOINT_NAMES",
    "COCO_KEYPOINT_COUNT",
]
