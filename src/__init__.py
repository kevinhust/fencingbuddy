"""
FencerAI: Spatio-Temporal Fencing Analysis Pipeline
==================================================
Version: 1.0 | Last Updated: 2026-03-27

High-performance edge-first pipeline for extracting 101-dimensional
spatio-temporal feature vectors from fencing videos.
"""

from __future__ import annotations

# Core utilities
from src.utils.schemas import (
    Keypoint,
    FencerPose,
    AudioEvent,
    FrameData,
    FeatureMatrix,
    create_empty_frame,
    create_touch_audio_event,
)

from src.utils.buffer import TimestampedBuffer, BufferEntry

from src.utils.config import (
    Config,
    DEFAULT_CONFIG,
    load_config,
    save_config,
    merge_config,
)

from src.utils.logging import (
    logger,
    configure_logging,
    setup_logger,
    LogLevel,
)

from src.utils.constants import (
    # Keypoint indices
    COCO_INDICES,
    FERA_INDICES,
    # Feature dimensions
    FEATURE_DIM,
    STATIC_GEOMETRY_START,
    STATIC_GEOMETRY_END,
    CENTER_OF_MASS_START,
    CENTER_OF_MASS_END,
    DISTANCE_START,
    DISTANCE_END,
    ANGULAR_START,
    ANGULAR_END,
    TORSO_ORIENTATION_START,
    TORSO_ORIENTATION_END,
    ARM_EXTENSION_START,
    ARM_EXTENSION_END,
    VELOCITY_START,
    VELOCITY_END,
    ACCELERATION_START,
    ACCELERATION_END,
    COM_VELOCITY_START,
    COM_VELOCITY_END,
    COM_ACCELERATION_INDEX,
    AUDIO_FLAG_INDEX,
    # Enums
    TrackerState,
    # Constants
    BLADE_TOUCH,
    PARRY_BEAT,
    REFEREE_HALT,
    FOOTSTEP,
    DEFAULT_PISTE_LENGTH,
    DEFAULT_PISTE_WIDTH,
    DEFAULT_VELOCITY_ALPHA,
    DEFAULT_ACCELERATION_ALPHA,
    MAX_DETECTION_DISTANCE,
    MAX_TRACKER_AGE,
    MIN_HIT_HITS,
    REFREE_FILTER_Y_THRESHOLD,
)

from src.utils.types import (
    KeypointArray,
    PoseKeypoints,
    HomographyMatrix,
    FeatureVector,
    FencerFeaturePair,
    FeatureSequence,
    AudioSample,
    AudioFlags,
)

# Package version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Version
    "__version__",
    # Schemas
    "Keypoint",
    "FencerPose",
    "AudioEvent",
    "FrameData",
    "FeatureMatrix",
    "create_empty_frame",
    "create_touch_audio_event",
    # Buffer
    "TimestampedBuffer",
    "BufferEntry",
    # Config
    "Config",
    "DEFAULT_CONFIG",
    "load_config",
    "save_config",
    "merge_config",
    # Logging
    "logger",
    "configure_logging",
    "setup_logger",
    "LogLevel",
    # Constants
    "COCO_INDICES",
    "FERA_INDICES",
    "FEATURE_DIM",
    "STATIC_GEOMETRY_START",
    "STATIC_GEOMETRY_END",
    "CENTER_OF_MASS_START",
    "CENTER_OF_MASS_END",
    "DISTANCE_START",
    "DISTANCE_END",
    "ANGULAR_START",
    "ANGULAR_END",
    "TORSO_ORIENTATION_START",
    "TORSO_ORIENTATION_END",
    "ARM_EXTENSION_START",
    "ARM_EXTENSION_END",
    "VELOCITY_START",
    "VELOCITY_END",
    "ACCELERATION_START",
    "ACCELERATION_END",
    "COM_VELOCITY_START",
    "COM_VELOCITY_END",
    "COM_ACCELERATION_INDEX",
    "AUDIO_FLAG_INDEX",
    "TrackerState",
    "BLADE_TOUCH",
    "PARRY_BEAT",
    "REFEREE_HALT",
    "FOOTSTEP",
    "DEFAULT_PISTE_LENGTH",
    "DEFAULT_PISTE_WIDTH",
    "DEFAULT_VELOCITY_ALPHA",
    "DEFAULT_ACCELERATION_ALPHA",
    "MAX_DETECTION_DISTANCE",
    "MAX_TRACKER_AGE",
    "MIN_HIT_HITS",
    "REFREE_FILTER_Y_THRESHOLD",
    # Types
    "KeypointArray",
    "PoseKeypoints",
    "HomographyMatrix",
    "FeatureVector",
    "FencerFeaturePair",
    "FeatureSequence",
    "AudioSample",
    "AudioFlags",
]
