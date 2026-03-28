"""
FencerAI Constants
=================
Version: 1.0 | Last Updated: 2026-03-27

Canonical constants for FencerAI pipeline.
Based on Architectural Decisions (AD4, AD6, AD7).
"""

from __future__ import annotations

from enum import Enum


# =============================================================================
# COCO-17 Keypoint Indices (AD4)
# =============================================================================

COCO_INDICES = {
    # Basic facial keypoints
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    # Upper body keypoints
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    # Lower body keypoints
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


# =============================================================================
# FERA 12-Keypoint Subset (AD4)
# =============================================================================

FERA_INDICES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


# =============================================================================
# 101-Dimensional Feature Vector Index Map (AD1)
# =============================================================================

FEATURE_DIM = 101

# Index ranges for feature groups
STATIC_GEOMETRY_START = 0
STATIC_GEOMETRY_END = 23  # 24 dims: 12 keypoints × 2 coords

CENTER_OF_MASS_START = 24
CENTER_OF_MASS_END = 25  # 2 dims: (x, y)

DISTANCE_START = 26
DISTANCE_END = 36  # 11 dims: physical distance features

ANGULAR_START = 37
ANGULAR_END = 40  # 4 dims: knee angles, weapon elbow, torso lean

TORSO_ORIENTATION_START = 41
TORSO_ORIENTATION_END = 42  # 2 dims

ARM_EXTENSION_START = 43
ARM_EXTENSION_END = 48  # 6 dims

VELOCITY_START = 49
VELOCITY_END = 72  # 24 dims: 1st derivative of static geometry

ACCELERATION_START = 73
ACCELERATION_END = 96  # 24 dims: 2nd derivative

COM_VELOCITY_START = 97
COM_VELOCITY_END = 98  # 2 dims

COM_ACCELERATION_INDEX = 99  # 1 dim

AUDIO_FLAG_INDEX = 100  # 1 dim


# =============================================================================
# Tracker State Enumerations (AD3)
# =============================================================================

class TrackerState(Enum):
    """Norfair tracker states."""
    SEARCHING = "searching"  # Looking for fencer
    TRACKING = "tracking"     # Actively tracking
    LOST = "lost"            # Tracking lost


# =============================================================================
# Audio Event Type Constants (AD7)
# =============================================================================

BLADE_TOUCH = "blade_touch"
PARRY_BEAT = "parry_beat"
REFEREE_HALT = "referee_halt"
FOOTSTEP = "footstep"


# =============================================================================
# Piste Dimensions (AD6)
# =============================================================================

DEFAULT_PISTE_LENGTH = 14.0  # meters (standard fencing)
DEFAULT_PISTE_WIDTH = 1.8  # meters (average of 1.5-2.0)


# =============================================================================
# EMA Alpha Values (AD1)
# =============================================================================

DEFAULT_VELOCITY_ALPHA = 0.7  # EMA smoothing for velocity
DEFAULT_ACCELERATION_ALPHA = 0.7  # EMA smoothing for acceleration


# =============================================================================
# Homography Calibration (AD6)
# =============================================================================

HOMOGRAPHY_MATRIX_SIZE = 3  # 3x3 matrix
MAX_REPROJECTION_ERROR = 10.0  # meters - max error for valid calibration


# =============================================================================
# Tracker Configuration (AD2, AD3)
# =============================================================================

MAX_DETECTION_DISTANCE = 30.0  # pixels
MAX_TRACKER_AGE = 30  # frames
MIN_HIT_HITS = 1  # minimum detections before tracking (1 = immediate)
REFREE_FILTER_Y_THRESHOLD = 0.3  # Y < 0.3 * frame_height = referee
