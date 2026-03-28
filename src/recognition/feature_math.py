"""
FencerAI Feature Math Engine
===========================
Version: 1.0 | Last Updated: 2026-03-27

Vectorized numpy operations for computing the 101-dimensional feature vector
from fencing pose data.

Feature Categories (per ARCHITECTURE.md Section 5):
    0-23:   Static Geometry (12 keypoints × 2 coords)
    24-25:  Center of Mass (CoM)
    26-36:  Distance (Interaction) - physical meters
    37-40:  Angles
    41-42:  Torso Orientation
    43-48:  Arm Extension
    49-72:  Velocity (1st derivative) - EMA smoothed
    73-96:  Acceleration (2nd derivative) - EMA smoothed
    97-98:  CoM Velocity
    99:     CoM Acceleration
    100:    Audio Touch Flag
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from src.utils.constants import (
    COCO_INDICES,
    FERA_INDICES,
    DEFAULT_VELOCITY_ALPHA,
    DEFAULT_ACCELERATION_ALPHA,
)


# =============================================================================
# Constants for Feature Extraction
# =============================================================================

# COCO keypoint indices for the 12 FERA keypoints
# Order: L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist,
#        L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle
FERA_12_INDICES = [
    COCO_INDICES["left_shoulder"],   # 0 -> 5
    COCO_INDICES["right_shoulder"],  # 1 -> 6
    COCO_INDICES["left_elbow"],      # 2 -> 7
    COCO_INDICES["right_elbow"],    # 3 -> 8
    COCO_INDICES["left_wrist"],      # 4 -> 9
    COCO_INDICES["right_wrist"],     # 5 -> 10
    COCO_INDICES["left_hip"],        # 6 -> 11
    COCO_INDICES["right_hip"],       # 7 -> 12
    COCO_INDICES["left_knee"],       # 8 -> 13
    COCO_INDICES["right_knee"],      # 9 -> 14
    COCO_INDICES["left_ankle"],     # 10 -> 15
    COCO_INDICES["right_ankle"],    # 11 -> 16
]

# Weapon arm indices (assuming right-handed fencer by default)
RIGHT_WEAPON_HAND = COCO_INDICES["right_wrist"]   # 10
RIGHT_WEAPON_ELBOW = COCO_INDICES["right_elbow"]  # 8
RIGHT_WEAPON_SHOULDER = COCO_INDICES["right_shoulder"]  # 6


# =============================================================================
# EMA Utilities
# =============================================================================

class EMASmoother:
    """Exponential Moving Average smoother for temporal features."""

    def __init__(self, alpha: float = DEFAULT_VELOCITY_ALPHA):
        """
        Initialize EMA smoother.

        Args:
            alpha: Smoothing factor (0.6-0.8 recommended per AD8)
        """
        self.alpha = alpha
        self._last_value: Optional[np.ndarray] = None

    def smooth(self, value: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing.

        Args:
            value: Current value array

        Returns:
            Smoothed value array
        """
        if self._last_value is None:
            self._last_value = value.copy()
            return value.copy()

        smoothed = self.alpha * value + (1 - self.alpha) * self._last_value
        self._last_value = smoothed.copy()
        return smoothed

    def smooth_scalar(self, value: float) -> float:
        """Apply EMA smoothing to scalar."""
        if self._last_value is None:
            self._last_value = np.array([value])
            return value

        smoothed = self.alpha * value + (1 - self.alpha) * float(self._last_value[0])
        self._last_value = np.array([smoothed])
        return smoothed

    def reset(self) -> None:
        """Reset EMA state."""
        self._last_value = None


# =============================================================================
# Static Geometry Features (Indices 0-23)
# =============================================================================

def extract_static_geometry(
    keypoints: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract static geometry features (12 keypoints × 2 coords = 24 dims).

    Uses 12 FERA keypoints normalized by shoulder width and centered at pelvis.

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf] for COCO keypoints
        normalize: If True, normalize by shoulder width and center at pelvis

    Returns:
        Array of shape (24,) with [x0, y0, x1, y1, ..., x11, y11]
    """
    if keypoints.shape[0] < 17:
        raise ValueError(f"Need 17 keypoints, got {keypoints.shape[0]}")

    # Extract 12 FERA keypoints
    kp_12 = keypoints[FERA_12_INDICES, :2]  # Shape: (12, 2)

    if normalize:
        # Compute shoulder width for normalization
        l_shoulder = keypoints[COCO_INDICES["left_shoulder"], :2]
        r_shoulder = keypoints[COCO_INDICES["right_shoulder"], :2]
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

        if shoulder_width < 1e-6:
            shoulder_width = 1.0  # Avoid division by zero

        # Compute pelvis center for centering
        l_hip = keypoints[COCO_INDICES["left_hip"], :2]
        r_hip = keypoints[COCO_INDICES["right_hip"], :2]
        pelvis_center = (l_hip + r_hip) / 2

        # Normalize and center
        kp_12 = (kp_12 - pelvis_center) / shoulder_width

    # Flatten to (24,)
    return kp_12.flatten().astype(np.float32)


def compute_center_of_mass(keypoints: np.ndarray) -> Tuple[float, float]:
    """
    Compute center of mass (pelvis/hip center).

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf]

    Returns:
        Tuple of (x, y) center of mass
    """
    l_hip = keypoints[COCO_INDICES["left_hip"], :2]
    r_hip = keypoints[COCO_INDICES["right_hip"], :2]
    com = (l_hip + r_hip) / 2
    return (float(com[0]), float(com[1]))


# =============================================================================
# Distance Features (Indices 26-36) - Physical Meters
# =============================================================================

def extract_distance_features(
    left_keypoints: np.ndarray,
    right_keypoints: np.ndarray,
    calibrator,  # HomographyCalibrator
) -> np.ndarray:
    """
    Extract distance features (11 dims) in physical meters.

    Requires homography calibrator for pixel-to-meter transformation.

    Args:
        left_keypoints: Left fencer keypoints (17, 3)
        right_keypoints: Right fencer keypoints (17, 3)
        calibrator: HomographyCalibrator instance

    Returns:
        Array of shape (11,) with distance features
    """
    if not calibrator.is_calibrated:
        raise RuntimeError("Calibrator must be calibrated for distance features")

    distances = []

    # Compute centers of mass
    l_com_x, l_com_y = compute_center_of_mass(left_keypoints)
    r_com_x, r_com_y = compute_center_of_mass(right_keypoints)

    # Transform to meters
    l_com_m = calibrator.pixel_to_meter(l_com_x, l_com_y)
    r_com_m = calibrator.pixel_to_meter(r_com_x, r_com_y)

    # 1. Inter-fencer pelvis distance (2D)
    inter_pelvis_dist = np.sqrt((l_com_m[0] - r_com_m[0])**2 + (l_com_m[1] - r_com_m[1])**2)
    distances.append(inter_pelvis_dist)

    # 2. Inter-fencer foot distance (min of 4 combinations)
    l_ankle = left_keypoints[COCO_INDICES["left_ankle"], :2]
    r_ankle = right_keypoints[COCO_INDICES["right_ankle"], :2]
    l_ankle_m = calibrator.pixel_to_meter(l_ankle[0], l_ankle[1])
    r_ankle_m = calibrator.pixel_to_meter(r_ankle[0], r_ankle[1])

    foot_dists = []
    for l_a in [left_keypoints[COCO_INDICES["left_ankle"], :2],
                left_keypoints[COCO_INDICES["right_ankle"], :2]]:
        for r_a in [right_keypoints[COCO_INDICES["left_ankle"], :2],
                    right_keypoints[COCO_INDICES["right_ankle"], :2]]:
            l_a_m = calibrator.pixel_to_meter(l_a[0], l_a[1])
            r_a_m = calibrator.pixel_to_meter(r_a[0], r_a[1])
            d = np.sqrt((l_a_m[0] - r_a_m[0])**2 + (l_a_m[1] - r_a_m[1])**2)
            foot_dists.append(d)
    distances.append(min(foot_dists))

    # 3. Left fencer stance width (left ankle to right ankle distance)
    l_l_ankle = left_keypoints[COCO_INDICES["left_ankle"], :2]
    l_r_ankle = left_keypoints[COCO_INDICES["right_ankle"], :2]
    l_l_ankle_m = calibrator.pixel_to_meter(l_l_ankle[0], l_l_ankle[1])
    l_r_ankle_m = calibrator.pixel_to_meter(l_r_ankle[0], l_r_ankle[1])
    l_stance = np.sqrt((l_l_ankle_m[0] - l_r_ankle_m[0])**2 + (l_l_ankle_m[1] - l_r_ankle_m[1])**2)
    distances.append(l_stance)

    # 4. Right fencer stance width
    r_l_ankle = right_keypoints[COCO_INDICES["left_ankle"], :2]
    r_r_ankle = right_keypoints[COCO_INDICES["right_ankle"], :2]
    r_l_ankle_m = calibrator.pixel_to_meter(r_l_ankle[0], r_l_ankle[1])
    r_r_ankle_m = calibrator.pixel_to_meter(r_r_ankle[0], r_r_ankle[1])
    r_stance = np.sqrt((r_l_ankle_m[0] - r_r_ankle_m[0])**2 + (r_l_ankle_m[1] - r_r_ankle_m[1])**2)
    distances.append(r_stance)

    # 5-7. Weapon hand to opponent torso distances (left fencer weapon = right hand)
    weapon_hand = right_keypoints[COCO_INDICES["right_wrist"], :2]
    weapon_hand_m = calibrator.pixel_to_meter(weapon_hand[0], weapon_hand[1])

    opponent_shoulder_center = (
        (left_keypoints[COCO_INDICES["left_shoulder"], :2] + left_keypoints[COCO_INDICES["right_shoulder"], :2]) / 2
    )
    opponent_hip_center = (
        (left_keypoints[COCO_INDICES["left_hip"], :2] + left_keypoints[COCO_INDICES["right_hip"], :2]) / 2
    )
    opponent_shoulder_m = calibrator.pixel_to_meter(opponent_shoulder_center[0], opponent_shoulder_center[1])
    opponent_hip_m = calibrator.pixel_to_meter(opponent_hip_center[0], opponent_hip_center[1])

    weapon_to_shoulder = np.sqrt((weapon_hand_m[0] - opponent_shoulder_m[0])**2 + (weapon_hand_m[1] - opponent_shoulder_m[1])**2)
    weapon_to_hip = np.sqrt((weapon_hand_m[0] - opponent_hip_m[0])**2 + (weapon_hand_m[1] - opponent_hip_m[1])**2)

    distances.append(weapon_to_shoulder)
    distances.append(weapon_to_hip)

    # 8. Weapon arm extension ratio (distance shoulder to wrist / reach)
    l_shoulder = left_keypoints[COCO_INDICES["left_shoulder"], :2]
    r_shoulder = right_keypoints[COCO_INDICES["right_shoulder"], :2]
    l_elbow = left_keypoints[COCO_INDICES["left_elbow"], :2]
    l_wrist = left_keypoints[COCO_INDICES["left_wrist"], :2]

    l_shoulder_m = calibrator.pixel_to_meter(l_shoulder[0], l_shoulder[1])
    l_wrist_m = calibrator.pixel_to_meter(l_wrist[0], l_wrist[1])
    l_arm_len = np.sqrt((l_shoulder_m[0] - l_wrist_m[0])**2 + (l_shoulder_m[1] - l_wrist_m[1])**2)
    distances.append(l_arm_len)

    # 9. Average fencer height (shoulder to ankle)
    l_knee = left_keypoints[COCO_INDICES["left_knee"], :2]
    l_ankle = left_keypoints[COCO_INDICES["left_ankle"], :2]
    l_knee_m = calibrator.pixel_to_meter(l_knee[0], l_knee[1])
    l_ankle_m = calibrator.pixel_to_meter(l_ankle[0], l_ankle[1])
    l_height = np.sqrt((l_shoulder_m[0] - l_ankle_m[0])**2 + (l_shoulder_m[1] - l_ankle_m[1])**2)
    distances.append(l_height)

    # 10-11. Reserved/margin features (zeros for now)
    distances.append(0.0)
    distances.append(0.0)

    return np.array(distances[:11], dtype=np.float32)


# =============================================================================
# Angular Features (Indices 37-40)
# =============================================================================

def extract_angle_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract angular features (4 dims).

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf]

    Returns:
        Array of shape (4,) with [front_knee_angle, back_knee_angle, weapon_elbow_angle, torso_lean_angle]
    """
    angles = []

    # Helper to compute angle at middle point of 3 points
    def compute_angle(p1, p2, p3):
        """Compute angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    # 1. Front knee angle (assuming left is front for now - canonical perspective)
    l_knee = keypoints[COCO_INDICES["left_knee"], :2]
    l_hip = keypoints[COCO_INDICES["left_hip"], :2]
    l_ankle = keypoints[COCO_INDICES["left_ankle"], :2]
    front_knee = compute_angle(l_hip, l_knee, l_ankle)
    angles.append(front_knee)

    # 2. Back knee angle
    r_knee = keypoints[COCO_INDICES["right_knee"], :2]
    r_hip = keypoints[COCO_INDICES["right_hip"], :2]
    r_ankle = keypoints[COCO_INDICES["right_ankle"], :2]
    back_knee = compute_angle(r_hip, r_knee, r_ankle)
    angles.append(back_knee)

    # 3. Weapon elbow angle (right elbow for right-handed)
    r_shoulder = keypoints[COCO_INDICES["right_shoulder"], :2]
    r_elbow = keypoints[COCO_INDICES["right_elbow"], :2]
    r_wrist = keypoints[COCO_INDICES["right_wrist"], :2]
    weapon_elbow = compute_angle(r_shoulder, r_elbow, r_wrist)
    angles.append(weapon_elbow)

    # 4. Torso lean angle (shoulder center to hip center)
    l_shoulder = keypoints[COCO_INDICES["left_shoulder"], :2]
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2
    torso_vector = shoulder_center - hip_center
    torso_lean = np.arctan2(torso_vector[1], torso_vector[0])
    angles.append(torso_lean)

    return np.array(angles, dtype=np.float32)


def extract_torso_orientation(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract torso orientation (2 dims).

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf]

    Returns:
        Array of shape (2,) with [dx, dy] direction vector from shoulder to hip
    """
    l_shoulder = keypoints[COCO_INDICES["left_shoulder"], :2]
    r_shoulder = keypoints[COCO_INDICES["right_shoulder"], :2]
    l_hip = keypoints[COCO_INDICES["left_hip"], :2]
    r_hip = keypoints[COCO_INDICES["right_hip"], :2]

    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2

    direction = shoulder_center - hip_center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 1.0])  # Default to vertical
    else:
        direction = direction / norm

    return direction.astype(np.float32)


# =============================================================================
# Arm Extension Features (Indices 43-48)
# =============================================================================

def extract_arm_extension_features(
    keypoints: np.ndarray,
    is_canonical: bool = False,
) -> np.ndarray:
    """
    Extract arm extension features (6 dims).

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf]
        is_canonical: If True, assumes left-handed weapon (canonical perspective)

    Returns:
        Array of shape (6,) with arm extension features
    """
    features = []

    # Weapon arm indices (depends on handedness)
    if is_canonical:
        weapon_shoulder = COCO_INDICES["left_shoulder"]
        weapon_elbow = COCO_INDICES["left_elbow"]
        weapon_wrist = COCO_INDICES["left_wrist"]
    else:
        weapon_shoulder = COCO_INDICES["right_shoulder"]
        weapon_elbow = COCO_INDICES["right_elbow"]
        weapon_wrist = COCO_INDICES["right_wrist"]

    # Compute arm segment lengths
    shoulder = keypoints[weapon_shoulder, :2]
    elbow = keypoints[weapon_elbow, :2]
    wrist = keypoints[weapon_wrist, :2]

    upper_arm = np.linalg.norm(elbow - shoulder)
    forearm = np.linalg.norm(wrist - elbow)
    total_reach = upper_arm + forearm

    # 1. Extension ratio (current reach / max reach estimate)
    features.append(forearm / (upper_arm + 1e-6))

    # 2. Total reach normalized
    features.append(total_reach / 1000.0)  # Normalize assuming ~1m total reach

    # 3. Weapon arm angle relative to horizontal
    arm_vector = wrist - shoulder
    features.append(np.arctan2(arm_vector[1], arm_vector[0]))

    # 4-5. Off-arm (non-weapon) extension
    if is_canonical:
        off_shoulder = COCO_INDICES["right_shoulder"]
        off_elbow = COCO_INDICES["right_elbow"]
        off_wrist = COCO_INDICES["right_wrist"]
    else:
        off_shoulder = COCO_INDICES["left_shoulder"]
        off_elbow = COCO_INDICES["left_elbow"]
        off_wrist = COCO_INDICES["left_wrist"]

    off_shoulder_pt = keypoints[off_shoulder, :2]
    off_elbow_pt = keypoints[off_elbow, :2]
    off_wrist_pt = keypoints[off_wrist, :2]

    off_upper = np.linalg.norm(off_elbow_pt - off_shoulder_pt)
    off_forearm = np.linalg.norm(off_wrist_pt - off_elbow_pt)

    features.append(off_forearm / (off_upper + 1e-6))
    features.append((off_upper + off_forearm) / 1000.0)

    # 6. Weapon arm elevation angle
    features.append(np.arctan2(arm_vector[1], np.abs(arm_vector[0]) + 1e-6))

    return np.array(features, dtype=np.float32)


# =============================================================================
# Temporal Derivatives (Velocity/Acceleration)
# =============================================================================

def compute_velocity(
    current_geometry: np.ndarray,
    previous_geometry: np.ndarray,
    dt: float,
    ema_smoother: Optional[EMASmoother] = None,
) -> np.ndarray:
    """
    Compute velocity from geometry features.

    Args:
        current_geometry: Current static geometry (24,)
        previous_geometry: Previous static geometry (24,)
        dt: Time delta in seconds
        ema_smoother: Optional EMA smoother

    Returns:
        Velocity array (24,)
    """
    if dt < 1e-6:
        return np.zeros(24, dtype=np.float32)

    velocity = (current_geometry - previous_geometry) / dt

    if ema_smoother is not None:
        velocity = ema_smoother.smooth(velocity)

    return velocity.astype(np.float32)


def compute_acceleration(
    current_velocity: np.ndarray,
    previous_velocity: np.ndarray,
    dt: float,
    ema_smoother: Optional[EMASmoother] = None,
) -> np.ndarray:
    """
    Compute acceleration from velocity features.

    Args:
        current_velocity: Current velocity (24,)
        previous_velocity: Previous velocity (24,)
        dt: Time delta in seconds
        ema_smoother: Optional EMA smoother

    Returns:
        Acceleration array (24,)
    """
    if dt < 1e-6:
        return np.zeros(24, dtype=np.float32)

    acceleration = (current_velocity - previous_velocity) / dt

    if ema_smoother is not None:
        acceleration = ema_smoother.smooth(acceleration)

    return acceleration.astype(np.float32)


# =============================================================================
# Meta Features (Indices 97-100)
# =============================================================================

def extract_meta_features(
    current_com: Tuple[float, float],
    previous_com: Optional[Tuple[float, float]],
    dt: float,
    com_velocity_ema: Optional[EMASmoother] = None,
) -> np.ndarray:
    """
    Extract meta features (4 dims): CoM velocity (2), CoM acceleration (1), Audio flag (1).

    Args:
        current_com: Current center of mass (x, y)
        previous_com: Previous center of mass or None
        dt: Time delta in seconds
        com_velocity_ema: EMA smoother for CoM velocity

    Returns:
        Array of shape (4,) with [com_vx, com_vy, com_acc, audio_flag]
    """
    features = np.zeros(4, dtype=np.float32)

    if previous_com is not None and dt > 1e-6:
        com_vx = (current_com[0] - previous_com[0]) / dt
        com_vy = (current_com[1] - previous_com[1]) / dt

        if com_velocity_ema is not None:
            com_vx = com_velocity_ema.smooth_scalar(com_vx)
            com_vy = com_velocity_ema.smooth_scalar(com_vy)

        features[0] = com_vx
        features[1] = com_vy

    # CoM acceleration (placeholder - would need previous velocity)
    features[2] = 0.0

    # Audio flag - placeholder, set by caller
    features[3] = 0.0

    return features


# =============================================================================
# Complete Feature Extraction
# =============================================================================

def extract_all_features(
    keypoints: np.ndarray,
    previous_geometry: Optional[np.ndarray] = None,
    previous_com: Optional[Tuple[float, float]] = None,
    previous_velocity: Optional[np.ndarray] = None,
    dt: float = 0.0,
    calibrator=None,
    audio_flag: float = 0.0,
    is_canonical: bool = False,
    velocity_ema_alpha: float = DEFAULT_VELOCITY_ALPHA,
    acceleration_ema_alpha: float = DEFAULT_ACCELERATION_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all 101 features from keypoints.

    Args:
        keypoints: Array of shape (17, 3) with [x, y, conf]
        previous_geometry: Previous static geometry for velocity
        previous_com: Previous center of mass
        previous_velocity: Previous velocity for acceleration (24,)
        dt: Time delta in seconds
        calibrator: HomographyCalibrator for distance features
        audio_flag: Audio touch flag (0.0 or 1.0)
        is_canonical: If True, use canonical (left-handed) perspective
        velocity_ema_alpha: EMA alpha for velocity smoothing
        acceleration_ema_alpha: EMA alpha for acceleration smoothing

    Returns:
        Tuple of (features_101, current_geometry_24, current_com, current_velocity)
    """
    features = np.zeros(101, dtype=np.float32)

    # Static geometry (0-23)
    current_geometry = extract_static_geometry(keypoints, normalize=True)
    features[0:24] = current_geometry

    # Center of mass (24-25)
    current_com = compute_center_of_mass(keypoints)
    features[24] = current_com[0]
    features[25] = current_com[1]

    # Distance features (26-36) - require calibrator and both fencers
    # These are inter-fencer distances, set to 0 for single-fencer extraction
    # Full implementation would need both fencers' keypoints
    features[26:37] = 0.0

    # Angular features (37-40)
    angles = extract_angle_features(keypoints)
    features[37:41] = angles

    # Torso orientation (41-42)
    torso = extract_torso_orientation(keypoints)
    features[41:43] = torso

    # Arm extension (43-48)
    arm_ext = extract_arm_extension_features(keypoints, is_canonical=is_canonical)
    features[43:49] = arm_ext

    # Velocity (49-72)
    current_velocity = np.zeros(24, dtype=np.float32)
    if previous_geometry is not None and dt > 1e-6:
        velocity_smoother = EMASmoother(alpha=velocity_ema_alpha)
        current_velocity = compute_velocity(
            current_geometry, previous_geometry, dt, velocity_smoother
        )
        features[49:73] = current_velocity

    # Acceleration (73-96)
    if previous_velocity is not None and dt > 1e-6:
        acceleration_smoother = EMASmoother(alpha=acceleration_ema_alpha)
        acceleration = compute_acceleration(
            current_velocity, previous_velocity, dt, acceleration_smoother
        )
        features[73:97] = acceleration

    # Meta features (97-100)
    meta = extract_meta_features(current_com, previous_com, dt)
    features[97] = meta[0]  # CoM vx
    features[98] = meta[1]  # CoM vy
    features[99] = meta[2]  # CoM acceleration
    features[100] = audio_flag  # Audio flag

    return features, current_geometry, current_com, current_velocity
