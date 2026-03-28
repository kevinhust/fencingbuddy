"""
FencerAI Recognition Module
=========================
101-Dimensional Feature Extraction, Canonicalization, and Feature Math.
"""

from __future__ import annotations

from src.recognition.feature_math import (
    extract_static_geometry,
    extract_distance_features,
    extract_angle_features,
    extract_torso_orientation,
    extract_arm_extension_features,
    extract_all_features,
    compute_velocity,
    compute_acceleration,
    EMASmoother,
    FERA_12_INDICES,
)

from src.recognition.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorState,
    canonicalize_pose,
    canonicalize_frame,
    extract_single_pose_features,
)

__all__ = [
    "extract_static_geometry",
    "extract_distance_features",
    "extract_angle_features",
    "extract_torso_orientation",
    "extract_arm_extension_features",
    "extract_all_features",
    "compute_velocity",
    "compute_acceleration",
    "EMASmoother",
    "FERA_12_INDICES",
    "FeatureExtractor",
    "FeatureExtractorState",
    "canonicalize_pose",
    "canonicalize_frame",
    "extract_single_pose_features",
]
