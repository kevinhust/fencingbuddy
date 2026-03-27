"""
FencerAI Type Aliases
=====================
Version: 1.0 | Last Updated: 2026-03-27

Type aliases for FencerAI pipeline for improved type safety.
"""

from __future__ import annotations

import numpy as np
from typing import Dict


# =============================================================================
# Keypoint Arrays
# =============================================================================

# Shape: (N, 3) - [x, y, confidence]
KeypointArray = np.ndarray

# Dict mapping keypoint name to keypoint array
PoseKeypoints = Dict[str, np.ndarray]


# =============================================================================
# Homography Matrix
# =============================================================================

# Shape: (3, 3) - 3x3 transformation matrix
HomographyMatrix = np.ndarray


# =============================================================================
# Feature Arrays
# =============================================================================

# Shape: (101,) - single feature vector per fencer
FeatureVector = np.ndarray

# Shape: (2, 101) - feature vector for both fencers
FencerFeaturePair = np.ndarray

# Shape: (N_frames, 2, 101) - feature matrix for sequence
FeatureSequence = np.ndarray


# =============================================================================
# Audio Arrays
# =============================================================================

# Shape: (N_samples,) - audio waveform
AudioSample = np.ndarray

# Shape: (N_frames, 2) - binary touch flags per frame
AudioFlags = np.ndarray
