"""
FencerAI Data Contracts - Pydantic Models
=========================================
Version: 1.1 | Last Updated: 2026-03-27

This module defines the canonical data structures for the FencerAI pipeline.
All data moving between the Perception Layer and Recognition Layer MUST be
validated by these Pydantic models.

Design Constraints:
- Target: <150ms end-to-end latency on iPhone-level hardware
- Type Safety: Python 3.9+ type hints for ALL fields
- No raw dicts: Always validate through these models

Milestone 1 Success Criteria:
    A runnable main_pipeline.py that processes a sample video and saves a
    (N_frames, 2, 101) .npy file with valid Pydantic-wrapped data.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, ClassVar
from functools import partial
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Core Data Models
# =============================================================================

class Keypoint(BaseModel):
    """
    Single body keypoint from pose estimation.
    
    Attributes:
        x: Horizontal coordinate (pixels or normalized [0,1])
        y: Vertical coordinate (pixels or normalized [0,1])
        conf: Confidence score [0.0, 1.0]
    """
    x: float = Field(..., ge=0.0, description="Horizontal coordinate")
    y: float = Field(..., ge=0.0, description="Vertical coordinate")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [x, y, conf]."""
        return np.array([self.x, self.y, self.conf], dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Keypoint:
        """Create from numpy array [x, y, conf]."""
        return cls(x=float(arr[0]), y=float(arr[1]), conf=float(arr[2]))


class FencerPose(BaseModel):
    """
    Complete pose estimate for a single fencer in a single frame.
    
    Attributes:
        fencer_id: 0 = Left (Canonical), 1 = Right
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        keypoints: List of 12-33 keypoints (FERA=12, MediaPipe/RTMPose=17/33)
        is_canonical_flipped: True if horizontally flipped for unified perspective
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    fencer_id: int = Field(..., description="0=Left(Canonical), 1=Right")
    bbox: Tuple[float, float, float, float] = Field(
        ..., 
        description="Bounding box [x1, y1, x2, y2]"
    )
    keypoints: List[Keypoint] = Field(
        ..., 
        min_length=12, 
        max_length=33,
        description="12-33 canonical keypoints"
    )
    is_canonical_flipped: bool = Field(
        default=False,
        description="True if horizontally flipped for unified perspective"
    )
    
    @field_validator('fencer_id')
    @classmethod
    def validate_fencer_id(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"fencer_id must be 0 or 1, got {v}")
        return v
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {v}. x2 must > x1, y2 must > y1")
        return v
    
    def get_keypoint_array(self) -> np.ndarray:
        """Convert all keypoints to (N, 3) numpy array [x, y, conf]."""
        return np.array([kp.to_numpy() for kp in self.keypoints], dtype=np.float32)
    
    def shoulder_width(self) -> float:
        """Calculate shoulder width (distance between L/R shoulders)."""
        if len(self.keypoints) < 6:
            return 1.0  # Fallback
        # Assumes indices: 5=LShoulder, 2=RShoulder (COCO format)
        l_shoulder = np.array([self.keypoints[5].x, self.keypoints[5].y])
        r_shoulder = np.array([self.keypoints[2].x, self.keypoints[2].y])
        return float(np.linalg.norm(l_shoulder - r_shoulder))


class AudioEvent(BaseModel):
    """
    Audio event detected from the microphone/buffer.
    
    Attributes:
        timestamp: Time in seconds from video start
        event_type: Type of sound event
        confidence: Detection confidence [0.0, 1.0]
    """
    timestamp: float = Field(..., ge=0.0, description="Time in seconds")
    event_type: str = Field(
        default="blade_touch",
        description="blade_touch, parry_beat, referee_halt, etc."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        valid_types = {"blade_touch", "parry_beat", "referee_halt", "footstep", "unknown"}
        # Accept any for flexibility, but document known types
        return v


class FrameData(BaseModel):
    """
    All data for a single video frame.

    Attributes:
        frame_id: Sequential frame index
        timestamp: Time in seconds
        poses: List of 0-2 FencerPose objects (empty if no detections)
        audio_event: Optional audio event detected in this frame
        homography_matrix: Optional 3x3 matrix for pixel-to-meter transformation
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int = Field(..., ge=0, description="Sequential frame index")
    timestamp: float = Field(..., ge=0.0, description="Time in seconds from video start")
    poses: List[FencerPose] = Field(
        default_factory=list,
        min_length=0,
        max_length=2,
        description="0-2 fencer poses per frame (empty if no detections)"
    )
    audio_event: Optional[AudioEvent] = Field(
        default=None,
        description="Audio event if detected in this frame"
    )
    homography_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 homography matrix for pixel-to-meter transformation"
    )
    
    @field_validator('homography_matrix')
    @classmethod
    def validate_homography(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if v is None:
            return None
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("homography_matrix must be 3x3")
        return v
    
    def get_pose_by_id(self, fencer_id: int) -> Optional[FencerPose]:
        """Get pose for specific fencer (0=Left, 1=Right)."""
        for pose in self.poses:
            if pose.fencer_id == fencer_id:
                return pose
        return None


class FeatureMatrix(BaseModel):
    """
    The ultimate output for Milestone 1 - Sequence Level Feature Matrix.
    
    Attributes:
        features: (N_frames, 2, 101) feature tensor
        timestamps: List of timestamps for each frame
        frame_ids: List of frame indices
        audio_flags: Optional (N_frames, 2) binary touch flags
    
    101-Dimensional Feature Vector Layout (per fencer, per frame):
        [0-23]:   Static Geometry (12 keypoints × 2 coords)
        [24-25]:  Center of Mass (CoM)
        [26-36]:  Distance Features (11 dims)
        [37-40]:  Angles (4 dims)
        [41-42]:  Torso Orientation (2 dims)
        [43-48]:  Arm Extension (6 dims)
        [49-72]:  Velocity - 1st Derivative (24 dims, EMA smoothed)
        [73-96]:  Acceleration - 2nd Derivative (24 dims, EMA smoothed)
        [97-98]:  CoM Velocity (2 dims)
        [99]:     CoM Acceleration (1 dim)
        [100]:    Audio Touch Flag (1 dim)
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={np.ndarray: lambda v: v.tolist()}
    )
    
    features: np.ndarray = Field(
        ..., 
        description="(N_frames, 2, 101) feature matrix, dtype=float32"
    )
    timestamps: List[float] = Field(
        ..., 
        description="Timestamps for each frame"
    )
    frame_ids: List[int] = Field(
        ..., 
        description="Frame indices"
    )
    audio_flags: Optional[np.ndarray] = Field(
        default=None,
        description="(N_frames, 2) binary touch flags"
    )
    
    # Class-level constants for feature dimensions
    FEATURE_DIM: ClassVar[int] = 101
    NUM_FENCERS: ClassVar[int] = 2
    
    @field_validator('features')
    @classmethod
    def validate_features_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 3:
            raise ValueError(f"features must be 3D (N_frames, 2, 101), got {v.ndim}D")
        if v.shape[1] != 2:
            raise ValueError(f"features.shape[1] must be 2 (fencers), got {v.shape[1]}")
        if v.shape[2] != 101:
            raise ValueError(f"features.shape[2] must be 101, got {v.shape[2]}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v
    
    @field_validator('timestamps', 'frame_ids')
    @classmethod
    def validate_lengths(cls, v: List) -> List:
        if len(v) == 0:
            raise ValueError("timestamps/frame_ids cannot be empty")
        return v
    
    def save(self, filepath: str) -> None:
        """Save feature matrix to .npy file."""
        np.save(filepath, self.features)
    
    @classmethod
    def load(cls, filepath: str) -> FeatureMatrix:
        """Load feature matrix from .npy file."""
        data = np.load(filepath)
        n_frames = data.shape[0]
        return cls(
            features=data,
            timestamps=[0.0] * n_frames,  # Placeholder
            frame_ids=list(range(n_frames))  # Placeholder
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_empty_frame(frame_id: int, timestamp: float) -> FrameData:
    """Factory for creating an empty frame placeholder."""
    return FrameData(
        frame_id=frame_id,
        timestamp=timestamp,
        poses=[],
        audio_event=None,
        homography_matrix=None
    )


def create_touch_audio_event(timestamp: float, confidence: float = 0.9) -> AudioEvent:
    """Factory for creating a blade touch audio event."""
    return AudioEvent(
        timestamp=timestamp,
        event_type="blade_touch",
        confidence=confidence
    )


# =============================================================================
# Type Aliases for Pipeline Clarity
# =============================================================================

FrameDataSequence = List[FrameData]
PoseSequence = List[FencerPose]
FeatureVector = np.ndarray  # Shape: (101,) or (2, 101)
FeatureSequence = np.ndarray  # Shape: (N_frames, 2, 101)