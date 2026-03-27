---

### 文件 2：`DATA_SCHEMA.md` (映射至 `src/utils/schemas.py`)
```python
# DATA_SCHEMA.md - FencerAI Data Contracts
# Version: 1.1 | Last Updated: 2026-03
# Please implement these using Pydantic in `src/utils/schemas.py`

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import numpy as np

class Keypoint(BaseModel):
    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    conf: float = Field(..., ge=0.0, le=1.0)

class FencerPose(BaseModel):
    fencer_id: int  # 0: Left (Canonical), 1: Right
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    # Flexible: 12 canonical keypoints for FERA, or up to 33 for MediaPipe/RTMPose full
    keypoints: List[Keypoint] = Field(..., min_items=12, max_items=33)
    is_canonical_flipped: bool = False  # True if horizontally flipped for unified perspective

class AudioEvent(BaseModel):
    timestamp: float
    event_type: str = "blade_touch" # e.g., "blade_touch", "parry_beat", "referee_halt"
    confidence: float = Field(..., ge=0.0, le=1.0)

class FrameData(BaseModel):
    frame_id: int
    timestamp: float
    # Can be 1 during extreme occlusion/infighting, ideally 2 after tracker filtering
    poses: List[FencerPose] = Field(..., max_items=2) 
    audio_event: Optional[AudioEvent] = None
    homography_matrix: Optional[List[List[float]]] = None  # 3x3 matrix for pixel-to-meter

class FeatureMatrix(BaseModel):
    """The ultimate output for Milestone 1 - Sequence Level"""
    features: np.ndarray  # Shape: (N_frames, 2, 101), dtype=float32
    timestamps: List[float]
    frame_ids: List[int]
    audio_flags: Optional[np.ndarray] = None  # Shape: (N_frames, 2) binary touch flags
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda v: v.tolist()}