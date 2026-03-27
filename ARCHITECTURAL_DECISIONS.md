# Architectural Decisions (AD)
*Version: 1.0 | Created: 2026-03-27*

This document records all architectural decisions for FencerAI. All implementation must follow these decisions. Deviations require an AD number change and documentation update.

---

## AD1: 101-Dimensional Feature Vector Canonical Index Map
**Status**: ACCEPTED
**Supersedes**: ARCHITECTURE.md lines 30-40 (text description only; index ranges were correct)

### Decision
The 101-dimensional feature vector per fencer per frame is defined as follows:

| Indices | Feature Group | Dimensions | Notes |
|---------|--------------|------------|-------|
| 0-23 | Static Geometry | 24 | 12 keypoints × 2 coords (x, y), normalized by shoulder-width, pelvis-centered |
| 24-25 | Center of Mass (CoM) | 2 | Pelvis/hip center (x, y) in normalized space |
| 26-36 | Distance Features | 11 | Physical meters via homography transformation |
| 37-40 | Angular Features | 4 | Front/back knee angles, weapon elbow angle, torso lean |
| 41-42 | Torso Orientation | 2 | Vector from shoulder center to hip center |
| 43-48 | Arm Extension | 6 | Weapon arm extension ratio + directional + secondary arm |
| 49-72 | Velocity (1st Deriv) | 24 | Derivative of static geometry, EMA smoothed (α=0.6-0.8), units: pixels/sec |
| 73-96 | Acceleration (2nd Deriv) | 24 | Derivative of velocity, EMA smoothed (α=0.6-0.8), units: pixels/sec² |
| 97-98 | CoM Velocity | 2 | Magnitude of center-of-mass velocity |
| 99 | CoM Acceleration | 1 | Magnitude of center-of-mass acceleration |
| 100 | Audio Touch Flag | 1 | 1.0 if blade_touch detected, 0.0 otherwise |

**Total: 101 dimensions**

### Velocity/Acceleration Clarification
- Indices 49-72 are the **first derivative** (velocity) of indices 0-23
- Indices 73-96 are the **second derivative** (acceleration) of indices 0-23
- Both use **actual timestamps (Δt)** for accurate derivatives across variable FPS
- EMA smoothing alpha range: 0.6-0.8

### Implementation
- `src/recognition/feature_math.py` must implement exact index mapping
- `src/utils/schemas.py` FeatureMatrix docstring is the **authoritative source**
- Test `tests/test_feature_math.py` must verify all 101 indices match exactly

---

## AD2: Y-Axis Convention for Tracker Initialization
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 43

### Decision
OpenCV coordinate convention is used:
- **Origin**: Top-left corner
- **Y-axis**: Increases downward (standard screen coordinates)
- **"Bottom 70% of Y-axis"** = pixels where `Y > 0.3 * frame_height`

### Rationale
Fencers' feet appear in the **lower portion** of the frame (higher Y values), not the upper portion. Referees standing upright appear in the upper 30%. By filtering to bottom 70%, we exclude referees and focus on fencers.

### Tracker Initialization Rule
```
IF frame has > 2 detections:
  1. Filter to detections in bottom 70% of frame (Y > 0.3 * frame_height)
  2. Select 2 largest BBoxes by area
  3. Assign fencer_id=0 to left-most (lower X), fencer_id=1 to right-most
```

### Implementation
- `src/perception/tracker.py`: FencerTracker class
- Referee Filter: ignore detections with centroid Y < 0.3 * frame_height

---

## AD3: Norfair Tracker Integration with EMA Position Prediction
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 45, DEVELOPMENT_PLAN.md Phase 2.2

### Decision
Norfair's Tracker does **NOT** natively support EMA-based position prediction. To implement ARCHITECTURE.md's graceful failure requirement:

1. **Extend Norfair Tracker** with custom position predictor
2. **EMA Position Predictor**: When fencer ID is lost due to occlusion:
   - Maintain EMA-smoothed position history
   - Predict next position using last velocity (ema_position + ema_velocity * dt)
3. **Fallback**: Return `None` if no history available

### Tracker Motion Model
```
Simple Exponential Smoothing (SES) for position prediction:
  - smoothed_pos = alpha * raw_pos + (1 - alpha) * predicted_pos
  - alpha = 0.7 (within 0.6-0.8 range)
  - Prediction: next_pos = smoothed_pos + smoothed_vel * dt
```

### Graceful Failure Behavior
| Scenario | Behavior |
|----------|----------|
| Single detection, missing partner | Return 1 fencer + 1 None placeholder |
| Occlusion < 5 frames | Use EMA-predicted position |
| Occlusion ≥ 5 frames | Treat as new detection, may reassign ID |
| >2 detections | Prioritize by pose-embedding similarity to locked IDs |

### Implementation
- `src/perception/tracker.py`: FencerTracker extends Norfair Tracker
- `src/perception/pose_embedder.py`: Cosine similarity for pose matching
- Norfair distance metric: Euclidean (override possible via custom distance_fn)

---

## AD4: Canonical Keypoint Format — COCO-17
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 32, schemas.py line 106-109

### Decision
RTMPose outputs **COCO-17 format** as canonical:

| Index | Keypoint | COCO-17 |
|-------|----------|---------|
| 0 | nose | ✓ |
| 1 | left_eye | ✓ |
| 2 | right_eye | ✓ |
| 3 | left_ear | ✓ |
| 4 | right_ear | ✓ |
| 5 | left_shoulder | ✓ |
| 6 | right_shoulder | ✓ |
| 7 | left_elbow | ✓ |
| 8 | right_elbow | ✓ |
| 9 | left_wrist | ✓ |
| 10 | right_wrist | ✓ |
| 11 | left_hip | ✓ |
| 12 | right_hip | ✓ |
| 13 | left_knee | ✓ |
| 14 | right_knee | ✓ |
| 15 | left_ankle | ✓ |
| 16 | right_ankle | ✓ |

### FERA 12-Keypoint Subset (for feature extraction)
Feature extraction uses these COCO-17 indices:
- L_Shoulder: 5, R_Shoulder: 6
- L_Elbow: 7, R_Elbow: 8
- L_Wrist: 9, R_Wrist: 10
- L_Hip: 11, R_Hip: 12
- L_Knee: 13, R_Knee: 14
- L_Ankle: 15, R_Ankle: 16

### Implementation
- `src/utils/constants.py`: Define COCO_INDICES dict mapping names to indices
- `src/recognition/feature_math.py`: Extract 12 FERA keypoints from COCO-17 via indices
- `schemas.py shoulder_width()`: Uses COCO indices [5, 2] — **CORRECT** for COCO-17

---

## AD5: Horizontal Canonicalization — Always Left Fencer
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 19, DEVELOPMENT_PLAN.md Phase 3.8

### Decision
The downstream classifier only knows about the **"Left" fencer**. The pipeline must canonicalize all input to this perspective:

```
IF fencer_id == 1 (Right):
  - Horizontally flip ALL x-coordinates: x_flipped = frame_width - x
  - Set is_canonical_flipped = True
  - Assign temporary fencer_id = 0 for feature extraction

IF fencer_id == 0 (Left):
  - No change
  - is_canonical_flipped = False (default)
```

### Canonicalization Order
1. After Norfair Tracker assigns fencer IDs
2. Before Calibrator transforms coordinates
3. Before Feature Extractor processes poses

### Implementation
- `src/recognition/canonicalize.py`: `canonicalize_pose()`, `canonicalize_frame()`
- Applied in `src/perception/pipeline.py` before passing to FeatureExtractor
- `is_canonical_flipped` flag preserved in FencerPose for potential inverse transform

---

## AD6: Homography Calibration — Piste-End Detection
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 25, DEVELOPMENT_PLAN.md Phase 2.3

### Decision
Pixel-to-meter transformation uses **manual marker-based calibration** with automatic piste-end detection as辅助:

1. **Primary**: User provides 4 corner markers of the piste (physical已知点)
2. **Secondary**: Automatic edge detection of white piste lines as fallback
3. **RANSAC**: Used for robust homography estimation from 4+ point correspondences
4. **Confidence**: Calibration confidence score based on reprojection error

### Homography Matrix Behavior
| Scenario | Output |
|----------|--------|
| Fully calibrated | Valid 3x3 matrix in FrameData.homography_matrix |
| Not calibrated | `None` — distance features return zeros |
| Partial calibration | Use available points, log warning |

### Calibration Points (Manual)
```
Physical piste corners (in meters, relative to left end):
  - Top-left: (0.0, 0.0)
  - Top-right: (piste_length, 0.0)
  - Bottom-right: (piste_length, piste_width)
  - Bottom-left: (0.0, piste_width)

Default piste dimensions:
  - Length: 14.0 meters (standard fencing)
  - Width: 1.5-2.0 meters (from scoring box)
```

### Implementation
- `src/perception/calibrator.py`: HomographyCalibrator class
- `pixel_to_meter()`: Apply homography, validate output < 10m
- `meter_to_pixel()`: Inverse transformation for visualization

---

## AD7: Audio Touch Flag — Binary Threshold
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 40, DEVELOPMENT_PLAN.md Phase 3.7.2

### Decision
The audio touch flag (index 100) is **binary** (0.0 or 1.0), not a confidence value:

```
IF blade_touch detected AND confidence >= 0.5:
  flag = 1.0
ELSE:
  flag = 0.0
```

### Rationale
- The feature vector is designed for classification, not regression
- Binary flag aligns with the ML task (touch vs no-touch)
- Confidence threshold 0.5 is a sensible default for "likely real event"
- Low-confidence events are logged but do not set the flag

### Audio Event Detection
| Event Type | Flag Behavior |
|------------|--------------|
| blade_touch | Sets flag=1.0 if confidence ≥ 0.5 |
| parry_beat | Does NOT set flag (different scoring event) |
| referee_halt | Does NOT set flag (stop, not touch) |
| Other | Ignored |

### Implementation
- `src/perception/audio.py`: AudioDetector class
- `src/recognition/feature_math.py`: `extract_audio_flag()` returns 0.0 or 1.0

---

## AD8: Latency Target — 500ms End-to-End
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md line 5, DEVELOPMENT_PLAN.md, schemas.py

### Decision
**500ms** is the authoritative end-to-end latency target (not 150ms).

| Target | Context |
|--------|---------|
| 500ms | End-to-end pipeline (video → features) — **AUTHORITATIVE** |
| 150ms | Per-frame processing budget (future optimization goal) |

### Rationale
- 500ms at 30fps = 15 frame buffer (reasonable for real-time analysis)
- 150ms is an aggressive optimization target for Phase 6, not Phase 1-4
- Edge hardware (iPhone-level) can achieve ~100-200ms per frame with optimization
- Full pipeline includes tracking and feature extraction overhead

### Phase Targets
| Phase | Latency Target |
|-------|---------------|
| Phase 1-4 (MVP) | <500ms end-to-end |
| Phase 5 (Validation) | Verify <500ms on target hardware |
| Phase 6 (Optimization) | <150ms per-frame on iPhone-level |

### Implementation
- Profile with `src/utils/profiling.py` (to be created in Phase 5)
- Log per-stage timing in PerceptionPipeline

---

## AD9: Phase 3 Dependency Refinement
**Status**: ACCEPTED
**Related to**: DEVELOPMENT_PLAN.md line 396

### Decision
Phase 3 (Recognition Layer) dependencies are **partially blocked**:

| Feature Group | Blocking Dependencies | Can Start |
|---------------|----------------------|-----------|
| 3.1 Feature Math Foundation | 1.2 (schemas) | Immediately |
| 3.2 Static Geometry (0-25) | 1.2 | Immediately |
| 3.3 Distance (26-36) | 1.2, 2.3 (Calibrator) | Blocked by 2.3 |
| 3.4 Angular (37-42) | 1.2 | Immediately |
| 3.5 Arm Extension (43-48) | 1.2 | Immediately |
| 3.6 Temporal Derivatives (49-96) | 1.2, 1.3 (Buffer) | After 1.3 |
| 3.7 Meta & Audio (97-100) | 1.2 | Immediately |
| 3.8 Canonicalization | 1.2, 2.2 (Tracker) | After 2.2 |
| 3.9 FeatureExtractor Pipeline | All above | After above |

### Implementation
Update DEVELOPMENT_PLAN.md dependency diagram to reflect partial unblocking.

---

## AD10: File Structure — `src/` Root
**Status**: ACCEPTED
**Related to**: README.md (outdated)

### Decision
The project root structure uses `src/` not `fencer_ai/`:

```
fecingbuddy/
├── src/
│   ├── __init__.py
│   ├── main_pipeline.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── rtmpose.py
│   │   ├── tracker.py
│   │   ├── pose_embedder.py
│   │   ├── calibrator.py
│   │   ├── audio.py
│   │   ├── audio_buffer.py
│   │   └── pipeline.py
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── feature_math.py
│   │   ├── feature_extractor.py
│   │   └── canonicalize.py
│   └── utils/
│       ├── __init__.py
│       ├── schemas.py
│       ├── config.py
│       ├── logging.py
│       ├── buffer.py
│       ├── constants.py
│       └── types.py
├── tests/
├── data/
├── configs/
└── outputs/
```

### Implementation
- README.md must be updated to reflect actual structure
- DEVELOPMENT_PLAN.md "Files to Create" section is **correct**

---

## AD11: Visualizer Module — Deferred to Phase 5
**Status**: ACCEPTED
**Related to**: ARCHITECTURE.md mentions visualizer in pipeline

### Decision
`src/utils/visualizer.py` is **not in the initial implementation plan** (Phase 1-4). It is a Phase 5 (P1) or Phase 6 (P2) item.

### When to Implement
- Phase 5.2: Integration tests with visualization
- Phase 6.1: Optional heatmap overlay for debugging

### Implementation (Future)
- `src/utils/visualizer.py`: KeypointOverlay, FeatureHeatmap classes
- `--visualize` flag in main_pipeline.py (Phase 4.2)

---

## Summary Table

| AD# | Title | Priority | Status |
|-----|-------|----------|--------|
| AD1 | 101-Dim Feature Vector Index Map | CRITICAL | ACCEPTED |
| AD2 | Y-Axis Convention for Tracker | HIGH | ACCEPTED |
| AD3 | Norfair + EMA Position Prediction | HIGH | ACCEPTED |
| AD4 | Canonical Keypoint Format (COCO-17) | HIGH | ACCEPTED |
| AD5 | Horizontal Canonicalization | CRITICAL | ACCEPTED |
| AD6 | Homography Calibration Method | HIGH | ACCEPTED |
| AD7 | Audio Touch Flag Binary Threshold | MEDIUM | ACCEPTED |
| AD8 | Latency Target (500ms authoritative) | MEDIUM | ACCEPTED |
| AD9 | Phase 3 Dependency Refinement | MEDIUM | ACCEPTED |
| AD10 | File Structure (`src/` root) | LOW | ACCEPTED |
| AD11 | Visualizer Module Deferred | LOW | ACCEPTED |

---

## Change Log

| Version | Date | AD# | Change |
|---------|------|-----|--------|
| 1.0 | 2026-03-27 | All | Initial architectural decisions from review |
