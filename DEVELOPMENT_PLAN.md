# FencerAI Development Plan
*Version: 1.0 | Created: 2026-03-27*

## Executive Summary

FencerAI is an edge-first, real-time fencing analysis system that extracts a strict 101-dimensional spatio-temporal feature vector from fencing videos, targeting <500ms end-to-end latency. The pipeline flows: Video/Audio → TimestampedBuffer → RTMPose → Norfair Dual Tracker → Calibrator (Homography) → Feature Extractor → (N, 2, 101) Matrix.

---

## Phase 1: Foundation & Infrastructure
**Goal:** Establish the project scaffolding, configuration management, logging, and critical Pydantic data contracts.

### 1.1 Project Configuration & Logging (P0)
- [ ] **1.1.1** Set up `src/utils/config.py` with Hydra-style config management using OmegaConf
  - Define config schema for: model paths, tracker params, homography settings, EMA alphas, feature extraction flags
  - Support config overrides via CLI/YAML
- [ ] **1.1.2** Configure `src/utils/logging.py` using Loguru
  - Define log levels: DEBUG (per-frame), INFO (pipeline stage), WARNING (graceful degradation), ERROR (failure)
  - Add temporal annotations (frame_id, timestamp) to log format
  - Ensure JSON output option for production logging
- [ ] **1.1.3** Create `src/__init__.py` with package-level exports
- [ ] **1.1.4** Verify all dependencies install correctly (CPU-only torch for edge)

### 1.2 Pydantic Data Schemas - SOURCE OF TRUTH (P0)
**Critical:** All Perception→Recognition data transfer MUST use these models.

- [ ] **1.2.1** Implement `src/utils/schemas.py` with exact models from `DATA_SCHEMA.md`:
  - `Keypoint(x: float, y: float, conf: float)` with validation 0.0-1.0
  - `FencerPose(fencer_id: int, bbox: Tuple[float,float,float,float], keypoints: List[Keypoint], is_canonical_flipped: bool)`
  - `AudioEvent(timestamp: float, event_type: str, confidence: float)`
  - `FrameData(frame_id: int, timestamp: float, poses: List[FencerPose], audio_event: Optional[AudioEvent], homography_matrix: Optional[List[List[float]]])`
  - `FeatureMatrix(features: np.ndarray shape=(N,2,101), timestamps: List[float], frame_ids: List[int], audio_flags: Optional[np.ndarray])`
- [ ] **1.2.2** Add custom validators for:
  - bbox format [x1, y1, x2, y2] with x2>x1, y2>y1
  - keypoints list length (min 12 for FERA, max 33 for full MediaPipe)
  - homography_matrix 3x3 structure
- [ ] **1.2.3** Add JSON encoders for numpy arrays
- [ ] **1.2.4** Write unit tests for schema validation:
  - Valid/invalid Keypoint ranges
  - FencerPose bbox validation
  - FrameData pose count limits (max 2)
  - FeatureMatrix shape enforcement (N,2,101)
  - Round-trip serialization/deserialization

### 1.3 TimestampedBuffer - Synchronization Primitives (P0)
**Purpose:** Synchronize video frames with audio events and provide temporal normalization.

- [ ] **1.3.1** Create `src/utils/buffer.py` with `TimestampedBuffer` class:
  - Store (frame_id, timestamp, frame_data, audio_sample) tuples
  - Support frame dropping when buffer exceeds max_size
  - Provide `get_frame_range(start_ts, end_ts)` method
  - Thread-safe with proper locking for edge deployment
- [ ] **1.3.2** Implement audio-video sync detection:
  - Use audio waveform cross-correlation to detect offset
  - Store sync_offset correction value
- [ ] **1.3.3** Write unit tests for buffer operations

### 1.4 Constants & Enumerations (P1)
- [ ] **1.4.1** Create `src/utils/constants.py`:
  - Keypoint indices mapping (canonical 12 FERA keypoints: shoulders, elbows, wrists, hips, knees, ankles)
  - Feature index constants for 101-dim vector (matching ARCHITECTURE.md exactly)
  - Tracker state enumerations
  - Audio event type constants
- [ ] **1.4.2** Define `src/utils/types.py` with custom types:
  - `KeypointArray = np.ndarray shape=(N, 3)` (x, y, conf)
  - `PoseKeypoints = Dict[str, KeypointArray]` (named keypoint access)
  - `HomographyMatrix = np.ndarray shape=(3, 3)`

---

## Phase 2: Perception Layer - RTMPose Integration (P0)
**Goal:** Integrate RTMPose for multi-person pose estimation with real-time performance.

### 2.1 RTMPose Wrapper (P0)
- [ ] **2.1.1** Create `src/perception/rtmpose.py` with `RTMPoseEstimator` class:
  - Initialize using rtmlib (ONNX-based, lightweight)
  - Support batch processing for efficiency
  - Output: List[FencerPose] with keypoints in pixel coordinates
  - Configurable confidence threshold (default 0.3)
- [ ] **2.1.2** Implement `estimate_from_frame(frame: np.ndarray) -> List[FencerPose]`:
  - Preprocess: BGR→RGB, normalize to [0,1]
  - Run inference with ONNX runtime
  - Postprocess: convert keypoints to Keypoint objects with confidence
  - Filter low-confidence detections
- [ ] **2.1.3** Handle edge cases:
  - No detections: return empty list (graceful)
  - >2 detections: return top-2 by average confidence
  - Single detection: return with fencer_id=0, handle missing fencer gracefully later

### 2.2 Norfair Dual-Tracker with Referee Filter (P0)
**Critical:** Must implement exact initialization rules from ARCHITECTURE.md.

- [ ] **2.2.1** Create `src/perception/tracker.py` with `FencerTracker` class extending Norfair's Tracker:
  - Override initialization to lock 2 largest BBoxes in bottom 70% of Y-axis
  - Assign ID 0 (Left) and ID 1 (Right) deterministically
  - Implement "Referee Filter" to ignore standing figures in upper 30%
- [ ] **2.2.2** Implement tracker maintenance logic:
  - When detections > 2: prioritize highest pose-embedding similarity to locked IDs
  - Track Y-variance to prefer stable candidates
  - Use IOU matching for detection-to-track association
- [ ] **2.2.3** Implement graceful failure handling:
  - When ID lost: Use EMA-predicted position for next valid match
  - Return `None` as sentinel when fencer completely occluded
  - Never crash on tracker failures
- [ ] **2.2.4** Create `src/perception/pose_embedder.py`:
  - Extract pose embedding from RTMPose keypoints for similarity matching
  - Use keypoint configuration as embedding vector
  - Cosine similarity for matching

### 2.3 Calibrator - Homography for Physical Distance (P0)
**Purpose:** Transform pixel coordinates to metric space (meters) for distance features.

- [ ] **2.3.1** Create `src/perception/calibrator.py` with `HomographyCalibrator` class:
  - Store 3x3 homography matrix
  - Support manual marker-based calibration
  - Implement automatic piste-end detection (edge case: may need manual override)
- [ ] **2.3.2** Implement `pixel_to_meter(pixel_coords: np.ndarray) -> np.ndarray`:
  - Apply homography transformation
  - Return (x_m, y_m) in physical space
  - Validate output is physically plausible (<10m range check)
- [ ] **2.3.3** Implement `meter_to_pixel(meter_coords: np.ndarray) -> np.ndarray`:
  - Inverse transformation for visualization
- [ ] **2.3.4** Calibration validation:
  - Require minimum 4 point correspondences
  - RANSAC for robust estimation if needed
  - Store calibration confidence score

### 2.4 Audio Event Detection (P1)
**Purpose:** Detect blade touches and other audio events for feature index 100.

- [ ] **2.4.1** Create `src/perception/audio.py` with `AudioDetector` class:
  - Use librosa for audio loading and feature extraction
  - Implement onset detection for blade sounds
  - Detect "touch" events (sharp transient) vs "parry" (double onset) vs "halt"
- [ ] **2.4.2** Implement `detect_events(audio_segment: np.ndarray, frame_times: List[float]) -> List[AudioEvent]`:
  - Return AudioEvent list with timestamps aligned to video frames
  - Confidence based on onset strength
- [ ] **2.4.3** Create `src/perception/audio_buffer.py`:
  - Circular buffer for real-time audio processing
  - Sync with video frames using TimestampedBuffer

### 2.5 Perception Layer Integration (P0)
- [ ] **2.5.1** Create `src/perception/pipeline.py` with `PerceptionPipeline` class:
  - Orchestrate: RTMPose → Tracker → Calibrator
  - Accept raw frame + timestamp, output FrameData
  - Apply Pydantic validation at boundary
- [ ] **2.5.2** Implement frame-by-frame processing:
  - `process_frame(frame, timestamp, audio_sample) -> FrameData`
  - Return `None` if critical failure (graceful degradation)
- [ ] **2.5.3** Write integration tests for perception pipeline:
  - Single frame processing
  - Multi-frame tracking consistency
  - Tracker ID persistence
  - Graceful handling of occlusions

---

## Phase 3: Recognition Layer - 101-Dimensional Feature Extraction (P0)
**Critical:** The 101-dim vector MUST exactly match ARCHITECTURE.md indices. No deviations.

### 3.1 Feature Math Engine Foundation (P0)
- [ ] **3.1.1** Create `src/recognition/feature_math.py`:
  - Vectorized numpy operations ONLY (no Python for loops for geometry)
  - All functions must be annotated with proper types
- [ ] **3.1.2** Implement keypoint selection for 12 canonical points:
  - Indices: L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle
  - Normalization by shoulder-width
  - Pelvis-centered coordinate system

### 3.2 Static Geometry Features (Indices 0-23, 24-25) (P0)
- [ ] **3.2.1** Implement `extract_static_geometry(pose: FencerPose) -> np.ndarray`:
  - Output shape: (24,) for 12 keypoints × 2 coords (x, y)
  - Normalized by shoulder-width (divide by shoulder distance)
  - Pelvis-centered (subtract hip center)
- [ ] **3.2.2** Implement `extract_center_of_mass(pose: FencerPose) -> np.ndarray`:
  - Output shape: (2,) - pelvis/hip center (x, y) in normalized space

### 3.3 Distance Features in Physical Meters (Indices 26-36) (P0)
- [ ] **3.3.1** Implement `extract_distance_features(frame_data: FrameData, calibrator: HomographyCalibrator) -> np.ndarray`:
  - Output shape: (11,)
  - Inter-fencer pelvis distance (meters)
  - Inter-fencer foot distance (meters)
  - Stance width (each fencer)
  - Weapon-hand to opponent-torso distance
  - All physical distances via homography transformation
- [ ] **3.3.2** Handle 1-fencer case gracefully (return zeros with valid shape)

### 3.4 Angular Features (Indices 37-40, 41-42) (P0)
- [ ] **3.4.1** Implement `extract_angle_features(pose: FencerPose) -> np.ndarray`:
  - Output shape: (4,) using `np.arctan2`
  - Front knee angle, Back knee angle, Weapon elbow angle, Torso lean angle
- [ ] **3.4.2** Implement `extract_torso_orientation(pose: FencerPose) -> np.ndarray`:
  - Output shape: (2,) - vector from shoulder center to hip center

### 3.5 Arm Extension Features (Indices 43-48) (P0)
- [ ] **3.5.1** Implement `extract_arm_extension(pose: FencerPose) -> np.ndarray`:
  - Output shape: (6,)
  - Weapon arm extension ratio (relative to max reach)
  - Extension directional angle
  - Secondary arm extension ratio (for balance)

### 3.6 Temporal Derivatives - Velocity & Acceleration (Indices 49-96) (P0)
**Critical:** Must use actual timestamps (Δt) for accurate derivatives across variable FPS.

- [ ] **3.6.1** Implement EMA smoothing utility:
  - `ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray`
  - Alpha range: 0.6-0.8 as specified in ARCHITECTURE.md
  - Vectorized implementation
- [ ] **3.6.2** Implement `extract_velocity_features(current_pose, previous_pose, dt) -> np.ndarray`:
  - Output shape: (24,) - 1st derivative of static geometry
  - Use EMA smoothing (alpha=0.6-0.8)
  - Divide by dt for true velocity (pixels/second)
- [ ] **3.6.3** Implement `extract_acceleration_features(velocity_current, velocity_previous, dt) -> np.ndarray`:
  - Output shape: (24,) - 2nd derivative
  - Use EMA smoothing
  - Divide by dt for true acceleration (pixels/second²)

### 3.7 Meta & Audio Features (Indices 97-100) (P0)
- [ ] **3.7.1** Implement `extract_meta_features(frame_data: FrameData) -> np.ndarray`:
  - Output shape: (3,)
  - CoM velocity (magnitude)
  - CoM acceleration (magnitude)
- [ ] **3.7.2** Implement `extract_audio_flag(audio_event: Optional[AudioEvent]) -> float`:
  - Output: 1.0 if blade_touch detected at this frame, 0.0 otherwise
  - Single scalar that fills index 100

### 3.8 Canonicalization - Horizontal Flip (P0)
**Critical:** Classifier only knows "Left" fencer. Right fencer MUST be flipped.

- [ ] **3.8.1** Implement `canonicalize_pose(pose: FencerPose) -> FencerPose`:
  - If fencer_id == 1 (Right), horizontally flip all x-coordinates
  - Set is_canonical_flipped = True when flipped
  - Apply before feature extraction
- [ ] **3.8.2** Implement `canonicalize_frame(frame_data: FrameData) -> FrameData`:
  - Apply canonicalization to both fencers
  - Ensure consistent pose ordering (ID 0 always Left perspective)

### 3.9 Complete Feature Extraction Pipeline (P0)
- [ ] **3.9.1** Create `src/recognition/feature_extractor.py` with `FeatureExtractor` class:
  - Orchestrate all feature extraction in correct order
  - Concatenate to 101-dim vector
  - Validate output shape: (101,)
- [ ] **3.9.2** Implement `extract_frame_features(frame_data: FrameData) -> Tuple[np.ndarray, np.ndarray]`:
  - Returns: (features_0, features_1) for each fencer slot
  - Use zeros for missing fencer (graceful degradation)
- [ ] **3.9.3** Unit tests for each feature group:
  - Test shape correctness for all 101 dimensions
  - Test index mapping matches ARCHITECTURE.md exactly
  - Test canonicalization correctness
  - Test EMA smoothing stability

---

## Phase 4: Main Pipeline Integration (P0)
**Goal:** End-to-end pipeline producing (N, 2, 101) feature matrix.

### 4.1 Main Pipeline Orchestration (P0)
- [ ] **4.1.1** Create `src/main_pipeline.py`:
  - Argument parser for: --video, --output, --config, --device
  - Initialize all components: PerceptionPipeline, FeatureExtractor, AudioDetector
  - Load video with OpenCV cv2.VideoCapture
- [ ] **4.1.2** Implement `process_video(video_path, output_path) -> FeatureMatrix`:
  - Frame-by-frame processing loop
  - Accumulate features into (N, 2, 101) matrix
  - Collect timestamps and frame_ids
  - Handle end-of-video gracefully
- [ ] **4.1.3** Implement audio processing:
  - Load audio with librosa
  - Sync with video frames
  - Pass AudioEvents to FrameData

### 4.2 Output & Persistence (P0)
- [ ] **4.2.1** Implement `save_features(features: FeatureMatrix, output_path: str)`:
  - Save as .npy binary file
  - Also save metadata as .json (timestamps, frame_ids)
  - Validate FeatureMatrix before saving
- [ ] **4.2.2** Add `--visualize` flag (P1):
  - Overlay keypoints on video
  - Draw feature vectors (optional heatmap)

### 4.3 Error Handling & Graceful Degradation (P0)
- [ ] **4.3.1** Implement comprehensive error handling:
  - Video load failure: clear error message
  - Model inference failure: skip frame, log warning
  - Tracker failure: use EMA-predicted positions
  - Audio load failure: continue without audio flags
- [ ] **4.3.2** Pipeline health monitoring:
  - Track frame drop rate
  - Track tracker ID switches (indicates tracking failure)
  - Log summary statistics at end

### 4.4 CLI Interface (P0)
- [ ] **4.4.1** Finalize command-line interface:
  ```bash
  python -m src.main_pipeline \
    --video data/samples/bout_clean.mp4 \
    --output outputs/features.npy \
    --config configs/default.yaml
  ```
- [ ] **4.4.2** Add progress bar with tqdm

---

## Phase 5: Testing & Validation (P1)
**Goal:** Ensure correctness, robustness, and 80%+ test coverage.

### 5.1 Unit Tests for Core Components (P1)
- [ ] **5.1.1** `tests/test_schemas.py`: Pydantic model validation
- [ ] **5.1.2** `tests/test_buffer.py`: TimestampedBuffer operations
- [ ] **5.1.3** `tests/test_tracker.py`: Norfair tracker logic
- [ ] **5.1.4** `tests/test_feature_math.py`: Each feature group
- [ ] **5.1.5** `tests/test_canonicalization.py`: Horizontal flip logic
- [ ] **5.1.6** `tests/test_ema.py`: EMA smoothing correctness

### 5.2 Integration Tests (P1)
- [ ] **5.2.1** `tests/test_perception_pipeline.py`:
  - Process single frame end-to-end
  - Multi-frame consistency
- [ ] **5.2.2** `tests/test_feature_extractor.py`:
  - Full 101-dim extraction
  - Index alignment verification
- [ ] **5.2.3** `tests/test_integration_pipeline.py`:
  - Full video processing
  - Output shape validation

### 5.3 E2E Tests with Sample Data (P1)
- [ ] **5.3.1** `tests/test_e2e_clean_bout.py`:
  - Process clean_bouts sample
  - Verify output shape (N, 2, 101)
  - No crashes, no exceptions
- [ ] **5.3.2** `tests/test_e2e_infighting.py`:
  - Process infighting sample
  - Handle occlusion gracefully
  - Audio events detected
- [ ] **5.3.3** `tests/test_e2e_club_noise.py`:
  - Process with referees in frame
  - Verify tracker ignores non-fencers

### 5.4 Performance Tests (P2)
- [ ] **5.4.1** Latency profiling:
  - Measure per-frame latency
  - Verify <500ms target achievable
  - Profile hot paths
- [ ] **5.4.2** Memory profiling for edge deployment

---

## Phase 6: Optimization & Edge Deployment (P2)
**Goal:** Achieve <150ms latency on iPhone-level hardware.

### 6.1 Performance Optimization (P2)
- [ ] **6.1.1** Batch processing optimization:
  - Process multiple frames in parallel when allowed
  - Balance latency vs throughput
- [ ] **6.1.2** Model optimization:
  - Quantize RTMPose ONNX model
  - Use CPU-optimized ONNX runtime
- [ ] **6.1.3** Memory optimization:
  - Reduce feature matrix memory footprint
  - Streaming mode for long videos

### 6.2 Edge Deployment Preparation (P2)
- [ ] **6.2.1** Mobile-friendly packaging
- [ ] **6.2.2** Reduce binary size
- [ ] **6.2.3** Power consumption optimization

---

## Task Dependencies Diagram

```
Phase 1 (Foundation)
├── 1.1 Config & Logging
│   └── 1.1.1 Config schema → 1.1.2 Logging → 1.1.3 Package init
├── 1.2 Pydantic Schemas [CRITICAL]
│   ├── 1.2.1 Schema implementation
│   ├── 1.2.2 Validators
│   ├── 1.2.3 JSON encoders
│   └── 1.2.4 Unit tests
└── 1.3 TimestampBuffer
    └── 1.3.1 Buffer → 1.3.2 Sync → 1.3.3 Tests

Phase 2 (Perception) - BLOCKED by Phase 1.2
├── 2.1 RTMPose Wrapper
│   └── 2.1.1 Class → 2.1.2 estimate → 2.1.3 Edge cases → Tests
├── 2.2 Norfair Tracker
│   ├── 2.2.1 Tracker class → 2.2.2 Maintenance → 2.2.3 Graceful failure
│   └── 2.2.4 Pose embedder
├── 2.3 Homography Calibrator
│   └── 2.3.1 Class → 2.3.2 pixel_to_meter → 2.3.3 inverse → 2.3.4 Validation
├── 2.4 Audio Detection [P1]
│   └── 2.4.1 Class → 2.4.2 detect_events → 2.4.3 Buffer
└── 2.5 Perception Pipeline Integration
    └── 2.5.1 Pipeline class → 2.5.2 process_frame → 2.5.3 Integration tests

Phase 3 (Recognition - Feature Math) - BLOCKED by Phase 1.2, 2.3
├── 3.1 Feature Math Foundation
│   └── 3.1.1 Vectorized operations → 3.1.2 Keypoint selection
├── 3.2 Static Geometry (0-25)
│   └── 3.2.1 extract_static_geometry → 3.2.2 extract_CoM
├── 3.3 Distance Features (26-36)
│   └── 3.3.1 extract_distance_features (requires calibrator from 2.3)
├── 3.4 Angular Features (37-42)
│   └── 3.4.1 extract_angle_features → 3.4.2 extract_torso_orientation
├── 3.5 Arm Extension (43-48)
│   └── 3.5.1 extract_arm_extension
├── 3.6 Temporal Derivatives (49-96)
│   ├── 3.6.1 EMA utility
│   ├── 3.6.2 velocity extraction
│   └── 3.6.3 acceleration extraction
├── 3.7 Meta & Audio (97-100)
│   └── 3.7.1 extract_meta → 3.7.2 extract_audio_flag
├── 3.8 Canonicalization
│   └── 3.8.1 canonicalize_pose → 3.8.2 canonicalize_frame
└── 3.9 Feature Extractor Pipeline
    └── 3.9.1 FeatureExtractor class → 3.9.2 extract_frame_features → 3.9.3 Tests

Phase 4 (Main Pipeline) - BLOCKED by Phase 2.5, 3.9
├── 4.1 Main Pipeline
│   ├── 4.1.1 main_pipeline.py
│   ├── 4.1.2 process_video
│   └── 4.1.3 Audio processing
├── 4.2 Output & Persistence
│   └── 4.2.1 save_features → 4.2.2 visualization [P1]
├── 4.3 Error Handling
│   └── 4.3.1 Comprehensive error handling → 4.3.2 Health monitoring
└── 4.4 CLI Interface
    └── 4.4.1 Argparse → 4.4.2 Progress bar

Phase 5 (Testing) - PARALLEL with Phase 1-4
└── Unit tests, Integration tests, E2E tests

Phase 6 (Optimization) - AFTER Phase 4
├── 6.1 Performance Optimization
└── 6.2 Edge Deployment
```

---

## Priority Summary

| Priority | Tasks | Description |
|----------|-------|-------------|
| **P0** | 1.1, 1.2, 1.3, 2.1-2.3, 2.5, 3.1-3.9, 4.1-4.4 | Core pipeline, feature extraction, main integration |
| **P1** | 1.4, 2.4, 5.1-5.3 | Audio, constants, all testing |
| **P2** | 5.4, 6.1, 6.2 | Performance profiling, edge optimization |

---

## Complexity Estimates

| Phase | Estimated Tasks | Complexity | Risk Level |
|-------|-----------------|------------|------------|
| Phase 1 | 11 | Medium | Low |
| Phase 2 | 14 | High | Medium (RTMPose/Norfair integration) |
| Phase 3 | 18 | Very High | High (101-dim exactness) |
| Phase 4 | 9 | Medium | Low |
| Phase 5 | 10 | Medium | Low |
| Phase 6 | 5 | High | Medium |

**Total: ~67 tasks across 6 phases**

---

## Critical Success Criteria

1. **101-dimensional vector MUST exactly match ARCHITECTURE.md indices** - This is non-negotiable
2. **Pydantic validation at Perception→Recognition boundary** - No raw dicts
3. **Canonicalization always produces "Left" fencer perspective** - Classifier assumption
4. **Graceful degradation** - No crashes on edge cases
5. **Vectorized numpy** - No Python for loops in geometric calculations
6. **Temporal accuracy** - Use actual Δt for velocity/acceleration
7. **Output shape (N, 2, 101)** - Valid FeatureMatrix from Milestone 1

---

## Files to Create

```
src/
├── __init__.py
├── main_pipeline.py
├── perception/
│   ├── __init__.py
│   ├── rtmpose.py          # RTMPoseEstimator
│   ├── tracker.py          # FencerTracker
│   ├── pose_embedder.py    # Pose embedding for similarity
│   ├── calibrator.py       # HomographyCalibrator
│   ├── audio.py            # AudioDetector
│   ├── audio_buffer.py     # Audio circular buffer
│   └── pipeline.py         # PerceptionPipeline
├── recognition/
│   ├── __init__.py
│   ├── feature_math.py     # All vectorized feature math
│   └── feature_extractor.py # FeatureExtractor orchestrator
└── utils/
    ├── __init__.py
    ├── schemas.py          # Pydantic models (SOURCE OF TRUTH)
    ├── config.py           # OmegaConf config management
    ├── logging.py          # Loguru setup
    ├── buffer.py           # TimestampedBuffer
    ├── constants.py        # Keypoint indices, feature indices
    └── types.py            # Custom type annotations

tests/
├── test_schemas.py
├── test_buffer.py
├── test_tracker.py
├── test_feature_math.py
├── test_canonicalization.py
├── test_ema.py
├── test_perception_pipeline.py
├── test_feature_extractor.py
├── test_integration_pipeline.py
├── test_e2e_clean_bout.py
├── test_e2e_infighting.py
└── test_e2e_club_noise.py
```
