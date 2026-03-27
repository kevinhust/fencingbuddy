# FencerAI Development Progress
*Version: 1.0 | Last Updated: 2026-03-27*

---

## Project Overview

**FencerAI** is an edge-first, real-time fencing analysis system that extracts a strict 101-dimensional spatio-temporal feature vector from fencing videos, targeting <500ms end-to-end latency.

### Pipeline Flow
```
Video/Audio → TimestampedBuffer → RTMPose → Norfair Dual Tracker → Calibrator (Homography) → Feature Extractor → (N, 2, 101) Matrix
```

### Reference Documents
| Document | Purpose |
|----------|---------|
| `ARCHITECTURE.md` | Pipeline flow, tracker rules, feature dictionary |
| `ARCHITECTURAL_DECISIONS.md` | 11 ADs - authoritative design decisions |
| `DATA_SCHEMA.md` | Pydantic model definitions |
| `DEVELOPMENT_PLAN.md` | 67 tasks across 6 phases |
| `CLAUDE.md` | Project instructions and commands |

---

## Phase 1: Foundation & Infrastructure
**Goal:** Project scaffolding, configuration management, logging, and Pydantic data contracts.

### 1.1 Config & Logging (P0)
| Task | Status | Description |
|------|--------|-------------|
| 1.1.1 | [ ] | `src/utils/config.py` - OmegaConf config management |
| 1.1.2 | [ ] | `src/utils/logging.py` - Loguru with temporal annotations |
| 1.1.3 | [ ] | `src/__init__.py` - Package-level exports |
| 1.1.4 | [ ] | Verify dependencies install (CPU-only torch) |

**Completed:**
- None yet

---

### 1.2 Pydantic Data Schemas (P0) ✅
**Status:** Core schemas implemented, unit tests pending

| Task | Status | Description |
|------|--------|-------------|
| 1.2.1 | ✅ | `src/utils/schemas.py` implemented |
| 1.2.2 | ✅ | Custom validators (bbox, keypoints, homography) |
| 1.2.3 | ✅ | JSON encoders for numpy arrays |
| 1.2.4 | [ ] | Unit tests for schema validation |

**Completed Work (2026-03-27):**
- Created `src/utils/schemas.py` with:
  - `Keypoint(x, y, conf)` - with ge=0.0 validation
  - `FencerPose(fencer_id, bbox, keypoints, is_canonical_flipped)`
  - `AudioEvent(timestamp, event_type, confidence)`
  - `FrameData(frame_id, timestamp, poses, audio_event, homography_matrix)`
  - `FeatureMatrix(features, timestamps, frame_ids, audio_flags)` - shape validated (N,2,101)
- Fixed stray syntax error at line 300
- Added `ge=0.0` to Keypoint.x and Keypoint.y per AD1 validation requirements

**Files Modified:**
- `src/utils/schemas.py` - created

**Pending:**
- `tests/test_schemas.py` - write unit tests

---

### 1.3 TimestampedBuffer (P0)
| Task | Status | Description |
|------|--------|-------------|
| 1.3.1 | [ ] | `src/utils/buffer.py` - TimestampedBuffer class |
| 1.3.2 | [ ] | Audio-video sync detection via cross-correlation |
| 1.3.3 | [ ] | Unit tests for buffer operations |

---

### 1.4 Constants & Enumerations (P1)
| Task | Status | Description |
|------|--------|-------------|
| 1.4.1 | [ ] | `src/utils/constants.py` - Keypoint indices, feature indices |
| 1.4.2 | [ ] | `src/utils/types.py` - Custom type annotations |

---

## Phase 2: Perception Layer - RTMPose Integration (P0)
**Goal:** Integrate RTMPose for multi-person pose estimation with real-time performance.

### 2.1 RTMPose Wrapper (P0)
| Task | Status | Description |
|------|--------|-------------|
| 2.1.1 | [ ] | `src/perception/rtmpose.py` - RTMPoseEstimator class |
| 2.1.2 | [ ] | `estimate_from_frame()` implementation |
| 2.1.3 | [ ] | Edge case handling (no detections, >2 detections) |
| Tests | [ ] | Unit tests for RTMPose wrapper |

### 2.2 Norfair Dual-Tracker with Referee Filter (P0)
| Task | Status | Description |
|------|--------|-------------|
| 2.2.1 | [ ] | `src/perception/tracker.py` - FencerTracker extending Norfair |
| 2.2.2 | [ ] | Tracker maintenance logic |
| 2.2.3 | [ ] | Graceful failure with EMA prediction (per AD3) |
| 2.2.4 | [ ] | `src/perception/pose_embedder.py` - Cosine similarity |

### 2.3 Calibrator - Homography (P0)
| Task | Status | Description |
|------|--------|-------------|
| 2.3.1 | [ ] | `src/perception/calibrator.py` - HomographyCalibrator |
| 2.3.2 | [ ] | `pixel_to_meter()` transformation |
| 2.3.3 | [ ] | `meter_to_pixel()` inverse transformation |
| 2.3.4 | [ ] | Calibration validation with RANSAC |

### 2.4 Audio Event Detection (P1)
| Task | Status | Description |
|------|--------|-------------|
| 2.4.1 | [ ] | `src/perception/audio.py` - AudioDetector |
| 2.4.2 | [ ] | `detect_events()` for blade touches |
| 2.4.3 | [ ] | `src/perception/audio_buffer.py` - Circular buffer |

### 2.5 Perception Layer Integration (P0)
| Task | Status | Description |
|------|--------|-------------|
| 2.5.1 | [ ] | `src/perception/pipeline.py` - PerceptionPipeline orchestration |
| 2.5.2 | [ ] | Frame-by-frame processing |
| 2.5.3 | [ ] | Integration tests |

---

## Phase 3: Recognition Layer - 101-Dimensional Feature Extraction (P0)
**Goal:** Extract exact 101-dim vector per AD1.

### 3.1 Feature Math Engine Foundation (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.1.1 | [ ] | `src/recognition/feature_math.py` - Vectorized numpy |
| 3.1.2 | [ ] | Keypoint selection for 12 canonical points (per AD4) |

### 3.2 Static Geometry Features (Indices 0-25) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.2.1 | [ ] | `extract_static_geometry()` - 24 dims |
| 3.2.2 | [ ] | `extract_center_of_mass()` - 2 dims |

### 3.3 Distance Features (Indices 26-36) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.3.1 | [ ] | `extract_distance_features()` - 11 dims (requires calibrator) |

### 3.4 Angular Features (Indices 37-42) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.4.1 | [ ] | `extract_angle_features()` - 4 dims |
| 3.4.2 | [ ] | `extract_torso_orientation()` - 2 dims |

### 3.5 Arm Extension Features (Indices 43-48) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.5.1 | [ ] | `extract_arm_extension()` - 6 dims |

### 3.6 Temporal Derivatives (Indices 49-96) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.6.1 | [ ] | EMA smoothing utility (α=0.6-0.8 per AD8) |
| 3.6.2 | [ ] | `extract_velocity_features()` - 24 dims |
| 3.6.3 | [ ] | `extract_acceleration_features()` - 24 dims |

### 3.7 Meta & Audio Features (Indices 97-100) (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.7.1 | [ ] | `extract_meta_features()` - 3 dims |
| 3.7.2 | [ ] | `extract_audio_flag()` - 1 dim (per AD7) |

### 3.8 Canonicalization (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.8.1 | [ ] | `canonicalize_pose()` - horizontal flip if Right fencer |
| 3.8.2 | [ ] | `canonicalize_frame()` - apply to both fencers |

### 3.9 Complete Feature Extraction Pipeline (P0)
| Task | Status | Description |
|------|--------|-------------|
| 3.9.1 | [ ] | `src/recognition/feature_extractor.py` - FeatureExtractor |
| 3.9.2 | [ ] | `extract_frame_features()` - returns (2, 101) per frame |
| 3.9.3 | [ ] | Unit tests for all 101 dimensions |

---

## Phase 4: Main Pipeline Integration (P0)
**Goal:** End-to-end pipeline producing (N, 2, 101) feature matrix.

### 4.1 Main Pipeline Orchestration (P0)
| Task | Status | Description |
|------|--------|-------------|
| 4.1.1 | [ ] | `src/main_pipeline.py` - CLI with argparse |
| 4.1.2 | [ ] | `process_video()` - frame-by-frame loop |
| 4.1.3 | [ ] | Audio processing with librosa |

### 4.2 Output & Persistence (P0)
| Task | Status | Description |
|------|--------|-------------|
| 4.2.1 | [ ] | `save_features()` - .npy + .json metadata |
| 4.2.2 | [ ] | `--visualize` flag (P1) |

### 4.3 Error Handling & Graceful Degradation (P0)
| Task | Status | Description |
|------|--------|-------------|
| 4.3.1 | [ ] | Comprehensive error handling |
| 4.3.2 | [ ] | Pipeline health monitoring |

### 4.4 CLI Interface (P0)
| Task | Status | Description |
|------|--------|-------------|
| 4.4.1 | [ ] | Finalize command-line interface |
| 4.4.2 | [ ] | Progress bar with tqdm |

---

## Phase 5: Testing & Validation (P1)
**Goal:** Ensure correctness, robustness, and 80%+ test coverage.

### 5.1 Unit Tests (P1)
| Task | Status | Description |
|------|--------|-------------|
| 5.1.1 | [ ] | `tests/test_schemas.py` |
| 5.1.2 | [ ] | `tests/test_buffer.py` |
| 5.1.3 | [ ] | `tests/test_tracker.py` |
| 5.1.4 | [ ] | `tests/test_feature_math.py` |
| 5.1.5 | [ ] | `tests/test_canonicalization.py` |
| 5.1.6 | [ ] | `tests/test_ema.py` |

### 5.2 Integration Tests (P1)
| Task | Status | Description |
|------|--------|-------------|
| 5.2.1 | [ ] | `tests/test_perception_pipeline.py` |
| 5.2.2 | [ ] | `tests/test_feature_extractor.py` |
| 5.2.3 | [ ] | `tests/test_integration_pipeline.py` |

### 5.3 E2E Tests with Sample Data (P1)
| Task | Status | Description |
|------|--------|-------------|
| 5.3.1 | [ ] | `tests/test_e2e_clean_bout.py` |
| 5.3.2 | [ ] | `tests/test_e2e_infighting.py` |
| 5.3.3 | [ ] | `tests/test_e2e_club_noise.py` |

### 5.4 Performance Tests (P2)
| Task | Status | Description |
|------|--------|-------------|
| 5.4.1 | [ ] | Latency profiling (<500ms target) |
| 5.4.2 | [ ] | Memory profiling for edge deployment |

---

## Phase 6: Optimization & Edge Deployment (P2)
**Goal:** Achieve <150ms latency on iPhone-level hardware.

### 6.1 Performance Optimization (P2)
| Task | Status | Description |
|------|--------|-------------|
| 6.1.1 | [ ] | Batch processing optimization |
| 6.1.2 | [ ] | RTMPose ONNX quantization |
| 6.1.3 | [ ] | Memory optimization |

### 6.2 Edge Deployment Preparation (P2)
| Task | Status | Description |
|------|--------|-------------|
| 6.2.1 | [ ] | Mobile-friendly packaging |
| 6.2.2 | [ ] | Reduce binary size |
| 6.2.3 | [ ] | Power consumption optimization |

---

## Progress Summary

### Overall: Phase 1 In Progress

| Phase | Tasks | Completed | In Progress | Pending |
|-------|-------|----------|------------|---------|
| Phase 1 | 11 | 1 | 0 | 10 |
| Phase 2 | 14 | 0 | 0 | 14 |
| Phase 3 | 18 | 0 | 0 | 18 |
| Phase 4 | 9 | 0 | 0 | 9 |
| Phase 5 | 10 | 0 | 0 | 10 |
| Phase 6 | 5 | 0 | 0 | 5 |
| **Total** | **67** | **1** | **0** | **66** |

### Completed Tasks
- ✅ 1.2.1-1.2.3: Pydantic schemas implemented (`src/utils/schemas.py`)

### Current Focus
- Phase 1.2.4: Write unit tests for schemas
- Phase 1.1: Config & Logging implementation
- Phase 1.3: TimestampBuffer implementation

---

## Change Log

| Date | Phase | Task | Change |
|------|-------|------|--------|
| 2026-03-27 | 1.2 | All | Initial schemas implementation |
| 2026-03-27 | 1.2 | 1.2.1 | Created `src/utils/schemas.py` with all Pydantic models |
| 2026-03-27 | 1.2 | 1.2.1 | Fixed `ge=0.0` validation on Keypoint.x and Keypoint.y |
| 2026-03-27 | 1.2 | 1.2.1 | Fixed syntax error (stray `"` at line 300) |
| 2026-03-27 | Docs | - | Created `docs/DEVELOPMENT_PROGRESS.md` |
| 2026-03-27 | Docs | - | Created `ARCHITECTURAL_DECISIONS.md` with 11 ADs |
