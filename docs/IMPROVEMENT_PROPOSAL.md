# FencerAI: Improvement Proposal & Technical Handoff
*Document Version: 1.0 | Date: 2026-04-02*

---

## 1. Project Overview

**FencerAI** is an edge-first, real-time fencing analysis system that extracts a 101-dimensional spatio-temporal feature vector from fencing videos.

### Current Capabilities
- Real-time pose estimation using RTMPose (lightweight mode: ~158ms/frame)
- Dual fencer tracking with Norfair tracker
- Referee filtering (bottom 70% of frame)
- Canonicalized feature extraction (left-fencer perspective)
- Audio event detection (blade touches)
- Visualization tools (skeleton overlay, heatmaps)
- Health monitoring (detection quality, latency)

### Repository
- **GitHub**: https://github.com/kevinhust/fecingbuddy
- **Main Branch**: main
- **Test Coverage**: 294 tests (288 unit + 6 E2E)

---

## 2. Technical Architecture

### Pipeline Flow
```
Video → RTMPose → Norfair Tracker → Feature Extractor → (N, 2, 101) Matrix
                ↓
           Audio Events → Audio Flag
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| RTMPose Wrapper | `src/perception/rtmpose.py` | 17-keypoint COCO pose estimation |
| Dual Tracker | `src/perception/tracker.py` | Fencer tracking + referee filter |
| Calibrator | `src/perception/calibrator.py` | Homography for pixel→meter |
| Audio Detector | `src/perception/audio.py` | Energy-based blade touch |
| Feature Math | `src/recognition/feature_math.py` | Vectorized geometry |
| Feature Extractor | `src/recognition/feature_extractor.py` | 101-dim extraction + canonicalization |
| Schemas | `src/utils/schemas.py` | Pydantic validation layer |

### Data Flow
1. Video frame → RTMPose (poses)
2. Poses → Norfair Tracker (ID assignment, referee filtering)
3. Tracked poses → Feature Extractor (canonicalization, 101-dim)
4. Features → (N, 2, 101) FeatureMatrix

---

## 3. Completed Features

### Phase 1: Foundation ✅
- Config management with dataclasses/YAML
- Loguru logging with temporal annotations
- Pydantic schemas (Keypoint, FencerPose, FrameData, FeatureMatrix)
- TimestampedBuffer with audio-video sync
- Constants and type aliases

### Phase 2: Perception Layer ✅
- RTMPose integration (rtmlib ONNX runtime)
- Norfair dual-tracker with pose embedding similarity
- Referee filter (bottom 70% Y-axis threshold)
- EMA predictor for graceful failure handling
- Homography calibrator (pixel↔meter)
- Audio event detection (blade touches)

### Phase 3: Recognition Layer ✅
- 101-dimensional feature extraction
- Static geometry (24 dims)
- Center of mass (2 dims)
- Distance features (11 dims, meters)
- Angular features (4 dims)
- Torso orientation (2 dims)
- Arm extension (6 dims)
- Velocity (24 dims, EMA smoothed)
- Acceleration (24 dims, EMA smoothed)
- Meta & audio (4 dims)
- Canonicalization (horizontal flip for right fencer)

### Phase 4: Pipeline Integration ✅
- CLI with argparse (src/main_pipeline.py)
- Frame-by-frame processing loop
- Output persistence (.npy + .json)
- `--visualize` flag (skeleton overlay)
- `--heatmap` flag (feature heatmaps)
- `--profile` flag (performance monitoring)
- HealthMonitor class

### Phase 5: Testing ✅
- 288 unit tests
- 6 E2E tests with sample video
- Integration tests

### Phase 6: Optimization ✅
- Lightweight RTMPose mode as default (158ms/frame)
- Performance profiling utilities
- Memory profiling utilities
- PipelineMonitor

### Phase 7: Visualization ✅
- Skeleton overlay with COCO-17 connections
- Color-coded fencers (red=left, blue=right)
- Info overlay (frame, fps, fencer count)
- Feature heatmap export (per-fencer + combined)

---

## 4. Known Limitations & Issues

### 4.1 Critical (Should Fix)

| Issue | Location | Description |
|-------|---------|-------------|
| **Tracker uses scalar distance function** | `src/perception/tracker.py:102` | Warning: "You are using a scalar distance function" - should use vectorized IOU for speed |
| **E2E tests need sample videos** | `tests/test_e2e_*.py` | Only 1 of 3 E2E tests has sample data |
| **No ONNX quantization** | Phase 6 | Model quantization not implemented (requires model files) |

### 4.2 Important (Should Address)

| Issue | Location | Description |
|-------|---------|-------------|
| **Homography calibration manual** | `src/perception/calibrator.py` | User must provide calibration file, no auto-calibration |
| **Audio detection is basic** | `src/perception/audio.py` | Simple energy threshold, not ML-based |
| **No real video testing** | E2E tests | Only tested with 1 video (498_1728950803.mp4) |
| **Lightweight mode accuracy** | RTMPose | 8x faster but may have lower accuracy than balanced |

### 4.3 Nice to Have (Future)

| Issue | Description |
|-------|-------------|
| **No GPU optimization** | Current profiling on CPU only |
| **No mobile deployment** | Edge deployment not implemented |
| **No WebAssembly** | Browser-based inference not available |
| **No REST API** | Real-time streaming not supported |
| **No visualization player** | Can't playback processed videos interactively |

---

## 5. Proposed Improvements

### 5.1 High Priority

#### P0-1: Vectorize Tracker Distance Function
**Problem**: Scalar distance function causes performance warning
**Solution**: Use Norfair's built-in `iou` or `yolox_iou` distance function
**Files**: `src/perception/tracker.py`
**Effort**: Low (configuration change)

```python
# Current (line 254-259):
self._tracker = Tracker(
    distance_function=fence_detection_distance,
    distance_threshold=distance_threshold,
    ...
)

# Proposed: Use Norfair's vectorized IOU
from norfair.tracker importiou_distance
self._tracker = Tracker(
    distance_function=iou_distance,
    distance_threshold=distance_threshold,
    ...
)
```

#### P0-2: Auto-Calibration from Piste Detection
**Problem**: Manual homography calibration required
**Solution**: Detect piste boundaries automatically using edge detection
**Files**: `src/perception/calibrator.py`
**Effort**: Medium

#### P0-3: Better Audio Classification
**Problem**: Simple energy threshold produces false positives
**Solution**: Use ML model for blade touch classification
**Files**: `src/perception/audio.py`
**Effort**: Medium

### 5.2 Medium Priority

#### P1-1: Multi-Video E2E Tests
**Problem**: Only 1 of 3 E2E test suites has sample data
**Solution**: Add sample videos for infighting and club_noise scenarios
**Files**: `tests/test_e2e_*.py`
**Effort**: Low (test writing, needs video data)

#### P1-2: Accuracy vs Speed Analysis
**Problem**: Lightweight mode may have lower accuracy
**Solution**: Compare pose accuracy across modes using keypoint consistency
**Files**: `tests/`
**Effort**: Medium

#### P1-3: ONNX Model Quantization
**Problem**: Full-precision models are larger
**Solution**: Quantize to INT8 for edge deployment
**Effort**: High (requires model training pipeline)

### 5.3 Low Priority

#### P2-1: Visualization Player
**Problem**: Static heatmaps not interactive
**Solution**: Create video player with feature overlays
**Effort**: Medium

#### P2-2: WebAssembly Runtime
**Problem**: Can't run in browser
**Solution**: Convert ONNX to ONNX.js or TensorFlow.js
**Effort**: High

#### P2-3: REST API for Streaming
**Problem**: Batch processing only
**Solution**: Add FastAPI endpoint for real-time streaming
**Effort**: Medium

---

## 6. Technical Debt

### 6.1 Code Quality

| Item | Description |
|------|-------------|
| **No type checking** | Should run mypy --strict |
| **No linting** | Should run ruff and black |
| **No pre-commit hooks** | Should add pre-commit config |
| **Deprecated Pydantic** | `json_encoders` deprecated in v2 |

### 6.2 Documentation

| Item | Description |
|------|-------------|
| **API docs missing** | No Sphinx/autodoc |
| **No changelog** | Should add keep a changelog |
| **No CONTRIBUTING guide** | No contribution guidelines |

### 6.3 Testing

| Item | Description |
|------|-------------|
| **No performance benchmarks** | Should add pytest-benchmark |
| **No mutation testing** | Should add mutmut or cosmic-ray |
| **Coverage could be higher** | Some branches not covered |

---

## 7. Recommended Next Steps

### Immediate (This Week)

1. **Fix tracker distance function** - Quick win, removes warning
2. **Add more E2E tests** - Need sample videos for infighting scenarios
3. **Run mypy --strict** - Improve type safety

### Short-term (This Month)

4. **Implement auto-calibration** - Improve usability
5. **Better audio detection** - Reduce false positives
6. **Add pre-commit hooks** - Maintain code quality

### Long-term (This Quarter)

7. **ONNX quantization** - Enable edge deployment
8. **WebAssembly runtime** - Browser-based inference
9. **REST API** - Real-time streaming capability

---

## 8. Architecture Decisions (ADs)

See `docs/ARCHITECTURAL_DECISIONS.md` for 11 documented decisions:

| AD | Topic |
|----|-------|
| AD1 | 101-Dimensional Feature Vector |
| AD2 | Tracker Configuration |
| AD3 | EMA Predictor |
| AD4 | COCO-17 Keypoint Selection |
| AD5 | Canonicalization Strategy |
| AD6 | Homography Calibration |
| AD7 | Audio Event Integration |
| AD8 | EMA Alpha Values |
| AD9 | Confidence Threshold |
| AD10 | Frame Width Canonicalization |
| AD11 | Vectorized Feature Math |

---

## 9. Key Files Reference

### Entry Points
- `src/main_pipeline.py` - CLI and main processing loop

### Perception Layer
- `src/perception/rtmpose.py` - RTMPose integration
- `src/perception/tracker.py` - Norfair tracking
- `src/perception/calibrator.py` - Homography
- `src/perception/audio.py` - Audio events
- `src/perception/pipeline.py` - Orchestration

### Recognition Layer
- `src/recognition/feature_math.py` - Geometry calculations
- `src/recognition/feature_extractor.py` - 101-dim extraction

### Utilities
- `src/utils/schemas.py` - Pydantic models
- `src/utils/profiling.py` - Latency/Memory/Health monitoring
- `src/utils/visualization.py` - Skeleton and heatmap drawing

### Tests
- `tests/` - 294 tests total
- `tests/test_e2e_clean_bout.py` - Main E2E suite

---

## 10. Contacts & References

- **Project Owner**: kevinhust
- **Repository**: https://github.com/kevinhust/fecingbuddy
- **Current Branch**: main
- **Latest Commit**: See `git log`

---

*This document is intended to help another AI or developer quickly understand the FencerAI project and identify concrete improvement opportunities.*
