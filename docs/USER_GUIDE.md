# FencerAI User Guide
*Version: 2.0 | Last Updated: 2026-04-02*

---

## Overview

**FencerAI** is a real-time fencing analysis system that extracts 101-dimensional spatio-temporal feature vectors from fencing videos. It provides:

- Real-time pose estimation using RTMPose
- 101-dim feature extraction per frame (per fencer)
- Dual-person tracking with referee filtering
- Rule-based coaching alerts
- Action classification (attack, parry, fleche, etc.)
- Session recording with SQLite storage
- HTML report generation

---

## Installation

### Requirements

- Python 3.10+
- macOS, Linux, or Windows

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kevinhust/fencingbuddy.git
cd fencingbuddy

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies for Live Mode

```bash
pip install opencv-python
```

---

## Quick Start

### Process a Video File

```bash
python -m src.main_pipeline --video data/samples/sample.mp4 --output outputs/features
```

This extracts features from `sample.mp4` and saves:
- `outputs/features.npy` - Feature matrix
- `outputs/features.json` - Metadata

### Live Streaming (with webcam)

```bash
python -m src.main_pipeline --live --camera 0
```

### Live Streaming from RTSP Camera

```bash
python -m src.main_pipeline --live --url rtsp://camera-ip:554/stream
```

---

## Command-Line Options

### Input Sources (required, mutually exclusive)

| Option | Description |
|--------|-------------|
| `--video PATH` | Path to input video file |
| `--live` | Enable live streaming mode |
| `--camera INDEX` | Camera index for live mode (default: 0) |
| `--url URL` | RTSP/HTTP stream URL |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | - | Output path (without extension) |
| `--visualize` | False | Enable skeleton overlay video |
| `--heatmap` | False | Export feature heatmap PNGs |

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--confidence-threshold` | 0.3 | Minimum pose detection confidence |
| `--device` | cpu | Device: cpu, cuda, or mps |
| `--skip-frames N` | 0 | Skip every N frames (0 = process all) |
| `--frame-width` | 1920 | Frame width for canonicalization |
| `--calibration-file PATH` | None | Path to homography calibration JSON |

### Live Mode Options

| Option | Default | Description |
|--------|---------|-------------|
| `--live-fps` | 30 | Target FPS for live capture |
| `--live-resolution` | 1920x1080 | Capture resolution |
| `--light` | False | Light mode for M1 Mac (reduced quality for speed) |

### Debugging Options

| Option | Description |
|--------|-------------|
| `--profile` | Enable performance profiling |

---

## Feature Extraction

### The 101-Dimensional Feature Vector

Each frame produces a `(N, 2, 101)` matrix where:
- `N` = number of frames
- `2` = two fencers
- `101` = feature dimensions

| Index Range | Category | Description |
|------------|----------|-------------|
| 0-23 | Static Geometry | 12 keypoints (x,y normalized) |
| 24-25 | Center of Mass | Pelvis/hip center |
| 26-36 | Distance | Inter-fencer distances (meters) |
| 37-40 | Angles | Knee, elbow, torso angles |
| 41-42 | Torso Orientation | Forward lean, lateral tilt |
| 43-48 | Arm Extension | Weapon arm extension ratio |
| 49-72 | Velocity | 1st derivative (EMA smoothed) |
| 73-96 | Acceleration | 2nd derivative (EMA smoothed) |
| 97-98 | CoM Velocity | Center of mass velocity |
| 99 | CoM Acceleration | Center of mass acceleration |
| 100 | Audio Flag | 1.0 = blade touch detected |

---

## Live Mode Features

### Real-Time Display

When running in live mode, the viewer shows:
- Skeleton overlay with keypoints
- Bounding boxes with fencer IDs (0=Left, 1=Right)
- FPS counter
- Processing time

### Coaching Alerts

The coaching engine evaluates metrics in real-time and displays alerts:

| Priority | Level | Example |
|----------|-------|---------|
| 1 | Critical | "Counter-attack opportunity" |
| 2 | Warning | "Attack now — distance open" |
| 3 | Info | "Extend arm fully" |
| 4 | Debug | "Watch for fleche attack" |
| 5 | Trace | "Both fencers tracked" |

### Available Coaching Alerts

1. "Shorten recovery — riposte risk"
2. "Attack now — distance open"
3. "Extend arm fully"
4. "Opponent favors 4th — attack 5th"
5. "Son drops guard on retreat"
6. "Counter-attack opportunity"
7. "Distance closing — parry-riposte ready"
8. "Watch for fleche attack"
9. "Recovery stance too wide"
10. "Trust your attack — you're fast enough"

### Action Classification

The system classifies fencer actions in real-time:

| Action | Detection Criteria |
|--------|-------------------|
| IDLE | Low movement (< 2.0 velocity) |
| ADVANCE | Positive forward velocity |
| RETREAT | Negative forward velocity |
| ATTACK | High lunge speed + arm extension |
| ATTACK_PREP | Medium speed + good extension |
| PARRY | Opponent attacking + good blade position |
| RIPOSTE | Post-parry forward motion |
| COUNTER_ATTACK | Quick forward on opponent prep |
| FLECHE | Deep torso lean + high speed |
| RECOVERY | Post-action deceleration |

---

## Session Recording

### Automatic Recording

When running live mode, the system records:
- Session start/end time
- Final score (manual entry)
- All coaching alerts triggered
- Action sequence log

### SQLite Database

Sessions are stored in `outputs/fencerai_history.db`:

```python
from src.reporting.history_db import HistoryDatabase

db = HistoryDatabase()
sessions = db.get_sessions(limit=10)
for session in sessions:
    print(f"{session.session_name}: {session.son_score}-{session.opp_score}")
```

### HTML Reports

Generate post-session reports:

```python
from src.reporting.report_generator import ReportGenerator, SessionReportData

data = SessionReportData(
    session_name="Practice Bout",
    date="2026-04-02T10:00:00",
    duration_seconds=600.0,
    son_score=5,
    opp_score=3,
    son_fencer_id="son_001",
    opp_fencer_id="opp_001",
    alerts=[...],
    action_stats={"attack": 10, "parry": 5},
    alert_stats={"attack": 3, "distance": 2},
    frequent_alerts=[("Extend arm", 5)],
)

gen = ReportGenerator()
html = gen.generate_session_report(data)
gen.save_report(html, "outputs/reports/bout_report.html")
```

---

## Calibration

### Why Calibrate?

Calibration transforms pixel coordinates to physical meters, enabling accurate distance measurements.

### How to Calibrate

1. Record a calibration video with known piste dimensions
2. Use the `HomographyCalibrator` class:

```python
from src.perception.calibrator import HomographyCalibrator

calibrator = HomographyCalibrator()
calibrator.add_points_from_piste_corners(
    pixel_corners=[[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    world_corners=[[0,0], [18,0], [18,1.5], [0,1.5]],  # Standard piste
)
error = calibrator.compute_reprojection_error()
if error < 5.0:  # pixels
    calibrator.save("calibration.json")
```

3. Use with pipeline:

```bash
python -m src.main_pipeline --video input.mp4 --calibration-file calibration.json
```

---

## Performance Tips

### Light Mode (M1 Mac)

For real-time performance on M1 Macs:

```bash
python -m src.main_pipeline --live --camera 0 --light
```

This enables:
- Lightweight RTMPose model
- Frame skipping (process every 2nd frame)

### Device Selection

| Device | Speed | Notes |
|--------|-------|-------|
| cpu | Slowest | Works on all machines |
| mps | Fast | Apple Silicon GPU (macOS) |
| cuda | Fastest | NVIDIA GPU (Linux/Windows) |

### Skip Frames

For longer videos where real-time isn't required:

```bash
python -m src.main_pipeline --video long_bout.mp4 --skip-frames 2
```

---

## Troubleshooting

### No Pose Detected

- Ensure good lighting
- Adjust `--confidence-threshold` lower (e.g., 0.2)
- Verify camera is pointed at the fencers

### Tracking Issues

- If fencers swap IDs, restart the session
- Ensure only 2 fencers in frame (referee filtering ignores bottom 70%)

### Import Errors

Missing packages:
```bash
pip install opencv-python numpy torch torchvision onnxruntime
```

### Slow Performance

- Enable light mode: `--light`
- Use mps/cuda device
- Reduce resolution: `--live-resolution 1280x720`

---

## File Structure

```
fencingbuddy/
├── src/
│   ├── main_pipeline.py      # CLI entry point
│   ├── perception/           # RTMPose, tracker, audio
│   ├── recognition/         # Feature extraction
│   ├── coaching/            # Coaching engine & alerts
│   ├── reporting/           # History DB & HTML reports
│   ├── ui/                  # Live viewer, HUD, alerts
│   ├── live/                # Live capture, session recorder
│   └── utils/               # Config, logging, schemas
├── tests/                   # Unit & integration tests
├── docs/                    # Documentation
└── outputs/                 # Generated outputs
    ├── features/             # Feature matrices
    ├── reports/              # HTML reports
    └── fencerai_history.db   # SQLite database
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `PerceptionPipeline` | `src.perception.pipeline` | Frame processing (pose, track, audio) |
| `FeatureExtractor` | `src.recognition.feature_extractor` | 101-dim feature extraction |
| `CoachingEngine` | `src.coaching.coaching_engine` | Alert evaluation |
| `ActionClassifier` | `src.recognition.action_classifier` | Action recognition |
| `HistoryDatabase` | `src.reporting.history_db` | Session storage |
| `ReportGenerator` | `src.reporting.report_generator` | HTML report generation |

---

## License

MIT License

---

## Support

- GitHub Issues: https://github.com/kevinhust/fencingbuddy/issues
- Documentation: See `docs/` directory
