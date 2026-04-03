# FencerAI Live Setup Guide
*Version: 1.0 | Date: 2026-04-02*

## Overview

FencerAI v2.0 supports live video streaming from iPhone (Continuity Camera) to Mac for real-time fencing analysis.

## Hardware Requirements

| Machine | Role | Target |
|---------|------|--------|
| **M4 Mac Mini** | Primary dev | ≥30 FPS |
| **MacBook Air M1 8GB** | On-site | ≥15 FPS, <6GB RAM |

## iPhone Continuity Camera Setup

### Option 1: USB Connection (Recommended)

**Lowest latency for M4 Mac Mini**

1. **Connect iPhone to Mac via USB cable**
2. **On iPhone**: Trust this computer (if prompted)
3. **On iPhone**: Go to Settings → FaceTime → and enable "Use as Camera"
4. **On Mac**: Open System Settings → Camera → Select your iPhone

```
Test camera connection:
    python -c "from src.live.live_capture import list_cameras; print(list_cameras())"
```

### Option 2: Wireless (Continuity Camera)

**Works well for M1 Air (cordless)**

1. **Connect iPhone and Mac to same WiFi network**
2. **On iPhone**: Enable FaceTime (Settings → FaceTime → ON)
3. **On iPhone**: Enable Continuity Camera (Settings → General → AirPlay & Handoff → ON)
4. **On Mac**: Open System Settings → Camera → Select your iPhone
5. **On iPhone**: Keep screen unlocked and in landscape orientation

Note: Wireless latency is typically 50-100ms higher than USB.

### Option 3: Camo Studio (Best Wireless Alternative)

[Camo Studio](https://reincubate.com/camo/) provides professional-grade wireless camera streaming.

1. Download Camo Studio on your Mac
2. Download Camo app on your iPhone
3. Connect via USB or WiFi
4. Configure in Camo:
   - Resolution: 1080p
   - Frame rate: 30 FPS
   - Enable "Pro" mode for lower latency

## Running Live Mode

### Quick Start

```bash
# List available cameras
python -c "from src.live.live_capture import list_cameras; print(list_cameras())"

# Start live with default camera
python -m src.main_pipeline --live

# Start with specific camera index
python -m src.main_pipeline --live --camera 1

# Start with 1080p @ 30fps (default)
python -m src.main_pipeline --live --camera 0

# Light mode for M1 Air (frame skipping)
python -m src.main_pipeline --live --light

# With RTSP stream URL
python -m src.main_pipeline --live --url rtsp://camera-ip:554/stream
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--live` | Enable live streaming mode | Required |
| `--camera N` | Camera index (0, 1, etc.) | Auto-detect |
| `--url URL` | RTSP/HTTP stream URL | None |
| `--live-fps N` | Target FPS for capture | 30 |
| `--live-resolution WxH` | Resolution (e.g., 1920x1080) | 1920x1080 |
| `--light` | Enable light mode for M1 Air | False |
| `--confidence-threshold` | Pose confidence threshold | 0.3 |

### Light Mode (M1 Air)

For MacBook Air M1 8GB, use `--light` flag to:
- Use lightweight RTMPose model (faster but less accurate)
- Skip every 2nd frame (15 FPS effective)
- Reduce memory usage

```bash
python -m src.main_pipeline --live --light --camera 0
```

## Live Viewer Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `r` | Toggle recording indicator |
| `s` | Son scores (+1) |
| `o` | Opponent scores (+1) |
| `z` | Undo son score |
| `x` | Undo opponent score |
| `c` | Clear scores |

## Troubleshooting

### Camera Not Found

```bash
# Check available cameras
python -c "from src.live.live_capture import list_cameras; print(list_cameras())"

# Should output something like:
# [0, 1, 2]  # Available camera indices
```

### Black Screen

1. Check if camera is being used by another app
2. Try closing FaceTime
3. Unplug/replug USB cable
4. Restart Continuity Camera: Toggle FaceTime off/on on iPhone

### Low FPS

1. Use USB instead of wireless
2. Reduce resolution: `--live-resolution 1280x720`
3. Enable light mode: `--light`
4. Close other apps using camera

### Memory Warning (>6GB)

1. Enable light mode: `--light`
2. Reduce buffer size (requires code change)
3. Close other memory-intensive apps

## Performance Tips

### For M4 Mac Mini (Development)
- Use USB connection for lowest latency
- Full resolution (1920x1080) at 30 FPS
- No frame skipping needed

### For M1 Air (On-site)
- Wireless is acceptable (less cable management)
- Light mode: `--light`
- Target 15 FPS (good enough for coaching)

## Testing Without Camera

You can test live mode with a video file:

```bash
# Process video in "live-like" mode (frame by frame)
python -m src.main_pipeline --video data/samples/498_1728950803.mp4 --output outputs/test

# Or just run the existing tests
python -m pytest tests/test_e2e_clean_bout.py -v
```

## Next Steps

After Phase 0 is working:
- **Phase 1**: Optimize performance for M1 Air
- **Phase 2**: Add coaching alerts
- **Phase 3**: Polish live UI

See [DEVELOPMENT_PLAN_V2.md](DEVELOPMENT_PLAN_V2.md) for full roadmap.
