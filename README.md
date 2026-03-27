# FencerAI - The Spatio-Temporal Fencing AI Coach
*Version: 1.1 | Last Updated: 2026-03*

FencerAI is an edge-first, real-time computer vision system designed to analyze fencing bouts using a single RGB camera. It extracts a 101-dimensional spatio-temporal feature vector to evaluate distances, tempos, and actions, eventually powering an automated Right-of-Way reasoning engine.

## 🎯 Milestone 1 Objective
Extract clean, synchronized, and noise-free `(N_frames, 2, 101)` feature matrices from raw fencing videos, handling dual-person tracking, referee occlusion, and high-speed motion.

## 🚀 Quick Start (Sprint 1 Demo)
```bash
# 1. Initialize environment
pip install -r requirements.txt

# 2. Run the end-to-end extraction pipeline
python -m src.main_pipeline --video data/samples/bout_clean.mp4 --output outputs/features.npy

fencer_ai/
├── src/
│   ├── perception/       # RTMPose wrapper, Norfair tracker, Audio buffer, Calibrator
│   ├── recognition/      # feature_math.py (101-dim math engine)
│   ├── utils/            # schemas.py (Pydantic), visualizer, TimestampedBuffer
├── data/                 
│   ├── clean_bouts/      # Standard 2-person matches without occlusion
│   ├── infighting/       # Close-quarters with heavy occlusion + audio touches
│   └── club_noise/       # Real-world footage with referees/coaches walking by
├── requirements.txt      # mmpose, norfair, pydantic, opencv-python, scipy, librosa