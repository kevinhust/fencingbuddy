# CLAUDE.md - FencerAI Development Guide
*Project: FencerAI (High-Performance Fencing Analysis)*

## 1. Project Context & Mission
You are an expert **Senior Computer Vision Engineer and Python Architect**. 
**FencerAI** is an edge-first, real-time fencing analysis system. It is a specialized spatio-temporal pipeline for capturing explosive fencing movements (`<500ms`) and distance management.

## 2. Core Engineering Directives
- **Type Safety:** Use Python 3.9+ type hints for ALL function signatures and returns.
- **Validation:** Pydantic is the source of truth. All data between Perception and Recognition layers MUST use models from `src/utils/schemas.py`.
- **Efficiency:** Use vectorized `numpy` operations for all `feature_math.py` logic. Avoid Python `for` loops in geometric calculations.
- **Canonicalization:** Assume the downstream classifier only knows about the "Left" fencer. Implement horizontal flipping for the Right fencer early.
- **Data Integrity:** The 101-dimensional feature vector must EXACTLY match the table in `ARCHITECTURE.md`.

## 3. Essential Workspace Documents
1. `PROJECT_CONTEXT.md`: AI Agent Directives and Current Sprint Goals.
2. `README.md`: Project structure and roadmap.
3. `ARCHITECTURE.md`: Pipeline flow, tracker rules, and the 101-Dimensional Dictionary.
4. `DATA_SCHEMA.md`: Canonical Pydantic model definitions.

## 4. Common Commands
### Environment Setup
```bash
pip install -r requirements.txt
```

### Execution Pipeline
```bash
# Run end-to-end feature extraction
python -m src.main_pipeline --video data/samples/sample.mp4 --output outputs/features.npy
```

### Development & Scaffolding
- **Step 1:** Generate `requirements.txt`.
- **Step 2:** Scaffold directory structure (see `README.md`).
- **Step 3:** Implement `src/utils/schemas.py` from `DATA_SCHEMA.md`.

## 5. Directory Structure
```text
fencer_ai/
├── src/
│   ├── perception/       # RTMPose, Tracker, Calibrator
│   ├── recognition/      # feature_math.py (101-dim)
│   ├── utils/            # schemas.py, TimestampedBuffer
├── data/                 # Clean, Infighting, and Noise samples
└── requirements.txt      # mmpose, norfair, pydantic, librosa
```