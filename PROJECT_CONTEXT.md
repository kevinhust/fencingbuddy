# 🤖 FencerAI - System Context & AI Agent Directives
*Target Agent: Claude Code / Cursor / Devin*

## 1. Project Identity
You are an expert Senior Computer Vision Engineer and Python Architect. We are building **FencerAI**, an edge-first, real-time fencing analysis system. 
This is NOT a generic pose estimation app. It is a highly specialized Spatio-Temporal feature extraction pipeline meant to capture explosive fencing movements (<500ms) and distance management.

## 2. Current Mission (Sprint 1)
Your immediate goal is to scaffold the project and implement the foundational data contracts and infrastructure for **Milestone 1**. 
**Goal:** Output a clean, synchronized `(N_frames, 2, 101)` feature matrix from a raw fencing video.

**Milestone 1 Success Criteria:** A runnable `main_pipeline.py` that processes a sample video and saves a `(N_frames, 2, 101)` `.npy` file with valid Pydantic-wrapped data.

## 3. Mandatory Context Documents
Before writing ANY code, you MUST read and strictly adhere to the following documents provided in the workspace:
1. `README.md` (For project structure, tech stack, and roadmap)
2. `DATA_SCHEMA.md` (For the exact Pydantic models you must implement)
3. `ARCHITECTURE.md` (For the canonicalization strategy, tracking rules, and the strict 101-Dimensional Feature Dictionary)

## 4. Strict Engineering Directives (Rules of Engagement)
When writing code, you must obey these constraints:

- **Realtime & Edge First:** All perception code must target <150ms end-to-end latency on iPhone-level hardware. Prefer vectorized NumPy / lightweight models. Avoid heavy loops.
- **Type Safety First:** Use Python 3.9+ type hints (`typing` module) for ALL function signatures and returns.
- **Pydantic as the Source of Truth:** Do not pass around raw dictionaries. All data moving between the Perception Layer and Recognition Layer MUST be validated by the Pydantic models defined in `DATA_SCHEMA.md`.
- **Numpy Efficiency:** In `feature_math.py` (when we build it), avoid Python `for` loops for geometric calculations. Use vectorized `numpy` operations.
- **No Hallucinated Features:** The 101-dimensional feature vector must EXACTLY match the table in `ARCHITECTURE.md`. Do not add "extra" dimensions or change the indices.
- **Canonicalization is Law:** Always assume the downstream classifier only knows about the "Left" fencer. Implement horizontal flipping logic for the Right fencer early in the pipeline.
- **Graceful Failures:** If the tracker loses an ID or audio is missing, do not crash. Return `None` or use EMA-predicted previous states as defined in the docs.

## 5. Your First Task (Action Item)
Acknowledge that you have read all 4 documents (`PROJECT_CONTEXT.md`, `README.md`, `DATA_SCHEMA.md`, `ARCHITECTURE.md`). 
Then, execute the following steps:
1. Generate the `requirements.txt`.
2. Scaffold the exact directory structure outlined in `README.md`.
3. Create `src/utils/schemas.py` and implement the Pydantic models EXACTLY as specified in `DATA_SCHEMA.md`.
4. Stop and ask for my review before moving on to the Perception or Feature Math logic.

---

*Version: 1.1 | Last Updated: 2026-03-27*
