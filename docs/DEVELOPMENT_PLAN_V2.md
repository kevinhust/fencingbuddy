# FencerAI Development Plan v2.0
*Live-Streaming Foil Coaching System*
*Version: 2.0 | Date: 2026-04-02*

---

## Executive Summary

**Goal**: Transform FencerAI from a batch-processing video analysis tool into a **live-streaming foil fencing coach**.

**Core addition**: Real-time video input (iPhone Continuity Camera) → Mac → Live skeleton overlay + coaching alerts → Post-session reports with historical tracking.

**Constraint**: No breaking changes to existing 101-dim feature vector, RTMPose lightweight mode, Norfair tracker, or canonicalization.

---

## 1. Current State Assessment

### 1.1 What We Have ✅

| Component | Status | Notes |
|-----------|--------|-------|
| **RTMPose Integration** | ✅ Complete | Lightweight mode: 158ms/frame |
| **Dual Fencer Tracker** | ✅ Complete | Norfair-based, referee filter |
| **101-Dim Feature Extraction** | ✅ Complete | All indices implemented |
| **Canonicalization** | ✅ Complete | Left-fencer perspective |
| **Audio Detection** | ✅ Basic | Energy threshold |
| **Visualization** | ✅ Complete | Skeleton overlay, heatmaps |
| **Health Monitoring** | ✅ Complete | Latency, confidence, FPS |
| **Unit Tests** | ✅ 294 tests | All passing |
| **CLI Pipeline** | ✅ Complete | File-based processing |

### 1.2 What We Need to Build 🆕

| Component | Status | Priority |
|-----------|--------|----------|
| **Live Video Input** | ❌ None | P0 |
| **Continuity Camera Support** | ❌ None | P0 |
| **Real-time Viewer** | ❌ None | P0 |
| **Coaching Engine** | ❌ None | P0 |
| **Action Classifier** | ❌ None | P1 |
| **Session Reports** | ❌ None | P1 |
| **Historical Database** | ❌ None | P1 |

### 1.3 Known Issues to Fix

| Issue | Priority | Effort |
|-------|----------|--------|
| Tracker scalar distance function warning | P0 | Low |
| No E2E test videos (infighting, club_noise) | P1 | Low |
| Basic audio detection (false positives) | P1 | Medium |
| No auto-calibration | P2 | Medium |

---

## 2. Target Hardware

| Machine | Role | Target |
|---------|------|--------|
| **M4 Mac Mini** | Primary dev machine | ≥30 FPS live |
| **MacBook Air M1 8GB** | On-site machine | ≥15 FPS, <6GB RAM |

### Performance Targets
- **Live input**: 1080p @ 15-30 FPS
- **Processing latency**: <200ms end-to-end
- **Memory**: <6 GB on M1 Air

---

## 3. New Architecture

### 3.1 Live Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FencerAI v2.0 Live Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  iPhone (Continuity Camera) ──USB/WiFi──► OpenCV VideoCapture              │
│                                               │                             │
│                                               ▼                             │
│                                      ┌────────────────┐                    │
│                                      │  Frame Buffer  │ ◄── 30 FPS input    │
│                                      └───────┬────────┘                    │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    Perception Pipeline                        │           │
│  │  RTMPose → Norfair Tracker → Audio Detection → FrameData   │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                            │                                              │
│                            ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                  Recognition Pipeline                        │           │
│  │  Feature Extractor → 101-dim Vector → Coaching Engine       │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                            │                                              │
│              ┌─────────────┴─────────────┐                              │
│              ▼                           ▼                                │
│  ┌─────────────────────┐     ┌─────────────────────┐                     │
│  │   Live Viewer       │     │   Session Storage   │                     │
│  │  (OpenCV Window)    │     │   (.npy + SQLite)   │                     │
│  │  - Red/Blue Skeleton│     └─────────────────────┘                     │
│  │  - Coaching Alerts  │                                               │
│  │  - FPS/Memory HUD   │                                               │
│  └─────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 New Directory Structure

```
fencer_ai/
├── src/
│   ├── perception/           # Existing (no changes)
│   │   ├── rtmpose.py
│   │   ├── tracker.py
│   │   ├── calibrator.py
│   │   ├── audio.py
│   │   ├── audio_buffer.py
│   │   └── pipeline.py
│   ├── recognition/           # Existing + New
│   │   ├── feature_math.py    # Existing
│   │   ├── feature_extractor.py  # Existing
│   │   ├── coaching_engine.py   # NEW: Rule-based coaching
│   │   └── action_classifier.py  # NEW: Attack/parry classification
│   ├── live/                  # NEW: Live input handling
│   │   ├── live_capture.py    # OpenCV VideoCapture wrapper
│   │   ├── frame_buffer.py    # Thread-safe frame buffer
│   │   └── device_manager.py  # Camera discovery + selection
│   ├── ui/                    # NEW: User interface
│   │   ├── live_viewer.py     # OpenCV live display
│   │   ├── alert_renderer.py  # Coaching text overlay
│   │   └── hud_overlay.py     # FPS, memory, status HUD
│   ├── reporting/              # NEW: Reports and history
│   │   ├── session_recorder.py # Save sessions to DB
│   │   ├── report_generator.py # HTML/PDF reports
│   │   ├── history_db.py      # SQLite historical data
│   │   └── drill_recommender.py # Training drill suggestions
│   ├── utils/                 # Existing (no changes)
│   └── main_pipeline.py       # Extended with --live mode
├── tests/
│   ├── unit/                  # Existing
│   ├── test_e2e_clean_bout.py # Existing
│   └── test_live/              # NEW: Live pipeline tests
├── docs/
│   ├── IMPROVEMENT_PROPOSAL.md
│   ├── DEVELOPMENT_PLAN_V2.md  # This document
│   └── LIVE_SETUP.md          # NEW: Camera setup guide
├── data/
│   ├── samples/                # Existing
│   └── sessions/               # NEW: Stored sessions
└── requirements.txt            # Add opencv-python, fastapi, etc.
```

---

## 4. Implementation Phases

### Phase 0: Live Input Foundation (Week 1)
**Milestone**: iPhone streams → Mac shows live skeletons on M4 Mini

#### Tasks

| Task | File | Description | Effort |
|------|------|-------------|--------|
| P0-1 | `src/live/live_capture.py` | OpenCV VideoCapture wrapper with camera index/URL support | Low |
| P0-2 | `src/live/frame_buffer.py` | Thread-safe circular buffer for live frames | Low |
| P0-3 | `src/main_pipeline.py` | Add `--live`, `--camera`, `--url` flags | Low |
| P0-4 | `src/ui/live_viewer.py` | OpenCV window with skeleton overlay | Medium |
| P0-5 | `src/ui/hud_overlay.py` | Real-time FPS + memory usage display | Low |
| P0-6 | Visualization | Update skeleton drawing for live mode (red=left/son, blue=right/opponent) | Low |
| P0-7 | `docs/LIVE_SETUP.md` | Continuity Camera setup instructions | Low |
| P0-8 | Tests | Basic live capture test | Low |

#### Acceptance Criteria
- [ ] `--live --camera 0` shows live video with skeletons
- [ ] FPS counter displays ≥15 FPS on M4 Mini
- [ ] Red skeleton = left fencer (your son), Blue = right fencer (opponent)
- [ ] Memory usage visible in HUD

---

### Phase 1: Performance & Stability (Week 1-2)
**Milestone**: Reliable ≥15 FPS on M1 Air 8GB

#### Tasks

| Task | File | Description | Priority |
|------|------|-------------|----------|
| P1-1 | `src/perception/tracker.py` | Use Norfair `iou_distance` instead of scalar | P0 |
| P1-2 | `src/live/frame_buffer.py` | Add frame-skip option for M1 Air | P0 |
| P1-3 | CLI | Add `--light` flag (force lightweight + skip frames) | P0 |
| P1-4 | Memory guard | Auto-skip if memory >5.5 GB | P1 |
| P1-5 | Tests | Fix/add E2E tests (need infighting + club_noise videos) | P1 |
| P1-6 | CI/CD | Add pre-commit hooks (ruff, black, mypy) | P1 |
| P1-7 | Profiling | Benchmark on M4 and M1 Air | P1 |

#### Acceptance Criteria
- [ ] Tracker warning removed (iou_distance used)
- [ ] `--light` mode achieves ≥15 FPS on M1 Air
- [ ] Memory stays <6 GB under continuous operation
- [ ] All 294 existing tests still pass

---

### Phase 2: Coaching Engine (Week 2-3)
**Milestone**: Real-time foil coaching alerts appear on screen

#### Tasks

| Task | File | Description |
|------|------|-------------|
| P2-1 | `src/recognition/coaching_metrics.py` | Extract 20 foil-specific metrics from 101-dim |
| P2-2 | `src/recognition/coaching_engine.py` | Rule-based if-then alert system |
| P2-3 | `src/ui/alert_renderer.py` | Render coaching text overlays |
| P2-4 | Audio sync | Link audio events to touch timestamps |
| P2-5 | Alert library | Implement first 10 coaching alerts |

#### Coaching Metrics (from 101-dim vector)

| Metric | Source Features | Coaching Meaning |
|--------|-----------------|-----------------|
| Lunge Speed | Velocity indices 49-72 | "Quick lunge — good attack" |
| Attack Prep Time | EMA smoothed velocity before lunge | "Too slow — opponent reads you" |
| Distance at Touch | Distance indices 26-36 | "Perfect distance for attack" |
| Recovery Speed | Acceleration after lunge | "Slow recovery — riposte risk" |
| Arm Extension % | Arm extension indices 43-48 | "Full extension = harder to parry" |
| Blade Control | Angular indices 37-40 | "Weak blade — beat attack" |
| Torso Lean | Torso orientation 41-42 | "Too leaning — balance risk" |
| Predictability | Velocity variance over last 5 frames | "Too predictable — vary your attacks" |

#### Initial Alert Library

1. **"Shorten recovery — riposte risk"** → Recovery speed < threshold
2. **"Attack now — opponent leaves distance open"** → Distance > optimal AND opponent stationary
3. **"Extend arm fully on next attack"** → Arm extension < 80%
4. **"Opponent favors 4th — attack on 5th"** → Opponent pattern detected
5. **"Your son drops guard on retreat"** → Son's torso lean + velocity indicates fatigue
6. **"Counter-attack opportunity — they overextend"** → Opponent acceleration pattern
7. **"Distance closing — parry-riposte ready"** → Son closing + opponent extended
8. **"Blade in prep — watch for fleche"** → Opponent weapon arm position
9. **"Recovery stance too wide — close guard"** → Son's stance after lunge
10. **"Trust your attack — you're fast enough"** → Son's lunge speed > opponent's reaction

#### Acceptance Criteria
- [ ] At least 10 coaching alerts implemented
- [ ] Alerts appear within 500ms of triggering event
- [ ] Alert text is plain English, actionable

---

### Phase 3: Live UI & Polish (Week 3-4)
**Milestone**: Polished live dashboard for sidelines use

#### Tasks

| Task | File | Description |
|------|------|-------------|
| P3-1 | `src/ui/live_viewer.py` | Score tracker (manual click or auto) |
| P3-2 | `src/ui/live_viewer.py` | Alert history panel |
| P3-3 | `src/ui/live_viewer.py` | "3 Things to Fix" summary |
| P3-4 | Session recording | One-click record session to file |
| P3-5 | Web UI (optional) | FastAPI + WebSocket for phone second-screen |

#### Live Viewer Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FencerAI Live  │  FPS: 18  │  MEM: 4.2GB  │  Mode: LIGHT  │  [REC]     │
├───────────────────────────────────────────────────────────┬─────────────────┤
│                                                           │  COACHING       │
│                                                           │                 │
│         ┌─────────────────────────────┐                   │  1. Shorten     │
│         │                             │                   │     recovery    │
│         │      LIVE VIDEO              │                   │  2. Attack now  │
│         │      + Red/Blue Skeletons   │                   │  3. Extend arm  │
│         │                             │                   │                 │
│         │                             │                   │─────────────────│
│         │                             │                   │  SCORE          │
│         │                             │                   │  Son:  12       │
│         └─────────────────────────────┘                   │  Opp:  10       │
│                                                           │                 │
├───────────────────────────────────────────────────────────┴─────────────────┤
│  Alert History: [1] 14:32 "Shorten recovery" [2] 14:35 "Attack now"     │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Acceptance Criteria
- [ ] Live viewer shows video + skeletons + HUD + alerts
- [ ] Score can be manually entered
- [ ] Session can be recorded with one click
- [ ] Alert history scrollable during session

---

### Phase 4: Action Classification & Bout Intelligence (Week 4-5)

#### Tasks

| Task | File | Description |
|------|------|-------------|
| P4-1 | `src/recognition/action_classifier.py` | Rule-based: attack, parry, riposte, counter, remise |
| P4-2 | Per-touch analysis | Link actions to score |
| P4-3 | Opponent profiling | Track opponent distance/attack habits |
| P4-4 | Practice vs Match | Compare session metrics across contexts |

#### Action Classification Rules

```
Attack:
  - Son velocity[49:72] > threshold
  - Distance to opponent decreasing
  - Arm extension increasing

Parry:
  - Opponent velocity spikes
  - Son arm position changes rapidly
  - Audio event (blade contact)

Riposte:
  - Son acceleration after parry
  - Forward movement following parry

Counter:
  - Opponent in attack prep
  - Son distance suddenly decreases
  - Son blade positioned for beat
```

#### Acceptance Criteria
- [ ] Actions correctly classified >80% of time
- [ ] Per-touch analysis links to score
- [ ] Opponent profile shows attack timing patterns

---

### Phase 5: Reports & History (Week 5-6)

#### Tasks

| Task | File | Description |
|------|------|-------------|
| P5-1 | `src/reporting/history_db.py` | SQLite database for sessions |
| P5-2 | `src/reporting/session_recorder.py` | Save full session data |
| P5-3 | `src/reporting/report_generator.py` | HTML report generation |
| P5-4 | `src/reporting/drill_recommender.py` | Suggest drills based on weaknesses |
| P5-5 | Historical graphs | Progress over weeks/months |

#### Report Template

```
═══════════════════════════════════════════════════════════════
                    FENCERAI SESSION REPORT
═══════════════════════════════════════════════════════════════
Date: 2026-04-02
Duration: 45 minutes
Score: Son 15 - Opponent 12

───────────────────────────────────────────────────────────────
                        TOP 3 DRILLS
───────────────────────────────────────────────────────────────
1. WALL DRILL (Recovery Speed)
   Problem: Recovery 23% slower than average
   Drill: 20x lunge + immediate retreat to line

2. DISTANCE CONTROL (Attack Timing)
   Problem: 3 attacks launched too early
   Drill: Partner holds distance, attack on command

3. BLADE WORK (Parry-5)
   Problem: Only 2/5 parries successful
   Drill: Parry-5 retreat, 30x repetitions

───────────────────────────────────────────────────────────────
                        MATCH HEATMAP
───────────────────────────────────────────────────────────────
[Distance heatmap across 27 touches]

───────────────────────────────────────────────────────────────
                    OPPONENT PROFILE
───────────────────────────────────────────────────────────────
- Prefers Attack: 65% (vs your 55%)
- Weakened after: 3rd touch (fatigue visible)
- Attack from Distance: 1.2m (close)
- Pattern: 3x Attack → 1x Retreat

═══════════════════════════════════════════════════════════════
```

#### Acceptance Criteria
- [ ] Session saved to SQLite automatically
- [ ] HTML report generates in <10 seconds
- [ ] Drill recommendations are specific and actionable
- [ ] Historical comparison works across sessions

---

### Phase 6: Testing, Documentation & Polish (Week 6-7)

#### Tasks

| Task | Description |
|------|-------------|
| P6-1 | Add pytest-benchmark for performance |
| P6-2 | Add 6+ E2E live tests (recorded sessions) |
| P6-3 | Full Sphinx documentation |
| P6-4 | Deprecate Pydantic json_encoders |
| P6-5 | Add CHANGELOG.md |
| P6-6 | Add CONTRIBUTING.md |

#### Acceptance Criteria
- [ ] All tests pass including new live tests
- [ ] Performance benchmarks track FPS over time
- [ ] Documentation complete for new features

---

## 5. Technical Decisions

### TD-1: Live Input Library
**Decision**: Use OpenCV VideoCapture (not ffmpeg or pyav)
**Rationale**: Works natively with Continuity Camera, minimal dependencies

### TD-2: Alert Display
**Decision**: OpenCV window (not Qt or tkinter)
**Rationale**: Lightweight, works cross-platform, simple text rendering

### TD-3: Session Storage
**Decision**: SQLite for metadata, .npy for features
**Rationale**: Simple, no server needed, works offline

### TD-4: Report Format
**Decision**: HTML + PDF (via weasyprint or similar)
**Rationale**: Rich formatting, printable, no external service

### TD-5: Coaching Rules
**Decision**: Rule-based first, ML later
**Rationale**: Interpretable, fast, low data requirements

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M1 Air memory pressure | High | High | Frame skip, lightweight mode, memory guard |
| Continuity Camera latency | Medium | Medium | USB preferred, fallback to Camo Studio |
| Pose accuracy in live lighting | Medium | Medium | Confidence threshold + EMA predictor |
| Coaching rules too simplistic | High | Low | Start rule-based, add ML later |
| Audio false positives | Medium | Medium | Correlate with pose for confirmation |

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Live FPS on M4 Mini | ≥30 FPS | FPS HUD |
| Live FPS on M1 Air | ≥15 FPS | FPS HUD |
| Memory on M1 Air | <6 GB | Memory HUD |
| Alert latency | <500ms | Timestamp delta |
| Action accuracy | >80% | Manual review |
| Report generation | <10s | Timer |
| All existing tests | Pass | pytest |

---

## 8. Implementation Order

```
Week 1:     Phase 0 (Live Input) + Phase 1 (Performance)
Week 2-3:   Phase 2 (Coaching Engine) + Phase 3 (Live UI)
Week 4-5:   Phase 4 (Action Classification) + Phase 5 (Reports)
Week 6-7:   Phase 6 (Testing & Polish)
```

**Quick Win**: After Week 1, you have live skeletons + FPS display.
**Minimum Viable**: After Week 3, you have live coaching alerts.

---

## 9. Next Action

Ready to implement. Which phase should we start with?

1. **"Start Phase 0"** → Live input foundation (live skeletons this week)
2. **"Start Phase 1"** → Performance optimization first
3. **"Start Phase 2"** → Coaching engine focus
4. **"Pick specific task"** → Tell me which task you want

---

*Document Version: 1.0 - Ready for implementation*
