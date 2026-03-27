# FencerAI: Spatio-Temporal Fencing Analysis Architecture
*Version: 1.2 | Last Updated: 2026-03*

## 1. Vision & Identity
FencerAI is a specialized, edge-first pipeline designed to capture explosive fencing movements (`<500ms`) and distance management. This is not a generic pose estimation app; it is a high-performance spatio-temporal engine for competitive fencing analysis.

## 2. Pipeline Architecture
The system follows a strict linear flow with data validation at each boundary:

`Video/Audio Input` -> `TimestampedBuffer (sync)` -> `RTMPose (multi-person)` -> `Norfair Dual Tracker + Referee Filter` -> `Calibrator (Homography)` -> `Feature Extractor (101-dim, Canonicalized)` -> `(N, 2, 101) Numpy Matrix`.

### 2.1 Data Integrity & Synchronization
- **Pydantic as Source of Truth:** All data moving between the **Perception Layer** (RTMPose/Norfair) and **Recognition Layer** (Feature Math) MUST be validated by Pydantic models in `src/utils/schemas.py`.
- **Type Safety:** All implementation logic must utilize strict Python 3.9+ type hints.
- **Temporal Normalization:** All velocity and acceleration calculations (1st/2nd derivatives) MUST use actual timestamps (`Δt`) to maintain accuracy across variable FPS recordings.

## 3. Canonicalization Strategy
To reduce classifier complexity, features are normalized to a **Left Fencer Perspective**:
- **Flip Logic:** If `fencer_id == 1` (Right Fencer), the coordinate system is horizontally flipped *before* feature extraction.
- **Directional Unity:** Advance/Retreat vectors are unified (e.g., positive X velocity always indicates closing the distance to the opponent).

## 4. Physical Metric Space (Homography)
Physical distance (Meters) is required for Distance Management features.
- **Method:** 3x3 Homography Matrix transforms 2D pixels into a top-down metric space.
- **Calibration:** Requires user-guided markers or piste-end detection.

## 5. The 101-Dimensional Feature Dictionary
Each frame generates a strict 101-dimensional vector for each fencer. **Dimensions must not be altered or added.**

| Index Range | Category | Dim | Description / Math Logic |
| :--- | :--- | :--- | :--- |
| **0 - 23** | Static Geometry | 24 | 12 selected canonical keypoints (L/R Shoulders, Elbows, Wrists, Hips, Knees, Ankles). Normalized by shoulder-width, pelvis-centered. |
| **24 - 25** | Center of Mass (CoM) | 2 | (x, y) of the calculated pelvis/hip center. |
| **26 - 36** | Distance (Interaction) | 11 | **Physical METERS.** Includes Inter-fencer pelvis/foot distance, stance width, and weapon-hand to torso distance. |
| **37 - 40** | Angles | 4 | `np.arctan2`: Front/Back Knee, Weapon Elbow, Torso Lean. |
| **41 - 42** | Torso Orientation | 2 | Vector direction from Shoulder center to Hip center. |
| **43 - 48** | Arm Extension | 6 | Weapon arm extension ratio (Relative to reach) + directional angle. |
| **49 - 72** | Velocity (1st Deriv) | 24 | EMA smoothed (`alpha=0.6~0.8`) 1st derivative of indices 0-23. |
| **73 - 96** | Accel (2nd Deriv) | 24 | EMA smoothed 2nd derivative of indices 0-23. Captures lunge force. |
| **97 - 100** | Meta & Audio | 4 | 97-98: CoM Velocity. 99: CoM Acceleration. 100: Audio Touch Flag (1.0 = Touch, 0.0 = Silence). |

## 6. Tracker Rules (The "Referee Filter")
- **Initialization:** Lock the 2 largest BBoxes in the bottom 70% of the Y-axis (ignoring standing referees). Assign ID 0 (Left) and ID 1 (Right).
- **Maintenance:** If detections > 2, prioritize candidates with highest pose-embedding similarity to locked IDs and lowest Y-variance.
- **Graceful Failures:** If an ID is lost due to occlusion, the system must NOT crash. Use EMA-predicted positions for the next valid match or return `None` as a sentinel.