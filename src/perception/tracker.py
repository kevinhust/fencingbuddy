"""
FencerAI FencerTracker - Norfair-based Dual Fencer Tracker with Referee Filter
============================================================================
Version: 1.0 | Last Updated: 2026-03-27

Norfair-based tracker with specialized initialization and maintenance for fencing:
- Lock 2 largest BBoxes in bottom 70% of Y-axis (referee filter)
- Assign ID 0 (Left) and ID 1 (Right) deterministically
- Graceful failure handling with EMA-predicted positions
- Pose embedding similarity for matching

Architecture (per ARCHITECTURE.md Section 6):
    - Initialization: Lock 2 largest BBoxes in bottom 70% of Y-axis
    - Maintenance: Prioritize pose-embedding similarity to locked IDs
    - Graceful Failures: EMA-predicted positions when occluded
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import numpy as np

from norfair import Tracker, Detection

from src.utils.schemas import FencerPose, Keypoint
from src.utils.constants import (
    REFREE_FILTER_Y_THRESHOLD,
    MAX_DETECTION_DISTANCE,
    MAX_TRACKER_AGE,
    MIN_HIT_HITS,
)


# =============================================================================
# Tracker State
# =============================================================================

@dataclass
class TrackedFencer:
    """Represents a tracked fencer with EMA-smoothed state."""
    fencer_id: int  # 0 = Left (Canonical), 1 = Right
    pose: FencerPose
    last_keypoints: np.ndarray  # For EMA prediction
    y_variance: float  # Stability metric
    hit_count: int = 0
    is_confirmed: bool = False


# =============================================================================
# Pose Embedder for Similarity Matching
# =============================================================================

class PoseEmbedder:
    """
    Extracts pose embedding from RTMPose keypoints for similarity matching.
    Uses cosine similarity for matching.
    """

    @staticmethod
    def compute_embedding(pose: FencerPose) -> np.ndarray:
        """
        Compute pose embedding vector from FencerPose.

        The embedding is the normalized keypoint configuration - a compact
        representation of the pose that can be used for similarity matching.

        Args:
            pose: FencerPose with keypoints

        Returns:
            np.ndarray: Normalized embedding vector
        """
        keypoint_array = pose.get_keypoint_array()  # Shape: (17, 3)
        if keypoint_array.shape[0] < 12:
            raise ValueError(f"Need at least 12 keypoints, got {keypoint_array.shape[0]}")

        # Use first 12 keypoints (FERA subset) for embedding
        # Shape: (12, 3) -> flatten to (36,)
        embedding = keypoint_array[:12, :2].flatten()  # x, y only (no conf)

        # Normalize by shoulder width for scale invariance
        shoulder_width = pose.shoulder_width()
        if shoulder_width > 0:
            embedding = embedding / shoulder_width

        return embedding.astype(np.float32)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Cosine similarity in range [-1, 1]
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# =============================================================================
# Fence Detection Filter
# =============================================================================

def fence_detection_distance(
    detection: Detection,
    tracked_object: "TrackedObject",
) -> float:
    """
    Custom distance function for Norfair tracker using pose similarity.

    This replaces IOU-based matching with pose embedding similarity,
    which is more robust for fencing poses.
    """
    # Get stored keypoints from detection
    if not hasattr(detection, 'data') or detection.data is None:
        return float('inf')

    det_keypoints = detection.data.get('keypoints')
    if det_keypoints is None:
        return float('inf')

    # Get stored keypoints from tracked object
    if not hasattr(tracked_object, 'last_detection') or tracked_object.last_detection is None:
        return float('inf')

    obj_det = tracked_object.last_detection
    obj_keypoints = obj_det.data.get('keypoints') if obj_det and obj_det.data else None
    if obj_keypoints is None:
        return float('inf')

    # Compute embeddings
    try:
        # Create temporary FencerPose objects for embedding
        det_pose = FencerPose(
            fencer_id=0,
            bbox=(0, 0, 1, 1),
            keypoints=[Keypoint(x=float(k[0]), y=float(k[1]), conf=float(k[2])) for k in det_keypoints]
        )
        obj_pose = FencerPose(
            fencer_id=0,
            bbox=(0, 0, 1, 1),
            keypoints=[Keypoint(x=float(k[0]), y=float(k[1]), conf=float(k[2])) for k in obj_keypoints]
        )

        # Compute cosine similarity
        emb1 = PoseEmbedder.compute_embedding(det_pose)
        emb2 = PoseEmbedder.compute_embedding(obj_pose)
        similarity = PoseEmbedder.cosine_similarity(emb1, emb2)

        # Convert similarity to distance (higher similarity = lower distance)
        # Range: [-1, 1] -> [0, 2] -> distance
        return 1.0 - similarity

    except Exception:
        return float('inf')


# =============================================================================
# EMA Predictor for Graceful Failure Handling
# =============================================================================

class EMAPredictor:
    """
    Exponential Moving Average predictor for graceful failure handling.
    Used to predict fencer position when tracking is lost.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize EMA predictor.

        Args:
            alpha: EMA smoothing factor (0.6-0.8 recommended)
        """
        self.alpha = alpha
        self._last_value: Optional[np.ndarray] = None

    def predict(self, current_value: np.ndarray) -> np.ndarray:
        """
        Predict next value using EMA smoothing.

        Args:
            current_value: Current keypoint array

        Returns:
            np.ndarray: Predicted next value
        """
        if self._last_value is None:
            self._last_value = current_value.copy()
            return current_value.copy()

        # EMA: predicted = alpha * current + (1 - alpha) * last
        predicted = self.alpha * current_value + (1 - self.alpha) * self._last_value
        self._last_value = current_value.copy()
        return predicted

    def reset(self) -> None:
        """Reset the predictor state."""
        self._last_value = None


# =============================================================================
# FencerTracker
# =============================================================================

class FencerTracker:
    """
    Norfair-based dual fencer tracker with referee filter.

    Specialized tracker for fencing that:
    - Locks 2 largest BBoxes in bottom 70% of Y-axis (referee filter)
    - Assigns ID 0 (Left) and ID 1 (Right) deterministically
    - Uses pose embedding similarity for matching
    - Implements graceful failure handling with EMA prediction

    Args:
        distance_threshold: Max distance for matching (pixels)
        max_age: Max frames to keep lost track alive
        min_hits: Frames before track is confirmed
        ref_y_threshold: Y-axis threshold for referee filter (default 0.3)

    Example:
        >>> tracker = FencerTracker()
        >>> # First frame: initialize
        >>> poses = tracker.initialize(detections, frame_height=480)
        >>> # Subsequent frames: update
        >>> poses = tracker.update(detections)
    """

    def __init__(
        self,
        distance_threshold: float = MAX_DETECTION_DISTANCE,
        max_age: int = MAX_TRACKER_AGE,
        min_hits: int = MIN_HIT_HITS,
        ref_y_threshold: float = REFREE_FILTER_Y_THRESHOLD,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.ref_y_threshold = ref_y_threshold

        # Initialize Norfair tracker with custom distance function
        self._tracker = Tracker(
            distance_function=fence_detection_distance,
            distance_threshold=distance_threshold,
            hit_counter_max=min_hits,
            past_detections_length=max_age,
        )

        # Track which IDs are locked to which fencer (0=Left, 1=Right)
        self._id_locked: Dict[int, int] = {}  # track_id -> fencer_id
        self._is_initialized = False

        # EMA predictors for graceful failure
        self._ema_predictors: Dict[int, EMAPredictor] = {
            0: EMAPredictor(alpha=0.7),
            1: EMAPredictor(alpha=0.7),
        }

        # Current tracked fencers
        self._tracked_fencers: Dict[int, TrackedFencer] = {}

    def initialize(
        self,
        detections: List[Detection],
        frame_height: float,
    ) -> List[FencerPose]:
        """
        Initialize tracker with first frame detections.

        Must be called on first frame to establish ID locking.

        Args:
            detections: List of Norfair Detection objects
            frame_height: Height of the video frame in pixels

        Returns:
            List[FencerPose] with assigned fencer_ids (0=Left, 1=Right)
        """
        # Filter detections by referee rule (bottom 70%)
        # Keep detections in lower 70% (Y >= ref_y_threshold * frame_height)
        y_threshold_px = frame_height * self.ref_y_threshold
        filtered_dets = self._filter_referees(detections, y_threshold_px)

        # Select top 2 by bbox area
        top_dets = self._select_top_by_bbox_area(filtered_dets, max_select=2)

        # Sort left-to-right (by centroid x coordinate)
        top_dets = sorted(top_dets, key=self._get_centroid_x)

        # Update tracker with initial detections
        tracked_objects = self._tracker.update(detections=top_dets)

        # Assign fencer IDs based on position (left=0, right=1)
        self._id_locked = {}
        for i, obj in enumerate(tracked_objects):
            self._id_locked[obj.id] = i  # 0 = Left, 1 = Right

        self._is_initialized = True

        # Build poses with assigned IDs
        return self._build_poses_from_tracked_objects(tracked_objects)

    def update(self, detections: List[Detection]) -> List[FencerPose]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Norfair Detection objects

        Returns:
            List[FencerPose] with assigned fencer_ids
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        # Update Norfair tracker
        tracked_objects = self._tracker.update(detections=detections)

        # Build poses with assigned IDs
        poses = self._build_poses_from_tracked_objects(tracked_objects)

        return poses

    def _filter_referees(
        self,
        detections: List[Detection],
        y_threshold: float,
    ) -> List[Detection]:
        """
        Filter out detections in upper 30% (likely referees).

        According to ARCHITECTURE.md Section 6:
        "Ignore standing figures in upper 30%"

        Args:
            detections: List of Norfair Detection objects
            y_threshold: Y coordinate threshold (below this = fencers)

        Returns:
            Filtered list of detections
        """
        filtered = []
        for det in detections:
            centroid_y = self._get_centroid_y(det)
            if centroid_y >= y_threshold:
                filtered.append(det)
        return filtered

    def _select_top_by_bbox_area(
        self,
        detections: List[Detection],
        max_select: int = 2,
    ) -> List[Detection]:
        """
        Select top N detections by bounding box area.

        Args:
            detections: List of Norfair Detection objects
            max_select: Maximum number to select

        Returns:
            List of top detections sorted by area descending
        """
        if not detections:
            return []

        # Compute areas
        areas = []
        for det in detections:
            points = det.points
            if points.shape[0] >= 2:
                # Use bbox area
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                area = (x_max - x_min) * (y_max - y_min)
            else:
                area = 0
            areas.append(area)

        # Sort by area descending
        sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        return [detections[i] for i in sorted_indices[:max_select]]

    def _get_centroid_x(self, detection: Detection) -> float:
        """Get centroid X coordinate of detection."""
        if detection.points.shape[0] >= 2:
            return float(detection.points[:, 0].mean())
        return 0.0

    def _get_centroid_y(self, detection: Detection) -> float:
        """Get centroid Y coordinate of detection."""
        if detection.points.shape[0] >= 2:
            return float(detection.points[:, 1].mean())
        return 0.0

    def _build_poses_from_tracked_objects(
        self,
        tracked_objects: List["TrackedObject"],
    ) -> List[FencerPose]:
        """
        Build FencerPose objects from Norfair tracked objects.

        Args:
            tracked_objects: List of Norfair TrackedObject

        Returns:
            List[FencerPose] with assigned fencer_ids
        """
        poses = []
        for obj in tracked_objects:
            # Get fencer_id from locked ID mapping
            fencer_id = self._id_locked.get(obj.id, -1)
            if fencer_id == -1:
                continue  # Skip if not in our mapping

            # Get keypoints from last detection
            if obj.last_detection and obj.last_detection.data:
                keypoints_data = obj.last_detection.data.get('keypoints', [])
            else:
                keypoints_data = []

            # Convert to Keypoint objects
            keypoints = []
            for kp in keypoints_data:
                if isinstance(kp, (list, np.ndarray)) and len(kp) >= 3:
                    keypoints.append(Keypoint(
                        x=float(kp[0]),
                        y=float(kp[1]),
                        conf=float(kp[2])
                    ))

            if len(keypoints) < 12:
                continue  # Skip invalid

            # Compute bbox from keypoints
            xs = [kp.x for kp in keypoints]
            ys = [kp.y for kp in keypoints]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            # Create FencerPose
            pose = FencerPose(
                fencer_id=fencer_id,
                bbox=(x1, y1, x2, y2),
                keypoints=keypoints,
                is_canonical_flipped=False,
            )
            poses.append(pose)

        return poses

    def get_fencer_pose(self, fencer_id: int) -> Optional[FencerPose]:
        """
        Get the current pose for a specific fencer.

        Args:
            fencer_id: 0 (Left) or 1 (Right)

        Returns:
            FencerPose or None if not tracked
        """
        for pose in self._tracked_fencers.values():
            if pose.fencer_id == fencer_id:
                return pose.pose
        return None

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._tracker = Tracker(
            distance_function=fence_detection_distance,
            distance_threshold=self.distance_threshold,
            hit_counter_max=self.min_hits,
            past_detections_length=self.max_age,
        )
        self._id_locked = {}
        self._is_initialized = False
        self._tracked_fencers = {}
        for predictor in self._ema_predictors.values():
            predictor.reset()

    def __repr__(self) -> str:
        return (
            f"FencerTracker("
            f"distance_threshold={self.distance_threshold}, "
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits}, "
            f"ref_y_threshold={self.ref_y_threshold})"
        )
