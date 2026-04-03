"""
FencerAI Session Recorder
=======================
Version: 2.0 | Last Updated: 2026-04-02

Records live sessions to file for later playback and analysis.
Saves video + metadata (scores, alerts, timestamps).

Example:
    recorder = SessionRecorder(output_dir="outputs/sessions")
    recorder.start(session_name="bout_20260402_1430")
    recorder.record_frame(frame, features, alerts)
    recorder.stop()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class SessionMetadata:
    """Metadata for a recorded session."""
    session_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    total_frames: int = 0
    son_final_score: int = 0
    opp_final_score: int = 0
    alert_count: int = 0
    recording_mode: str = "live"


@dataclass
class AlertRecord:
    """A recorded coaching alert."""
    timestamp: float
    message: str
    priority: int
    fencer_id: Optional[int] = None
    category: str = "general"


class SessionRecorder:
    """
    Records live fencing sessions to file.

    Records:
    - Video frames (if enabled)
    - Feature matrices (.npy)
    - Score events
    - Coaching alerts
    - Session metadata (.json)

    Example:
        >>> recorder = SessionRecorder(output_dir="outputs/sessions")
        >>> recorder.start("bout_20260402_1430")
        >>> recorder.record_frame(frame, features, score=(3, 2), alerts=alerts)
        >>> recorder.stop()
    """

    def __init__(
        self,
        output_dir: str = "outputs/sessions",
        record_video: bool = False,
        record_features: bool = True,
        fps: int = 30,
    ):
        """
        Initialize session recorder.

        Args:
            output_dir: Directory to save sessions
            record_video: Whether to record video frames (large file size)
            record_features: Whether to record feature matrices
            fps: Frames per second for video recording
        """
        self.output_dir = Path(output_dir)
        self.record_video = record_video
        self.record_features = record_features
        self.fps = fps

        self._is_recording = False
        self._session_name: Optional[str] = None
        self._session_start_time: float = 0.0
        self._frame_count = 0

        # Feature storage
        self._feature_history: List[np.ndarray] = []
        self._timestamp_history: List[float] = []
        self._score_history: List[Dict[str, int]] = []
        self._alert_history: List[AlertRecord] = []

        # Current score
        self._son_score = 0
        self._opp_score = 0

        # Video writer
        self._video_writer: Optional[cv2.VideoWriter] = None

    def start(self, session_name: str) -> Path:
        """
        Start recording a new session.

        Args:
            session_name: Name for the session (used in filename)

        Returns:
            Path to the session directory
        """
        self._session_name = session_name
        self._session_start_time = time.time()
        self._frame_count = 0

        # Clear history
        self._feature_history.clear()
        self._timestamp_history.clear()
        self._score_history.clear()
        self._alert_history.clear()
        self._son_score = 0
        self._opp_score = 0

        # Create session directory
        session_dir = self.output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        self._is_recording = True
        self._session_dir = session_dir

        return session_dir

    def record_frame(
        self,
        frame: np.ndarray,
        features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        alerts: Optional[List[AlertRecord]] = None,
    ) -> None:
        """
        Record a single frame.

        Args:
            frame: Video frame (H, W, 3) in BGR
            features: Optional (2, 101) feature matrix
            timestamp: Optional frame timestamp
            alerts: Optional list of alerts for this frame
        """
        if not self._is_recording:
            return

        self._frame_count += 1
        current_time = timestamp if timestamp is not None else time.time() - self._session_start_time

        # Record features
        if features is not None and self.record_features:
            self._feature_history.append(features.copy())
            self._timestamp_history.append(current_time)

        # Record alerts
        if alerts:
            for alert in alerts:
                self._alert_history.append(AlertRecord(
                    timestamp=current_time,
                    message=alert.message,
                    priority=alert.priority,
                    fencer_id=alert.fencer_id,
                    category=getattr(alert, 'category', 'general'),
                ))

        # Write video frame
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def update_score(self, son: int, opp: int) -> None:
        """
        Update current score (records a score change event).

        Args:
            son: Son's new score
            opp: Opponent's new score
        """
        if not self._is_recording:
            return

        # Record score change
        current_time = time.time() - self._session_start_time
        self._score_history.append({
            "timestamp": current_time,
            "son": son,
            "opp": opp,
        })

        self._son_score = son
        self._opp_score = opp

    def stop(self) -> Optional[Path]:
        """
        Stop recording and save session.

        Returns:
            Path to the session directory, or None if not recording
        """
        if not self._is_recording:
            return None

        self._is_recording = False
        duration = time.time() - self._session_start_time

        # Save features
        if self._feature_history and self.record_features:
            features_array = np.stack(self._feature_history)
            npy_path = self._session_dir / f"{self._session_name}_features.npy"
            np.save(npy_path, features_array)

            # Save timestamps
            ts_path = self._session_dir / f"{self._session_name}_timestamps.json"
            with open(ts_path, 'w') as f:
                json.dump(self._timestamp_history, f)

        # Save score history
        if self._score_history:
            score_path = self._session_dir / f"{self._session_name}_scores.json"
            with open(score_path, 'w') as f:
                json.dump(self._score_history, f, indent=2)

        # Save alert history
        if self._alert_history:
            alert_data = [
                {
                    "timestamp": a.timestamp,
                    "message": a.message,
                    "priority": a.priority,
                    "fencer_id": a.fencer_id,
                    "category": a.category,
                }
                for a in self._alert_history
            ]
            alert_path = self._session_dir / f"{self._session_name}_alerts.json"
            with open(alert_path, 'w') as f:
                json.dump(alert_data, f, indent=2)

        # Save metadata
        metadata = SessionMetadata(
            session_name=self._session_name or "unknown",
            start_time=self._session_start_time,
            end_time=time.time(),
            duration_seconds=duration,
            total_frames=self._frame_count,
            son_final_score=self._son_score,
            opp_final_score=self._opp_score,
            alert_count=len(self._alert_history),
            recording_mode="live",
        )
        meta_path = self._session_dir / f"{self._session_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "session_name": metadata.session_name,
                "start_time": metadata.start_time,
                "end_time": metadata.end_time,
                "duration_seconds": metadata.duration_seconds,
                "total_frames": metadata.total_frames,
                "son_final_score": metadata.son_final_score,
                "opp_final_score": metadata.opp_final_score,
                "alert_count": metadata.alert_count,
                "recording_mode": metadata.recording_mode,
            }, f, indent=2)

        # Release video writer
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        return self._session_dir

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def get_session_dir(self) -> Optional[Path]:
        """Get current session directory."""
        return getattr(self, '_session_dir', None)

    def set_video_writer(self, width: int, height: int) -> None:
        """
        Initialize video writer for recording.

        Args:
            width: Frame width
            height: Frame height
        """
        if not self._is_recording or self._session_dir is None:
            return

        if not self.record_video:
            return

        video_path = self._session_dir / f"{self._session_name}_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.fps, (width, height)
        )
