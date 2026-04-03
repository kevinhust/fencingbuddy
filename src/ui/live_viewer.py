"""
FencerAI Live Viewer Module
============================
Version: 2.0 | Last Updated: 2026-04-02

OpenCV-based live viewer for real-time fencing analysis.
Displays video with skeleton overlay, coaching alerts, and HUD.

Features:
- Red/Blue skeleton overlay (red=left/son, blue=right/opponent)
- Real-time coaching alerts
- FPS and memory HUD
- Score tracking (manual)
- Session recording indicator
- Alert history
- "3 Things to Fix" summary panel

Example:
    viewer = LiveViewer()
    viewer.show(frame, poses=poses, alerts=alerts)
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from src.utils.schemas import FencerPose, FrameData
from src.utils.visualization import (
    draw_frame_overlay,
    FENCER_COLORS,
)
from src.ui.hud_overlay import HUDOverlay, HUDConfig
from src.ui.alert_renderer import AlertRenderer, CoachingAlert


@dataclass
class LiveViewerConfig:
    """Configuration for live viewer."""
    window_name: str = "FencerAI Live"
    width: int = 1280
    height: int = 720
    fullscreen: bool = False
    show_hud: bool = True
    show_alerts: bool = True
    show_skeletons: bool = True
    show_score: bool = True
    alert_duration: float = 5.0


class LiveViewer:
    """
    Real-time live viewer for fencing analysis.

    Displays:
    - Video frame with skeleton overlay
    - Real-time coaching alerts
    - FPS and memory HUD
    - Score tracker
    - Recording indicator

    Example:
        viewer = LiveViewer()
        while True:
            frame, poses, alerts = process_frame()
            viewer.show(frame, poses=poses, alerts=alerts)
            if viewer.is_closed():
                break
    """

    def __init__(self, config: Optional[LiveViewerConfig] = None):
        """
        Initialize live viewer.

        Args:
            config: Viewer configuration
        """
        self.config = config or LiveViewerConfig()

        # Create window
        cv2.namedWindow(
            self.config.window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        if self.config.fullscreen:
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.resizeWindow(self.config.window_name, self.config.width, self.config.height)

        # Initialize components
        self._hud = HUDOverlay()
        self._alert_renderer = AlertRenderer(max_display=5)

        # Score tracking
        self._son_score = 0
        self._opp_score = 0

        # Recording state
        self._is_recording = False
        self._recording_start_time = 0.0

        # Frame info
        self._current_frame: Optional[np.ndarray] = None
        self._current_poses: List[FencerPose] = []
        self._is_closed = False

        # "3 Things to Fix" tracking - alert frequency
        self._alert_counter: Counter = Counter()
        self._things_to_fix_max: int = 3
        self._things_to_fix_min_count: int = 2  # Minimum occurrences to show

    def show(
        self,
        frame: np.ndarray,
        poses: Optional[List[FencerPose]] = None,
        alerts: Optional[List[CoachingAlert]] = None,
        frame_data: Optional[FrameData] = None,
    ) -> None:
        """
        Show a frame with overlays.

        Args:
            frame: Video frame (H, W, 3) in BGR
            poses: List of FencerPose objects
            alerts: List of CoachingAlert objects
            frame_data: Optional FrameData (extracts poses and audio events)
        """
        # Extract poses from frame_data if provided
        if frame_data is not None:
            poses = frame_data.poses

        # Start timing
        self._hud.update(frame)
        self._current_frame = frame.copy()

        # Get display frame
        display = frame.copy()

        # Draw skeleton overlay
        if self.config.show_skeletons and poses:
            display = self._draw_skeletons(display, poses)

        # Draw HUD
        if self.config.show_hud:
            display = self._hud.draw(display)

        # Draw alerts
        if self.config.show_alerts and alerts:
            for alert in alerts:
                self._alert_renderer.add_alert(
                    message=alert.message,
                    priority=alert.priority,
                    fencer_id=alert.fencer_id,
                )
                # Track alert for "3 Things to Fix" (only son's alerts)
                if alert.fencer_id == 0 or alert.fencer_id is None:
                    self._alert_counter[alert.message] += 1
        self._alert_renderer.update()
        if self.config.show_alerts:
            display = self._alert_renderer.draw(display)

        # Draw "3 Things to Fix" summary
        display = self._draw_things_to_fix(display)

        # Draw score
        if self.config.show_score:
            display = self._draw_score(display)

        # Draw recording indicator
        if self._is_recording:
            display = self._draw_recording(display)

        # Show frame
        cv2.imshow(self.config.window_name, display)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            self._is_closed = True

        # Handle keypresses
        self._handle_keypress(key)

    def _draw_skeletons(
        self,
        frame: np.ndarray,
        poses: List[FencerPose],
    ) -> np.ndarray:
        """
        Draw skeletons on frame.

        Args:
            frame: Video frame
            poses: List of FencerPose

        Returns:
            Frame with skeletons drawn
        """
        for pose in poses:
            color = FENCER_COLORS.get(pose.fencer_id, (200, 200, 200))
            frame = draw_frame_overlay(frame, [pose], min_conf=0.3)
        return frame

    def _draw_score(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw score overlay.

        Args:
            frame: Video frame

        Returns:
            Frame with score drawn
        """
        h, w = frame.shape[:2]

        # Score panel position (top right)
        panel_w = 150
        panel_h = 60
        x = w - panel_w - 10
        y = 60

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay[y:y + panel_h, x:x + panel_w],
            0.6,
            frame[y:y + panel_h, x:x + panel_w],
            0.4,
            0,
            frame[y:y + panel_h, x:x + panel_w],
        )
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (100, 100, 100), 1)

        # Score text
        son_text = f"SON:  {self._son_score}"
        opp_text = f"OPP:  {self._opp_score}"

        # Son score (red)
        cv2.putText(frame, son_text, (x + 10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        # Opponent score (blue)
        cv2.putText(frame, opp_text, (x + 10, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        return frame

    def _draw_recording(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw recording indicator.

        Args:
            frame: Video frame

        Returns:
            Frame with recording indicator
        """
        h, w = frame.shape[:2]

        # Blinking red dot + "REC"
        elapsed = time.time() - self._recording_start_time
        blink_on = int(elapsed * 2) % 2 == 0  # Blink every 0.5s

        if blink_on:
            # Red circle
            cv2.circle(frame, (w - 30, 30), 8, (0, 0, 255), -1)
            # "REC" text
            cv2.putText(frame, "REC", (w - 70, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def _draw_things_to_fix(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw "3 Things to Fix" panel based on frequent alerts.

        Args:
            frame: Video frame

        Returns:
            Frame with things-to-fix panel
        """
        h, w = frame.shape[:2]

        # Get top alerts by frequency (only son's alerts count)
        frequent_alerts = [
            (msg, count) for msg, count in self._alert_counter.most_common(10)
            if count >= self._things_to_fix_min_count
        ][:self._things_to_fix_max]

        if not frequent_alerts:
            return frame

        # Panel dimensions (bottom left area)
        panel_w = 300
        panel_h = 80 + len(frequent_alerts) * 25
        x = 10
        y = h - panel_h - 10

        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay[y:y + panel_h, x:x + panel_w],
            0.7,
            frame[y:y + panel_h, x:x + panel_w],
            0.3,
            0,
            frame[y:y + panel_h, x:x + panel_w],
        )

        # Draw border
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (100, 100, 100), 1)

        # Draw title
        cv2.putText(
            frame,
            "3 THINGS TO FIX",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw frequent alerts
        for i, (msg, count) in enumerate(frequent_alerts):
            y_pos = y + 40 + i * 22
            # Truncate message if too long
            if len(msg) > 30:
                display_msg = msg[:27] + "..."
            else:
                display_msg = msg

            text = f"({count}x) {display_msg}"
            cv2.putText(
                frame,
                text,
                (x + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 200, 100),  # Orange-ish for visibility
                1,
            )

        return frame

    def _handle_keypress(self, key: int) -> None:
        """Handle keypresses."""
        if key == ord('r'):  # Toggle recording
            self._is_recording = not self._is_recording
            if self._is_recording:
                self._recording_start_time = time.time()
        elif key == ord('s'):  # Son scores
            self._son_score += 1
        elif key == ord('o'):  # Opponent scores
            self._opp_score += 1
        elif key == ord('z'):  # Undo last score
            if self._son_score > 0:
                self._son_score -= 1
        elif key == ord('x'):  # Undo opponent score
            if self._opp_score > 0:
                self._opp_score -= 1
        elif key == ord('c'):  # Clear scores
            self._son_score = 0
            self._opp_score = 0

    def add_alert(
        self,
        message: str,
        priority: int = 3,
        fencer_id: Optional[int] = None,
    ) -> None:
        """
        Add a coaching alert.

        Args:
            message: Alert text
            priority: 1=critical, 5=info
            fencer_id: 0=son, 1=opponent
        """
        self._alert_renderer.add_alert(
            message=message,
            priority=priority,
            fencer_id=fencer_id,
        )

    def set_score(self, son: int, opponent: int) -> None:
        """
        Set the score.

        Args:
            son: Son's score
            opponent: Opponent's score
        """
        self._son_score = son
        self._opp_score = opponent

    def increment_son_score(self) -> None:
        """Increment son's score."""
        self._son_score += 1

    def increment_opp_score(self) -> None:
        """Increment opponent's score."""
        self._opp_score += 1

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._is_recording

    @property
    def recording_duration(self) -> float:
        """Get recording duration in seconds."""
        if self._is_recording:
            return time.time() - self._recording_start_time
        return 0.0

    def is_closed(self) -> bool:
        """Check if viewer window was closed."""
        return self._is_closed

    def reset_session(self) -> None:
        """Reset session data for a new bout."""
        self._son_score = 0
        self._opp_score = 0
        self._alert_counter.clear()
        self._alert_renderer.clear()

    def close(self) -> None:
        """Close the viewer window."""
        cv2.destroyWindow(self.config.window_name)
        self._is_closed = True

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            cv2.destroyWindow(self.config.window_name)
        except Exception:
            pass
