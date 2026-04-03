"""
FencerAI Alert Renderer Module
==============================
Version: 2.0 | Last Updated: 2026-04-02

Coaching alert display for live viewer.
Renders real-time fencing advice as text overlays.

Features:
- Priority-based alert display
- Alert history with timestamps
- Maximum alerts on screen
- Fade-out for old alerts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

import cv2
import numpy as np


@dataclass
class CoachingAlert:
    """A single coaching alert."""
    message: str
    priority: int = 1  # 1=highest, 5=lowest
    timestamp: float = field(default_factory=time.time)
    duration: float = 5.0  # seconds to display
    fencer_id: Optional[int] = None  # 0=son, 1=opponent, None=both
    category: str = "general"  # general, attack, defense, distance, recovery

    @property
    def is_expired(self) -> float:
        """Check if alert has expired."""
        return time.time() - self.timestamp > self.duration


class AlertRenderer:
    """
    Renders coaching alerts on video frames.

    Features:
    - Shows up to max_display alerts simultaneously
    - Color-coded by priority
    - Stores alert history
    - Auto-expires old alerts

    Example:
        renderer = AlertRenderer(max_display=3)
        renderer.add_alert("Shorten recovery!", priority=1, fencer_id=0)
        frame = renderer.draw(frame)
    """

    # Alert colors by priority (BGR for OpenCV)
    PRIORITY_COLORS = {
        1: (0, 0, 255),      # Red - critical
        2: (0, 165, 255),     # Orange - high
        3: (0, 255, 255),     # Yellow - medium
        4: (0, 255, 0),       # Green - low
        5: (255, 255, 255),   # White - info
    }

    def __init__(
        self,
        max_display: int = 5,
        history_size: int = 50,
    ):
        """
        Initialize alert renderer.

        Args:
            max_display: Maximum alerts to show on screen
            history_size: Maximum alerts to keep in history
        """
        self.max_display = max_display
        self._active_alerts: deque[CoachingAlert] = deque()
        self._history: deque[CoachingAlert] = deque(maxlen=history_size)

    def add_alert(
        self,
        message: str,
        priority: int = 3,
        fencer_id: Optional[int] = None,
        category: str = "general",
        duration: float = 5.0,
    ) -> None:
        """
        Add a new coaching alert.

        Args:
            message: Alert text (plain English)
            priority: 1=critical, 5=info
            fencer_id: 0=son, 1=opponent, None=both
            category: Alert category for filtering
            duration: How long to display (seconds)
        """
        alert = CoachingAlert(
            message=message,
            priority=priority,
            fencer_id=fencer_id,
            category=category,
            duration=duration,
        )
        self._active_alerts.append(alert)
        self._history.append(alert)

        # Sort active alerts by priority
        self._active_alerts = deque(
            sorted(self._active_alerts, key=lambda a: a.priority)
        )

    def update(self) -> None:
        """Remove expired alerts."""
        self._active_alerts = deque(
            [a for a in self._active_alerts if not a.is_expired]
        )

    def clear(self) -> None:
        """Clear all active alerts."""
        self._active_alerts.clear()

    def get_active_alerts(self) -> List[CoachingAlert]:
        """Get list of active (non-expired) alerts."""
        return list(self._active_alerts)

    def get_history(self) -> List[CoachingAlert]:
        """Get alert history."""
        return list(self._history)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw alerts on frame.

        Args:
            frame: Video frame (H, W, 3) in BGR

        Returns:
            Frame with alerts drawn
        """
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_width = min(400, w // 3)
        panel_x = w - panel_width - 10
        panel_y = 60  # Below HUD
        line_height = 30
        panel_height = self.max_display * line_height + 10

        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(
            overlay[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width],
            0.6,
            frame[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width],
            0.4,
            0,
            frame[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width],
        )

        # Draw border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            1,
        )

        # Draw title
        cv2.putText(
            frame,
            "COACHING",
            (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw alerts
        for i, alert in enumerate(self._active_alerts[:self.max_display]):
            y = panel_y + 20 + (i + 1) * line_height

            # Color based on priority
            color = self.PRIORITY_COLORS.get(alert.priority, (255, 255, 255))

            # Get fencer indicator
            fencer_tag = ""
            if alert.fencer_id == 0:
                fencer_tag = "[SON] "
            elif alert.fencer_id == 1:
                fencer_tag = "[OPP] "

            # Draw alert text
            text = fencer_tag + alert.message
            # Truncate if too long
            if len(text) > 25:
                text = text[:22] + "..."

            cv2.putText(
                frame,
                text,
                (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        return frame

    def draw_alert_history(self, frame: np.ndarray, max_rows: int = 5) -> np.ndarray:
        """
        Draw recent alert history at bottom of frame.

        Args:
            frame: Video frame
            max_rows: Maximum history rows to show

        Returns:
            Frame with history drawn
        """
        h, w = frame.shape[:2]

        # Start from bottom
        y_start = h - 20
        line_height = 18

        # Get recent history
        recent = self._history[-max_rows:] if self._history else []

        for i, alert in enumerate(reversed(recent)):
            y = y_start - i * line_height
            if y < 0:
                break

            # Time since alert
            age = time.time() - alert.timestamp
            age_str = f"{age:.0f}s ago"

            color = self.PRIORITY_COLORS.get(alert.priority, (200, 200, 200))
            text = f"[{age_str}] {alert.message[:30]}"

            # Semi-transparent background
            cv2.rectangle(frame, (5, y - 12), (w // 2, y + 5), (0, 0, 0), -1)
            cv2.addWeighted(
                frame[y - 12:y + 5, 5:w // 2],
                0.5,
                frame[y - 12:y + 5, 5:w // 2],
                0.5,
                0,
                frame[y - 12:y + 5, 5:w // 2],
            )

            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        return frame


# =============================================================================
# Default Alert Messages (Fencing-Specific)
# =============================================================================

DEFAULT_ALERTS = {
    # Distance alerts
    "distance_open": "Attack now - distance open",
    "distance_closed": "Too close - parry ready",
    "distance_far": "Close distance first",

    # Recovery alerts
    "slow_recovery": "Shorten recovery - riposte risk",
    "fast_recovery": "Good recovery speed",

    # Attack alerts
    "attack_prep": "Attack in prep - now!",
    "arm_extension": "Extend arm fully",
    "predictable": "Too predictable - vary attacks",

    # Defense alerts
    "guard_low": "Watch your guard position",
    "dropped_guard": "Dropped guard on retreat",

    # Blade work
    "weak_blade": "Weak blade control",
    "fleche_prep": "Watch for fleche attack",

    # Opponent
    "opp_overextended": "Opponent overextended - counter",
    "opp_weak_recovery": "Opponent slow recovery - attack",
}


def draw_alerts(frame: np.ndarray, alerts: List[CoachingAlert]) -> np.ndarray:
    """
    Simple function to draw alerts on frame.

    Args:
        frame: Video frame
        alerts: List of CoachingAlert objects

    Returns:
        Frame with alerts drawn
    """
    renderer = AlertRenderer()
    for alert in alerts:
        renderer.add_alert(
            message=alert.message,
            priority=alert.priority,
            fencer_id=alert.fencer_id,
        )
    return renderer.draw(frame)
