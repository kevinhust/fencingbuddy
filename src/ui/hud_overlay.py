"""
FencerAI HUD Overlay Module
===========================
Version: 2.0 | Last Updated: 2026-04-02

Heads-Up Display (HUD) overlay for live video.
Shows FPS, memory usage, processing status, and other metrics.

Features:
- Real-time FPS counter
- Memory usage display
- Processing mode indicator
- Recording status
- Frame buffer status
"""

from __future__ import annotations

import time
import psutil
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np


@dataclass
class HUDConfig:
    """Configuration for HUD display."""
    show_fps: bool = True
    show_memory: bool = True
    show_mode: bool = True
    show_buffer: bool = True
    show_recording: bool = True
    position: str = "top"  # "top" or "bottom"
    bg_color: tuple = (0, 0, 0)
    text_color: tuple = (255, 255, 255)
    font_scale: float = 0.6
    font_thickness: int = 2


class HUDOverlay:
    """
    Real-time HUD overlay for displaying processing metrics.

    Example:
        hud = HUDOverlay()
        frame = hud.draw(frame)
        cv2.imshow('Live', frame)
    """

    def __init__(self, config: Optional[HUDConfig] = None):
        """
        Initialize HUD overlay.

        Args:
            config: HUD configuration (uses defaults if None)
        """
        self.config = config or HUDConfig()
        self._fps_history: List[float] = []
        self._fps_window_size = 30
        self._last_frame_time = time.time()
        self._frame_times: List[float] = []
        self._process = psutil.Process()

    def update(self, frame: np.ndarray) -> None:
        """
        Update HUD with new frame timing.

        Call this once per frame before drawing.

        Args:
            frame: Current video frame
        """
        current_time = time.time()
        frame_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        if frame_time > 0:
            instant_fps = 1.0 / frame_time
            self._fps_history.append(instant_fps)
            if len(self._fps_history) > self._fps_window_size:
                self._fps_history.pop(0)

    def get_fps(self) -> float:
        """Get smoothed FPS value."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        mem_info = self._process.memory_info()
        return mem_info.rss / (1024 * 1024)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw HUD overlay on frame.

        Args:
            frame: Input frame (H, W, 3) in BGR

        Returns:
            Frame with HUD overlay
        """
        fps = self.get_fps()
        memory_mb = self.get_memory_mb()

        # Build HUD text
        lines = []
        if self.config.show_fps:
            fps_str = f"FPS: {fps:.1f}"
            if fps < 10:
                fps_str += " ⚠"
            lines.append(fps_str)

        if self.config.show_memory:
            mem_str = f"MEM: {memory_mb:.1f}MB"
            if memory_mb > 6000:  # > 6GB warning
                mem_str += " ⚠"
            lines.append(mem_str)

        if self.config.show_mode:
            lines.append("Mode: LIVE")

        if self.config.show_recording:
            lines.append("[REC]")

        # Draw HUD background and text
        return self._draw_text_panel(
            frame,
            lines,
            position=self.config.position,
        )

    def _draw_text_panel(
        self,
        frame: np.ndarray,
        lines: List[str],
        position: str = "top",
    ) -> np.ndarray:
        """
        Draw text panel on frame.

        Args:
            frame: Input frame
            lines: List of text lines
            position: "top" or "bottom"

        Returns:
            Frame with text panel
        """
        if not lines:
            return frame

        h, w = frame.shape[:2]

        # Calculate panel size
        line_height = int(25 * self.config.font_scale + 5)
        panel_height = len(lines) * line_height + 10
        panel_width = w

        # Create overlay
        overlay = frame.copy()

        # Draw background
        if position == "top":
            y_start = 0
        else:
            y_start = h - panel_height

        cv2.rectangle(
            overlay,
            (0, y_start),
            (panel_width, y_start + panel_height),
            self.config.bg_color,
            -1,
        )

        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(
            overlay[y_start:y_start + panel_height, 0:panel_width],
            alpha,
            frame[y_start:y_start + panel_height, 0:panel_width],
            1 - alpha,
            0,
            frame[y_start:y_start + panel_height, 0:panel_width],
        )

        # Draw text lines
        for i, line in enumerate(lines):
            y = y_start + 20 + i * line_height
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.text_color,
                self.config.font_thickness,
            )

        return frame

    def draw_buffer_status(self, frame: np.ndarray, buffer_size: int, max_size: int) -> np.ndarray:
        """
        Draw buffer status indicator.

        Args:
            frame: Input frame
            buffer_size: Current buffer size
            max_size: Maximum buffer size

        Returns:
            Frame with buffer indicator
        """
        h, w = frame.shape[:2]

        # Buffer bar position (bottom right)
        bar_width = 100
        bar_height = 10
        x = w - bar_width - 10
        y = h - bar_height - 10

        # Draw background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)

        # Draw fill
        fill_width = int(bar_width * (buffer_size / max_size))
        if fill_width > 0:
            # Color based on fill level
            if buffer_size / max_size > 0.8:
                color = (0, 0, 255)  # Red - almost full
            elif buffer_size / max_size > 0.5:
                color = (0, 255, 255)  # Yellow - half full
            else:
                color = (0, 255, 0)  # Green - healthy
            cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 1)

        return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    memory_mb: float,
    mode: str = "LIVE",
    recording: bool = False,
    buffer_size: int = 0,
    max_buffer: int = 30,
) -> np.ndarray:
    """
    Simple function to draw HUD on frame.

    Args:
        frame: Video frame
        fps: Current FPS
        memory_mb: Memory usage in MB
        mode: Processing mode string
        recording: Whether recording is active
        buffer_size: Current buffer size
        max_buffer: Maximum buffer size

    Returns:
        Frame with HUD drawn
    """
    hud = HUDOverlay()
    return hud.draw(frame)
