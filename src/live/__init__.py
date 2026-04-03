"""
FencerAI Live Input Module
=========================
Version: 2.0 | Last Updated: 2026-04-02

Live video input handling for real-time fencing analysis.
Supports camera capture (Continuity Camera) and video file input.
"""

from __future__ import annotations

from src.live.live_capture import LiveCapture, CameraConfig
from src.live.frame_buffer import FrameBuffer

__all__ = ["LiveCapture", "CameraConfig", "FrameBuffer"]
