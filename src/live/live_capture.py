"""
FencerAI Live Capture Module
=============================
Version: 2.0 | Last Updated: 2026-04-02

OpenCV-based video capture for live streaming and file input.
Supports:
- Camera index (0, 1 for Continuity Camera)
- RTSP/HTTP URL streams
- Video file input
- Resolution and FPS configuration

Usage:
    capture = LiveCapture(camera_index=0, width=1920, height=1080, fps=30)
    for frame in capture:
        # process frame
"""

from __future__ import annotations

import cv2
import time
from dataclasses import dataclass
from typing import Optional, Iterator, Union
from enum import Enum

import numpy as np


class CaptureSource(Enum):
    """Video capture source type."""
    CAMERA = "camera"
    VIDEO_FILE = "video_file"
    STREAM_URL = "stream_url"


@dataclass
class CameraConfig:
    """Configuration for video capture."""
    source: Union[int, str]  # Camera index or file/URL path
    width: int = 1920
    height: int = 1080
    fps: int = 30
    source_type: CaptureSource = CaptureSource.CAMERA

    def __post_init__(self):
        """Determine source type from source."""
        if isinstance(self.source, int):
            self.source_type = CaptureSource.CAMERA
        elif self.source.endswith(('.mp4', '.avi', '.mov')):
            self.source_type = CaptureSource.VIDEO_FILE
        else:
            self.source_type = CaptureSource.STREAM_URL


class LiveCapture:
    """
    Video capture wrapper for live streaming and file input.

    Supports:
    - Camera index (0, 1 for Continuity Camera)
    - Video file paths
    - RTSP/HTTP stream URLs

    Example:
        # From camera
        capture = LiveCapture(camera_index=0)
        for frame in capture:
            cv2.imshow('Live', frame)

        # From video file
        capture = LiveCapture(video_path='bout.mp4')
        for frame in capture:
            # process
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        video_path: Optional[str] = None,
        stream_url: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ):
        """
        Initialize video capture.

        Args:
            camera_index: Camera index (0, 1, etc.). Use None if not using camera.
            video_path: Path to video file. Use None if not reading from file.
            stream_url: RTSP/HTTP stream URL. Use None if not streaming.
            width: Desired frame width (default 1920)
            height: Desired frame height (default 1080)
            fps: Desired frames per second (default 30)

        Note: Provide exactly one of camera_index, video_path, or stream_url.
        """
        if camera_index is not None:
            self.source = camera_index
            self.source_type = CaptureSource.CAMERA
        elif video_path is not None:
            self.source = video_path
            self.source_type = CaptureSource.VIDEO_FILE
        elif stream_url is not None:
            self.source = stream_url
            self.source_type = CaptureSource.STREAM_URL
        else:
            raise ValueError("Must provide camera_index, video_path, or stream_url")

        self.width = width
        self.height = height
        self.fps = fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._is_opened = False

    def _open(self) -> bool:
        """Open the video capture."""
        if self._capture is not None:
            return True

        self._capture = cv2.VideoCapture(self.source)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            return False

        # Configure camera settings if available
        if self.source_type == CaptureSource.CAMERA:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

            # Check actual settings
            actual_width = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            # Use actual values if requested not supported
            if actual_width > 0 and actual_width != self.width:
                self.width = int(actual_width)
            if actual_height > 0 and actual_height != self.height:
                self.height = int(actual_height)
            if actual_fps > 0 and actual_fps != self.fps:
                self.fps = actual_fps

        self._is_opened = True
        self._start_time = time.time()
        return True

    def __enter__(self) -> "LiveCapture":
        """Context manager entry."""
        if not self._open():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        if self._capture is None:
            if not self._open():
                return
        return self

    def __next__(self) -> np.ndarray:
        """Get next frame."""
        if self._capture is None or not self._capture.isOpened():
            raise StopIteration

        ret, frame = self._capture.read()
        if not ret:
            raise StopIteration

        self._frame_count += 1
        return frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame manually.

        Returns:
            Tuple of (success, frame)
        """
        if self._capture is None or not self._capture.isOpened():
            return False, None
        ret, frame = self._capture.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def close(self) -> None:
        """Close the video capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._is_opened = False

    @property
    def is_opened(self) -> bool:
        """Check if capture is open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def frame_count(self) -> int:
        """Get number of frames read."""
        return self._frame_count

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since capture started."""
        return time.time() - self._start_time

    @property
    def actual_fps(self) -> float:
        """Get actual FPS achieved."""
        elapsed = self.elapsed_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0

    @property
    def total_frames(self) -> int:
        """Get total frames in video (for file/stream sources)."""
        if self._capture is not None and self.source_type == CaptureSource.VIDEO_FILE:
            return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1  # Unknown for live sources

    @property
    def resolution(self) -> tuple[int, int]:
        """Get actual frame resolution."""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"LiveCapture(source={self.source}, "
            f"type={self.source_type.value}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps})"
        )


def list_cameras(max_cameras: int = 5) -> list[int]:
    """
    List available camera indices.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_camera_info(camera_index: int) -> dict:
    """
    Get information about a camera.

    Args:
        camera_index: Camera index to query

    Returns:
        Dictionary with camera information
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"index": camera_index, "available": False}

    info = {
        "index": camera_index,
        "available": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.get(cv2.CAP_PROP_BACKEND),
    }
    cap.release()
    return info
