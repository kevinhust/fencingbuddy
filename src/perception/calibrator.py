"""
FencerAI Homography Calibrator
==============================
Version: 1.0 | Last Updated: 2026-03-27

Homography-based calibration for pixel-to-meter transformation.
Uses RANSAC for robust calibration with automatic outlier rejection.

Architecture (per ARCHITECTURE.md Section 4):
    - 3x3 Homography Matrix transforms 2D pixels into top-down metric space
    - Requires user-guided markers or piste-end detection for calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.utils.constants import (
    DEFAULT_PISTE_LENGTH,
    DEFAULT_PISTE_WIDTH,
    MAX_REPROJECTION_ERROR,
)


# =============================================================================
# Calibration Point
# =============================================================================

@dataclass
class CalibrationPoint:
    """Single calibration point with pixel and meter coordinates."""
    pixel_x: float
    pixel_y: float
    meter_x: float
    meter_y: float


from dataclasses import dataclass


# =============================================================================
# HomographyCalibrator
# =============================================================================

class HomographyCalibrator:
    """
    Homography-based calibrator for pixel-to-meter transformation.

    The calibrator computes a 3x3 homography matrix that maps pixel coordinates
    to physical meter coordinates in a top-down view of the fencing piste.

    Args:
        piste_length: Length of the piste in meters (default 14.0m)
        piste_width: Width of the piste in meters (default 1.8m)
        max_reprojection_error: Max reprojection error for RANSAC in meters

    Example:
        >>> calibrator = HomographyCalibrator()
        >>> # Add known point correspondences
        >>> calibrator.add_point(pixel_x=100, pixel_y=200, meter_x=0.0, meter_y=0.0)
        >>> calibrator.add_point(pixel_x=500, pixel_y=200, meter_x=14.0, meter_y=0.0)
        >>> calibrator.calibrate()
        >>> # Transform a pixel point to meters
        >>> meter_x, meter_y = calibrator.pixel_to_meter(300, 400)
    """

    def __init__(
        self,
        piste_length: float = DEFAULT_PISTE_LENGTH,
        piste_width: float = DEFAULT_PISTE_WIDTH,
        max_reprojection_error: float = MAX_REPROJECTION_ERROR,
    ) -> None:
        self.piste_length = piste_length
        self.piste_width = piste_width
        self.max_reprojection_error = max_reprojection_error

        self._points_pixel: List[np.ndarray] = []
        self._points_meter: List[np.ndarray] = []
        self._homography: Optional[np.ndarray] = None
        self._is_calibrated = False

    def add_point(
        self,
        pixel_x: float,
        pixel_y: float,
        meter_x: float,
        meter_y: float,
    ) -> None:
        """
        Add a calibration point.

        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate
            meter_x: Physical X coordinate in meters
            meter_y: Physical Y coordinate in meters
        """
        self._points_pixel.append(np.array([pixel_x, pixel_y], dtype=np.float32))
        self._points_meter.append(np.array([meter_x, meter_y], dtype=np.float32))
        self._is_calibrated = False  # Need to recalibrate

    def add_points_from_piste_corners(
        self,
        top_left: Tuple[float, float],
        top_right: Tuple[float, float],
        bottom_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
    ) -> None:
        """
        Add calibration points from piste corner markers.

        Args:
            top_left: Pixel (x, y) of top-left corner
            top_right: Pixel (x, y) of top-right corner
            bottom_left: Pixel (x, y) of bottom-left corner
            bottom_right: Pixel (x, y) of bottom-right corner
        """
        # Top-left corner = origin (0, 0)
        self.add_point(top_left[0], top_left[1], 0.0, 0.0)
        # Top-right corner = (piste_length, 0)
        self.add_point(top_right[0], top_right[1], self.piste_length, 0.0)
        # Bottom-left corner = (0, piste_width)
        self.add_point(bottom_left[0], bottom_left[1], 0.0, self.piste_width)
        # Bottom-right corner = (piste_length, piste_width)
        self.add_point(bottom_right[0], bottom_right[1], self.piste_length, self.piste_width)

    def calibrate(self) -> bool:
        """
        Compute homography matrix from calibration points.

        Uses RANSAC for robust estimation with automatic outlier rejection.

        Returns:
            True if calibration succeeded, False otherwise
        """
        if len(self._points_pixel) < 4:
            return False

        # Stack points
        src_points = np.array(self._points_pixel, dtype=np.float32)
        dst_points = np.array(self._points_meter, dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.max_reprojection_error,
        )

        if H is None:
            return False

        # Verify all inliers
        if mask is not None:
            inlier_count = np.sum(mask)
            if inlier_count < 4:
                return False

        self._homography = H
        self._is_calibrated = True
        return True

    def pixel_to_meter(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Transform pixel coordinates to meter coordinates.

        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate

        Returns:
            Tuple of (meter_x, meter_y)

        Raises:
            RuntimeError: If calibrator is not calibrated
        """
        if not self._is_calibrated or self._homography is None:
            raise RuntimeError("Calibrator not calibrated. Call calibrate() first.")

        # Convert to homogeneous coordinates
        pixel_hom = np.array([pixel_x, pixel_y, 1.0], dtype=np.float32)

        # Apply homography
        meter_hom = self._homography @ pixel_hom

        # Convert from homogeneous coordinates
        if meter_hom[2] == 0:
            return (float(meter_hom[0]), float(meter_hom[1]))

        meter_x = meter_hom[0] / meter_hom[2]
        meter_y = meter_hom[1] / meter_hom[2]

        return (float(meter_x), float(meter_y))

    def meter_to_pixel(self, meter_x: float, meter_y: float) -> Tuple[float, float]:
        """
        Transform meter coordinates to pixel coordinates.

        Args:
            meter_x: Physical X coordinate in meters
            meter_y: Physical Y coordinate in meters

        Returns:
            Tuple of (pixel_x, pixel_y)

        Raises:
            RuntimeError: If calibrator is not calibrated
        """
        if not self._is_calibrated or self._homography is None:
            raise RuntimeError("Calibrator not calibrated. Call calibrate() first.")

        # Compute inverse homography
        H_inv = np.linalg.inv(self._homography)

        # Convert to homogeneous coordinates
        meter_hom = np.array([meter_x, meter_y, 1.0], dtype=np.float32)

        # Apply inverse homography
        pixel_hom = H_inv @ meter_hom

        # Convert from homogeneous coordinates
        if pixel_hom[2] == 0:
            return (float(pixel_hom[0]), float(pixel_hom[1]))

        pixel_x = pixel_hom[0] / pixel_hom[2]
        pixel_y = pixel_hom[1] / pixel_hom[2]

        return (float(pixel_x), float(pixel_y))

    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """
        Get the current homography matrix.

        Returns:
            3x3 homography matrix or None if not calibrated
        """
        return self._homography.copy() if self._homography is not None else None

    def compute_reprojection_error(
        self,
        pixel_x: float,
        pixel_y: float,
        meter_x: float,
        meter_y: float,
    ) -> float:
        """
        Compute reprojection error for a calibration point.

        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate
            meter_x: Expected physical X in meters
            meter_y: Expected physical Y in meters

        Returns:
            Reprojection error in meters
        """
        if not self._is_calibrated:
            return float('inf')

        # Transform pixel to meter
        est_meter_x, est_meter_y = self.pixel_to_meter(pixel_x, pixel_y)

        # Compute Euclidean distance
        error = np.sqrt(
            (est_meter_x - meter_x) ** 2 + (est_meter_y - meter_y) ** 2
        )
        return float(error)

    def reset(self) -> None:
        """Clear all calibration points and reset state."""
        self._points_pixel.clear()
        self._points_meter.clear()
        self._homography = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibrator has been successfully calibrated."""
        return self._is_calibrated

    @property
    def num_points(self) -> int:
        """Return number of calibration points added."""
        return len(self._points_pixel)

    def __repr__(self) -> str:
        return (
            f"HomographyCalibrator("
            f"piste_length={self.piste_length}, "
            f"piste_width={self.piste_width}, "
            f"is_calibrated={self._is_calibrated}, "
            f"num_points={self.num_points})"
        )
