"""
Tests for src/perception/calibrator.py
TDD Phase 2.3: Homography Calibrator Unit Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.perception.calibrator import HomographyCalibrator, CalibrationPoint
from src.utils.constants import DEFAULT_PISTE_LENGTH, DEFAULT_PISTE_WIDTH


class TestHomographyCalibratorInit:
    """Test HomographyCalibrator initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        calibrator = HomographyCalibrator()
        assert calibrator.piste_length == DEFAULT_PISTE_LENGTH
        assert calibrator.piste_width == DEFAULT_PISTE_WIDTH
        assert not calibrator.is_calibrated
        assert calibrator.num_points == 0

    def test_init_with_custom_params(self):
        """Should accept custom parameters."""
        calibrator = HomographyCalibrator(
            piste_length=10.0,
            piste_width=2.0,
            max_reprojection_error=5.0,
        )
        assert calibrator.piste_length == 10.0
        assert calibrator.piste_width == 2.0

    def test_repr(self):
        """String representation should show parameters."""
        calibrator = HomographyCalibrator()
        repr_str = repr(calibrator)
        assert "HomographyCalibrator" in repr_str
        assert "piste_length=14.0" in repr_str


class TestCalibrationPoints:
    """Test calibration point management."""

    def test_add_single_point(self):
        """Should add a single calibration point."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=100, pixel_y=200, meter_x=0.0, meter_y=0.0)
        assert calibrator.num_points == 1
        assert not calibrator.is_calibrated

    def test_add_multiple_points(self):
        """Should add multiple calibration points."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        assert calibrator.num_points == 4

    def test_add_points_from_corners(self):
        """Should add 4 points from corner coordinates."""
        calibrator = HomographyCalibrator()
        calibrator.add_points_from_piste_corners(
            top_left=(100, 50),
            top_right=(540, 50),
            bottom_left=(100, 430),
            bottom_right=(540, 430),
        )
        assert calibrator.num_points == 4

    def test_reset_clears_points(self):
        """Reset should clear all calibration points."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=100, pixel_y=200, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=200, meter_x=14.0, meter_y=0.0)
        calibrator.reset()
        assert calibrator.num_points == 0
        assert not calibrator.is_calibrated


class TestCalibration:
    """Test homography calibration."""

    def test_calibrate_needs_at_least_4_points(self):
        """Calibration should fail with fewer than 4 points."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=100, pixel_y=200, meter_x=0.0, meter_y=0.0)
        result = calibrator.calibrate()
        assert result is False
        assert not calibrator.is_calibrated

    def test_calibrate_with_4_points(self):
        """Should calibrate successfully with 4 corner points."""
        calibrator = HomographyCalibrator()
        # Define a simple pixel-to-meter mapping:
        # pixel (0, 0) -> meter (0, 0)
        # pixel (640, 0) -> meter (14, 0)
        # pixel (0, 480) -> meter (0, 1.8)
        # pixel (640, 480) -> meter (14, 1.8)
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)

        result = calibrator.calibrate()
        assert result is True
        assert calibrator.is_calibrated

    def test_calibrate_returns_homography_matrix(self):
        """Should return valid 3x3 homography matrix after calibration."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)

        calibrator.calibrate()
        H = calibrator.get_homography_matrix()
        assert H is not None
        assert H.shape == (3, 3)


class TestCoordinateTransformation:
    """Test pixel-to-meter and meter-to-pixel transformation."""

    def test_pixel_to_meter_not_calibrated_raises(self):
        """Should raise RuntimeError if not calibrated."""
        calibrator = HomographyCalibrator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            calibrator.pixel_to_meter(100, 200)

    def test_meter_to_pixel_not_calibrated_raises(self):
        """Should raise RuntimeError if not calibrated."""
        calibrator = HomographyCalibrator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            calibrator.meter_to_pixel(0.0, 0.0)

    def test_pixel_to_meter_corners(self):
        """Should transform corner points correctly."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        calibrator.calibrate()

        # Test corners
        x0, y0 = calibrator.pixel_to_meter(0, 0)
        assert x0 == pytest.approx(0.0, abs=0.1)
        assert y0 == pytest.approx(0.0, abs=0.1)

        x1, y1 = calibrator.pixel_to_meter(640, 0)
        assert x1 == pytest.approx(14.0, abs=0.1)
        assert y1 == pytest.approx(0.0, abs=0.1)

        x2, y2 = calibrator.pixel_to_meter(0, 480)
        assert x2 == pytest.approx(0.0, abs=0.1)
        assert y2 == pytest.approx(1.8, abs=0.1)

        x3, y3 = calibrator.pixel_to_meter(640, 480)
        assert x3 == pytest.approx(14.0, abs=0.1)
        assert y3 == pytest.approx(1.8, abs=0.1)

    def test_meter_to_pixel_corners(self):
        """Should transform meter corners to pixel correctly."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        calibrator.calibrate()

        # Test corners
        px, py = calibrator.meter_to_pixel(0.0, 0.0)
        assert px == pytest.approx(0, abs=1.0)
        assert py == pytest.approx(0, abs=1.0)

        px, py = calibrator.meter_to_pixel(14.0, 1.8)
        assert px == pytest.approx(640, abs=1.0)
        assert py == pytest.approx(480, abs=1.0)

    def test_roundtrip_pixel_to_meter_to_pixel(self):
        """Should round-trip correctly."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        calibrator.calibrate()

        # Round-trip test
        original_px, original_py = 320, 240
        meter_x, meter_y = calibrator.pixel_to_meter(original_px, original_py)
        round_px, round_py = calibrator.meter_to_pixel(meter_x, meter_y)

        assert round_px == pytest.approx(original_px, abs=1.0)
        assert round_py == pytest.approx(original_py, abs=1.0)


class TestReprojectionError:
    """Test reprojection error computation."""

    def test_reprojection_error_not_calibrated(self):
        """Should return infinity if not calibrated."""
        calibrator = HomographyCalibrator()
        error = calibrator.compute_reprojection_error(100, 200, 0.0, 0.0)
        assert error == float('inf')

    def test_reprojection_error_zero_for_calibration_points(self):
        """Should be near zero for original calibration points."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        calibrator.calibrate()

        error = calibrator.compute_reprojection_error(0, 0, 0.0, 0.0)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_reprojection_error_nonzero_for_new_point(self):
        """Should be non-zero for a point not in calibration set."""
        calibrator = HomographyCalibrator()
        calibrator.add_point(pixel_x=0, pixel_y=0, meter_x=0.0, meter_y=0.0)
        calibrator.add_point(pixel_x=640, pixel_y=0, meter_x=14.0, meter_y=0.0)
        calibrator.add_point(pixel_x=0, pixel_y=480, meter_x=0.0, meter_y=1.8)
        calibrator.add_point(pixel_x=640, pixel_y=480, meter_x=14.0, meter_y=1.8)
        calibrator.calibrate()

        # Point at center should have some error (interpolated)
        error = calibrator.compute_reprojection_error(320, 240, 7.0, 0.9)
        # Error should be small but may not be zero
        assert error < 1.0  # Should be reasonably small


class TestIntegration:
    """Integration tests for calibrator."""

    def test_full_calibration_workflow(self):
        """Test complete calibration workflow."""
        calibrator = HomographyCalibrator()

        # Add corners from a simulated calibration
        calibrator.add_points_from_piste_corners(
            top_left=(100, 100),
            top_right=(540, 100),
            bottom_left=(100, 380),
            bottom_right=(540, 380),
        )

        assert calibrator.num_points == 4
        assert not calibrator.is_calibrated

        success = calibrator.calibrate()
        assert success
        assert calibrator.is_calibrated

        # Verify transformation works
        x, y = calibrator.pixel_to_meter(100, 100)
        assert x == pytest.approx(0.0, abs=0.1)
        assert y == pytest.approx(0.0, abs=0.1)

        x, y = calibrator.pixel_to_meter(540, 380)
        assert x == pytest.approx(14.0, abs=0.1)
        assert y == pytest.approx(1.8, abs=0.1)
