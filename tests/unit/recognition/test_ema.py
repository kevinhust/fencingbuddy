"""
Tests for EMA smoothing utilities
TDD Phase 5.1.6: Dedicated EMA Tests
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.recognition.feature_math import EMASmoother


class TestEMASmootherBasics:
    """Test EMASmoother basic functionality."""

    def test_initial_value_returned_as_is(self):
        """First value should be returned unchanged."""
        smoother = EMASmoother(alpha=0.5)
        value = np.array([1.0, 2.0, 3.0])
        result = smoother.smooth(value)
        np.testing.assert_array_equal(result, value)

    def test_alpha_values_work(self):
        """Various alpha values should be accepted and work correctly."""
        # Valid alpha values should work
        smoother1 = EMASmoother(alpha=0.0)
        smoother2 = EMASmoother(alpha=1.0)
        smoother3 = EMASmoother(alpha=0.5)

        assert smoother1.alpha == 0.0
        assert smoother2.alpha == 1.0
        assert smoother3.alpha == 0.5

    def test_default_alpha(self):
        """Default alpha should be 0.7."""
        smoother = EMASmoother()
        assert smoother.alpha == 0.7

    def test_custom_alpha(self):
        """Should accept custom alpha values."""
        smoother = EMASmoother(alpha=0.9)
        assert smoother.alpha == 0.9

        smoother = EMASmoother(alpha=0.3)
        assert smoother.alpha == 0.3


class TestEMASmootherSmoothing:
    """Test EMA smoothing behavior."""

    def test_second_value_is_ema_blend(self):
        """Second value should be EMA blend of current and previous."""
        smoother = EMASmoother(alpha=0.5)
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 1.0])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        # Expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_ema_formula_with_different_alpha(self):
        """Verify EMA formula: result = alpha * current + (1-alpha) * previous."""
        smoother = EMASmoother(alpha=0.8)
        v1 = np.array([10.0, 20.0])
        v2 = np.array([20.0, 40.0])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        # Expected: 0.8 * 20 + 0.2 * 10 = 18
        #           0.8 * 40 + 0.2 * 20 = 36
        np.testing.assert_array_almost_equal(result, [18.0, 36.0])

    def test_multiple_iterations(self):
        """Multiple iterations should progressively smooth."""
        smoother = EMASmoother(alpha=0.5)
        values = [
            np.array([0.0]),
            np.array([4.0]),
            np.array([4.0]),
            np.array([4.0]),
        ]

        results = []
        for v in values:
            result = smoother.smooth(v)
            results.append(result.copy())

        # Iteration 1: 0.0 (first value)
        assert results[0][0] == 0.0
        # Iteration 2: 0.5 * 4 + 0.5 * 0 = 2.0
        assert results[1][0] == 2.0
        # Iteration 3: 0.5 * 4 + 0.5 * 2 = 3.0
        assert results[2][0] == 3.0
        # Iteration 4: 0.5 * 4 + 0.5 * 3 = 3.5
        assert results[3][0] == 3.5

    def test_high_alpha_reacts_quickly(self):
        """High alpha (close to 1) should react quickly to changes."""
        smoother = EMASmoother(alpha=0.95)
        v1 = np.array([0.0])
        v2 = np.array([100.0])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        # 0.95 * 100 + 0.05 * 0 = 95
        assert result[0] == pytest.approx(95.0, abs=0.1)

    def test_low_alpha_smooths_strongly(self):
        """Low alpha should strongly smooth changes."""
        smoother = EMASmoother(alpha=0.1)
        v1 = np.array([0.0])
        v2 = np.array([100.0])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        # 0.1 * 100 + 0.9 * 0 = 10
        assert result[0] == pytest.approx(10.0, abs=0.1)


class TestEMASmootherReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Reset should clear previous value."""
        smoother = EMASmoother(alpha=0.5)
        smoother.smooth(np.array([10.0]))
        smoother.reset()

        result = smoother.smooth(np.array([10.0]))
        # After reset, first value is returned as-is
        assert result[0] == 10.0

    def test_reset_allows_fresh_start(self):
        """After reset, next smooth should be like first call."""
        smoother = EMASmoother(alpha=0.5)

        # First sequence
        smoother.smooth(np.array([5.0]))
        smoother.smooth(np.array([10.0]))

        # Reset
        smoother.reset()

        # New sequence should start fresh
        result = smoother.smooth(np.array([20.0]))
        assert result[0] == 20.0  # First value returned as-is


class TestEMASmootherScalar:
    """Test scalar smoothing methods."""

    def test_smooth_scalar_first_value(self):
        """First scalar value returned as-is."""
        smoother = EMASmoother(alpha=0.5)
        result = smoother.smooth_scalar(100.0)
        assert result == 100.0

    def test_smooth_scalar_formula(self):
        """Verify scalar EMA formula."""
        smoother = EMASmoother(alpha=0.6)

        smoother.smooth_scalar(0.0)
        result = smoother.smooth_scalar(10.0)

        # Expected: 0.6 * 10 + 0.4 * 0 = 6
        assert result == pytest.approx(6.0, abs=0.01)

    def test_smooth_scalar_multiple_iterations(self):
        """Multiple scalar iterations should converge."""
        smoother = EMASmoother(alpha=0.3)

        smoother.smooth_scalar(0.0)
        r1 = smoother.smooth_scalar(100.0)
        r2 = smoother.smooth_scalar(100.0)
        r3 = smoother.smooth_scalar(100.0)

        # After enough iterations, should converge toward 100
        assert r1 < 50  # First iteration
        assert r2 > r1  # Should increase
        assert r3 > r2  # Should continue increasing


class TestEMASmootherEdgeCases:
    """Test edge cases."""

    def test_multidimensional_arrays(self):
        """Should handle multi-dimensional arrays."""
        smoother = EMASmoother(alpha=0.5)
        v1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        v2 = np.array([[5.0, 6.0], [7.0, 8.0]])

        smoother.smooth(v1)
        result = smoother.smooth(v2)

        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_element_array(self):
        """Should handle single-element arrays."""
        smoother = EMASmoother(alpha=0.5)

        result1 = smoother.smooth(np.array([1.0]))
        assert result1[0] == 1.0

        result2 = smoother.smooth(np.array([3.0]))
        assert result2[0] == 2.0

    def test_negative_values(self):
        """Should handle negative values."""
        smoother = EMASmoother(alpha=0.5)

        smoother.smooth(np.array([-10.0]))
        result = smoother.smooth(np.array([10.0]))

        # 0.5 * 10 + 0.5 * (-10) = 0
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_very_small_alpha(self):
        """Should handle very small alpha values."""
        smoother = EMASmoother(alpha=1e-6)

        smoother.smooth(np.array([0.0]))
        result = smoother.smooth(np.array([100.0]))

        # Almost all weight on previous value
        assert result[0] < 1.0

    def test_alpha_close_to_one(self):
        """Should handle alpha close to 1."""
        smoother = EMASmoother(alpha=1.0 - 1e-10)

        smoother.smooth(np.array([0.0]))
        result = smoother.smooth(np.array([100.0]))

        # Almost all weight on current value
        assert result[0] > 99.0
