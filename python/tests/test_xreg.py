"""Tests for greybox.xreg module."""

import numpy as np
import pytest

from greybox.xreg import xreg_expander, xreg_multiplier, temporal_dummy


class TestXregExpander:
    def test_lag_minus_one(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1])
        assert result.shape == (5, 1)
        # lag=-1 shifts forward: row[1]=x[0], row[2]=x[1], ...
        np.testing.assert_array_equal(result[1:, 0], [10, 20, 30, 40])

    def test_lead_plus_one(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[1])
        assert result.shape == (5, 1)
        # lead=1: row[0]=x[1], row[1]=x[2], ...
        np.testing.assert_array_equal(result[:4, 0], [20, 30, 40, 50])

    def test_multiple_lags(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1, -2])
        assert result.shape == (5, 2)

    def test_zero_lag_only_returns_original(self):
        x = np.array([10, 20, 30], dtype=float)
        result = xreg_expander(x, lags=[0])
        # lag=0 is removed, so returns original
        assert result.shape == (3, 1)

    def test_gaps_zero(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1], gaps="zero")
        assert result[0, 0] == 0.0

    def test_gaps_naive(self):
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        result = xreg_expander(x, lags=[-1], gaps="naive")
        # First row should be filled with first valid value (10)
        assert result[0, 0] == 10.0

    def test_matrix_input(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        result = xreg_expander(x, lags=[-1])
        assert result.shape == (4, 2)


class TestXregMultiplier:
    def test_basic(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = xreg_multiplier(x)
        # Should have original cols + 1 cross-product
        assert result.shape == (3, 3)
        # Cross product: col0 * col1
        np.testing.assert_array_equal(result[:, 2], [2, 12, 30])

    def test_three_cols(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = xreg_multiplier(x)
        # 3 original + C(3,2)=3 combinations = 6
        assert result.shape == (2, 6)
        # Originals preserved
        np.testing.assert_array_equal(result[:, 0], [1, 4])
        np.testing.assert_array_equal(result[:, 1], [2, 5])
        np.testing.assert_array_equal(result[:, 2], [3, 6])

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            xreg_multiplier(np.array([1, 2, 3]))

    def test_single_col_raises(self):
        with pytest.raises(ValueError):
            xreg_multiplier(np.array([[1], [2], [3]]))


class TestTemporalDummy:
    def test_monthly_shape(self):
        x = np.arange(24)
        result = temporal_dummy(x, freq=12)
        assert result.shape == (24, 12)

    def test_monthly_one_hot(self):
        x = np.arange(12)
        result = temporal_dummy(x, freq=12)
        # Each row should sum to 1 (one-hot)
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(12))
        # First row: month 0 is 1
        assert result[0, 0] == 1.0
        assert result[0, 1] == 0.0

    def test_quarterly(self):
        x = np.arange(8)
        result = temporal_dummy(x, freq=4)
        assert result.shape == (8, 4)
        # q0, q1, q2, q3, q0, q1, q2, q3
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(8))

    def test_weekly(self):
        x = np.arange(14)
        result = temporal_dummy(x, freq=7)
        assert result.shape == (14, 7)

    def test_horizon(self):
        x = np.arange(6)
        result = temporal_dummy(x, freq=4, h=4)
        assert result.shape == (10, 4)

    def test_cycling(self):
        x = np.arange(5)
        result = temporal_dummy(x, freq=3)
        # 0->col0, 1->col1, 2->col2, 3->col0, 4->col1
        assert result[3, 0] == 1.0
        assert result[4, 1] == 1.0
