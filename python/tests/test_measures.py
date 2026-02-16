"""Tests for greybox.measures module."""

import numpy as np
import pytest

from greybox.measures import (
    mae,
    mse,
    rmse,
    mpe,
    mape,
    mase,
    accuracy,
    determination,
    association,
)

# Simple test arrays
actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
# errors: -0.1, -0.2, 0.2, -0.1, -0.3
# abs errors: 0.1, 0.2, 0.2, 0.1, 0.3  -> mean = 0.18
# sq errors: 0.01, 0.04, 0.04, 0.01, 0.09 -> mean = 0.038


class TestMAE:
    def test_basic(self):
        result = mae(actual, predicted)
        expected = np.mean([0.1, 0.2, 0.2, 0.1, 0.3])
        np.testing.assert_almost_equal(result, expected)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mae(np.array([1, 2]), np.array([1, 2, 3]))

    def test_with_nan(self):
        a = np.array([1.0, np.nan, 3.0])
        f = np.array([1.1, 2.2, 2.8])
        result = mae(a, f)
        np.testing.assert_almost_equal(result, np.mean([0.1, 0.2]))


class TestMSE:
    def test_basic(self):
        result = mse(actual, predicted)
        expected = np.mean([0.01, 0.04, 0.04, 0.01, 0.09])
        np.testing.assert_almost_equal(result, expected)


class TestRMSE:
    def test_basic(self):
        result = rmse(actual, predicted)
        expected = np.sqrt(np.mean([0.01, 0.04, 0.04, 0.01, 0.09]))
        np.testing.assert_almost_equal(result, expected)


class TestMPE:
    def test_basic(self):
        # (actual - forecast) / actual * 100
        # (1-1.1)/1*100=-10, (2-2.2)/2*100=-10, (3-2.8)/3*100=6.667,
        # (4-4.1)/4*100=-2.5, (5-5.3)/5*100=-6.0
        errors = [
            (1 - 1.1) / 1 * 100,
            (2 - 2.2) / 2 * 100,
            (3 - 2.8) / 3 * 100,
            (4 - 4.1) / 4 * 100,
            (5 - 5.3) / 5 * 100,
        ]
        result = mpe(actual, predicted)
        np.testing.assert_almost_equal(result, np.mean(errors))

    def test_skips_zero_actual(self):
        a = np.array([0.0, 2.0, 3.0])
        f = np.array([0.1, 2.2, 2.8])
        result = mpe(a, f)
        expected = np.mean([(2 - 2.2) / 2 * 100, (3 - 2.8) / 3 * 100])
        np.testing.assert_almost_equal(result, expected)


class TestMAPE:
    def test_basic(self):
        abs_pct = [
            abs((1 - 1.1) / 1) * 100,
            abs((2 - 2.2) / 2) * 100,
            abs((3 - 2.8) / 3) * 100,
            abs((4 - 4.1) / 4) * 100,
            abs((5 - 5.3) / 5) * 100,
        ]
        result = mape(actual, predicted)
        np.testing.assert_almost_equal(result, np.mean(abs_pct))


class TestMASE:
    def test_with_scale(self):
        result = mase(actual, predicted, scale=1.0)
        expected = mae(actual, predicted)
        np.testing.assert_almost_equal(result, expected)

    def test_auto_scale(self):
        # scale = mean(|diff(actual)|) = mean([1,1,1,1]) = 1.0
        result = mase(actual, predicted)
        expected = mae(actual, predicted) / 1.0
        np.testing.assert_almost_equal(result, expected)

    def test_zero_scale(self):
        a = np.array([3.0, 3.0, 3.0])
        f = np.array([3.1, 2.9, 3.0])
        result = mase(a, f)
        assert np.isnan(result)


class TestAccuracy:
    def test_keys(self):
        result = accuracy(actual, predicted)
        expected_keys = {"ME", "MAE", "MSE", "RMSE", "MPE", "MAPE", "MASE"}
        assert set(result.keys()) == expected_keys

    def test_values(self):
        result = accuracy(actual, predicted)
        np.testing.assert_almost_equal(result["MAE"], mae(actual, predicted))
        np.testing.assert_almost_equal(result["MSE"], mse(actual, predicted))
        np.testing.assert_almost_equal(result["RMSE"], rmse(actual, predicted))
        # ME = mean(actual - forecast)
        np.testing.assert_almost_equal(result["ME"], np.mean(actual - predicted))


class TestDetermination:
    def test_perfect_fit(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = determination(a, a)
        np.testing.assert_almost_equal(result["r2"], 1.0)

    def test_basic(self):
        result = determination(actual, predicted)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        expected_r2 = 1 - ss_res / ss_tot
        np.testing.assert_almost_equal(result["r2"], expected_r2)

    def test_adj_r2(self):
        result = determination(actual, predicted, k=1)
        n = 5
        k = 1
        expected_adj = 1 - (1 - result["r2"]) * (n - 1) / (n - k - 1)
        np.testing.assert_almost_equal(result["adjR2"], expected_adj)


class TestAssociation:
    def test_perfect_correlation(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])])
        result = association(x)
        np.testing.assert_almost_equal(result["value"][0, 1], 1.0)
        np.testing.assert_almost_equal(result["value"][1, 0], 1.0)

    def test_returns_keys(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([5, 3, 4, 2, 1])])
        result = association(x)
        assert "value" in result
        assert "p.value" in result
        assert "type" in result

    def test_diagonal_is_one(self):
        x = np.column_stack([np.array([1, 2, 3, 4, 5]), np.array([5, 3, 4, 2, 1])])
        result = association(x)
        np.testing.assert_almost_equal(result["value"][0, 0], 1.0)
        np.testing.assert_almost_equal(result["value"][1, 1], 1.0)

    def test_with_y(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = association(x, y)
        assert result["value"].shape == (2, 2)
