"""Tests for transform functions."""

import numpy as np
from greybox.transforms import bc_transform, bc_transform_inv, mean_fast


class TestBCTransform:
    """Tests for Box-Cox transformation."""

    def test_bc_transform_lambda_zero(self):
        """Test lambda=0 returns log(y)."""
        y = np.array([1.0, 2.0, 3.0])
        result = bc_transform(y, 0)
        expected = np.log(y)
        np.testing.assert_array_almost_equal(result, expected)

    def test_bc_transform_lambda_one(self):
        """Test lambda=1 returns y-1."""
        y = np.array([1.0, 2.0, 3.0])
        result = bc_transform(y, 1)
        expected = y - 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_bc_transform_lambda_half(self):
        """Test lambda=0.5."""
        y = np.array([1.0, 4.0, 9.0])
        result = bc_transform(y, 0.5)
        expected = (np.sqrt(y) - 1) / 0.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_bc_transform_with_array(self):
        """Test with numpy array input."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bc_transform(y, 0.5)
        assert result.shape == y.shape
        assert np.all(np.isfinite(result))


class TestBCTransformInv:
    """Tests for inverse Box-Cox transformation."""

    def test_bc_transform_inv_lambda_zero(self):
        """Test lambda=0 returns exp(y)."""
        y = np.array([0.0, 1.0, 2.0])
        result = bc_transform_inv(y, 0)
        expected = np.exp(y)
        np.testing.assert_array_almost_equal(result, expected)

    def test_bc_transform_inv_roundtrip(self):
        """Test roundtrip: transform then inverse."""
        y = np.array([1.0, 2.0, 3.0])
        lambda_bc = 0.5
        transformed = bc_transform(y, lambda_bc)
        result = bc_transform_inv(transformed, lambda_bc)
        np.testing.assert_array_almost_equal(result, y, decimal=10)

    def test_bc_transform_inv_lambda_one(self):
        """Test lambda=1."""
        y = np.array([0.0, 1.0, 2.0])
        result = bc_transform_inv(y, 1)
        expected = y * 1 + 1
        expected = expected ** (1 / 1)
        np.testing.assert_array_almost_equal(result, expected)


class TestMeanFast:
    """Tests for fast mean calculation."""

    def test_mean_fast_basic(self):
        """Test basic mean without trimming."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_fast(x)
        expected = 3.0
        assert result == expected

    def test_mean_fast_with_df(self):
        """Test with custom degrees of freedom."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_fast(x, df=10)
        expected = 15.0 / 10
        assert result == expected

    def test_mean_fast_trim_both(self):
        """Test trimming from both sides."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_fast(x, trim=0.2, side="both")
        assert 2.5 < result < 3.5

    def test_mean_fast_trim_lower(self):
        """Test trimming from lower end."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_fast(x, trim=0.2, side="lower")
        assert result > 3.0

    def test_mean_fast_trim_upper(self):
        """Test trimming from upper end."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_fast(x, trim=0.2, side="upper")
        assert result < 3.0
