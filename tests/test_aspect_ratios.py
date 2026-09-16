import numpy as np
import pytest

from fiberoripy.aspect_ratios import get_cox_aspect_ratio, get_zhang_aspect_ratio


@pytest.mark.parametrize("aspect_ratio", [2.0, 5.0, 10.0, 50.0])
def test_cox_matches_the_published_expression(aspect_ratio):
    """Cox: 1.24 * r / sqrt(ln(r))."""
    expected = 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))
    assert get_cox_aspect_ratio(aspect_ratio) == pytest.approx(expected)


@pytest.mark.parametrize("aspect_ratio", [2.0, 5.0, 10.0, 50.0])
def test_zhang_matches_the_published_polynomial(aspect_ratio):
    """Zhang: a cubic fit in the aspect ratio."""
    r = aspect_ratio
    expected = 0.000035 * r**3 - 0.00467 * r**2 + 0.764 * r + 0.404
    assert get_zhang_aspect_ratio(r) == pytest.approx(expected)


@pytest.mark.parametrize("func", [get_cox_aspect_ratio, get_zhang_aspect_ratio])
def test_equivalent_ratio_increases_with_the_geometric_one(func):
    """Both approximations must be monotonic over the usual range."""
    ratios = np.linspace(2.0, 50.0, 25)
    values = np.array([func(r) for r in ratios])
    assert np.all(np.diff(values) > 0.0)


@pytest.mark.parametrize("func", [get_cox_aspect_ratio, get_zhang_aspect_ratio])
def test_accepts_an_array_of_aspect_ratios(func):
    """Both are plain expressions, so they broadcast."""
    ratios = np.array([5.0, 10.0, 20.0])
    assert np.allclose(func(ratios), [func(r) for r in ratios])
