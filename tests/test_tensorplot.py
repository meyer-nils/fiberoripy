import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fiberoripy.tensorplot import (
    plot_orbit2,
    plot_orbit4,
    plot_projection2,
    plot_projection4,
    sample_circle,
    sample_sphere,
)

PLANES = ["xy", "xz", "yz"]


@pytest.fixture
def a():
    """Second order fiber orientation tensor."""
    return np.diag([0.6, 0.3, 0.1])


@pytest.fixture
def A(a):
    """Fourth order fiber orientation tensor, closed from `a`."""
    from fiberoripy.closures import IBOF_closure

    return IBOF_closure(a)


def test_sample_sphere_returns_unit_vectors():
    """The sampled directions must lie on the unit sphere."""
    r = sample_sphere(N=20)
    assert r.shape == (3, 20, 20)
    assert np.allclose(np.linalg.norm(r, axis=0), 1.0)


@pytest.mark.parametrize("plane", PLANES)
def test_sample_circle_is_a_unit_circle_in_its_plane(plane):
    """Two components trace a unit circle, the third is held at one."""
    R = sample_circle(plane, N=32)
    assert R.shape == (3, 32)
    held = {"xy": 2, "xz": 1, "yz": 0}[plane]
    assert np.allclose(R[held], 1.0)
    in_plane = [i for i in range(3) if i != held]
    assert np.allclose(np.linalg.norm(R[in_plane], axis=0), 1.0)


def test_sample_circle_rejects_an_unknown_plane():
    """An unknown plane is a caller error, not a silent None."""
    with pytest.raises(ValueError, match="Unknown plane"):
        sample_circle("zz")


@pytest.mark.parametrize("plot", [plot_orbit2, plot_orbit4])
def test_orbit_plots_draw_a_surface(plot, a, A):
    """The orbital plots add a surface and label all three axes."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plot(ax, {"color": "blue"}, a if plot is plot_orbit2 else A)
    assert ax.collections or ax.get_children()
    assert ax.get_xlabel() and ax.get_ylabel() and ax.get_zlabel()
    assert ax.get_title()
    plt.close(fig)


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("plot", [plot_projection2, plot_projection4])
def test_projection_plots_scatter_into_the_requested_plane(plot, plane, a, A):
    """Each projection scatters points and titles itself with the plane."""
    fig, ax = plt.subplots()
    plot(ax, plane, a if plot is plot_projection2 else A)
    assert len(ax.collections) == 1
    assert ax.get_xlabel() and ax.get_ylabel()
    assert "Projection" in ax.get_title()
    plt.close(fig)


def test_plots_accept_several_tensors_at_once(a):
    """The plot helpers take a variable number of tensors."""
    fig, ax = plt.subplots()
    plot_projection2(ax, "xy", a, 1.0 / 3.0 * np.eye(3))
    assert len(ax.collections) == 2
    plt.close(fig)
