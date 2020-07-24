# -*- coding: utf-8 -*-
"""Tools to plot tensors."""
import numpy as np


def sample_sphere(N=100):
    """Define all spherical angles."""
    phi = np.linspace(0, 2 * np.pi, N)
    theta = np.linspace(0, np.pi, N)

    # Cartesian coordinates that correspond to the spherical angles:
    r = np.array(
        [
            np.outer(np.cos(phi), np.sin(theta)),
            np.outer(np.sin(phi), np.sin(theta)),
            np.outer(np.ones_like(phi), np.cos(theta)),
        ]
    )
    return r


def sample_circle(plane="xy", N=100):
    """Define all angles in a certain plane."""
    phi = np.linspace(0, 2 * np.pi, N)
    if plane == "xy":
        return np.array([np.cos(phi), np.sin(phi), np.ones_like(phi)])
    elif plane == "xz":
        return np.array([np.cos(phi), np.ones_like(phi), np.sin(phi)])
    elif plane == "yz":
        return np.array([np.ones_like(phi), np.cos(phi), np.sin(phi)])


def plot_orbit2(ax, *tensors):
    """Orbital plot of a second order."""
    r = sample_sphere()
    for a in tensors:
        assert np.shape(a) == (3, 3)
        x, y, z = np.einsum("ij,jkl->ikl", a, r)
        ax.plot_surface(x, y, z, rstride=4, cstride=4)
        m = max(
            np.max(x),
            np.max(y),
            np.max(z),
            abs(np.min(x)),
            abs(np.min(y)),
            abs(np.min(z)),
        )
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_zlim(-m, m)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax.set_title("Second Order Orientation Tensor")
        # ax.set_aspect(1.0)


def plot_orbit4(ax, *tensors):
    """Orbital plot of a second order."""
    r = sample_sphere()
    for A in tensors:
        assert np.shape(A) == (3, 3, 3, 3)
        x, y, z = np.einsum("ijkl,jmn,kmn,lmn->imn", A, r, r, r)
        ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.1)
        m = max(
            np.max(x),
            np.max(y),
            np.max(z),
            abs(np.min(x)),
            abs(np.min(y)),
            abs(np.min(z)),
        )
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_zlim(-m, m)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax.set_title("Fourth Order Orientation Tensor")
        # ax.set_aspect(1.0)


def plot_projection2(ax, plane, *tensors):
    """Orbital plot of a second order."""
    R = sample_circle(plane)
    for a in tensors:
        assert np.shape(a) == (3, 3)
        x, y, z = np.einsum("ij,jk->ik", a, R)
        if plane == "xy":
            ax.scatter(x, y, s=1)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_title("Projection to $x_1$-$x_2$ plane")
        elif plane == "xz":
            ax.scatter(x, z, s=1)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_3$")
            ax.set_title("Projection to $x_1$-$x_3$ plane")
        elif plane == "yz":
            ax.scatter(y, z, s=1)
            ax.set_xlabel("$x_2$")
            ax.set_ylabel("$x_3$")
            ax.set_title("Projection to $x_2$-$x_3$ plane")
        m = max(np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y)))
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect(1.0)
        ax.grid()


def plot_projection4(ax, plane, *tensors):
    """Orbital plot of a second order."""
    R = sample_circle(plane)
    for A in tensors:
        assert np.shape(A) == (3, 3, 3, 3)
        x, y, z = np.einsum("ijkl,jm,km,lm->im", A, R, R, R)
        if plane == "xy":
            ax.scatter(x, y, s=4)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_title("Projection to $x_1$-$x_2$ plane")
        elif plane == "xz":
            ax.scatter(x, z, s=4)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_3$")
            ax.set_title("Projection to $x_1$-$x_3$ plane")
        elif plane == "yz":
            ax.scatter(y, z, s=4)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_3$")
            ax.set_title("Projection to $x_2$-$x_3$ plane")
        m = max(np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y)))
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect(1.0)
        ax.grid()
