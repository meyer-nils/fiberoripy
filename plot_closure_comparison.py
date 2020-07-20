# -*- coding: utf-8 -*-
"""Plotting fiber orientation tensors and closure."""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.tensoroperations import (IBOF_closure, compute_closure,
                                      hybrid_closure, linear_closure,
                                      quadratic_closure)
from fiberpy.tensorplot import (plot_orbit2, plot_orbit4, plot_projection2,
                                plot_projection4)
from mpl_toolkits.mplot3d import Axes3D

a = 1.0 / 3.0 * np.eye(3)
A = compute_closure(a, "RANDOM", N=100)

A_hybrid = hybrid_closure(a)
A_ibof = IBOF_closure(a)
A_lin = linear_closure(a)
A_quad = quadratic_closure(a)

# Figure setup
fig = plt.figure(figsize=(12, 8))

# Second order tensor plot
ax = fig.add_subplot(221, projection=Axes3D.name)
plot_orbit2(ax, a)

ax = fig.add_subplot(222)
plot_projection2(ax, "xy", a)

ax = fig.add_subplot(223)
plot_projection2(ax, "xz", a)

ax = fig.add_subplot(224)
plot_projection2(ax, "yz", a)

plt.show()

fig = plt.figure(figsize=(12, 8))
# Fourth order tensor plot
ax = fig.add_subplot(221, projection=Axes3D.name)
plot_orbit4(ax, A, A_hybrid, A_lin, A_quad, A_ibof)

ax = fig.add_subplot(222)
plot_projection4(ax, "xy", A, A_hybrid, A_lin, A_quad, A_ibof)

ax = fig.add_subplot(223)
plot_projection4(ax, "xz", A, A_hybrid, A_lin, A_quad, A_ibof)

ax = fig.add_subplot(224)
plot_projection4(ax, "yz", A, A_hybrid, A_lin, A_quad, A_ibof)

plt.legend(["Original", "Hybrid", "Linear", "Quadratic", "IBOF"])

plt.show()
