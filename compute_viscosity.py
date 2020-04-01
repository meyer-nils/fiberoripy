"""Evaluate long range interaction term utilizing IBOF closure."""
# -*- coding: utf-8 -*-
import numpy as np

from fiberpy.tensoroperations import IBOF_closure

h0 = 0.005
v0 = -0.001666
t = 0

D = np.array(
    [
        [-v0 / (h0 + v0 * t), 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, v0 / (h0 + v0 * t)],
    ]
)

d = np.eye(3)
a = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])

A = IBOF_closure(a)


m = 1000

s = m * np.einsum(
    "ijkl, kl -> ij", A - 1.0 / 3.0 * np.einsum("ij, kl->ijkl", d, a), D
)

print(s)
