# -*- coding: utf-8 -*-
"""Testing re-orientation in compression flow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import folgar_tucker_ode


h0 = 0.0062
hf = 0.002
v = 0.001
Ci = 0.0001
ar = 25.0
xi = (ar ** 2 - 1) / (ar ** 2 + 1)

T = (h0 - hf) / v
t = np.linspace(0, T, 500)


def L(t):
    """Velocity gradient."""
    return np.array(
        [[v / (h0 - v * t), 0, 0], [0, 0, 0], [0, 0, -v / (h0 - v * t)]]
    )


A0 = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])

A = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, Ci))
# plots
labels = [
    "$A_{11}$",
    "$A_{12}$",
    "$A_{13}$",
    "$A_{21}$",
    "$A_{22}$",
    "$A_{23}$",
    "$A_{31}$",
    "$A_{32}$",
    "$A_{33}$",
]

subplots = [0, 1, 2, 4, 5, 8]

legend_list = ["Folgar-Tucker with IBOF"]

plt.figure()
for i in subplots:
    plt.subplot("33" + str(i + 1))
    p = plt.plot(t, A[:, i])
    plt.xlabel("Time $t$ in s")
    plt.ylabel(labels[i])
    plt.ylim([-1, 1])
plt.legend(legend_list)
plt.tight_layout()
plt.show()
