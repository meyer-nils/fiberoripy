# -*- coding: utf-8 -*-
"""Testing re-orientation in compression flow."""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.constants import COMPS
from fiberpy.orientation import folgar_tucker_ode
from scipy.integrate import odeint

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

plt.figure()
for c in ["A11", "A12", "A13", "A22", "A23", "A33"]:
    i = COMPS[c]
    plt.subplot("33" + str(i + 1))
    p = plt.plot(t, A[:, i])
    plt.xlabel("Time $t$ in s")
    plt.ylabel(c)
    plt.ylim([-1, 1])
plt.tight_layout()
plt.show()
