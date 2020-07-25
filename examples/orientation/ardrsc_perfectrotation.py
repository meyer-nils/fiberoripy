# -*- coding: utf-8 -*-
"""Testing re-orientation in perfectly rotating flow.

The result should be a sine shape.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from fiberoripy.orientation import ard_rsc_ode

xi = 1.0
# time steps
t = np.linspace(0, np.pi, 500)


A0 = np.zeros((3, 3))
A0[0, 0] = 0.5
A0[2, 2] = 0.5


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


# computed solution
A = odeint(
    ard_rsc_ode,
    A0.ravel(),
    t,
    args=(xi, L, 1.924e-4, 1.0 / 30.0, 5.839e-3, 0.04, 1.168e-5, 0.0),
)

plt.plot(t, A[:, 0], "-", label="A11")
plt.plot(t, A[:, 4], "--", label="A22")
plt.plot(t, A[:, 8], "-.", label="A33")
plt.plot(t, A[:, 2], ":", label="A31")

plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
