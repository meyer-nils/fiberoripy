# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow.

The result should match Figure 1 in
Huan-Chang Tseng, Rong-Yeu Chang, Chia-Hsiang Hsu: "An objective tensor to predict
anisotropic fiber orientation in concentrated suspensions",
Journal of Rheology 60, 215 (2016):
DOI: 10.1122/1.4939098
"""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.orientation import ard_rsc_ode, iardrpr_ode
from scipy.integrate import odeint

xi = 1.0
# time steps
t = np.linspace(0, 2000, 100)


A0 = 1.0 / 3.0 * np.eye(3)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


# computed solution
Aref = odeint(
    ard_rsc_ode,
    A0.ravel(),
    t,
    args=(xi, L, 3.842e-4, 1.0 / 30.0, -1.786e-3, 5.25e-2, 1.168e-5, -5.0e-4),
)
A = odeint(iardrpr_ode, A0.ravel(), t, args=(xi, L, 0.025, 1.0, 0.967),)

plt.plot(t, Aref[:, 0], "k-", label="A11 (ARD-RSC)")
plt.plot(t, Aref[:, 4], "r-", label="A22 (ARD-RSC)")
plt.plot(t, Aref[:, 8], "b-", label="A33 (ARD-RSC)")
plt.plot(t, Aref[:, 2], "g-", label="A31 (ARD-RSC)")
plt.plot(t, A[:, 0], "k.", label="A11 (iARD-RPR)")
plt.plot(t, A[:, 4], "r.", label="A22 (iARD-RPR)")
plt.plot(t, A[:, 8], "b.", label="A33 (iARD-RPR)")
plt.plot(t, A[:, 2], "g.", label="A31 (iARD-RPR)")

plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
