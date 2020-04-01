# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow."""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.constants import COMPS
from fiberpy.orientation import folgar_tucker_ode, get_zhang_aspect_ratio
from scipy.integrate import odeint

# shear rate
gamma = 1.0
# aspect ratio
ar = 5
# equivalent aspect ratio
are = get_zhang_aspect_ratio(ar)
print("Equivalent aspect ratio is %f" % are)
xi = (are ** 2 - 1) / (are ** 2 + 1)
# period fo rotation
T = 2.0 * np.pi / gamma * (are + 1.0 / are)
print("Period for a quarter rotation is %f" % T)
# time steps
t = np.linspace(0, T, 500)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0], [gamma, 0.0, 0.0], [0.0, 0.0, 0.0]])


A0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

# computed solution
N = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, 0.0))


plt.figure()
for c in ["A11", "A12", "A13", "A22", "A23", "A33"]:
    i = COMPS[c]
    plt.subplot("33" + str(i + 1))
    p = plt.plot(t, N[:, i])
    plt.xlabel("Time $t$ in s")
    plt.ylabel(c)
    plt.ylim([-1, 1])
plt.tight_layout()
plt.show()
