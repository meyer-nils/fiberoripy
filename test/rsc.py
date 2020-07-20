# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow.

The result should match Figure 5 in
Wang, Jin, O’Gara, John F., Tucker, Charles L: "An objective model for slow
orientation kinetics in concentrated fiber suspensions: Theory and rheological
evidence", Journal of Rheology, 2008: DOI: 10.1122/1.2946437.
"""
import matplotlib.pyplot as plt
import numpy as np
from fiberpy.orientation import rsc_ode
from scipy.integrate import odeint

xi = 1.0
# time steps
t = np.linspace(0, 400, 500)


A0 = np.array([[0.333, 0.0, 0.0], [0.0, 0.333, 0.0], [0.0, 0.0, 0.333]])


fig, axes = plt.subplots(3, 1, figsize=(4, 6))

for ax, ratio in zip(axes, [0.1, 0.12, 0.2]):

    def L(t):
        """Velocity gradient."""
        return np.array(
            [[-ratio, 0.0, 1.0], [0.0, ratio, 0.0], [0.0, 0.0, 0.0]]
        )

    # computed solution
    N_rsc = odeint(rsc_ode, A0.ravel(), t, args=(xi, L, 0.01, 0.1))

    ax.plot(t, N_rsc[:, 0], label="N11 RSC")
    ax.plot(t, N_rsc[:, 4], label="N22 RSC")
    ax.plot(t, N_rsc[:, 2], label="N13 RSC")

    ax.set_xlabel("Time $t$ in s")
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend()
plt.tight_layout()
plt.show()
