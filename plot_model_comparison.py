# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow.

You may want to compare this to Fig 5 in
Jin Wang, John F. O'Gara, and Charles L. Tucker, 'An objective model
for slow orientation kinetics in concentrated fiber suspensions:
Theory and rheological evidence', Journal of Rheology 52, 1179, 2008.
https://doi.org/10.1122/1.2946437
"""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.constants import COMPS
from fiberpy.orientation import (ard_rsc_ode, folgar_tucker_ode, iard_ode,
                                 iardrpr_ode, jeffery_ode, maier_saupe_ode,
                                 rsc_ode)
from matplotlib import cm
from scipy.integrate import odeint

dark2 = cm.get_cmap("Dark2", 9)

# shape factor
xi = 1.0
# time steps
t = np.linspace(0, 400, 1000)
# strain rate
G = 1.0
E = 0.12


def L(t):
    """Velocity gradient."""
    return np.array([[-E, 0.0, G], [0.0, E, 0.0], [0.0, 0.0, 0.0]])


A0 = 1.0 / 3.0 * np.eye(3)

models = [folgar_tucker_ode, rsc_ode]
labels = ["Folgar-Tucker", "RSC"]
symbols = ["--", ":"]
args = {"Ci": 0.01, "kappa": 0.1}
components = ["A11", "A22", "A13"]

for ode, lbl, smb in zip(models, labels, symbols):
    N = odeint(lambda a, t: ode(a, t, xi, L, **args), A0.ravel(), t)
    for c in components:
        i = COMPS[c]
        plt.plot(t, N[:, i], smb, color=dark2(i), label="%s %s" % (c, lbl))

plt.xlabel("Time t ")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
