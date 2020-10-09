"""Different orthotropic fitted closure approximations in simple shear flow.

Model should reproduce Figure 6(a) in
Du Hwan Chung and Tai Hun Kwon (2001),
'Improved model of orthotropic closure approximation for flow induced fiber
orientation', Polymer Composites, 22(5), 636-649,
https://doi.org/10.1002/pc.10566
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from fiberoripy.orientation import folgar_tucker_ode


def L(t):
    """Velocity gradient.

    Parameter:
    ---------
    t : float
        time to evaluate

    Return:
    ------
    3x3 numpy array
        velocity gradient

    """
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


# initial fiber orientation state
a0 = 1.0 / 3.0 * np.eye(3)

# fiber aspect ratio
xi = 1.0
# Phenomenological fiber-fiber interaction coefficient
C1 = 0.01

# time points
t = np.linspace(0, 30, 60)

# compute the solution
a_ibof = odeint(
    folgar_tucker_ode,
    a0.ravel(),
    t,
    args=(xi, L, C1, "HYBRID"),
)
a_orf = odeint(
    folgar_tucker_ode,
    a0.ravel(),
    t,
    args=(xi, L, C1, "ORF"),
)
a_orw = odeint(
    folgar_tucker_ode,
    a0.ravel(),
    t,
    args=(xi, L, C1, "ORW"),
)
a_orw3 = odeint(
    folgar_tucker_ode,
    a0.ravel(),
    t,
    args=(xi, L, C1, "ORW3"),
)

# Plotting the results
fig, ax = plt.subplots(nrows=1, ncols=1)

l3 = ax.plot(
    t, a_ibof[:, 0], linestyle="-", label="$a_{11}$ Hybrid", color="b"
)
l4 = ax.plot(t, a_orf[:, 0], linestyle="-", label="$a_{11}$ ORF", color="r")
l5 = ax.plot(t, a_orw[:, 0], linestyle="-", label="$a_{11}$ ORW", color="g")
l6 = ax.plot(t, a_orw3[:, 0], linestyle="-.", label="$a_{11}$ ORW3", color="k")

l7 = ax.plot(
    t, a_ibof[:, 1], linestyle="--", label="$a_{12}$ Hybrid", color="b"
)
l8 = ax.plot(t, a_orf[:, 1], linestyle="--", label="$a_{12}$ ORF", color="r")
l9 = ax.plot(t, a_orw[:, 1], linestyle="--", label="$a_{12}$ ORW", color="g")
l10 = ax.plot(
    t, a_orw3[:, 1], linestyle="-.", label="$a_{12}$ ORW3", color="k"
)
ax.set(ylim=([-0.2, 1]))
ax.grid(b=True, which="major", linestyle="-")
ax.minorticks_on()
ax.grid(b=True, which="minor", linestyle="--", alpha=0.2)
ax.set(xlabel="Time $t$ in $s$")
ax.set(ylabel="$a_{11}, a_{12}$")
ax.legend(loc="center right")
plt.title(r"Simple shear flow, $\xi = 1$, $C_1 = 0,01$")
plt.show()
