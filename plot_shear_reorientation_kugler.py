"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import folgar_tucker_ode, get_equivalent_aspect_ratio

# Folgar-Tucker constnat
Ci = 0.0008
# shear rate
gamma = 1.0
# aspect ratio
ar = 25.0
# equivalent aspect ratio
are = get_equivalent_aspect_ratio(ar)
xi = (are**2 - 1)/(are**2 + 1)
# time steps
t = np.linspace(0, 50, 500)

# load simulation data
sim_developed = np.loadtxt("data/N_developed.csv", delimiter=',', skiprows=1)
sim_rest = np.loadtxt("data/N_rest.csv", delimiter=',', skiprows=1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, gamma],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = sim_developed[0, 1:10]
# computed solution
N = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, Ci))


plt.plot(sim_developed[:, 0], sim_developed[:, 1], '-r',
         label='A11 SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 1], ':r', label='A11 SPH (rest)')
plt.plot(t, N[:, 0], '--r', label='A11 Folgar-Tucker with $C_i$=%.4f' % Ci)

plt.plot(sim_developed[:, 0], sim_developed[:, 5], '-b',
         label='A12 SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 5], ':b', label='A12 SPH (rest)')
plt.plot(t, N[:, 4], '--b', label='A12 Folgar-Tucker with $C_i$=%.4f' % Ci)

plt.plot(sim_developed[:, 0], sim_developed[:, 9], '-g',
         label='A33 SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 9], ':g', label='A22 SPH (rest)')
plt.plot(t, N[:, 8], '--g', label='A33 Folgar-Tucker with $C_i$=%.4f' % Ci)
plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
