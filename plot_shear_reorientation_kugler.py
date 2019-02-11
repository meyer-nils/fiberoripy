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
sim_data = np.loadtxt("data/N.csv", delimiter=',', skiprows=1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, gamma],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = sim_data[0, 1:10]
# computed solution
N = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, Ci))


plt.plot(sim_data[:, 0], sim_data[:, 1], '-r', label='A11 SPH')
plt.plot(t, N[:, 0], '--r', label='A11 Folgar-Tucker with $C_i$=%.4f' % Ci)

plt.plot(sim_data[:, 0], sim_data[:, 5], '-b', label='A12 SPH')
plt.plot(t, N[:, 4], '--b', label='A12 Folgar-Tucker with $C_i$=%.4f' % Ci)

plt.plot(sim_data[:, 0], sim_data[:, 9], '-g', label='A33 SPH')
plt.plot(t, N[:, 8], '--g', label='A33 Folgar-Tucker with $C_i$=%.4f' % Ci)
plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
