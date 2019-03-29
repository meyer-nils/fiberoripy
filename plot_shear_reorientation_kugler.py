"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import folgar_tucker_ode, get_equivalent_aspect_ratio

# Folgar-Tucker constants
C0 = 0
C1 = 0.001
C2 = 0.01
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
sim_rigid = np.loadtxt("data/N_rigid.csv", delimiter=',', skiprows=1)
A11_kugler = np.loadtxt("data/A11_kugler.csv", delimiter=',')
A22_kugler = np.loadtxt("data/A22_kugler.csv", delimiter=',')
A33_kugler = np.loadtxt("data/A33_kugler.csv", delimiter=',')


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, gamma],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = sim_developed[0, 1:10]
# computed solution
N0 = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, C0))
N1 = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, C1))
N2 = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, C2))

plt.subplot(1, 3, 1)
plt.plot(sim_developed[:, 0], sim_developed[:, 1], label='SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 1], label='SPH (rest)')
plt.plot(sim_rigid[:, 0], sim_rigid[:, 1], label='SPH (rigid)')
plt.plot(A11_kugler[:, 0], A11_kugler[:, 1], label='Kugler')
plt.plot(t, N0[:, 0], '-k', label='Folgar-Tucker with $C_i$=%.4f' % C0)
plt.plot(t, N1[:, 0], '--k', label='Folgar-Tucker with $C_i$=%.4f' % C1)
plt.plot(t, N2[:, 0], ':k', label='Folgar-Tucker with $C_i$=%.4f' % C2)
plt.xlabel("Time $t$ in s")
plt.title("A11")
plt.ylim([0, 1])
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(sim_developed[:, 0], sim_developed[:, 5], label='SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 5], label='SPH (rest)')
plt.plot(sim_rigid[:, 0], sim_rigid[:, 5], label='SPH (rigid)')
plt.plot(A22_kugler[:, 0], A22_kugler[:, 1], label='Kugler')
plt.plot(t, N0[:, 4], '-k', label='Folgar-Tucker with $C_i$=%.4f' % C0)
plt.plot(t, N1[:, 4], '--k', label='Folgar-Tucker with $C_i$=%.4f' % C1)
plt.plot(t, N2[:, 4], ':k', label='Folgar-Tucker with $C_i$=%.4f' % C2)
plt.xlabel("Time $t$ in s")
plt.title("A22")
plt.ylim([0, 1])
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(sim_developed[:, 0], sim_developed[:, 9], label='SPH (developed)')
plt.plot(sim_rest[:, 0], sim_rest[:, 9], label='SPH (rest)')
plt.plot(sim_rigid[:, 0], sim_rigid[:, 9], label='SPH (rigid)')
plt.plot(A33_kugler[:, 0], A33_kugler[:, 1], label='Kugler')
plt.plot(t, N0[:, 8], '-k', label='Folgar-Tucker with $C_i$=%.4f' % C0)
plt.plot(t, N1[:, 8], '--k', label='Folgar-Tucker with $C_i$=%.4f' % C1)
plt.plot(t, N2[:, 8], ':k', label='Folgar-Tucker with $C_i$=%.4f' % C2)
plt.xlabel("Time $t$ in s")
plt.title("A33")
plt.ylim([0, 1])
plt.grid()

plt.legend()
plt.tight_layout()
plt.show()
