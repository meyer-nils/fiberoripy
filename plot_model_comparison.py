"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import rsc_ode, jeffery_ode, folgar_tucker_ode

xi = 1.0
# time steps
t = np.linspace(0, 100, 500)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 3.3, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = np.array([[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]])

# computed solution
N_jef = odeint(jeffery_ode, A0.ravel(), t, args=(xi, L))
N_ft = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, 0.01))
N_rsc = odeint(rsc_ode, A0.ravel(), t, args=(xi, L, 0.01, 0.1))


plt.plot(t, N_jef[:, 0], '-r', label='N11 Jeffery')
plt.plot(t, N_ft[:, 0], '--r', label='N11 Folgar-Tucker')
plt.plot(t, N_rsc[:, 0], ':r', label='N11 RSC')

plt.plot(t, N_jef[:, 1], '-b', label='N12 Jeffery')
plt.plot(t, N_ft[:, 1], '--b', label='N12 Folgar-Tucker')
plt.plot(t, N_rsc[:, 1], ':b', label='N12 RSC')

plt.plot(t, N_jef[:, 4], '-g', label='N22 Jeffery')
plt.plot(t, N_ft[:, 4], '--g', label='N22 Folgar-Tucker')
plt.plot(t, N_rsc[:, 4], ':g', label='N22 RSC')
plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.tight_layout()
plt.show()
