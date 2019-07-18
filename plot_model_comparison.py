"""Testing re-orientation in shearflow."""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from fiberpy.orientation import (ard_rsc_ode, rsc_ode, jeffery_ode,
                                 folgar_tucker_ode, iard_ode, maier_saupe_ode,
                                 iardrpr_ode,
                                 get_zhang_aspect_ratio, get_cox_aspect_ratio)

ar = 5.0
xi = (ar**2 - 1)/(ar**2 + 1)
# time steps
t = np.linspace(0, 10, 500)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0],
                     [3.3, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


A0 = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])

# computed solution
N_jef = odeint(jeffery_ode, A0.ravel(), t, args=(xi, L))
N_ft = odeint(folgar_tucker_ode, A0.ravel(), t, args=(xi, L, 0.01))
N_rsc = odeint(rsc_ode, A0.ravel(), t, args=(xi, L, 0.01, 0.0))
N_ard_rsc = odeint(ard_rsc_ode, A0.ravel(), t, args=(xi, L, 0.01, 1))
N_iard = odeint(iard_ode, A0.ravel(), t, args=(xi, L, 0.01, 0))
N_iardrpr = odeint(iardrpr_ode, A0.ravel(), t, args=(xi, L, 0.01, 0, 0.0))
N_ms = odeint(maier_saupe_ode, A0.ravel(), t, args=(xi, L, 0.01, 0))


plt.plot(t, N_jef[:, 0], '-r', label='N11 Jeffery')
plt.plot(t, N_ft[:, 0], '--r', label='N11 Folgar-Tucker')
plt.plot(t, N_rsc[:, 0], ':r', label='N11 RSC')
plt.plot(t, N_ard_rsc[:, 0], '-.r', label='N11 ARD-RSC')
plt.plot(t, N_iard[:, 0], '*r', label='N11 iARD')
plt.plot(t, N_iardrpr[:, 0], '.r', label='N11 iARD-RPR')
plt.plot(t, N_ms[:, 0], '+r', label='N11 Maier-Saupe')

plt.plot(t, N_jef[:, 1], '-b', label='N12 Jeffery')
plt.plot(t, N_ft[:, 1], '--b', label='N12 Folgar-Tucker')
plt.plot(t, N_rsc[:, 1], ':b', label='N12 RSC')
plt.plot(t, N_ard_rsc[:, 1], '-.b', label='N12 ARD-RSC')
plt.plot(t, N_iard[:, 1], '*b', label='N12 iARD')
plt.plot(t, N_iardrpr[:, 1], '.b', label='N12 iARD-RPR')
plt.plot(t, N_ms[:, 1], '+b', label='N12 Maier-Saupe')

plt.plot(t, N_jef[:, 4], '-g', label='N22 Jeffery')
plt.plot(t, N_ft[:, 4], '--g', label='N22 Folgar-Tucker')
plt.plot(t, N_rsc[:, 4], ':g', label='N22 RSC')
plt.plot(t, N_ard_rsc[:, 4], '-.g', label='N22 ARD-RSC')
plt.plot(t, N_iard[:, 4], '*g', label='N12 iARD')
plt.plot(t, N_iardrpr[:, 4], '.g', label='N12 iARD-RPR')
plt.plot(t, N_ms[:, 4], '+g', label='N12 Maier-Saupe')

plt.xlabel("Time $t$ in s")
plt.ylim([0, 1])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
