"""Testing re-orientation in shearflow."""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

from fiberpy.orientation import (rsc_ode, iard_ode, maier_saupe_ode,
                                 folgar_tucker_ode,
                                 get_zhang_aspect_ratio)
from fiberpy.fit import fit_optimal_params

from scipy.integrate import odeint


volfrac = ["10"]
ar = 5.0
xi = (ar**2 - 1)/(ar**2 + 1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0],
                     [3.3, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


p_opt = {"1": (0.00262122, 0.85292257),
         "10": (0.00515386, 0.76019708),
         "30": (0.01103383, 0.58449658)}

# load simulation data
N = []
t = []
N_rsc = []
for v in volfrac:
    data = np.loadtxt("data/volfrac%s/%d/rve_output/N.csv" % (v, 6),
                      delimiter=',')
    t.append(data[:, 0])
    N.append(data[:, 1:10])
    N_rsc.append(odeint(rsc_ode, data[0, 1:10], data[:, 0],
                        args=(xi, L) + tuple(p_opt[v])))
N = np.array(N)
t = np.array(t)
N_rsc = np.array(N_rsc)

labels = ["$A_{11}$", "$A_{12}$", "$A_{13}$",
          "$A_{21}$", "$A_{22}$", "$A_{23}$",
          "$A_{31}$", "$A_{32}$", "$A_{33}$"]

subplots = [0, 4, 8, 1]

legend_list = ["RSC",
               r"10 \% SPH simulation"]

plt.figure(figsize=(10, 10))
for j, i in enumerate(subplots):
    plt.subplot("41"+str(j+1))
    for k, v in enumerate(volfrac):
        p = plt.plot(t[k, :], N_rsc[k, :, i], '-k',
                     t[k, :], N[k, :, i])
    plt.xlabel("Time $t$ in s")
    plt.title(labels[i])
    if i % 2 == 0:
        plt.ylim([0, 1])
    else:
        plt.ylim([-1, 1])
plt.legend(legend_list)
plt.tight_layout()

# save tikz figure (width means individual subplot width!)
tikz_save('longterm.tex', figurewidth='5cm')
plt.show()
