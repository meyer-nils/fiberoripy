"""Testing re-orientation in shearflow."""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

from fiberpy.orientation import (rsc_ode, iard_ode, maier_saupe_ode,
                                 folgar_tucker_ode,
                                 get_zhang_aspect_ratio)

from scipy.integrate import odeint


volfrac = ["10", "30"]
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

tt = np.linspace(0, 50, 100)

# load simulation data
N = []
t = []
N_rsc = []
N_longterm = []
for v in volfrac:
    data = np.loadtxt("data/volfrac%s/%d/rve_output/N.csv" % (v, 7),
                      delimiter=',')
    t.append(data[:, 0])
    N.append(data[:, 1:10])
    N_rsc.append(odeint(rsc_ode, data[0, 1:10], data[:, 0],
                        args=(xi, L) + tuple(p_opt[v])))
    N_longterm.append(odeint(rsc_ode, data[0, 1:10], tt,
                             args=(xi, L) + tuple(p_opt[v])))
N = np.array(N)
t = np.array(t)
N_rsc = np.array(N_rsc)
N_longterm = np.array(N_longterm)

labels = ["$A_{11}$", "$A_{12}$", "$A_{13}$",
          "$A_{21}$", "$A_{22}$", "$A_{23}$",
          "$A_{31}$", "$A_{32}$", "$A_{33}$"]

subplots = [0, 4, 8, 1]

legend_list = [r"RSC (10 \%)",
               r"SPH (10 \%)",
               r"RSC (30 \%)",
               r"SPH (30 \%)"]

plt.figure(figsize=(12, 3))
for j, i in enumerate(subplots):
    plt.subplot("14"+str(j+1))
    for k, v in enumerate(volfrac):
        p = plt.plot(t[k, :], N_rsc[k, :, i], '-k',
                     t[k, :], N[k, :, i])
    plt.plot(tt[:50], np.ones_like(tt[:50])*N_longterm[0, -1, i], ':k')
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
