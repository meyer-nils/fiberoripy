"""Testing re-orientation in shearflow."""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

from fiberpy.orientation import (rsc_ode, get_equivalent_aspect_ratio)
from fiberpy.fit import fit_optimal_params


volfrac = 40
ar = get_equivalent_aspect_ratio(10.0)
xi = (ar**2 - 1)/(ar**2 + 1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 3.3, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


# load simulation data
data_list = []
for i in range(10):
    data_list.append(
        np.loadtxt("data/volfrac%d/%d/N.csv" % (volfrac, i+1), delimiter=','))

# array with shape: N_simulation, time_step, data_index .
# data index 0 is time, others are orientation tensor components
data = np.array(data_list)

# Compute mean values of simulation result at given times t.
t = np.nanmean(data[:, :, 0], axis=0)
mean = np.nanmean(data[:, :, 1:10], axis=0)
std = np.std(data[:, :, 1:10], axis=0)

p0 = [0.0, 1.0]

p_opt, N_rsc = fit_optimal_params(t, mean, rsc_ode, xi, L,
                                  p0, ([0.0, 0.0], [1.0, 1.0]))
# p_opt = np.array([3.19232144e-05, 9.99691868e-01])  # volfrac 1
# p_opt = np.array([5.96691495e-04, 1.0])  # volfrac 4
# p_opt = np.array([0.00268519, 0.65202155])  # volfrac 10
# p_opt = np.array([0.00309768, 0.99992701])  # volfrac 40
print(p_opt)


labels = ["$A_{11}$", "$A_{12}$", "$A_{13}$",
          "$A_{21}$", "$A_{22}$", "$A_{23}$",
          "$A_{31}$", "$A_{32}$", "$A_{33}$"]

subplots = [0, 4, 8, 1]

legend_list = ["RSC model", "SPH simulation"]

plt.figure(figsize=(12, 3))
for j, i in enumerate(subplots):
    plt.subplot("14"+str(j+1))
    p = plt.plot(t, N_rsc[:, i], t, mean[:, i])
    color = p[1].get_color()
    plt.fill_between(t, mean[:, i] + std[:, i], mean[:, i] - std[:, i],
                     color=color, alpha=0.3)
    plt.xlabel("Time $t$ in s")
    plt.title(labels[i])
    if i % 2 == 0:
        plt.ylim([0, 1])
    else:
        plt.ylim([-1, 1])
plt.legend(legend_list)
plt.tight_layout()

# save tikz figure (width means individual subplot width!)
tikz_save('volfrac%d.tex' % volfrac, figurewidth='5cm')
plt.show()
