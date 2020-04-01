# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow."""
import numpy as np

import matplotlib.pyplot as plt
from fiberpy.constants import COMPS
from fiberpy.fit import fit_optimal_params
from fiberpy.orientation import get_zhang_aspect_ratio, rsc_ode
from matplotlib import cm

# import tikzplotlib

dark2 = cm.get_cmap("Dark2", 3)

volfrac = "30"
ar = get_zhang_aspect_ratio(5)
xi = (ar ** 2 - 1) / (ar ** 2 + 1)
G = 1.0


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0], [G, 0.0, 0.0], [0.0, 0.0, 0.0]])


# load simulation data
data_list = []
for i in range(1):
    data_list.append(
        np.loadtxt(
            "data/volfrac%s/%d/rve_output/N.csv" % (volfrac, i + 1),
            delimiter=",",
        )
    )

# array with shape: N_simulation, time_step, data_index .
# data index 0 is time, others are orientation tensor components
data = np.array(data_list)

# Compute mean values of simulation result at given times t.
t = np.nanmean(data[:, :, 0], axis=0)
mean = np.nanmean(data[:, :, 1:10], axis=0)
std = np.std(data[:, :, 1:10], axis=0)

# p0 = [0.0]
# p_opt, N_ft = fit_optimal_params(t, mean, folgar_tucker_ode, xi, L,
#                                  p0, ([0.0], [0.1]))
# print("Optimal parameters for Folgar-Tucker: " + str(p_opt))
# # 1%  -> 0.00197425
# # 10% -> 0.00337344
# # 30% -> 0.00664119

p0 = [0.005, 0.7]
p_opt, N_rsc = fit_optimal_params(
    t, mean, rsc_ode, xi, L, p0, ([0.0, 0.0], [0.1, 1.0])
)
print("Optimal parameters for RSC: " + str(p_opt))
# 1% ->  0.00262122  0.85292257
# 10% -> 0.00515386  0.76019708
# 30% -> 0.01103383  0.58449658

# p0 = [0.01, 0.0, 1.0]
# p_opt, N_iard = fit_optimal_params(t, mean, iardrpr_ode, xi, L,
#                                    p0, ([0.0, 0.0, 0.0], [0.1, 1.0, 1.0]))
# print("Optimal parameters for iARD-RPR: " + str(p_opt))
# 1% -> 1.97419337e-03   9.98953936e-11
# 10% -> 3.37341057e-03   9.98750917e-11
# 30% -> 6.64142473e-03   9.95918971e-11

# p0 = [0.0, 0.0]
# p_opt, N_ms = fit_optimal_params(t, mean, maier_saupe_ode, xi, L,
#                                  p0, ([0.0, 0.0], [0.1, 0.8]))
# print("Optimal parameters for Maier-Saupe: " + str(p_opt))
# # 1% -> 0.00235344  0.03221782
# # 10% -> 0.00409046  0.03973743
# # 30% -> 0.01098597  0.15316572

subplots = ["A11", "A22", "A33", "A12"]

legend_list = ["RSC", "SPH simulation"]

plt.figure(figsize=(12, 3))
for j, c in enumerate(subplots):
    i = COMPS[c]
    plt.subplot("14" + str(j + 1))
    plt.plot(G * t, N_rsc[:, i], color=dark2(0))
    plt.plot(G * t, mean[:, i], color=dark2(1))
    # for d in range(len(data_list)):
    #    p = plt.plot(G * t, data[d, :, i + 1], "k")
    plt.fill_between(
        G * t,
        mean[:, i] + std[:, i],
        mean[:, i] - std[:, i],
        color=dark2(1),
        alpha=0.3,
    )
    plt.xlabel("Strains")
    plt.title(c)
    plt.ylim([-(i % 2), 1])

plt.legend(legend_list)
plt.tight_layout()

# save tikz figure (width means individual subplot width!)
# tikzplotlib.save('volfrac%s.tex' % volfrac, figurewidth='5cm')
plt.show()
