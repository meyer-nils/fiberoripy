"""Testing re-orientation in shearflow."""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

from fiberpy.orientation import (rsc_ode, iard_ode, maier_saupe_ode,
                                 folgar_tucker_ode,
                                 get_zhang_aspect_ratio)
from fiberpy.fit import fit_optimal_params

from scipy.integrate import odeint


volfrac = "30"
ar = 5.0
xi = (ar**2 - 1)/(ar**2 + 1)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0],
                     [3.3, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


# load simulation data
data_list = []
for i in range(5):
    data_list.append(
        np.loadtxt("data/volfrac%s/%d/rve_output/N.csv" % (volfrac, i+1),
                   delimiter=','))

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

p0 = [0.01, 0.5]
p_opt, N_rsc = fit_optimal_params(t, mean, rsc_ode, xi, L,
                                  p0, ([0.0, 0.0], [0.1, 1.0]))
print("Optimal parameters for RSC: " + str(p_opt))
# 1% ->  0.00262122  0.85292257
# 10% -> 0.00515386  0.76019708
# 30% -> 0.01103383  0.58449658

# p0 = [0.0, 0.0]
# p_opt, N_iard = fit_optimal_params(t, mean, iard_ode, xi, L,
#                                    p0, ([0.0, 0.0], [0.1, 1.0]))
# print("Optimal parameters for iARD: " + str(p_opt))
# # 1% -> 1.97419337e-03   9.98953936e-11
# # 10% -> 3.37341057e-03   9.98750917e-11
# # 30% -> 6.64142473e-03   9.95918971e-11

# p0 = [0.0, 0.0]
# p_opt, N_ms = fit_optimal_params(t, mean, maier_saupe_ode, xi, L,
#                                  p0, ([0.0, 0.0], [0.1, 0.8]))
# print("Optimal parameters for Maier-Saupe: " + str(p_opt))
# # 1% -> 0.00235344  0.03221782
# # 10% -> 0.00409046  0.03973743
# # 30% -> 0.01098597  0.15316572

labels = ["$A_{11}$", "$A_{12}$", "$A_{13}$",
          "$A_{21}$", "$A_{22}$", "$A_{23}$",
          "$A_{31}$", "$A_{32}$", "$A_{33}$"]

subplots = [0, 4, 8, 1]

legend_list = ["RSC",
               "SPH simulation"]

plt.figure(figsize=(12, 3))
for j, i in enumerate(subplots):
    plt.subplot("14"+str(j+1))
    p = plt.plot(
                 # t, N_ft[:, i],
                 t, N_rsc[:, i],
                 # t, N_iard[:, i],
                 # t, N_ms[:, i],
                 t, mean[:, i])
    color = p[1].get_color()
    # for d in range(len(data_list)):
    #     p = plt.plot(t, data[d, :, i+1], "k")
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
tikz_save('volfrac%s.tex' % volfrac, figurewidth='5cm')
plt.show()
