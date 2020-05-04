# -*- coding: utf-8 -*-
"""Testing re-orientation in shearflow."""
import os

import numpy as np

import matplotlib.pyplot as plt
import tikzplotlib
from fiberpy.constants import COMPS
from fiberpy.fit import fit_optimal_params
from fiberpy.orientation import get_zhang_aspect_ratio, rsc_ode
from matplotlib import cm

dark2 = cm.get_cmap("Dark2", 2)

ar = get_zhang_aspect_ratio(5)
xi = (ar ** 2 - 1) / (ar ** 2 + 1)
G = 1.0


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, 0.0, 0.0], [G, 0.0, 0.0], [0.0, 0.0, 0.0]])


# load simulation data
data_list = []
rootdir = "data/volfrac30"
file_name = os.path.join(rootdir, "README.md")
pic_name = os.path.join(rootdir, "plot.png")
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == "A.csv":
            name = os.path.join(subdir, file)
            data_list.append(np.loadtxt(name, delimiter=","))
data = np.array(data_list)

# array with shape: N_simulation, time_step, data_index .
# data index 0 is time, others are orientation tensor components
data = np.array(data_list)

# Compute mean values of simulation result at given times t.
t = np.nanmean(data[:, :, 0], axis=0)
mean = np.nanmean(data[:, :, 1:10], axis=0)
std = np.std(data[:, :, 1:10], axis=0)

with open(file_name, "w") as f:
    f.write("# Fitting Results #\n")

p0 = [0.005, 0.7]
p_opt, N_rsc, msg = fit_optimal_params(
    t, mean, rsc_ode, xi, L, p0, ([0.0, 0.0], [0.1, 1.0])
)
with open(file_name, "a") as f:
    f.write("## RSC Model ##\n")
    f.write(msg + "\n")
    f.write("C_i:   %f\n" % p_opt[0])
    f.write("Kappa: %f\n" % p_opt[1])

subplots = ["A11", "A22", "A33", "A12"]

legend_list = ["SPH simulation", "RSC"]

plt.figure(figsize=(12, 3))
for j, c in enumerate(subplots):
    i = COMPS[c]
    plt.subplot("14" + str(j + 1))
    plt.plot(G * t, mean[:, i], color=dark2(0))
    plt.fill_between(
        G * t,
        mean[:, i] + std[:, i],
        mean[:, i] - std[:, i],
        color=dark2(0),
        alpha=0.3,
    )
    plt.xlabel("Strains")
    plt.title(c)
    plt.ylim([-(i % 2), 1])
    plt.plot(G * t, N_rsc[:, i], color=dark2(1))

plt.legend(legend_list)
plt.tight_layout()
plt.savefig(pic_name)

# save tikz figure (width means individual subplot width!)
tikzplotlib.save(pic_name.replace(".png", ".tex"), figurewidth='4.5cm')
plt.show()
