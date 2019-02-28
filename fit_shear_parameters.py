"""Testing re-orientation in shearflow."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from fiberpy.orientation import rsc_ode, get_equivalent_aspect_ratio


T = 19
G = 3.3
A0 = np.array([[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]])
ar = get_equivalent_aspect_ratio(10.0)
xi = (ar**2 - 1)/(ar**2 + 1)

t = np.linspace(0, T, 500)

# load simulation data
data_list = []
for i in range(10):
    data_list.append(
        np.loadtxt("data/volfrac10/%d/N.csv" % (i+1), delimiter=','))

# array with shape: N_simulation, time_step, data_index .
# data index 0 is time, others are orientation tensor components
data = np.array(data_list)


def L(t):
    """Velocity gradient."""
    return np.array([[0.0, G, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def compute_rsc_solution(p):
    """Compute solution of ODE given a paramter set p."""
    c = p[0]
    kappa = p[1]
    sol = odeint(rsc_ode, A0.ravel(), t, args=(xi, L, c, kappa))
    return sol


def compute_mean(t):
    """Compute mean values of simulation result at given times t."""
    t_sim = np.nanmean(data[:, :, 0], axis=0)
    N_sim = np.nanmean(data[:, :, 1:10], axis=0)
    sim = interp1d(t_sim, N_sim, axis=0)
    return sim(t)


def compute_std(t):
    """Compute standard deviation of simulation result at given times t."""
    t_sim = np.mean(data[:, :, 0], axis=0)
    N_sim = np.std(data[:, :, 1:10], axis=0)
    sim = interp1d(t_sim, N_sim, axis=0)
    return sim(t)


def mean_rsc_error(p):
    """Compute the error between dataset mean and solution."""
    sol = compute_rsc_solution(p)
    mean = compute_mean(t)
    return np.linalg.norm(sol-mean, axis=1)


p0 = [0.0, 1.0]
mean = compute_mean(t)
std = compute_std(t)
opt = least_squares(mean_rsc_error, p0,
                    bounds=([0.0, 0.0], [1.0, 1.0]),
                    verbose=2,
                    max_nfev=100)
p_opt = opt.x
# p_opt = np.array([[3.19232144e-05 9.99691868e-01])  # volfrac 1
# p_opt = np.array([5.96691495e-04 1.00000000e+00])  # volfrac 4
# p_opt = np.array([0.00268519 0.65202155])  # volfrac 10
# p_opt = np.array([0.00309768 0.99992701])  # volfrac 40
print(p_opt)

N_rsc = compute_rsc_solution(p_opt)


labels = ["$N_{11}$", "$N_{12}$", "$N_{13}$",
          "$N_{21}$", "$N_{22}$", "$N_{23}$",
          "$N_{31}$", "$N_{32}$", "$N_{33}$"]

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
    plt.ylabel(labels[i])
    if i % 2 == 0:
        plt.ylim([0, 1])
    else:
        plt.ylim([-1, 1])
plt.legend(legend_list)
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(t, std)
# plt.title("Standard deviation in simulation dataset")
# plt.show()
#
# plt.figure()
# plt.plot(t, mean_error(p_opt))
# plt.title("Error over time")
# plt.show()
