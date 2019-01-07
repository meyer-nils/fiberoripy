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

t = np.linspace(0, T, 500)
data_list = []
for i in range(10):
    data_list.append(
        np.loadtxt("data/volfrac10/%d/N.csv" % (i+1), delimiter=','))

# array with shape: N_simulation, time_step, data_index .
# data index 0 is time, others are orientation tensor components
data = np.array(data_list)


def D(t):
    """Symmetric strain rate tensor."""
    return np.array([[0.0, G/2, 0.0],
                     [G/2, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def W(t):
    """Skew-symmetric strain rate tensor."""
    return np.array([[0.0, G/2, 0.0],
                     [-G/2, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def compute_solution(p):
    c = p[0]
    kappa = p[1]
    sol = odeint(rsc_ode, A0.ravel(), t, args=(ar, D, W, c, kappa))
    return sol


def compute_mean(t):
    t_sim = np.mean(data[:, :, 0], axis=0)
    N_sim = np.mean(data[:, :, 1:10], axis=0)
    sim = interp1d(t_sim, N_sim, axis=0)
    return sim(t)


def compute_std(t):
    t_sim = np.mean(data[:, :, 0], axis=0)
    N_sim = np.std(data[:, :, 1:10], axis=0)
    sim = interp1d(t_sim, N_sim, axis=0)
    return sim(t)


def error(p):
    """Compute the scalar error between dataset and solution."""
    sol = compute_solution(p)
    sim = interp1d(data[:, :, 0], data[:, :, 1:10], axis=1)
    print(sol.shape)
    print(sim.shape)
    return np.linalg.norm(sol-sim)


def mean_error(p):
    """Compute the error between dataset mean and solution."""
    sol = compute_solution(p)
    mean = compute_mean(t)
    return np.linalg.norm(sol-mean, axis=1)


p0 = [0.0, 1.0]
A = compute_solution(p0)
mean = compute_mean(t)
std = compute_std(t)
# p_min = least_squares(mean_error, p0, verbose=2)


labels = ["A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"]

for i in range(9):
    plt.subplot("33"+str(i+1))
    p = plt.plot(t, A[:, i], t, mean[:, i])
    color = p[1].get_color()
    plt.fill_between(t, mean[:, i] + std[:, i], mean[:, i] - std[:, i],
                     color=color, alpha=0.3)
    plt.xlabel("Time in s")
    plt.ylabel(labels[i])
    plt.ylim([-1, 1])

plt.suptitle("Folgar Tucker and IBOF Closure")
plt.tight_layout()
plt.show()

# plt.plot(t, mean_error(p0))
# plt.show()
