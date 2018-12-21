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
data = np.loadtxt("data/N1.csv", delimiter=',')


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


def error(p):
    """Compute the scalar error between dataset and solution."""
    c = p[0]
    kappa = p[1]
    sol = odeint(rsc_ode, A0.ravel(), t, args=(ar, D, W, c, kappa))
    sim = interp1d(data[:, 0], data[:, 1:10], axis=0)
    return np.linalg.norm(sol-sim(t))


def mean_error(p):
    """Compute the error between dataset mean and solution."""
    c = p[0]
    kappa = p[1]
    sol = odeint(rsc_ode, A0.ravel(), t, args=(ar, D, W, c, kappa))
    sim = interp1d(data[:, 0], data[:, 1:10], axis=0)
    return np.linalg.norm(sol-sim(t), axis=1)


p0 = [0.0, 1.0]
p_min = least_squares(error, p0, verbose=2)

plt.plot(t, mean_error(p_min))
plt.show()
