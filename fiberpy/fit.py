# -*- coding: utf-8 -*-
"""Fit parameters to model."""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares


def compute_error(params, t, reference, ode, xi, L):
    """Compute the error between dataset mean and solution."""
    A0 = reference[0, :]
    sol = odeint(ode, A0, t, args=(xi, L) + tuple(params))
    return np.linalg.norm(sol - reference, axis=1)


def fit_optimal_params(t, reference, ode, xi, L, params, bounds):
    """Apply least-squares optimization to find optimal parameters.

    Args
    ----
        reference: Reference solution for each time step to be fitted.

        ode: function that describes the fiber orientation model

        xi (double): shape factor

        L: function that describes the velocity gradient

    """
    opt = least_squares(
        compute_error,
        params,
        bounds=bounds,
        verbose=2,
        args=[t, reference, ode, xi, L],
    )

    A0 = reference[0, :]
    N = odeint(ode, A0, t, args=(xi, L) + tuple(opt.x))
    return [opt.x, N]
