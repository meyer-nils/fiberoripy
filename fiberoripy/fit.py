# -*- coding: utf-8 -*-
"""Fit parameters to model."""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares


def compute_error(params, t, reference, ode, xi, L):
    """Compute the error between dataset mean and solution."""
    A0 = reference[0, :]
    mask = [0, 1, 2, 4, 5, 8]
    sol = odeint(ode, A0, t, args=(xi, L) + tuple(params))
    diff = np.abs(sol[:, mask] - reference[:, mask])
    return np.linalg.norm(diff, axis=1)


def fit_optimal_params(t, reference, ode, xi, L, params, bounds):
    """Apply least-squares optimization to find optimal parameters.

    Parameters
    ----------
    t : numpy array
        Desired output times.
    reference : numpy array
        Reference solution for each time step to be fitted.
    ode : function handle
        function that describes the fiber orientation model, e.g. 'folgar_tucker_ode'.
    xi : float
        shape_factor
    L : function hanlde
        Velocity gradient as function of time.
    params : list of float
        parameters passed for optimization.
    bounds : tuple of list of floats
        Upper and lower bounds of the parameters for optimization.

    Returns
    -------
    list
        Optimal parameter set, resulting fiber orientation evolution, optimization message.

    """
    opt = least_squares(
        compute_error,
        params,
        bounds=bounds,
        verbose=2,
        args=[t, reference, ode, xi, L],
    )

    A0 = reference[0, :]
    A = odeint(ode, A0, t, args=(xi, L) + tuple(opt.x))
    return [opt.x, A, opt.message]
