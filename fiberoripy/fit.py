import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from fiberoripy.closures import IBOF_closure
from fiberoripy.orientation import integrate_ori_ode


def compute_error(values, keys, t, reference, ode, xi, L):
    """Compute the error between dataset mean and solution."""
    kwargs = dict(zip(keys, values))
    kwargs["xi"] = xi

    A0 = reference[0, :]
    mask = [0, 1, 2, 4, 5, 8]
    sol = solve_ivp(
        integrate_ori_ode,
        (t.min(), t.max()),
        A0,
        t_eval=t,
        args=(L, IBOF_closure, ode, kwargs),
    )
    diff = np.abs(sol.y[mask, :].T - reference[:, mask])
    return np.linalg.norm(diff, axis=1)


def fit_optimal_params(t, reference, ode, xi, L, keys, values, bounds):
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
    keys : list of str
        parameters names passed for optimization.
    values: list of floats
        Initial guess for the parameters to be optimized.
    bounds : tuple of list of floats
        Upper and lower bounds of the parameters for optimization.

    Returns
    -------
    list
        Optimal parameter set, resulting fiber orientation evolution, optimizer message.

    """

    opt = least_squares(
        compute_error,
        values,
        bounds=bounds,
        verbose=2,
        args=[keys, t, reference, ode, xi, L],
    )

    kwargs = dict(zip(keys, opt.x))
    kwargs["xi"] = xi

    A0 = reference[0, :]
    sol = solve_ivp(
        integrate_ori_ode,
        (t.min(), t.max()),
        A0.ravel(),
        t_eval=t,
        args=(L, IBOF_closure, ode, kwargs),
    )
    return [opt.x, sol.y.T, opt.message]
