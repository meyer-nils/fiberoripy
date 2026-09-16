import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fiberoripy.closures import IBOF_closure
from fiberoripy.fit import compute_error, fit_optimal_params
from fiberoripy.orientation import integrate_ori_ode, rsc_ode

XI = 1.0
KEYS = ["Ci", "kappa"]
TRUTH = [0.02, 0.4]
BOUNDS = ([0.0, 0.0], [0.1, 1.0])


def L(t):
    """Velocity gradient of a simple shear flow."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.fixture
def t():
    """Output times."""
    return np.linspace(0.0, 30.0, 40)


@pytest.fixture
def reference(t):
    """Orientation evolution generated from the known parameters in `TRUTH`."""
    kwargs = dict(zip(KEYS, TRUTH, strict=True))
    kwargs["xi"] = XI
    return solve_ivp(
        integrate_ori_ode,
        (t.min(), t.max()),
        (np.eye(3) / 3.0).ravel(),
        t_eval=t,
        args=(L, IBOF_closure, rsc_ode, kwargs),
    ).y.T


def test_error_vanishes_at_the_true_parameters(t, reference):
    """The reference was generated with these parameters, so the error is zero."""
    error = compute_error(TRUTH, KEYS, t, reference, rsc_ode, XI, L)
    assert error.shape == (len(t),)
    assert np.allclose(error, 0.0)


def test_error_grows_when_the_parameters_are_wrong(t, reference):
    """A worse parameter set must give a larger residual."""
    good = np.linalg.norm(compute_error(TRUTH, KEYS, t, reference, rsc_ode, XI, L))
    bad = np.linalg.norm(compute_error([0.08, 0.9], KEYS, t, reference, rsc_ode, XI, L))
    assert bad > good


def test_fit_recovers_the_parameters_it_was_generated_with(t, reference):
    """A round trip through the optimizer must return the original parameters."""
    p_opt, solution, message = fit_optimal_params(
        t, reference, rsc_ode, XI, L, KEYS, [0.01, 0.8], BOUNDS
    )
    assert np.allclose(p_opt, TRUTH, atol=1e-3)
    assert solution.shape == (len(t), 9)
    assert isinstance(message, str)


def test_mismatched_parameter_names_and_values_are_rejected(t, reference):
    """Silently fitting fewer parameters than asked for would be a trap."""
    with pytest.raises(ValueError):
        compute_error([0.02], KEYS, t, reference, rsc_ode, XI, L)
