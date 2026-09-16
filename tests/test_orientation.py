import numpy as np
import pytest

from fiberoripy.closures import IBOF_closure
from fiberoripy.orientation import (
    ard_rsc_ode,
    folgar_tucker_ode,
    iard_ode,
    iardrpr_ode,
    integrate_ori_ode,
    jeffery_ode,
    maier_saupe_ode,
    mori_tanaka_ode,
    mrd_ode,
    pard_ode,
    pardrpr_ode,
    rsc_ode,
)


@pytest.mark.parametrize(
    "model",
    [
        ard_rsc_ode,
        rsc_ode,
        folgar_tucker_ode,
        maier_saupe_ode,
        iard_ode,
        mrd_ode,
        iardrpr_ode,
        pard_ode,
        pardrpr_ode,
        mori_tanaka_ode,
    ],
)
def test_default_case(model):
    """The default argument set of all functions should yield to Jeffery's solution."""
    from scipy.integrate import solve_ivp

    t = np.linspace(0, 1, 100)

    a0 = 1.0 / 3.0 * np.eye(3)

    def L(t):
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def solve(ode):
        return solve_ivp(
            integrate_ori_ode,
            (t.min(), t.max()),
            a0.ravel(),
            t_eval=t,
            args=(L, IBOF_closure, ode, {"xi": 1.0}),
        ).y

    assert np.allclose(solve(jeffery_ode), solve(model), atol=1e-12)


@pytest.mark.parametrize("model", [ard_rsc_ode, folgar_tucker_ode, jeffery_ode])
def test_quiescent_flow_leaves_the_orientation_unchanged(model):
    """Without a velocity gradient there is no reorientation.

    This also exercises the branch that `ard_rsc_ode` takes when the shear rate
    vanishes and its rotary diffusion tensor cannot be normalised by it.
    """
    a = np.diag([0.6, 0.3, 0.1])
    A = IBOF_closure(a)
    zero = np.zeros((3, 3))

    dadt = model(a, A, zero, zero, xi=1.0)
    assert np.allclose(dadt, 0.0)
