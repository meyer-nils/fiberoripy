#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing the orientation models."""
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
    from scipy.integrate import odeint

    t = np.linspace(0, 1, 100)

    a0 = 1.0 / 3.0 * np.eye(3)

    def L(t):
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    kwargs = {"xi": 1.0}
    a_ref = odeint(
        integrate_ori_ode,
        a0.ravel(),
        t,
        args=(L, IBOF_closure, jeffery_ode, kwargs),
        tfirst=True,
    )
    a_test = odeint(
        integrate_ori_ode,
        a0.ravel(),
        t,
        args=(L, IBOF_closure, model, kwargs),
        tfirst=True,
    )
    assert np.allclose(a_ref, a_test, atol=1e-12)
