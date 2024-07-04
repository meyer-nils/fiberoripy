#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing the closures."""
import numpy as np
import pytest


def get_test_tensors():
    """Generate a random board and key position."""
    a_iso = 1.0 / 3.0 * np.eye(3)
    a_uni1 = np.zeros((3, 3))
    a_uni2 = np.zeros((3, 3))
    a_uni3 = np.zeros((3, 3))
    a_uni1[0, 0] = 1.0
    a_uni2[1, 1] = 1.0
    a_uni3[2, 2] = 1.0
    rands = []
    for a in range(10):
        a_rand = np.random.rand(3, 3)
        a_rand_symm = np.dot(a_rand.T, a_rand) + np.eye(3)
        rands.append(a_rand_symm / np.trace(a_rand_symm))

    return [a_iso, a_uni1, a_uni2, a_uni3] + rands


@pytest.mark.parametrize("a", get_test_tensors())
@pytest.mark.parametrize(
    "type",
    ["IBOF", "LINEAR", "HYBRID", "QUADRATIC", "ORF", "SIQ", "SIHYB"],
)
def test_reduction(a, type):
    """Test contraction property."""
    from fiberoripy.closures import compute_closure

    A = compute_closure(a, type)
    a_contract = np.einsum("ijkk", A)
    assert np.allclose(a_contract, a, atol=1e-5)


@pytest.mark.parametrize("a", get_test_tensors())
@pytest.mark.parametrize(
    "type",
    ["IBOF", "LINEAR", "HYBRID", "QUADRATIC", "ORF", "SIQ", "SIHYB", "SQC"],
)
def test_contraction(a, type):
    """Test first contraction property."""
    from fiberoripy.closures import compute_closure

    A = compute_closure(a, type)
    a_contract = np.einsum("ii", np.einsum("ijkk", A))
    assert a_contract == pytest.approx(1.0, 0.0001)
