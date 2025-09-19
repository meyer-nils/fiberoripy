import numpy as np
import pytest

# Helper function to generate test tensors


def create_random_spherical_distribution(N=100):
    """
    Create a random uniform distribution on S2.

    Parameters
    ----------
    N : int, optional
        Number of fibers in the random distribution. The default is 100.

    Returns
    -------
    fiber_vecs : Nx3 Array
        Normalized fiber directions.

    """
    phi_rand = np.random.uniform(0.0, 2.0 * np.pi, N)
    theta_rand = np.arccos(1.0 - 2.0 * np.random.uniform(0.0, 1.0, N))
    fiber_vecs = np.array(
        [
            np.cos(phi_rand) * np.sin(theta_rand),
            np.sin(phi_rand) * np.sin(theta_rand),
            np.cos(theta_rand),
        ]
    ).T
    return fiber_vecs


def get_test_tensor_FOT2(N=100):
    """Create random fiber orientation tensor second order.

    Args:
        N (int, optional): Number of random fiber vectors. Defaults to 100.

    Returns:
        3x3 Array: Fiber orientation tensor second order.
    """
    fibervecs = create_random_spherical_distribution(N)
    a = np.einsum(
        "Ii,Ij->ij",
        fibervecs,
        fibervecs,
    ) / len(fibervecs)
    return a


def get_test_tensor_FOT4(N=100):
    """Create random fiber orientation tensor fourth order.

    Args:
        N (int, optional): Number of random fiber vectors. Defaults to 100.

    Returns:
        3x3x3x3 Array: Fiber orientation tensor fourth order.
    """
    fibervecs = create_random_spherical_distribution(N)
    A = np.einsum(
        "Ii,Ij,Ik,Il->ijkl", fibervecs, fibervecs, fibervecs, fibervecs
    ) / len(fibervecs)
    return A


def get_isotropic_FOT4():
    """Generate the isotropic fiberorientation tensor fourth order.

    Returns:
        3x3x3x3 Array: Isotropic fiberorientation tensor fourth order.
    """
    A_iso = np.array(
        [
            [
                [
                    [0.2, 0.0, 0.0],
                    [0.0, 0.06666667, 0.0],
                    [0.0, 0.0, 0.06666667],
                ],
                [
                    [0.0, 0.06666667, 0.0],
                    [0.06666667, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.06666667],
                    [0.0, 0.0, 0.0],
                    [0.06666667, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.06666667, 0.0],
                    [0.06666667, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.06666667, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.06666667],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.06666667],
                    [0.0, 0.06666667, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.06666667],
                    [0.0, 0.0, 0.0],
                    [0.06666667, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.06666667],
                    [0.0, 0.06666667, 0.0],
                ],
                [
                    [0.06666667, 0.0, 0.0],
                    [0.0, 0.06666667, 0.0],
                    [0.0, 0.0, 0.2],
                ],
            ],
        ]
    )
    return A_iso


def get_test_tensors():
    """Generate a random board and key position."""
    a_iso = 1.0 / 3.0 * np.eye(3)
    a_uni1 = np.zeros((3, 3))
    a_uni2 = np.zeros((3, 3))
    a_uni3 = np.zeros((3, 3))
    a_uni1[0, 0] = 1.0
    a_uni2[1, 1] = 1.0
    a_uni3[2, 2] = 1.0
    rands = [get_test_tensor_FOT2(N=10) for _ in range(10)]
    return [a_iso, a_uni1, a_uni2, a_uni3] + rands


def get_test_tensors_FOT4():
    """Generate a random board and key position."""
    A_iso = get_isotropic_FOT4()
    A_uni1 = np.zeros((3, 3, 3, 3))
    A_uni2 = np.zeros((3, 3, 3, 3))
    A_uni3 = np.zeros((3, 3, 3, 3))
    A_uni1[0, 0, 0, 0] = 1.0
    A_uni2[1, 1, 1, 1] = 1.0
    A_uni3[2, 2, 2, 2] = 1.0
    rands = [get_test_tensor_FOT4(N=10) for _ in range(10)]

    return [A_iso, A_uni1, A_uni2, A_uni3] + rands


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


@pytest.mark.parametrize("A", get_test_tensors_FOT4())
@pytest.mark.parametrize(
    "type",
    ["QUADRATIC", "LINEAR", "HYBRID"],
    # "type", ["IBOF", "LINEAR", "HYBRID", "QUADRATIC", "RANDOM"]
)
def test_reduction_FOT4(A, type):
    """Test contraction property."""
    from fiberoripy.closures import compute_closure_FOT4

    A6 = compute_closure_FOT4(A, type)
    A_contract = np.einsum("ijklmm", A6)
    assert np.allclose(A_contract, A, atol=1e-5)


@pytest.mark.parametrize("A", get_test_tensors_FOT4())
@pytest.mark.parametrize(
    "type",
    ["QUADRATIC", "LINEAR", "HYBRID"],
    # "type", ["IBOF", "LINEAR", "HYBRID", "QUADRATIC", "RANDOM"]
)
def test_contraction_FOT4(A, type):
    """Test first contraction property."""
    from fiberoripy.closures import compute_closure_FOT4

    A6 = compute_closure_FOT4(A, type)
    A_contract = np.einsum("iikkmm", A6)
    assert A_contract == pytest.approx(1.0, 0.0001)
