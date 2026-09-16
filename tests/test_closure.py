import numpy as np
import pytest

# Seeded so that the parametrised test tensors are identical on every run.
RNG = np.random.default_rng(20240914)

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
    phi_rand = RNG.uniform(0.0, 2.0 * np.pi, N)
    theta_rand = np.arccos(1.0 - 2.0 * RNG.uniform(0.0, 1.0, N))
    fiber_vecs = np.array(
        [
            np.cos(phi_rand) * np.sin(theta_rand),
            np.sin(phi_rand) * np.sin(theta_rand),
            np.cos(theta_rand),
        ]
    ).T
    return fiber_vecs


def get_test_tensor_FOT2(N=100):
    """Create a random second order fiber orientation tensor.

    Parameters
    ----------
    N : int, optional
        Number of random fiber vectors. The default is 100.

    Returns
    -------
    3x3 numpy array
        Second order fiber orientation tensor.

    """
    fibervecs = create_random_spherical_distribution(N)
    a = np.einsum(
        "Ii,Ij->ij",
        fibervecs,
        fibervecs,
    ) / len(fibervecs)
    return a


def get_test_tensor_FOT4(N=100):
    """Create a random fourth order fiber orientation tensor.

    Parameters
    ----------
    N : int, optional
        Number of random fiber vectors. The default is 100.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.

    """
    fibervecs = create_random_spherical_distribution(N)
    A = np.einsum(
        "Ii,Ij,Ik,Il->ijkl", fibervecs, fibervecs, fibervecs, fibervecs
    ) / len(fibervecs)
    return A


def get_isotropic_FOT4():
    """Generate the isotropic fourth order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Isotropic fourth order fiber orientation tensor.

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
    return [a_iso, a_uni1, a_uni2, a_uni3, *rands]


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

    return [A_iso, A_uni1, A_uni2, A_uni3, *rands]


@pytest.mark.parametrize("a", get_test_tensors())
@pytest.mark.parametrize(
    "type",
    ["IBOF", "LINEAR", "HYBRID", "QUADRATIC", "ORF", "SIQ", "SIHYB"],
)
def test_reduction_FOT2(a, type):
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
def test_contraction_FOT2(a, type):
    """Test first contraction property."""
    from fiberoripy.closures import compute_closure

    A = compute_closure(a, type)
    a_contract = np.einsum("ii", np.einsum("ijkk", A))
    assert a_contract == pytest.approx(1.0, 0.0001)


@pytest.mark.parametrize("A", get_test_tensors_FOT4())
@pytest.mark.parametrize(
    "type",
    ["QUADRATIC", "LINEAR", "HYBRID"],
)
def test_reduction_FOT4(A, type):
    """Test contraction property."""
    from fiberoripy.closures import compute_closure

    A6 = compute_closure(A, type)
    A_contract = np.einsum("ijklmm", A6)
    assert np.allclose(A_contract, A, atol=1e-5)


@pytest.mark.parametrize("A", get_test_tensors_FOT4())
@pytest.mark.parametrize(
    "type",
    ["QUADRATIC", "LINEAR", "HYBRID"],
)
def test_contraction_FOT4(A, type):
    """Test first contraction property."""
    from fiberoripy.closures import compute_closure

    A6 = compute_closure(A, type)
    A_contract = np.einsum("iikkmm", A6)
    assert A_contract == pytest.approx(1.0, 0.0001)


@pytest.mark.parametrize(
    "type",
    [
        "IBOF",
        "LINEAR",
        "HYBRID",
        "QUADRATIC",
        "ORF",
        "ORW",
        "ORW3",
        "SIQ",
        "SIHYB",
        "SQC",
    ],
)
def test_batch_FOT2(type):
    """A batched call must equal the individual results, stacked."""
    from fiberoripy.closures import compute_closure

    tensors = get_test_tensors()
    batched = compute_closure(np.array(tensors), type)
    individual = np.array([compute_closure(a, type) for a in tensors])

    assert batched.shape == (len(tensors), 3, 3, 3, 3)
    assert np.allclose(batched, individual, atol=1e-10)


@pytest.mark.parametrize(
    "type",
    ["QUADRATIC", "LINEAR", "HYBRID"],
)
def test_batch_FOT4(type):
    """A batched call must equal the individual results, stacked."""
    from fiberoripy.closures import compute_closure

    tensors = get_test_tensors_FOT4()[:4]
    batched = compute_closure(np.array(tensors), type)
    individual = np.array([compute_closure(A, type) for A in tensors])

    assert batched.shape == (len(tensors), 3, 3, 3, 3, 3, 3)
    assert np.allclose(batched, individual, atol=1e-10)


def test_get_random_tensor_pair_returns_a_consistent_pair():
    """The fourth order tensor must contract to the second order one."""
    from fiberoripy.closures import get_random_tensor_pair

    a, A = get_random_tensor_pair(N=500, rng=np.random.default_rng(0))
    assert a.shape == (3, 3) and A.shape == (3, 3, 3, 3)
    assert np.trace(a) == pytest.approx(1.0)
    assert np.allclose(np.einsum("ijkk", A), a)
    assert np.allclose(a, a.T)


def test_get_random_tensor_pair_is_reproducible():
    """Passing a seeded generator must make the sample repeatable."""
    from fiberoripy.closures import get_random_tensor_pair

    first = get_random_tensor_pair(N=50, rng=np.random.default_rng(7))
    second = get_random_tensor_pair(N=50, rng=np.random.default_rng(7))
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_get_random_tensor_pair_follows_its_seed_tensor():
    """A uniaxial seed must concentrate the sampled directions on that axis."""
    from fiberoripy.closures import get_random_tensor_pair

    seed = np.diag([1.0, 0.05, 0.05])
    a, _ = get_random_tensor_pair(seed=seed, N=2000, rng=np.random.default_rng(1))
    assert a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]


@pytest.mark.parametrize(
    "tensor, closure, match",
    [
        (np.eye(3) / 3.0, "NOPE", "2nd-order"),
        (np.zeros((3, 3, 3, 3)), "NOPE", "4th-order"),
    ],
)
def test_compute_closure_rejects_unknown_names(tensor, closure, match):
    """An unrecognised closure name must not fall through to None."""
    from fiberoripy.closures import compute_closure

    with pytest.raises(ValueError, match=match):
        compute_closure(tensor, closure)


def test_compute_closure_rejects_an_unusable_shape():
    """Only 3x3 and 3x3x3x3 trailing dimensions can be closed."""
    from fiberoripy.closures import compute_closure

    with pytest.raises(ValueError, match="Expected input shape"):
        compute_closure(np.zeros((4, 4)))


@pytest.mark.parametrize(
    "func, shape",
    [("assert_fot_properties", (2, 2)), ("assert_fot4_properties", (3, 3))],
)
def test_shape_assertions_report_the_offending_shape(func, shape):
    """The error message has to name what was passed in."""
    import fiberoripy.closures as closures

    with pytest.raises(ValueError, match="Expected input shape"):
        getattr(closures, func)(np.zeros(shape))


@pytest.mark.parametrize("type", ["SIQ", "SIHYB"])
def test_implicit_closures_report_non_convergence(type):
    """The Newton iteration must fail loudly rather than return a wrong tensor."""
    from fiberoripy.closures import (
        compute_closure_FOT2,
        implicit_hybrid_closure,
        symmetric_implicit_closure,
    )

    func = symmetric_implicit_closure if type == "SIQ" else implicit_hybrid_closure
    with pytest.raises(ValueError, match="did not converge"):
        func(np.diag([0.6, 0.3, 0.1]), n_iter_newton=0)
    assert compute_closure_FOT2(np.diag([0.6, 0.3, 0.1]), type) is not None


@pytest.mark.parametrize(
    "func, tensor, match",
    [
        ("compute_closure_FOT2", np.eye(3) / 3.0, "2nd-order"),
        ("compute_closure_FOT4", np.zeros((3, 3, 3, 3)), "4th-order"),
    ],
)
def test_dispatch_helpers_reject_unknown_names(func, tensor, match):
    """The helpers are public, so they guard their own input too."""
    import fiberoripy.closures as closures

    with pytest.raises(ValueError, match=match):
        getattr(closures, func)(tensor, "NOPE")


def test_get_random_tensor_pair_works_without_an_explicit_generator():
    """Omitting `rng` must still produce a valid tensor pair."""
    from fiberoripy.closures import get_random_tensor_pair

    a, A = get_random_tensor_pair(N=200)
    assert np.trace(a) == pytest.approx(1.0)
    assert np.allclose(np.einsum("ijkk", A), a)
