from itertools import permutations

import numpy as np

fullsym6_permutations = np.array(
    ["".join(perm) for perm in list(permutations("ijklmn"))]
)

_FOT2_CLOSURES = {
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
}

_FOT4_CLOSURES = {
    "LINEAR",
    "HYBRID",
    "QUADRATIC",
}


def compute_closure(a, closure="IBOF"):
    """Compute closure for second or fourth order fiber orientation tensor.

    Parameters
    ----------
    a : np.ndarray (Nx)3x3 or (Nx)3x3x3x3
        Fiber orientation tensor second or fourth order.
    closure : str, optional
        Closure type, by default "IBOF"

    Returns
    -------
    np.ndarray (Nx)3x3x3x3 or (Nx)3x3x3x3x3x3
        Fiber orientation tensor fourth or sixth order.

    Raises
    ------
    ValueError
        If the closure type is not recognized.
    """
    # Identify order from trailing dimensions
    is_4th = a.ndim >= 4 and a.shape[-4:] == (3, 3, 3, 3)
    is_2nd = a.ndim >= 2 and a.shape[-2:] == (3, 3)

    if not (is_2nd or is_4th):
        raise ValueError(
            f"Expected input shape (..., 3, 3) or (..., 3, 3, 3, 3); got {a.shape}."
        )

    # Giving priority to 4th order tensors because otherwise a list of
    # 4th order tensors would also be recognized as a 2nd order tensors.
    # There might be a better way to do this.
    if is_4th:
        if closure not in _FOT4_CLOSURES:
            raise ValueError(f"Unsupported closure for 4th-order tensor: {closure}")
        return compute_closure_FOT4(a, closure)

    if is_2nd:
        if closure not in _FOT2_CLOSURES:
            raise ValueError(f"Unsupported closure for 2nd-order tensor: {closure}")
        return compute_closure_FOT2(a, closure)

    raise ValueError("Closure type not recognized.")


def compute_closure_FOT2(a, closure="IBOF"):
    """Create a fourth order tensor from a second order tensor.
    This is essentially a wrapper around all closures.
    Parameters
    ----------
    a : 3x3 numpy array
        Second order fiber orientation tensor.
    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    """
    if closure == "IBOF":
        return IBOF_closure(a)
    if closure == "HYBRID":
        return hybrid_closure(a)
    if closure == "LINEAR":
        return linear_closure(a)
    if closure == "QUADRATIC":
        return quadratic_closure(a)
    if closure == "ORF":
        return orthotropic_fitted_closures(a, "ORF")
    if closure == "ORW":
        return orthotropic_fitted_closures(a, "ORW")
    if closure == "ORW3":
        return orthotropic_fitted_closures(a, "ORW3")
    if closure == "SIQ":
        return symmetric_implicit_closure(a)
    if closure == "SIHYB":
        return implicit_hybrid_closure(a)
    if closure == "SQC":
        return symmetric_quadratic_closure(a)


def assert_fot_properties(a):
    """Assert properties of second order input tensor.
    Parameters
    ----------
    a : 3x3 numpy array
        Second order fiber orientation tensor.
    """
    # assert symmetry and shape
    assert np.shape(a)[-2:] == (3, 3)
    # assert(A[0, 1] == A[1, 0])
    # assert(A[0, 2] == A[2, 0])
    # assert(A[1, 2] == A[2, 1])


def get_random_tensor_pair(seed=(1.0 / 3.0 * np.eye(3)), N=1000):
    """Sample a random fiber orientation and compute second and fourth order tensors.
    Parameters
    ----------
    seed : 3x3 numpy array
        Second order fiber orientation tensor describing the seed distribution.
    N : int, optional
        number of random fibers. The default is 1000.
    Returns
    -------
    a : 3x3 numpy array
        Second order fiber orientation tensor
    A : 3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    """

    phi = np.random.uniform(-np.pi, np.pi, N)
    costheta = np.random.uniform(-1.0, 1.0, N)
    theta = np.arccos(costheta)

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    p = np.array([x, y, z]).transpose()
    p = np.einsum("ij, Ij -> Ii", seed, p)
    p /= np.linalg.norm(p, axis=1)[:, np.newaxis]

    a = np.einsum("Ii, Ij -> ij", p, p) / N
    A = np.einsum("Ii, Ij, Ik, Il -> ijkl", *4 * (p,)) / N

    return [a, A]


def linear_closure(a):
    """Generate a linear closure.
    Parameters
    ----------
    a : (Mx)3x3 numpy array
        (Array of) Second order fiber orientation tensor.
    Returns
    -------
    A : (Mx)3x3x3x3 numpy array
        (Array of) Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002
    """
    assert_fot_properties(a)

    delta = np.eye(3)
    A = (
        1.0
        / 7.0
        * (
            np.einsum("...ij,kl->...ijkl", a, delta)
            + np.einsum("...ik,jl->...ijkl", a, delta)
            + np.einsum("...il,jk->...ijkl", a, delta)
            + np.einsum("...kl,ij->...ijkl", a, delta)
            + np.einsum("...jl,ik->...ijkl", a, delta)
            + np.einsum("...jk,il->...ijkl", a, delta)
        )
    )
    A -= (
        1.0
        / 35.0
        * (
            np.einsum("ij,kl->ijkl", delta, delta)
            + np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )

    return A


def quadratic_closure(a):
    """Generate a quadratic closure.
    Parameters
    ----------
    a : (Mx)3x3 numpy array
        (Array of) Second order fiber orientation tensor.
    Returns
    -------
    A : (Mx)3x3x3x3 numpy arrayF
        (Array of) Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002
    """
    assert_fot_properties(a)

    return np.einsum("...ij, ...kl -> ...ijkl", a, a)


def hybrid_closure(a):
    """Generate a hybrid closure.
    Parameters
    ----------
    a : (Mx)3x3 numpy array
        (Array of) Second order fiber orientation tensor.
    Returns
    -------
    A : (Mx)3x3x3x3 numpy array
        (Array of) Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002
    """
    assert_fot_properties(a)

    f = 1.0 - 27.0 * np.linalg.det(a)
    A = np.einsum("..., ...ijkl -> ...ijkl", 1.0 - f, linear_closure(a))
    A += np.einsum("..., ...ijkl -> ...ijkl", f, quadratic_closure(a))

    return A


def IBOF_closure(a):
    """Generate IBOF closure.
    Parameters
    ----------
    a : 3x3 numpy array
        Second order fiber orientation tensor.
    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Du Hwan Chung and Tai Hun Kwon,
       'Invariant-based optimal fitting closure approximation for the numerical
       prediction of flow-induced fiber orientation',
       Journal of Rheology 46(1):169-194,
       https://doi.org/10.1122/1.1423312
    """
    assert_fot_properties(a)

    # second invariant
    II = (
        a[..., 0, 0] * a[..., 1, 1]
        + a[..., 1, 1] * a[..., 2, 2]
        + a[..., 0, 0] * a[..., 2, 2]
        - a[..., 0, 1] * a[..., 1, 0]
        - a[..., 1, 2] * a[..., 2, 1]
        - a[..., 0, 2] * a[..., 2, 0]
    )

    # third invariant
    III = np.linalg.det(a)

    # coefficients from Chung & Kwon paper
    C1 = np.zeros((1, 21))

    C2 = np.zeros((1, 21))

    C3 = np.array(
        [
            [
                0.24940908165786e2,
                -0.435101153160329e3,
                0.372389335663877e4,
                0.703443657916476e4,
                0.823995187366106e6,
                -0.133931929894245e6,
                0.880683515327916e6,
                -0.991630690741981e7,
                -0.159392396237307e5,
                0.800970026849796e7,
                -0.237010458689252e7,
                0.379010599355267e8,
                -0.337010820273821e8,
                0.322219416256417e5,
                -0.257258805870567e9,
                0.214419090344474e7,
                -0.449275591851490e8,
                -0.213133920223355e8,
                0.157076702372204e10,
                -0.232153488525298e5,
                -0.395769398304473e10,
            ]
        ]
    )

    C4 = np.array(
        [
            [
                -0.497217790110754e0,
                0.234980797511405e2,
                -0.391044251397838e3,
                0.153965820593506e3,
                0.152772950743819e6,
                -0.213755248785646e4,
                -0.400138947092812e4,
                -0.185949305922308e7,
                0.296004865275814e4,
                0.247717810054366e7,
                0.101013983339062e6,
                0.732341494213578e7,
                -0.147919027644202e8,
                -0.104092072189767e5,
                -0.635149929624336e8,
                -0.247435106210237e6,
                -0.902980378929272e7,
                0.724969796807399e7,
                0.487093452892595e9,
                0.138088690964946e5,
                -0.160162178614234e10,
            ]
        ]
    )

    C5 = np.zeros((1, 21))

    C6 = np.array(
        [
            [
                0.234146291570999e2,
                -0.412048043372534e3,
                0.319553200392089e4,
                0.573259594331015e4,
                -0.485212803064813e5,
                -0.605006113515592e5,
                -0.477173740017567e5,
                0.599066486689836e7,
                -0.110656935176569e5,
                -0.460543580680696e8,
                0.203042960322874e7,
                -0.556606156734835e8,
                0.567424911007837e9,
                0.128967058686204e5,
                -0.152752854956514e10,
                -0.499321746092534e7,
                0.132124828143333e9,
                -0.162359994620983e10,
                0.792526849882218e10,
                0.466767581292985e4,
                -0.128050778279459e11,
            ]
        ]
    )

    # build matrix of coefficients by stacking vectors
    C = np.vstack((C1, C2, C3, C4, C5, C6))

    # compute parameters as fith order polynom based on invariants
    beta3 = (
        C[2, 0]
        + C[2, 1] * II
        + C[2, 2] * II**2
        + C[2, 3] * III
        + C[2, 4] * III**2
        + C[2, 5] * II * III
        + C[2, 6] * II**2 * III
        + C[2, 7] * II * III**2
        + C[2, 8] * II**3
        + C[2, 9] * III**3
        + C[2, 10] * II**3 * III
        + C[2, 11] * II**2 * III**2
        + C[2, 12] * II * III**3
        + C[2, 13] * II**4
        + C[2, 14] * III**4
        + C[2, 15] * II**4 * III
        + C[2, 16] * II**3 * III**2
        + C[2, 17] * II**2 * III**3
        + C[2, 18] * II * III**4
        + C[2, 19] * II**5
        + C[2, 20] * III**5
    )

    beta4 = (
        C[3, 0]
        + C[3, 1] * II
        + C[3, 2] * II**2
        + C[3, 3] * III
        + C[3, 4] * III**2
        + C[3, 5] * II * III
        + C[3, 6] * II**2 * III
        + C[3, 7] * II * III**2
        + C[3, 8] * II**3
        + C[3, 9] * III**3
        + C[3, 10] * II**3 * III
        + C[3, 11] * II**2 * III**2
        + C[3, 12] * II * III**3
        + C[3, 13] * II**4
        + C[3, 14] * III**4
        + C[3, 15] * II**4 * III
        + C[3, 16] * II**3 * III**2
        + C[3, 17] * II**2 * III**3
        + C[3, 18] * II * III**4
        + C[3, 19] * II**5
        + C[3, 20] * III**5
    )

    beta6 = (
        C[5, 0]
        + C[5, 1] * II
        + C[5, 2] * II**2
        + C[5, 3] * III
        + C[5, 4] * III**2
        + C[5, 5] * II * III
        + C[5, 6] * II**2 * III
        + C[5, 7] * II * III**2
        + C[5, 8] * II**3
        + C[5, 9] * III**3
        + C[5, 10] * II**3 * III
        + C[5, 11] * II**2 * III**2
        + C[5, 12] * II * III**3
        + C[5, 13] * II**4
        + C[5, 14] * III**4
        + C[5, 15] * II**4 * III
        + C[5, 16] * II**3 * III**2
        + C[5, 17] * II**2 * III**3
        + C[5, 18] * II * III**4
        + C[5, 19] * II**5
        + C[5, 20] * III**5
    )

    beta1 = (
        3.0
        / 5.0
        * (
            -1.0 / 7.0
            + 1.0 / 5.0 * beta3 * (1.0 / 7.0 + 4.0 / 7.0 * II + 8.0 / 3.0 * III)
            - beta4 * (1.0 / 5.0 - 8.0 / 15.0 * II - 14.0 / 15.0 * III)
            - beta6
            * (
                1.0 / 35.0
                - 24.0 / 105.0 * III
                - 4.0 / 35.0 * II
                + 16.0 / 15.0 * II * III
                + 8.0 / 35.0 * II**2
            )
        )
    )

    beta2 = (
        6.0
        / 7.0
        * (
            1.0
            - 1.0 / 5.0 * beta3 * (1.0 + 4.0 * II)
            + 7.0 / 5.0 * beta4 * (1.0 / 6.0 - II)
            - beta6
            * (-1.0 / 5.0 + 2.0 / 3.0 * III + 4.0 / 5.0 * II - 8.0 / 5.0 * II**2)
        )
    )

    beta5 = (
        -4.0 / 5.0 * beta3
        - 7.0 / 5.0 * beta4
        - 6.0 / 5.0 * beta6 * (1.0 - 4.0 / 3.0 * II)
    )

    # second order identy matrix
    delta = np.eye(3)

    # generate fourth order tensor with parameters and tensor algebra
    return (
        symm(np.einsum("..., ij,kl->...ijkl", beta1, delta, delta))
        + symm(np.einsum("..., ij, ...kl-> ...ijkl", beta2, delta, a))
        + symm(np.einsum("..., ...ij, ...kl -> ...ijkl", beta3, a, a))
        + symm(np.einsum("..., ij, ...km, ...ml -> ...ijkl", beta4, delta, a, a))
        + symm(np.einsum("..., ...ij, ...km, ...ml -> ...ijkl", beta5, a, a, a))
        + symm(
            np.einsum("..., ...im, ...mj, ...kn, ...nl -> ...ijkl", beta6, a, a, a, a)
        )
    )


def symm(A):
    """Symmetrize the fourth order tensor.
    This function computes the symmetric part of a fourth order Tensor A
    and returns a symmetric fourth order tensor S.
    """
    # if A.ndim == 4:
    #     S = np.stack(
    #         [np.transpose(A, axes=p) for p in permutations([0, 1, 2, 3])]
    #     )sy
    S = np.stack(
        [
            np.transpose(A, axes=(*range(A.ndim - 4), *p))
            for p in permutations([A.ndim - 4, A.ndim - 3, A.ndim - 2, A.ndim - 1])
        ]
    )
    return S.sum(axis=0) / 24


def orthotropic_fitted_closures(a, closure="ORF"):
    """Generate a orthotropic fitted closure.
    Parameter
    ---------
    a : 3x3 array
        2. order orientation tensor
    closure: string
        Defines the used closure ("ORF", "ORW", "ORW3")
    Return:
    ------
    A : 3x3x3x3 numpy array
        4. order orientation tensor
    References
    ----------
    .. [1] Joaquim S. Cintra and Charles L. Tucker III (1995),
    'Orthotropic closure approximations for flow-induced fiber orientation',
    Journal of Rheology, 39(6), 1095-1122,
    https://doi.org/10.1122/1.550630
    .. [2] Du Hwan Chung and Tai Hun Kwon (2001),
    'Improved model of orthotropic closure approximation for flow induced fiber
    orientation', Polymer Composites, 22(5), 636-649,
    https://doi.org/10.1002/pc.10566
    """
    assert_fot_properties(a)

    A = np.zeros([*a.shape[:-2], 3, 3, 3, 3])

    # Calculate eigenvalues w and eigenvector-matrix R of a
    w, R = np.linalg.eigh(a)
    # Sort eigenvalues and eigenvectors in descending order
    w = w[..., ::-1]
    R = R[..., ::-1]

    ev1 = w[..., 0]
    ev2 = w[..., 1]
    ev3 = w[..., 2]

    if closure == "ORF":
        C = np.array(
            [
                [0.060964, 0.371243, 0.555301, -0.369160, 0.318266, 0.371218],
                [0.124711, -0.389402, 0.258844, 0.086169, 0.796080, 0.544992],
                [1.228982, -2.054116, 0.821548, -2.260574, 1.053907, 1.819756],
            ]
        )
        W = np.array(
            [
                np.ones(ev1.shape),
                ev1,
                ev1**2.0,
                ev2,
                ev2**2.0,
                ev1 * ev2,
            ]
        )

    elif closure == "ORW":
        C = np.array(
            [
                [0.070055, 0.339376, 0.590331, -0.396796, 0.333693, 0.411944],
                [0.115177, -0.368267, 0.252880, 0.094820, 0.800181, 0.535224],
                [1.249811, -2.148297, 0.898521, -2.290157, 1.044147, 1.934914],
            ]
        )
        W = np.array(
            [
                np.ones(ev1.shape),
                ev1,
                ev1**2,
                ev2,
                ev2**2,
                ev1 * ev2,
            ]
        )

    elif closure == "ORW3":
        C = np.array(
            [
                [
                    -0.1480648093,
                    0.8084618453,
                    0.3722003446,
                    0.7765597096,
                    -1.3431772379,
                    -1.7366749542,
                    0.8895946393,
                    1.7367571741,
                    -0.0324756095,
                    0.6631716575,
                ],
                [
                    -0.2106349673,
                    0.9092350296,
                    -1.2840654776,
                    1.1104441966,
                    0.1260059291,
                    -2.5375632310,
                    1.9988098293,
                    1.4863151577,
                    0.5856304774,
                    -0.0756740034,
                ],
                [
                    0.4868019601,
                    0.5776328438,
                    -2.2462007509,
                    0.4605743789,
                    -1.9088154281,
                    -4.8900459209,
                    4.0544348937,
                    3.8542602127,
                    1.1817992322,
                    0.9512305286,
                ],
            ]
        )
        W = np.array(
            [
                np.ones(ev1.shape),
                ev1,
                ev1**2,
                ev2,
                ev2**2,
                ev1 * ev2,
                ev1 * ev1 * ev2,
                ev1 * ev2 * ev2,
                ev1 * ev1 * ev1,
                ev2 * ev2 * ev2,
            ]
        )

    W = np.einsum("I... -> ...I", W)
    # A_sol = ([A11, A22, A33, A44, A55, A66])
    A_sol = np.zeros((*a.shape[:-2], 6))
    # [A_sol[0], A_sol[1], A_sol[2]] = np.einsum("ij,j->i", C, W)
    for i in range(3):
        if closure == "ORW3":
            A_sol[..., i] = (
                C[i, 0] * W[..., 0]
                + C[i, 1] * W[..., 1]
                + C[i, 2] * W[..., 2]
                + C[i, 3] * W[..., 3]
                + C[i, 4] * W[..., 4]
                + C[i, 5] * W[..., 5]
                + C[i, 6] * W[..., 6]
                + C[i, 7] * W[..., 7]
                + C[i, 8] * W[..., 8]
                + C[i, 9] * W[..., 9]
            )
        else:
            A_sol[..., i] = (
                C[i, 0] * W[..., 0]
                + C[i, 1] * W[..., 1]
                + C[i, 2] * W[..., 2]
                + C[i, 3] * W[..., 3]
                + C[i, 4] * W[..., 4]
                + C[i, 5] * W[..., 5]
            )

    A_sol[..., 3] = 0.5 * (
        ev3 - A_sol[..., 2] - ev1 + A_sol[..., 0] + ev2 - A_sol[..., 1]
    )
    # A_sol[4] = a[0, 0] - A_sol[0] - A_sol[5]
    # A_sol[5] = a[1, 1] - A_sol[1] - A_sol[3]
    A_sol[..., 4] = 0.5 * (
        -A_sol[..., 0] + A_sol[..., 1] - A_sol[..., 2] + ev1 - ev2 + ev3
    )
    A_sol[..., 5] = 0.5 * (
        -A_sol[..., 0] - A_sol[..., 1] + A_sol[..., 2] + ev1 + ev2 - ev3
    )

    for i in range(3):
        A[..., i, i, i, i] = A_sol[..., i]

    A[..., 0, 0, 1, 1] = A_sol[..., 5]
    A[..., 1, 1, 0, 0] = A_sol[..., 5]
    # A1133 = A13 = A55
    A[..., 0, 0, 2, 2] = A_sol[..., 4]
    A[..., 2, 2, 0, 0] = A_sol[..., 4]
    # A2233 = A23 = A44
    A[..., 1, 1, 2, 2] = A_sol[..., 3]
    A[..., 2, 2, 1, 1] = A_sol[..., 3]
    # A2323 = A44
    A[..., 1, 2, 1, 2] = A_sol[..., 3]
    A[..., 2, 1, 2, 1] = A_sol[..., 3]
    A[..., 2, 1, 1, 2] = A_sol[..., 3]
    A[..., 1, 2, 2, 1] = A_sol[..., 3]
    # A1313 = A55
    A[..., 0, 2, 0, 2] = A_sol[..., 4]
    A[..., 2, 0, 2, 0] = A_sol[..., 4]
    A[..., 2, 0, 0, 2] = A_sol[..., 4]
    A[..., 0, 2, 2, 0] = A_sol[..., 4]
    # A1212 = A66
    A[..., 0, 1, 0, 1] = A_sol[..., 5]
    A[..., 1, 0, 1, 0] = A_sol[..., 5]
    A[..., 1, 0, 0, 1] = A_sol[..., 5]
    A[..., 0, 1, 1, 0] = A_sol[..., 5]
    A = np.einsum("...im, ...jn, ...ko, ...lp, ...mnop -> ...ijkl", R, R, R, R, A)

    return A


def symmetric_quadratic_closure(a):
    """Generate a symmetric quadratic closure.
    Parameters
    ----------
    a : (Mx)3x3 numpy array
        (Array of) Second order fiber orientation tensor.
    Returns
    -------
    A : (Mx)3x3x3x3 numpy array
        (Array of) Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Karl, Tobias and Gatti, Davide and Frohnapfel, Bettina and Böhlke, Thomas,
       'Asymptotic fiber orientation states of the quadratically
       closed Folgar--Tucker equation and a subsequent closure
       improvement',
       Journal of Rheology 65(5) : 999-1022,
       https://doi.org/10.1122/8.0000245
    Notes
    ----------
        In general, the SQC does not contract to its second-order input.
    """
    assert_fot_properties(a)

    a4 = (
        1.0
        / 3.0
        * (
            np.einsum("...ij, ...kl -> ...ijkl", a, a)
            + np.einsum("...ik, ...lj -> ...ijkl", a, a)
            + np.einsum("...il, ...kj -> ...ijkl", a, a)
        )
    )

    a4 /= 1.0 / 3.0 * (1.0 + 2.0 * np.einsum("...ij, ...ij -> ...", a, a))

    return a4


def symmetric_implicit_closure(a, eps_newton=1.0e-12, n_iter_newton=25):
    """Generate SIQ closure.
    Parameters
    ----------
    a : ...x3x3 numpy array
        (Array of) second order fiber orientation tensor.
    eps_newton : float
        convergence criterion for newton algorithm
        optional: default 1.0e-12
    n_iter_newton : int
        number of maximum iterations in newton algorithm
        optional: default 25
    Returns
    -------
    ...x3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Karl, Tobias, Matti Schneider, and Thomas Böhlke,
       'On fully symmetric implicit closure approximations for fiber orientation
       tensors', Journal of Non-Newtonian Fluid Mechanics 318 : 105049,
       https://doi.org/10.1016/j.jnnfm.2023.105049
    """

    assert_fot_properties(a)

    evs, r = np.linalg.eigh(a)

    d = evs.shape[-1]
    s = np.ones(a.shape[:-2])
    err, it = 1.0e12, 0

    while err > eps_newton and it < n_iter_newton:
        f = (d + 4.0) * s - np.sum(np.sqrt(1.5 * evs + s[..., None] ** 2.0), axis=-1)
        f_prime = 4.0 + np.sum(
            1.0 - s[..., None] / np.sqrt(1.5 * evs + s[..., None] ** 2.0),
            axis=-1,
        )

        s -= f / f_prime

        err = np.linalg.norm(f)
        it += 1

    if err > eps_newton:
        raise ValueError("Newton algorithm did not converge.")

    mus = np.sqrt(1.5 * evs + s[..., None] ** 2) - s[..., None]
    b = np.einsum("...ij, ...j, ...kj -> ...ik", r, mus, r)

    return (
        1.0
        / 3.0
        * (
            np.einsum("...ij, ...kl -> ...ijkl", b, b)
            + np.einsum("...ik, ...lj -> ...ijkl", b, b)
            + np.einsum("...il, ...kj -> ...ijkl", b, b)
        )
    )


def implicit_hybrid_closure(a, eps_newton=1.0e-12, n_iter_newton=25):
    """Generate implicit hybrid closure.
    Parameters
    ----------
    a : ...x3x3 numpy array
        (Array of) second order fiber orientation tensor.
    eps_newton : float
        convergence criterion for newton algorithm
        optional: default 1.0e-12
    n_iter_newton : int
        number of maximum iterations in newton algorithm
        optional: default 25
    Returns
    -------
    ...x3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Karl, Tobias, Matti Schneider, and Thomas Böhlke,
       'On fully symmetric implicit closure approximations for fiber orientation
       tensors', Journal of Non-Newtonian Fluid Mechanics 318 : 105049,
       https://doi.org/10.1016/j.jnnfm.2023.105049
    Notes
    ----------
    There seems to be a typo in the expression for f_prime in the original work.
    """

    assert_fot_properties(a)

    evs, r = np.linalg.eigh(a)

    d = evs.shape[-1]
    s = np.ones(a.shape[:-2])

    k = 1.0 - 27.0 * np.prod(evs, axis=-1)

    err, it = 1.0e12, 0

    #  ignore warnings when input is isotropic (results in k=0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while err > eps_newton and it < n_iter_newton:
            # l1 and l2 correspond to a, b in the original work
            l1 = np.nan_to_num(0.5 * (3.0 + (4.0 * s - 3.0) * k) / k)
            l2 = np.nan_to_num((3.0 * (1.0 - k) * (4.0 * s - 1.0)) / (14.0 * k))

            l1_prime = 4.0 * np.ones(l1.shape)
            l2_prime = np.nan_to_num((6.0 * (1.0 - k)) / (7.0 * k))

            f = np.nan_to_num(
                (
                    4.0 * s
                    + 0.5 * l1 * d
                    - np.sum(
                        np.sqrt(
                            1.5 * evs / k[..., None]
                            + 0.25 * l1[..., None] ** 2.0
                            - l2[..., None]
                        ),
                        axis=-1,
                    )
                )
            )

            f_prime = np.nan_to_num(
                (
                    4.0
                    + 0.5 * (l1_prime * d)
                    - 0.25
                    * np.sum(
                        (
                            2.0 * l1[..., None] * l1_prime[..., None]
                            - 4.0 * l2_prime[..., None]
                        )
                        / np.sqrt(
                            2.0 * evs / k[..., None]
                            + l1[..., None] ** 2.0
                            - 4.0 * l2[..., None]
                        ),
                        axis=-1,
                    )
                )
            )

            s -= f / f_prime

            err = np.linalg.norm(f)
            it += 1

        if err > eps_newton:
            raise ValueError("Newton algorithm did not converge.")

        l1 = np.nan_to_num(0.5 * (3.0 + (4.0 * s - 3.0) * k) / k)
        l2 = np.nan_to_num((3.0 * (1.0 - k) * (4.0 * s - 1.0)) / (14.0 * k))

        mus = -0.5 * l1[..., None] + np.sqrt(
            1.5 * evs / k[..., None] + 0.25 * l1[..., None] ** 2 - l2[..., None]
        )

    b = np.einsum("...ij, ...j, ...kj -> ...ik", r, mus, r)

    w1 = 1.0 - k
    sym_ii = -3.0 / 35.0 * symm(np.einsum("ij, kl -> ijkl", np.eye(3), np.eye(3)))
    sym_ib = 6.0 / 7.0 * symm(np.einsum("ij, ...kl -> ...ijkl", np.eye(3), b))
    sym_bb = symm(np.einsum("...ij, ...kl -> ...ijkl", b, b))

    linear_term = np.einsum("..., ...ijkl -> ...ijkl", w1, sym_ii + sym_ib)
    quadratic_term = np.einsum("..., ...ijkl -> ...ijkl", k, sym_bb)

    a4_iso = 0.2 * symm(np.einsum("ij, kl -> ijkl", *2 * (np.eye(3),)))
    a4 = linear_term + quadratic_term

    # replace numerical garbage with true isotropic 4th-order FOT
    return np.where(
        np.isclose(k, 0.0)[..., None, None, None, None],
        a4_iso[..., :, :, :, :],
        a4,
    )


# Simple closures from FOT4 to
def assert_fot4_properties(A):
    """Assert properties of fourth order fiber orientation tensor.

    Parameters
    ----------
    A : np.ndarray
        Fourth order fiber orientation tensor.
    """
    assert np.shape(A)[-4:] == (3, 3, 3, 3)


def compute_closure_FOT4(A, closure="QUADRATIC"):
    """Create a sixth order tensor from a fourth order tensor.
    This is essentially a wrapper around all closures.
    Parameters
    ----------
    a : 3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    Returns
    -------
    3x3x3x3x3x3 numpy array
        Sixth order fiber orientation tensor.
    """
    # assertation
    assert closure in (
        "LINEAR",
        "HYBRID",
        "QUADRATIC",  # A x a
    )
    if closure == "HYBRID":
        return hybrid_closure_FOT4(A)
    if closure == "LINEAR":
        return linear_closure_FOT4(A)
    if closure == "QUADRATIC":
        return quadratic_closure_FOT4(A)


def quadratic_closure_FOT4(A):
    """Generate quadratic closure. A6 = A x a

    Args:
        A (Mx)3x3x3x3 Array: (Array of) Fourth order fiber orientation tensor.

    Returns:
        (Mx)3x3x3x3x3x3: (Array of) Sixth order fiber orientation tensor.
    References
    ----------
    .. [1] Advani, Suresh G.; Tucker, Charles L. (1987),
    The Use of Tensors to Describe and Predict Fiber Orientation in Short Fiber Comp.
    In: Journal of Rheology 31 (8), S. 751-784
    https://doi.org/10.1122/1.549945
    """
    assert_fot4_properties(A)
    return np.einsum("...ijkl, ...mnoo -> ...ijklmn", A, A)


def linear_closure_FOT4(A):
    """Generate linear closure.

    Args:
        A (Mx)3x3x3x3 Array: (Array of) Fourth order fiber orientation tensor.

    Returns:
        (Mx)3x3x3x3x3x3: (Array of) Sixth order fiber orientation tensor.
        References
    ----------
    .. [1] Advani, Suresh G.; Tucker, Charles L. (1987),
    The Use of Tensors to Describe and Predict Fiber Orientation in Short Fiber Comp.
    In: Journal of Rheology 31 (8), S. 751-784
    https://doi.org/10.1122/1.549945

    """
    assert_fot4_properties(A)
    a = np.einsum("...ijkl,kl->...ij", A, np.eye(3))
    IxIxI = np.einsum("...ij,kl,mn->...ijklmn", np.eye(3), np.eye(3), np.eye(3))
    IxIxI_fullsym = sum(
        np.array(
            [
                np.einsum("..." + string + "->...ijklmn", IxIxI)
                for string in fullsym6_permutations
            ]
        )
    )
    IxIxI_fullsym = IxIxI_fullsym / len(fullsym6_permutations)
    axIxI = np.einsum("...ij,kl,mn->...ijklmn", a, np.eye(3), np.eye(3))
    axIxI_fullsym = sum(
        np.array(
            [
                np.einsum("..." + string + "->...ijklmn", axIxI)
                for string in fullsym6_permutations
            ]
        )
    )
    IxaxI = np.einsum("...ij,kl,mn->...ijklmn", np.eye(3), a, np.eye(3))
    IxaxI_fullsym = sum(
        np.array(
            [
                np.einsum("..." + string + "->...ijklmn", IxaxI)
                for string in fullsym6_permutations
            ]
        )
    )
    IxIxa = np.einsum("...ij,kl,mn->...ijklmn", np.eye(3), np.eye(3), a)
    IxIxa_fullsym = sum(
        np.array(
            [
                np.einsum("..." + string + "->...ijklmn", IxIxa)
                for string in fullsym6_permutations
            ]
        )
    )
    axIxI_fullsym_combine = (
        (axIxI_fullsym + IxaxI_fullsym + IxIxa_fullsym)
        / 3.0
        / len(fullsym6_permutations)
    )

    AxI = np.einsum("...ijkl,mn->...ijklmn", A, np.eye(3))
    AxI_fullsym = sum(
        np.array(
            [
                np.einsum("..." + string + "->...ijklmn", AxI)
                for string in fullsym6_permutations
            ]
        )
    )
    AxI_fullsym = AxI_fullsym / len(fullsym6_permutations)
    A6 = (
        15.0 / 693.0 * IxIxI_fullsym
        - 45.0 / 99.0 * axIxI_fullsym_combine
        + 15.0 / 11.0 * AxI_fullsym
    )
    return A6


def hybrid_closure_FOT4(A):
    """Generate hybrid closure for FOT4.

    Args:
        A (Mx)3x3x3x3 Array: (Array of) Fourth order fiber orientation tensor.

    Returns:
        (Mx)3x3x3x3x3x3: (Array of) Sixth order fiber orientation tensor.
    References
    ----------
    .. [1] Advani, Suresh G.; Tucker, Charles L. (1987),
    The Use of Tensors to Describe and Predict Fiber Orientation in Short Fiber Comp.
    In: Journal of Rheology 31 (8), S. 751-784
    https://doi.org/10.1122/1.549945

    """
    assert_fot4_properties(A)
    a = np.einsum("...ijkl,kl->...ij", A, np.eye(3))
    f = 1.0 - 27.0 * np.linalg.det(a)
    A6 = np.einsum("..., ...ijklmn -> ...ijklmn", 1.0 - f, linear_closure_FOT4(A))
    A6 += np.einsum("..., ...ijklmn -> ...ijklmn", f, quadratic_closure_FOT4(A))

    return A6
