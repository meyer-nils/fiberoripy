# -*- coding: utf-8 -*-
"""Collection of closure approximations."""
import numpy as np


def compute_closure(a, closure="IBOF", N=1000):
    """Create a fourth order tensor from a second order tensor.

    This is essentially a wrapper around all closures.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.

    """
    # assertation
    assert closure in (
        "IBOF",
        "LINEAR",
        "HYBRID",
        "QUADRATIC",
        "RANDOM",
        "ORF",
        "ORW",
        "ORW3",
    )
    if closure == "IBOF":
        return IBOF_closure(a)
    if closure == "HYBRID":
        return hybrid_closure(a)
    if closure == "LINEAR":
        return linear_closure(a)
    if closure == "QUADRATIC":
        return quadratic_closure(a)
    if closure == "RANDOM":
        return random_closure(a, N)
    if closure == "ORF":
        return orthotropic_fitted_closures(a, "ORF")
    if closure == "ORW":
        return orthotropic_fitted_closures(a, "ORW")
    if closure == "ORW3":
        return orthotropic_fitted_closures(a, "ORW3")


def assert_fot_properties(A):
    """Assert properties of second order input tensor.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    """
    # assert symmetry and shape
    assert np.shape(A) == (3, 3)
    # assert(A[0, 1] == A[1, 0])
    # assert(A[0, 2] == A[2, 0])
    # assert(A[1, 2] == A[2, 1])


def random_closure(a, N=1000):
    """Sample a random fiber orientation and compute fourth order tensor.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.

    """
    orientations = []
    for i in range(N):
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        rand_orientation = np.array([x, y, z])
        p = np.dot(a, rand_orientation)
        p = p / np.linalg.norm(p)
        orientations.append(p)

    a.fill(0.0)
    A = np.zeros((3, 3, 3, 3))
    for o in orientations:
        a += 1.0 / N * np.einsum("i,j->ij", o, o)
        A += 1.0 / N * np.einsum("i,j,k,l->ijkl", o, o, o, o)
    return A


def linear_closure(A):
    """Generate a linear closure.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.


    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002

    """
    assert_fot_properties(A)

    delta = np.eye(3)
    return 1.0 / 7.0 * (
        np.einsum("ij,kl->ijkl", A, delta)
        + np.einsum("ik,jl->ijkl", A, delta)
        + np.einsum("il,jk->ijkl", A, delta)
        + np.einsum("kl,ij->ijkl", A, delta)
        + np.einsum("jl,ik->ijkl", A, delta)
        + np.einsum("jk,il->ijkl", A, delta)
    ) - 1.0 / 35.0 * (
        np.einsum("ij,kl->ijkl", delta, delta)
        + np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    )


def quadratic_closure(A):
    """Generate a quadratic closure.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.


    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002

    """
    assert_fot_properties(A)
    return np.einsum("ij,kl->ijkl", A, A)


def hybrid_closure(A):
    """Generate a hybrid closure.

    Parameters
    ----------
    A : 3x3 numpy array
        Second order fiber orientation tensor.

    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.


    References
    ----------
    .. [1] Kyeong-Hee Han and Yong-Taek Im,
       'Modified hybrid closure approximation for prediction of flow-induced
       fiber orientation'
       Journal of Rheology 43, 569 (1999)
       https://doi.org/10.1122/1.551002

    """
    assert_fot_properties(A)
    f = 1.0 - 27.0 * np.linalg.det(A)
    return (1.0 - f) * linear_closure(A) + f * quadratic_closure(A)


def IBOF_closure(A):
    """Generate IBOF closure.

    Parameters
    ----------
    A : 3x3 numpy array
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
    assert_fot_properties(A)

    # second invariant
    II = (
        A[0, 0] * A[1, 1]
        + A[1, 1] * A[2, 2]
        + A[0, 0] * A[2, 2]
        - A[0, 1] * A[1, 0]
        - A[1, 2] * A[2, 1]
        - A[0, 2] * A[2, 0]
    )

    # third invariant
    III = np.linalg.det(A)

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
        + C[2, 2] * II ** 2
        + C[2, 3] * III
        + C[2, 4] * III ** 2
        + C[2, 5] * II * III
        + C[2, 6] * II ** 2 * III
        + C[2, 7] * II * III ** 2
        + C[2, 8] * II ** 3
        + C[2, 9] * III ** 3
        + C[2, 10] * II ** 3 * III
        + C[2, 11] * II ** 2 * III ** 2
        + C[2, 12] * II * III ** 3
        + C[2, 13] * II ** 4
        + C[2, 14] * III ** 4
        + C[2, 15] * II ** 4 * III
        + C[2, 16] * II ** 3 * III ** 2
        + C[2, 17] * II ** 2 * III ** 3
        + C[2, 18] * II * III ** 4
        + C[2, 19] * II ** 5
        + C[2, 20] * III ** 5
    )

    beta4 = (
        C[3, 0]
        + C[3, 1] * II
        + C[3, 2] * II ** 2
        + C[3, 3] * III
        + C[3, 4] * III ** 2
        + C[3, 5] * II * III
        + C[3, 6] * II ** 2 * III
        + C[3, 7] * II * III ** 2
        + C[3, 8] * II ** 3
        + C[3, 9] * III ** 3
        + C[3, 10] * II ** 3 * III
        + C[3, 11] * II ** 2 * III ** 2
        + C[3, 12] * II * III ** 3
        + C[3, 13] * II ** 4
        + C[3, 14] * III ** 4
        + C[3, 15] * II ** 4 * III
        + C[3, 16] * II ** 3 * III ** 2
        + C[3, 17] * II ** 2 * III ** 3
        + C[3, 18] * II * III ** 4
        + C[3, 19] * II ** 5
        + C[3, 20] * III ** 5
    )

    beta6 = (
        C[5, 0]
        + C[5, 1] * II
        + C[5, 2] * II ** 2
        + C[5, 3] * III
        + C[5, 4] * III ** 2
        + C[5, 5] * II * III
        + C[5, 6] * II ** 2 * III
        + C[5, 7] * II * III ** 2
        + C[5, 8] * II ** 3
        + C[5, 9] * III ** 3
        + C[5, 10] * II ** 3 * III
        + C[5, 11] * II ** 2 * III ** 2
        + C[5, 12] * II * III ** 3
        + C[5, 13] * II ** 4
        + C[5, 14] * III ** 4
        + C[5, 15] * II ** 4 * III
        + C[5, 16] * II ** 3 * III ** 2
        + C[5, 17] * II ** 2 * III ** 3
        + C[5, 18] * II * III ** 4
        + C[5, 19] * II ** 5
        + C[5, 20] * III ** 5
    )

    beta1 = (
        3.0
        / 5.0
        * (
            -1.0 / 7.0
            + 1.0
            / 5.0
            * beta3
            * (1.0 / 7.0 + 4.0 / 7.0 * II + 8.0 / 3.0 * III)
            - beta4 * (1.0 / 5.0 - 8.0 / 15.0 * II - 14.0 / 15.0 * III)
            - beta6
            * (
                1.0 / 35.0
                - 24.0 / 105.0 * III
                - 4.0 / 35.0 * II
                + 16.0 / 15.0 * II * III
                + 8.0 / 35.0 * II ** 2
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
            * (
                -1.0 / 5.0
                + 2.0 / 3.0 * III
                + 4.0 / 5.0 * II
                - 8.0 / 5.0 * II ** 2
            )
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
        beta1 * symm(np.einsum("ij,kl->ijkl", delta, delta))
        + beta2 * symm(np.einsum("ij,kl->ijkl", delta, A))
        + beta3 * symm(np.einsum("ij,kl->ijkl", A, A))
        + beta4 * symm(np.einsum("ij,km,ml->ijkl", delta, A, A))
        + beta5 * symm(np.einsum("ij,km,ml->ijkl", A, A, A))
        + beta6 * symm(np.einsum("im,mj,kn,nl->ijkl", A, A, A, A))
    )


def symm(A):
    """Symmetrize the fourth order tensor.

    This function computes the symmetric part of a fourth order Tensor A
    and returns a symmetric fourth order tensor S.
    """
    # initial symmetric tensor with zeros
    S = np.zeros((3, 3, 3, 3))

    # Einsteins summation
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    # sum of all permutations divided by 4!=24
                    S[i, j, k, l] = (
                        1.0
                        / 24.0
                        * (
                            A[i, j, k, l]
                            + A[j, i, k, l]
                            + A[i, j, l, k]
                            + A[j, i, l, k]
                            + A[k, l, i, j]
                            + A[l, k, i, j]
                            + A[k, l, j, i]
                            + A[l, k, j, i]
                            + A[i, k, j, l]
                            + A[k, i, j, l]
                            + A[i, k, l, j]
                            + A[k, i, l, j]
                            + A[j, l, i, k]
                            + A[l, j, i, k]
                            + A[j, l, k, i]
                            + A[l, j, k, i]
                            + A[i, l, j, k]
                            + A[l, i, j, k]
                            + A[i, l, k, j]
                            + A[l, i, k, j]
                            + A[j, k, i, l]
                            + A[k, j, i, l]
                            + A[j, k, l, i]
                            + A[k, j, l, i]
                        )
                    )
    return S


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

    A = np.zeros([3, 3, 3, 3])

    # Calculate eigenvalues w and eigenvector-matrix R of a
    w, R = np.linalg.eigh(a)
    # Sort eigenvalues and eigenvectors in descending order
    idx = w.argsort()[::-1]
    w = w[idx]
    R = R[:, idx]

    if closure == "ORF":
        C = np.array(
            [
                [0.060964, 0.371243, 0.555301, -0.369160, 0.318266, 0.371218],
                [0.124711, -0.389402, 0.258844, 0.086169, 0.796080, 0.544992],
                [1.228982, -2.054116, 0.821548, -2.260574, 1.053907, 1.819756],
            ]
        )
        W = np.array([1.0, w[0], w[0] ** 2, w[1], w[1] ** 2, w[0] * w[1]])

    elif closure == "ORW":
        C = np.array(
            [
                [0.070055, 0.339376, 0.590331, -0.396796, 0.333693, 0.411944],
                [0.115177, -0.368267, 0.252880, 0.094820, 0.800181, 0.535224],
                [1.249811, -2.148297, 0.898521, -2.290157, 1.044147, 1.934914],
            ]
        )
        W = np.array([1.0, w[0], w[0] ** 2, w[1], w[1] ** 2, w[0] * w[1]])

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
                1.0,
                w[0],
                w[0] ** 2,
                w[1],
                w[1] ** 2,
                w[0] * w[1],
                w[0] * w[0] * w[1],
                w[0] * w[1] * w[1],
                w[0] * w[0] * w[0],
                w[1] * w[1] * w[1],
            ]
        )
    # A_sol = ([A11, A22, A33, A44, A55, A66])
    A_sol = np.zeros(6)
    [A_sol[0], A_sol[1], A_sol[2]] = np.einsum("ij,j->i", C, W)
    A_sol[3] = 0.5 * (w[2] - A_sol[2] - w[0] + A_sol[0] + w[1] - A_sol[1])
    # A_sol[5] = a[1, 1] - A_sol[1] - A_sol[3]
    # A_sol[4] = a[0, 0] - A_sol[0] - A_sol[5]
    A_sol[4] = 0.5 * (-A_sol[0] + A_sol[1] - A_sol[2] + w[0] - w[1] + w[2])
    A_sol[5] = 0.5 * (-A_sol[0] - A_sol[1] + A_sol[2] + w[0] + w[1] - w[2])

    for i in range(3):
        A[i, i, i, i] = A_sol[i]

    # A1122 = A12 = A66
    A[0, 0, 1, 1] = A_sol[5]
    A[1, 1, 0, 0] = A_sol[5]
    # A1133 = A13 = A55
    A[0, 0, 2, 2] = A_sol[4]
    A[2, 2, 0, 0] = A_sol[4]
    # A2233 = A23 = A44
    A[1, 1, 2, 2] = A_sol[3]
    A[2, 2, 1, 1] = A_sol[3]
    # A2323 = A44
    A[1, 2, 1, 2] = A_sol[3]
    A[2, 1, 2, 1] = A_sol[3]
    A[2, 1, 1, 2] = A_sol[3]
    A[1, 2, 2, 1] = A_sol[3]
    # A1313 = A55
    A[0, 2, 0, 2] = A_sol[4]
    A[2, 0, 2, 0] = A_sol[4]
    A[2, 0, 0, 2] = A_sol[4]
    A[0, 2, 2, 0] = A_sol[4]
    # A1212 = A66
    A[0, 1, 0, 1] = A_sol[5]
    A[1, 0, 1, 0] = A_sol[5]
    A[1, 0, 0, 1] = A_sol[5]
    A[0, 1, 1, 0] = A_sol[5]

    A = np.einsum("im,jn,ko,lp,mnop->ijkl", R, R, R, R, A)

    return A
