# -*- coding: utf-8 -*-
"""Orientation models."""

import numpy as np


def jeffery_ode(a, A, D, W, xi, **kwargs):
    """ODE describing Jeffery's model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] G.B. Jeffery
       'The motion of ellipsoidal particles immersed in a viscous fluid',
       Proceedings of the Royal Society A, 1922.
       https://doi.org/10.1098/rspa.1922.0078

    """
    return (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
    )


def folgar_tucker_ode(a, A, D, W, xi, Ci=0.0, **kwargs):
    """ODE describing the Folgar-Tucker model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] F. Folgar, C.L. Tucker III,
       'Orientation behavior of fibers in concentrated suspensions',
       Journal of Reinforced Plastic Composites 3, 98-119, 1984.
       https://doi.org/10.1177%2F073168448400300201

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    dadt = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
        + 2 * Ci * G * (delta - 3 * a)
    )
    return dadt


def maier_saupe_ode(a, A, D, W, xi, Ci=0.0, U0=0.0, **kwargs):
    """ODE using Folgar-Tucker constant and Maier-Saupe potential.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.1).
    U0 : float
        Maier-Saupe Potential (in 3D stable for y U0 < 8 Ci).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Arnulf Latz, Uldis Strautins, Dariusz Niedziela,
       'Comparative numerical study of two concentrated fiber suspension models',
       Journal of Non-Newtonian Fluid Mechanics 165, 764-781, 2010.
       https://doi.org/10.1016/j.jnnfm.2010.04.001

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    dadt = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
        + 2
        * G
        * (
            Ci * (delta - 3 * a)
            + U0 * (np.einsum("ik,kj->ij", a, a) - np.einsum("ijkl,kl->ij", A, a))
        )
    )
    return dadt


def iard_ode(a, A, D, W, xi, Ci=0.0, Cm=0.0, **kwargs):
    """ODE describing iARD model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.1).
    Cm : float
        Anisotropy factor (0 < Cm < 1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Tseng, Huan-Chang; Chang, Rong-Yeu; Hsu, Chia-Hsiang,
       'An objective tensor to predict anisotropic fiber orientation in concentrated
       suspensions',
       Journal of Rheology 60, 215, 2016.
       https://doi.org/10.1122/1.4939098

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    D2 = np.einsum("ik,kj->ij", D, D)
    D2_norm = np.sqrt(1.0 / 2.0 * np.einsum("ij,ij", D2, D2))

    Dr = Ci * (delta - Cm * D2 / D2_norm)

    dadt_HD = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
    )

    dadt_iard = G * (
        2 * Dr
        - 2 * np.trace(Dr) * a
        - 5 * np.einsum("ik,kj->ij", Dr, a)
        - 5 * np.einsum("ik,kj->ij", a, Dr)
        + 10 * np.einsum("ijkl,kl->ij", A, Dr)
    )

    dadt = dadt_HD + dadt_iard
    return dadt


def iardrpr_ode(a, A, D, W, xi, Ci=0.0, Cm=0.0, alpha=0.0, beta=0.0, **kwargs):
    """ODE describing iARD-RPR model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.1).
    Cm : float
        Anisotropy factor (0 < Cm < 1).
    alpha : float
        Retardance rate (0 < alpha < 1).
    beta : float
        Retardance tuning factor (0 < beta < 1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Tseng, Huan-Chang; Chang, Rong-Yeu; Hsu, Chia-Hsiang,
       'An objective tensor to predict anisotropic fiber orientation in
       concentrated suspensions',
       Journal of Rheology 60, 215, 2016.
       https://doi.org/10.1122/1.4939098

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    D2 = np.einsum("ik,kj->ij", D, D)
    D2_norm = np.sqrt(1.0 / 2.0 * np.einsum("ij,ij", D2, D2))

    Dr = Ci * (delta - Cm * D2 / D2_norm)

    dadt_HD = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2.0 * np.einsum("ijkl,kl->ij", A, D)
        )
    )

    dadt_iard = G * (
        2.0 * Dr
        - 2.0 * np.trace(Dr) * a
        - 5.0 * np.einsum("ik,kj->ij", Dr, a)
        - 5.0 * np.einsum("ik,kj->ij", a, Dr)
        + 10.0 * np.einsum("ijkl,kl->ij", A, Dr)
    )

    dadt_temp = dadt_HD + dadt_iard

    # Spectral Decomposition
    eigenValues, eigenVectors = np.linalg.eig(a)
    idx = eigenValues.argsort()[::-1]
    R = eigenVectors[:, idx]

    # Estimation of eigenvalue rates (rotated back)
    dadt_diag = np.einsum("ik, kl, lj->ij", np.transpose(R), dadt_temp, R)

    lbd0 = dadt_diag[0, 0]
    lbd1 = dadt_diag[1, 1]
    lbd2 = dadt_diag[2, 2]

    # Computation of IOK tensor by rotation
    IOK = np.zeros((3, 3))
    IOK[0, 0] = alpha * (lbd0 - beta * (lbd0**2.0 + 2.0 * lbd1 * lbd2))
    IOK[1, 1] = alpha * (lbd1 - beta * (lbd1**2.0 + 2.0 * lbd0 * lbd2))
    IOK[2, 2] = alpha * (lbd2 - beta * (lbd2**2.0 + 2.0 * lbd0 * lbd1))

    dadt_rpr = -np.einsum("ik, kl, lj->ij", R, IOK, np.transpose(R))

    dadt = dadt_temp + dadt_rpr
    return dadt


def mrd_ode(a, A, D, W, xi, Ci=0.0, D1=1.0, D2=0.8, D3=0.15, **kwargs):
    """ODE describing MRD model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.1).
    D1 : type
        Anisotropy factors (D1 > 0).
    D2 : type
        Anisotropy factors (D2 > 0).
    D3 : type
        Anisotropy factors (D3 > 0).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] A. Bakharev, H. Yu, R. Speight and J. Wang,
       'Using New Anisotropic Rotational Diffusion Model To Improve Prediction Of Short
       Fibers in Thermoplastic InjectionMolding',
       ANTEC, Orlando, 2018.

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))

    C_hat = np.array([[D1, 0.0, 0.0], [0.0, D2, 0.0], [0.0, 0.0, D3]])

    # Spectral Decomposition
    eigenValues, eigenVectors = np.linalg.eig(a)
    idx = eigenValues.argsort()[::-1]
    R = eigenVectors[:, idx]

    C = Ci * np.einsum("ij,jk,kl->il", R, C_hat, np.transpose(R))

    dadt_HD = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
    )

    dadt_mrd = G * (
        2 * C
        - 2 * np.trace(C) * a
        - 5 * np.einsum("ik,kj->ij", C, a)
        - 5 * np.einsum("ik,kj->ij", a, C)
        + 10 * np.einsum("ijkl,kl->ij", A, C)
    )

    dadt = dadt_HD + dadt_mrd
    return dadt


def pard_ode(a, A, D, W, xi, Ci=0.0, Omega=0.0, **kwargs):
    """ODE describing pARD model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.05).
    Omega : type
        Anisotropy factor (0.5 < Omega < 1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Tseng, Huan-Chang; Chang, Rong-Yeu; Hsu, Chia-Hsiang,
       'The use of principal spatial tensor to predict anisotropic fiber orientation in
       concentrated fiber suspensions',
       Journal of Rheology 62, 313, 2017.
       https://doi.org/10.1122/1.4998520

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))

    C_hat = np.array([[1.0, 0.0, 0.0], [0.0, Omega, 0.0], [0.0, 0.0, 1.0 - Omega]])

    # Spectral Decomposition
    eigenValues, eigenVectors = np.linalg.eig(a)
    idx = eigenValues.argsort()[::-1]
    R = eigenVectors[:, idx]

    C = Ci * np.einsum("ij,jk,kl->il", R, C_hat, np.transpose(R))

    dadt_HD = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
    )

    dadt_pard = G * (
        2 * C
        - 2 * np.trace(C) * a
        - 5 * np.einsum("ik,kj->ij", C, a)
        - 5 * np.einsum("ik,kj->ij", a, C)
        + 10 * np.einsum("ijkl,kl->ij", A, C)
    )

    dadt = dadt_HD + dadt_pard
    return dadt


def pardrpr_ode(a, A, D, W, xi, Ci=0.0, Omega=0.0, alpha=0.0, **kwargs):
    """ODE describing pARD-RPR model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.05).
    Omega : type
        Anisotropy factor (0.5 < Omega < 1).
    alpha : float
        Retardance rate (0 < alpha < 1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Tseng, Huan-Chang; Chang, Rong-Yeu; Hsu, Chia-Hsiang,
       'The use of principal spatial tensor to predict anisotropic fiber orientation in
       concentrated fiber suspensions',
       Journal of Rheology 62, 313, 2017.
       https://doi.org/10.1122/1.4998520

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))

    C_hat = np.array([[1.0, 0.0, 0.0], [0.0, Omega, 0.0], [0.0, 0.0, 1.0 - Omega]])

    # Spectral Decomposition
    eigenValues, eigenVectors = np.linalg.eig(a)
    idx = eigenValues.argsort()[::-1]
    R = eigenVectors[:, idx]

    C = Ci * np.einsum("ij,jk,kl->il", R, C_hat, np.transpose(R))

    dadt_HD = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", A, D)
        )
    )

    dadt_pard = G * (
        2 * C
        - 2 * np.trace(C) * a
        - 5 * np.einsum("ik,kj->ij", C, a)
        - 5 * np.einsum("ik,kj->ij", a, C)
        + 10 * np.einsum("ijkl,kl->ij", A, C)
    )

    dadt_temp = dadt_HD + dadt_pard

    # Estimation of eigenvalue rates (rotated back)
    dadt_diag = np.einsum("ik, kl, lj->ij", np.transpose(R), dadt_temp, R)

    lbd0 = dadt_diag[0, 0]
    lbd1 = dadt_diag[1, 1]
    lbd2 = dadt_diag[2, 2]

    # Computation of IOK tensor by rotation
    IOK = np.zeros((3, 3))
    IOK[0, 0] = alpha * lbd0
    IOK[1, 1] = alpha * lbd1
    IOK[2, 2] = alpha * lbd2

    dadt_rpr = -np.einsum("ik, kl, lj->ij", R, IOK, np.transpose(R))

    dadt = dadt_temp + dadt_rpr
    return dadt


def rsc_ode(a, A, D, W, xi, Ci=0.0, kappa=1.0, **kwargs):
    """ODE describing RSC model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    Ci : float
        Fiber interaction constant (typically 0 < Ci < 0.05).
    kappa : float
        Strain reduction factor (0 < kappa < 1).

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] Jin Wang, John F. O'Gara, and Charles L. Tucker,
       'An objective model for slow orientation kinetics in concentrated fiber
       suspensions:
       Theory and rheological evidence',
       Journal of Rheology 52, 1179, 2008.
       https://doi.org/10.1122/1.2946437

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    w, v = np.linalg.eig(a)
    L = (
        w[0] * np.einsum("i,j,k,l->ijkl", v[:, 0], v[:, 0], v[:, 0], v[:, 0])
        + w[1] * np.einsum("i,j,k,l->ijkl", v[:, 1], v[:, 1], v[:, 1], v[:, 1])
        + w[2] * np.einsum("i,j,k,l->ijkl", v[:, 2], v[:, 2], v[:, 2], v[:, 2])
    )
    M = (
        np.einsum("i,j,k,l->ijkl", v[:, 0], v[:, 0], v[:, 0], v[:, 0])
        + np.einsum("i,j,k,l->ijkl", v[:, 1], v[:, 1], v[:, 1], v[:, 1])
        + np.einsum("i,j,k,l->ijkl", v[:, 2], v[:, 2], v[:, 2], v[:, 2])
    )

    tensor4 = A + (1.0 - kappa) * (L - np.einsum("ijmn,mnkl->ijkl", M, A))

    dadt = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", tensor4, D)
        )
        + 2 * kappa * Ci * G * (delta - 3 * a)
    )
    return dadt


def ard_rsc_ode(a, A, D, W, xi, b1=0.0, kappa=1.0, b2=0, b3=0, b4=0, b5=0, **kwargs):
    """ODE describing ARD-RSC model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    b1 : float
        First parameter of rotary diffusion tensor (0 < b1 < 0.1).
    kappa : float
        Strain reduction factor (0 < kappa < 1).
    b2 : type
        Second parameter of rotary diffusion tensor.
    b3 : type
        Third parameter of rotary diffusion tensor.
    b4 : type
        Fourth parameter of rotary diffusion tensor.
    b5 : type
        Fith parameter of rotary diffusion tensor.

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] J. H. Phelps,  C. L. Tucker,
       'An anisotropic rotary diffusion model for fiber orientation in short- and
       long-fiber thermoplastics',
       Journal of Non-Newtonian Fluid Mechanics 156, 165-176, 2009.
       https://doi.org/10.1016/j.jnnfm.2008.08.002

    """
    G = np.sqrt(2.0 * np.einsum("ij,ij", D, D))
    delta = np.eye(3)

    w, v = np.linalg.eig(a)
    L = (
        w[0] * np.einsum("i,j,k,l->ijkl", v[:, 0], v[:, 0], v[:, 0], v[:, 0])
        + w[1] * np.einsum("i,j,k,l->ijkl", v[:, 1], v[:, 1], v[:, 1], v[:, 1])
        + w[2] * np.einsum("i,j,k,l->ijkl", v[:, 2], v[:, 2], v[:, 2], v[:, 2])
    )
    M = (
        np.einsum("i,j,k,l->ijkl", v[:, 0], v[:, 0], v[:, 0], v[:, 0])
        + np.einsum("i,j,k,l->ijkl", v[:, 1], v[:, 1], v[:, 1], v[:, 1])
        + np.einsum("i,j,k,l->ijkl", v[:, 2], v[:, 2], v[:, 2], v[:, 2])
    )
    if G > 0.0:
        C = (
            b1 * delta
            + b2 * a
            + b3 * np.einsum("ik,kj->ij", a, a)
            + b4 * D / G
            + b5 * np.einsum("ik,kj->ij", D, D) / (G * G)
        )
    else:
        C = np.eye(3)

    tensor4 = A + (1.0 - kappa) * (L - np.einsum("ijmn,mnkl->ijkl", M, A))

    dadt = (
        np.einsum("ik,kj->ij", W, a)
        - np.einsum("ik,kj->ij", a, W)
        + xi
        * (
            np.einsum("ik,kj->ij", D, a)
            + np.einsum("ik,kj->ij", a, D)
            - 2 * np.einsum("ijkl,kl->ij", tensor4, D)
        )
        + G
        * (
            2 * (C - (1 - kappa) * np.einsum("ijkl,kl->ij", M, C))
            - 2 * kappa * np.trace(C) * a
            - 5 * (np.einsum("ik,kj->ij", C, a) + np.einsum("ik,kj->ij", a, C))
            + 10 * np.einsum("ijkl,kl->ij", tensor4, C)
        )
    )
    return dadt


def mori_tanaka_ode(a, A, D, W, xi, c_f=0.0, **kwargs):
    """ODE describing the modified Jeffery equation based on the Mori-Tanaka model.

    Parameters
    ----------
    a : 3x3 numpy array
        Second-order fiber orientation tensor.
    A : 3x3x3x3 numpy array
        Fourth-order fiber orientation tensor.
    D : 3x3 numpy array
        Symmetric part of velocity gradient tensor.
    W : 3x3 numpy array
        Skew-symmetric part of velocity gradient tensor.
    xi : float
        Shape factor computed from aspect ratio.
    c_f : float
        Fiber volume fraction.

    Returns
    -------
    3x3 numpy array
        Orientation tensor rate.

    References
    ----------
    .. [1] T. Karl, T. Böhlke,
       'Generalized Micromechanical Formulation of Fiber Orientation Tensor Evolution
       Equations',
       International Journal of Mechanical Sciences 2023.
       https://doi.org/10.1016/j.ijmecsci.2023.108771
    """
    c_m_inv = 1.0 / (1.0 - c_f)
    return (
        np.einsum("ij, jk -> ik", W, a)
        - np.einsum("ij, jk -> ik", a, W)
        + xi
        * c_m_inv
        * (
            np.einsum("ij, jk -> ik", D, a)
            + np.einsum("ij, jk -> ik", a, D)
            - 2.0 * np.einsum("ijkl, kl -> ij", A, D)
        )
        - xi
        * c_f
        * c_m_inv
        * (
            np.einsum("ij, jk, kl -> il", D, a, a)
            + np.einsum("ij, jk, kl -> il", a, a, D)
            - 2.0 * np.einsum("ij, jk, kl -> il", a, D, a)
        )
    )


def integrate_ori_ode(t, a_flat, L, closure, ori_model, kwargs):
    """Wrapper to solve fiber reorientation ODE using `scipy` solvers.

    Parameters
    ----------
    t : float
        Time of evaluation.
    a_flat : 9x1 numpy array
        Flattened second-order fiber orientation tensor.
    L : function handle
        Function `L(t)` to retrieve velocity gradient at time `t`. Must return 3x3 numpy array.
    closure: function handle
        Function `closure(a)` to compute closure approximation. Must return 3x3x3x3 numpy array.
    ori_model: function handle
        Function `ori_model(a,A,D,W,**kwargs)` computing the rate of the orientation tensor.
    kwargs : dict
        Keyword arguments for function `ori_model`.

    Returns
    -------
    9x1 numpy array
        Orientation tensor rate.
    """
    a = a_flat.reshape((3, 3))
    A = closure(a)

    D = 0.5 * (L(t) + np.transpose(L(t)))
    W = 0.5 * (L(t) - np.transpose(L(t)))

    return ori_model(a, A, D, W, **kwargs).ravel()
