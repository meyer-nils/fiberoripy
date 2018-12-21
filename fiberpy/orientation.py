"""Orientation models."""
import numpy as np

from tensoroperations import generate_fourth_order_tensor


def get_equivalent_aspect_ratio(aspect_ratio):
    """Get Jeffrey's equivalent aspect ratio.

    Compare Cox et al.
    """
    return 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))


def rsc_ode(a, t, ar, D, W, Ci=0.0, kappa=1.0):
    """ODE describing RSC model."""
    a = np.reshape(a, (3, 3))
    A = generate_fourth_order_tensor(a)
    G = np.linalg.norm(D(t), ord='fro')
    lbd = (ar**2 - 1) / (ar**2 + 1)
    delta = np.eye(3)

    w, v = np.linalg.eig(a)
    L = (w[0]*np.einsum('i,j,k,l->ijkl', v[:, 0], v[:, 0], v[:, 0], v[:, 0])
         + w[1]*np.einsum('i,j,k,l->ijkl', v[:, 1], v[:, 1], v[:, 1], v[:, 1])
         + w[2]*np.einsum('i,j,k,l->ijkl', v[:, 2], v[:, 2], v[:, 2], v[:, 2]))
    M = (np.einsum('i,j,k,l->ijkl', v[:, 0], v[:, 0], v[:, 0], v[:, 0])
         + np.einsum('i,j,k,l->ijkl', v[:, 1], v[:, 1], v[:, 1], v[:, 1])
         + np.einsum('i,j,k,l->ijkl', v[:, 2], v[:, 2], v[:, 2], v[:, 2]))

    closure = A + (1.0-kappa)*(L-np.einsum('ijmn,mnkl->ijkl', M, A))

    dadt = (np.einsum('ik,kj->ij', W(t), a)
            - np.einsum('ik,kj->ij', a, W(t))
            + lbd*(np.einsum('ik,kj->ij', D(t), a)
                   + np.einsum('ik,kj->ij', a, D(t))
                   - 2*np.einsum('ijkl,kl->ij', closure, D(t)))
            + 2*Ci*G*(delta-3*a))
    return dadt.ravel()


def folgar_tucker_ode(a, t, ar, D, W, Ci=0.0):
    """ODE describing Folgar-Tucker model."""
    a = np.reshape(a, (3, 3))
    A = generate_fourth_order_tensor(a)
    G = np.linalg.norm(D(t), ord='fro')
    lbd = (ar**2 - 1) / (ar**2 + 1)
    delta = np.eye(3)

    dadt = (np.einsum('ik,kj->ij', W(t), a)
            - np.einsum('ik,kj->ij', a, W(t))
            + lbd*(np.einsum('ik,kj->ij', D(t), a)
                   + np.einsum('ik,kj->ij', a, D(t))
                   - 2*np.einsum('ijkl,kl->ij', A, D(t)))
            + 2*Ci*G*(delta-3*a))
    return dadt.ravel()
