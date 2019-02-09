"""Collection of tools to handle fiber orientation tensors."""
import numpy as np


def compute_closure(a, closure='IBOF', N=1000):
    """Create a fourth order tensor from a second order tensor."""
    # assertation
    assert(closure in ('IBOF', 'LINEAR', 'HYBRID', 'QUADRATIC', 'RANDOM'))
    if closure == 'IBOF':
        return IBOF_closure(a)
    if closure == 'HYBRID':
        return hybrid_closure(a)
    if closure == 'LINEAR':
        return linear_closure(a)
    if closure == 'QUADRATIC':
        return quadratic_closure(a)
    if closure == 'RANDOM':
        return random_closure(a, N)


def assert_fot_properties(A):
    """Assert fiber properties."""
    # assert symmetry and shape
    assert(np.shape(A) == (3, 3))
    # assert(A[0, 1] == A[1, 0])
    # assert(A[0, 2] == A[2, 0])
    # assert(A[1, 2] == A[2, 1])


def random_closure(a, N=1000):
    """Sample a random fiber orientaion given a."""
    orientations = []
    for i in range(N):
        phi = np.random.uniform(0, np.pi*2)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        rand_orientation = np.array([x, y, z])
        p = np.dot(a, rand_orientation)
        p = p/np.linalg.norm(p)
        orientations.append(p)

    a.fill(0.0)
    A = np.zeros((3, 3, 3, 3))
    for o in orientations:
        a += 1.0/N*np.einsum('i,j->ij', o, o)
        A += 1.0/N*np.einsum('i,j,k,l->ijkl', o, o, o, o)
    return A


def linear_closure(A):
    """Generate a linear closure.

    This is appropiate for UD directed fibers. See 'Modified hybrid closure
    approximation for prediction of flow-induced fiber orientation'
    - Kyeong-Hee Han and Yong-Taek Im
    """
    assert_fot_properties(A)

    delta = np.eye(3)
    return (1.0/7.0*(np.einsum('ij,kl->ijkl', A, delta) +
                     np.einsum('ik,jl->ijkl', A, delta) +
                     np.einsum('il,jk->ijkl', A, delta) +
                     np.einsum('kl,ij->ijkl', A, delta) +
                     np.einsum('jl,ik->ijkl', A, delta) +
                     np.einsum('jk,il->ijkl', A, delta))
            - 1.0 / 35.0 * (np.einsum('ij,kl->ijkl', delta, delta) +
                            np.einsum('ik,jl->ijkl', delta, delta) +
                            np.einsum('il,jk->ijkl', delta, delta))
            )


def quadratic_closure(A):
    """Generate a linear closure.

    This is appropiate for isotropic orientation states. See 'Modified hybrid
    closure approximation for prediction of flow-induced fiber orientation'
    ~ Kyeong-Hee Han and Yong-Taek Im
    """
    assert_fot_properties(A)
    return np.einsum('ij,kl->ijkl', A, A)


def hybrid_closure(A):
    """Generate a hybrid closure.

    See 'Modified hybrid closure approximation for prediction of flow-induced
    fiber orientation' - Kyeong-Hee Han and Yong-Taek Im
    """
    assert_fot_properties(A)
    f = 1.0-27.0*np.linalg.det(A)
    return (1.0-f)*linear_closure(A)+f*quadratic_closure(A)


def IBOF_closure(A):
    """Generate IBOF closure.

    This function utilizes a invariant based optimal fitting closure to
    generate a fourth order tensor from a second order tensor.
    Reference: Chung & Kwon paper about 'Invariant-based optimal fitting
    closure approximation for the numerical prediction of flow-induced fiber
    orientation

    Input:
    A : 3x3 fiber orientation tensor
    """
    assert_fot_properties(A)

    # second invariant
    II = (A[0, 0] * A[1, 1] + A[1, 1] * A[2, 2] + A[0, 0] * A[2, 2] -
          A[0, 1] * A[1, 0] - A[1, 2] * A[2, 1] - A[0, 2] * A[2, 0])

    # third invariant
    III = np.linalg.det(A)

    # coefficients from Chung & Kwon paper
    C1 = np.zeros((1, 21))

    C2 = np.zeros((1, 21))

    C3 = np.array([[0.24940908165786E2,
                    -0.435101153160329E3,
                    0.372389335663877E4,
                    0.703443657916476E4,
                    0.823995187366106E6,
                    -0.133931929894245E6,
                    0.880683515327916E6,
                    -0.991630690741981E7,
                    -0.159392396237307E5,
                    0.800970026849796E7,
                    -0.237010458689252E7,
                    0.379010599355267E8,
                    -0.337010820273821E8,
                    0.322219416256417E5,
                    -0.257258805870567E9,
                    0.214419090344474E7,
                    -0.449275591851490E8,
                    -0.213133920223355E8,
                    0.157076702372204E10,
                    -0.232153488525298E5,
                    -0.395769398304473E10]])

    C4 = np.array([[-0.497217790110754E0,
                    0.234980797511405E2,
                    -0.391044251397838E3,
                    0.153965820593506E3,
                    0.152772950743819E6,
                    -0.213755248785646E4,
                    -0.400138947092812E4,
                    -0.185949305922308E7,
                    0.296004865275814E4,
                    0.247717810054366E7,
                    0.101013983339062E6,
                    0.732341494213578E7,
                    -0.147919027644202E8,
                    -0.104092072189767E5,
                    -0.635149929624336E8,
                    -0.247435106210237E6,
                    -0.902980378929272E7,
                    0.724969796807399E7,
                    0.487093452892595E9,
                    0.138088690964946E5,
                    -0.160162178614234E10]])

    C5 = np.zeros((1, 21))

    C6 = np.array([[0.234146291570999E2,
                    -0.412048043372534E3,
                    0.319553200392089E4,
                    0.573259594331015E4,
                    -0.485212803064813E5,
                    -0.605006113515592E5,
                    -0.477173740017567E5,
                    0.599066486689836E7,
                    -0.110656935176569E5,
                    -0.460543580680696E8,
                    0.203042960322874E7,
                    -0.556606156734835E8,
                    0.567424911007837E9,
                    0.128967058686204E5,
                    -0.152752854956514E10,
                    -0.499321746092534E7,
                    0.132124828143333E9,
                    -0.162359994620983E10,
                    0.792526849882218E10,
                    0.466767581292985E4,
                    -0.128050778279459E11]])

    # build matrix of coefficients by stacking vectors
    C = np.vstack((C1, C2, C3, C4, C5, C6))

    # compute parameters as fith order polynom based on invariants
    beta3 = (C[2, 0]+C[2, 1]*II+C[2, 2]*II**2+C[2, 3]*III+C[2, 4]*III**2
             + C[2, 5]*II*III+C[2, 6]*II**2*III+C[2, 7]*II*III**2+C[2, 8]*II**3
             + C[2, 9]*III**3+C[2, 10]*II**3*III+C[2, 11]*II**2*III**2
             + C[2, 12]*II*III**3+C[2, 13]*II**4+C[2, 14]*III**4
             + C[2, 15]*II**4*III+C[2, 16]*II**3*III**2+C[2, 17]*II**2*III**3
             + C[2, 18]*II*III**4+C[2, 19]*II**5+C[2, 20]*III**5)

    beta4 = (C[3, 0]+C[3, 1]*II+C[3, 2]*II**2+C[3, 3]*III+C[3, 4]*III**2
             + C[3, 5]*II*III+C[3, 6]*II**2*III+C[3, 7]*II*III**2+C[3, 8]*II**3
             + C[3, 9]*III**3+C[3, 10]*II**3*III+C[3, 11]*II**2*III**2
             + C[3, 12]*II*III**3+C[3, 13]*II**4+C[3, 14]*III**4
             + C[3, 15]*II**4*III+C[3, 16]*II**3*III**2+C[3, 17]*II**2*III**3
             + C[3, 18]*II*III**4+C[3, 19]*II**5+C[3, 20]*III**5)

    beta6 = (C[5, 0]+C[5, 1]*II+C[5, 2]*II**2+C[5, 3]*III+C[5, 4]*III**2
             + C[5, 5]*II*III+C[5, 6]*II**2*III+C[5, 7]*II*III**2+C[5, 8]*II**3
             + C[5, 9]*III**3+C[5, 10]*II**3*III+C[5, 11]*II**2*III**2
             + C[5, 12]*II*III**3+C[5, 13]*II**4+C[5, 14]*III**4
             + C[5, 15]*II**4*III+C[5, 16]*II**3*III**2+C[5, 17]*II**2*III**3
             + C[5, 18]*II*III**4+C[5, 19]*II**5+C[5, 20]*III**5)

    beta1 = 3.0/5.0*(-1.0/7.0
                     + 1.0/5.0*beta3*(1.0/7.0+4.0/7.0*II+8.0/3.0*III)
                     - beta4*(1.0/5.0-8.0/15.0*II-14.0/15.0*III)
                     - beta6*(1.0/35.0
                              - 24.0/105.0*III
                              - 4.0/35.0*II
                              + 16.0/15.0*II*III
                              + 8.0/35.0*II**2))

    beta2 = 6.0/7.0*(1.0
                     - 1.0/5.0*beta3*(1.0+4.0*II)
                     + 7.0/5.0*beta4*(1.0/6.0-II)
                     - beta6*(-1.0/5.0+2.0/3.0*III+4.0/5.0*II-8.0/5.0*II**2))

    beta5 = -4.0/5.0*beta3-7.0/5.0*beta4-6.0/5.0*beta6*(1.0-4.0/3.0*II)

    # second order identy matrix
    delta = np.eye(3)

    # generate fourth order tensor with parameters and tensor algebra
    return (beta1*symm(np.einsum('ij,kl->ijkl', delta, delta))
            + beta2*symm(np.einsum('ij,kl->ijkl', delta, A))
            + beta3*symm(np.einsum('ij,kl->ijkl', A, A))
            + beta4*symm(np.einsum('ij,km,ml->ijkl', delta, A, A))
            + beta5*symm(np.einsum('ij,km,ml->ijkl', A, A, A))
            + beta6*symm(np.einsum('im,mj,kn,nl->ijkl', A, A, A, A)))


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
                    S[i, j, k, l] = 1.0/24.0*(A[i, j, k, l]
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
                                              + A[k, j, l, i])
    return S
