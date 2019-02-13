import numpy as np
import matplotlib.pyplot as plt


def fit_bay(phir):
    """Compute interaction coefficient with Bay's fit.

    This is valid for Phi*r = O(1).

    Reference: R.S. Bay, 'Fibre Orientation in injection molded composites; a
    comparison of theory and experiment', PhD Thesis, Univ. Illinois, 1991
    """
    # Set nan's outside applicable domain
    phir[phir > 10] = np.nan
    Ci = 0.0184*np.exp(-0.7148*phir)
    return Ci


def fit_phan_tien(phir):
    """Compute interaction coefficient with Phan-Tien's fit.

    This is valid for ...

    Reference:
    N.Phan-Thien, X.-J.Fan, R.I.Tanner, R.Zheng, 'Folgar-Tucker constant for a
    fibre suspension in a Newtonian fluid', Journal of Non-Newtonian Fluid
    Mechanics, 2002.
    https://doi.org/10.1016/S0377-0257(02)00006-X
    """
    A = 0.030
    B = 0.224
    Ci = A*(1.0-np.exp(-B*phir))
    return Ci


phir = np.linspace(0, 4, 100)

sph_phir = 10*np.array([0.01, 0.04, 0.10, 0.40])
sph_fits = np.array([3.19108377e-05, 5.88520754e-04, 0.00176328,  0.00309774])

plt.plot(phir, fit_bay(phir), phir, fit_phan_tien(phir),
         sph_phir, sph_fits, '*')
plt.show()
