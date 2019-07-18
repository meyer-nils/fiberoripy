import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save


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


phir = np.linspace(0, 3, 100)

sph_phir = 5*np.array([0.01, 0.10, 0.30])
ft_phir = np.array([0.16, 1.28, 2.56, 0.0332, 0.249, 0.498, 1.079])
sph_fits = np.array([0.00262122, 0.00515386, 0.01103383])
ft_fit = np.array([0.0038, 0.0081, 0.0165, 0.0032, 0.0035, 0.0042, 0.0044])

plt.plot(phir, fit_phan_tien(phir), 'k',
         ft_phir, ft_fit, '+k',
         sph_phir, sph_fits, '.k')
plt.yscale('log')
plt.xlabel(r"$\phi r_p$")
plt.ylabel("Interaction coefficient $C_i$")
plt.legend(["Phan-Tien et al.", "Folgar & Tucker", "SPH Simulation"],
           loc="lower right")
plt.tight_layout()

# save tikz figure (width means individual subplot width!)
tikz_save('ci_comparison.tex', figurewidth='10cm')
plt.show()
