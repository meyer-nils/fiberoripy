[![LICENSE](https://black.readthedocs.io/en/stable/_static/license.svg)](https://raw.github.com/nilsmeyerkit/fiberoripy/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/fiberoripy/badge/?version=latest)](https://fiberoripy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/fiberoripy.svg)](https://badge.fury.io/py/fiberoripy)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Fiberoripy
This python package provides basic functionality and tools for fiber orientations and
closure models.

For example,

    example/orientation/comparison_perfectshear.py

should reproduce Figure 2 in *Favaloro, A.J., Tucker III, C.L., Composites Part A, 126 (2019)*:

  ![example_image](https://raw.github.com/nilsmeyerkit/fiberoripy/master/docs/images/example.png)


## Orientation models
Following models have been implemented:

 * __Jeffery__:\
 G.B. Jeffery,\
 'The motion of ellipsoidal particles immersed in a viscous fluid',\
 Proceedings of the Royal Society A, 1922.\
 (https://doi.org/10.1098/rspa.1922.0078)
 * __Folgar-Tucker__:\
 F. Folgar, C.L. Tucker,\
 'Orientation behavior of fibers in concentrated suspensions',\
 Journal of Reinforced Plastic Composites 3, 98-119, 1984.\
 (https://doi.org/10.1177%2F073168448400300201)
 * __FTMS__:\
 A. Latz, U. Strautins, D. Niedziela,\
 'Comparative numerical study of two concentrated fiber suspension models',\
 Journal of Non-Newtonian Fluid Mechanics 165, 764-781, 2010.\
 (https://doi.org/10.1016/j.jnnfm.2010.04.001)
 * __iARD(-RPR)__:\
 H.C. Tseng, R.Y. Chang, C.H. Hsu,\
 'An objective tensor to predict anisotropic fiber orientation in concentrated susp ensions',\
 Journal of Rheology 60, 215, 2016.\
 (https://doi.org/10.1122/1.4939098)
 * __pARD(-RPR)__:\
 H.C. Tseng, R.Y. Chang, C.H. Hsu,\
 'The use of principal spatial tensor to predict anisotropic fiber orientation in concentrated fiber suspensions',\
 Journal of Rheology 62, 313, 2017.\
 (https://doi.org/10.1122/1.4998520)
 * __MRD__:\
 A. Bakharev, H. Yu, R. Speight and J. Wang,\
 “Using New Anisotropic Rotational Diffusion Model To Improve Prediction Of Short Fibers in Thermoplastic Injection Molding",\
 ANTEC, Orlando, 2018.
 * __RSC__:\
 J. Wang, J.F. O'Gara, and C.L. Tucker,\
 'An objective model for slow orientation kinetics in concentrated fiber suspensions: Theory and rheological evidence',\
 Journal of Rheology 52, 1179, 2008.\
 (https://doi.org/10.1122/1.2946437)
 * __ARD-RSC__:\
 J. H. Phelps,  C. L. Tucker,\
 'An anisotropic rotary diffusion model for fiber orientation in short- and long-fiber thermoplastics',\
 Journal of Non-Newtonian Fluid Mechanics 156, 165-176, 2009.\
 (https://doi.org/10.1016/j.jnnfm.2008.08.002)

## Closures
 * Linear
 * Quadratic
 * IBOF
 * Sampling from directions

## Approximations for equivalent aspect ratios
 * __Cox__:\
 R.G. Cox,\
 'The motion of long slender bodies in a viscous fluid. Part 2. Shear flow.',\
 J. Fluid Mech. 1971, 45, 625–657.\
 (http://doi.org/10.1017/S0022112071000259)
 * __Zhang__:\
 D. Zhang, D.E. Smith, D.A. Jack, S. Montgomery-Smith,\
 'Numerical Evaluation of Single Fiber Motion for Short-Fiber-Reinforced Composite Materials Processing',\
 J. Manuf. Sci. Eng. 2011, 133, 51002.\
 (http://doi.org/10.1115/1.4004831)
