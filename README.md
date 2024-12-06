[![LICENSE](https://black.readthedocs.io/en/stable/_static/license.svg)](https://raw.github.com/nilsmeyerkit/fiberoripy/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/fiberoripy/badge/?version=latest)](https://fiberoripy.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fiberoripy)
[![PyPI version](https://badge.fury.io/py/fiberoripy.svg)](https://badge.fury.io/py/fiberoripy)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/282262907.svg)](https://zenodo.org/badge/latestdoi/282262907)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nilsmeyerkit/fiberoripy/HEAD)

# Fiberoripy
This python package provides basic functionality and tools for fiber orientations and
closure models.

For example, the Jupyter Notebook

    example/orientation/comparison_perfectshear.ipynb

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
 * __Mori-Tanaka__:\
 T. Karl, T. Böhlke,\
 'Generalized Micromechanical Formulation of Fiber Orientation Tensor Evolution Equations',\
 International Journal of Mechanical Sciences, 2023.\
 (https://doi.org/10.1016/j.ijmecsci.2023.108771)

## Closures
* __Linear__, __Quadratic__, __Hybrid__:\
Kyeong-Hee Han and Yong-Taek Im,\
'Modified hybrid closure approximation for prediction of flow-induced fiber orientation', \
Journal of Rheology 43, 569, 1999.\
(https://doi.org/10.1122/1.551002)
* __IBOF__:\
Du Hwan Chung and Tai Hun Kwon,\
'Invariant-based optimal fitting closure approximation for the numerical prediction of flow-induced fiber orientation',\
Journal of Rheology 46(1), 169-194, 2002.\
(https://doi.org/10.1122/1.1423312)
* __ORF__, __ORW__, __ORW3__:\
Joaquim S. Cintra and Charles L. Tucker III,\
'Orthotropic closure approximations for flow-induced fiber orientation',\
Journal of Rheology, 39(6), 1095-1122, 1995.
(https://doi.org/10.1122/1.550630) \
Du Hwan Chung and Tai Hun Kwon, \
'Improved model of orthotropic closure approximation for flow induced fiber orientation', \
Polymer Composites, 22(5), 636-649, 2001.\
(https://doi.org/10.1002/pc.10566)
* __SQC__:\
Tobias Karl, Davide Gatti, Bettina Frohnapfel and Thomas Böhlke,\
'Asymptotic fiber orientation states of the quadratically closed Folgar-Tucker equation and a subsequent closure improvement',\
Journal of Rheology 65(5) : 999-1022, 2021\
(https://doi.org/10.1122/8.0000245)
* __SIC__, __SIHYB__:\
Tobias Karl,  Matti Schneider and Thomas Böhlke,\
'On fully symmetric implicit closure approximations for fiber orientation tensors',\
Journal of Non-Newtonian Fluid Mechanics 318 : 105049, 2023.\
(https://doi.org/10.1016/j.jnnfm.2023.105049)

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
