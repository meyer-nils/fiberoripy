# -*- coding: utf-8 -*-
"""Compute equivalent aspect ratios."""
import numpy as np


def get_cox_aspect_ratio(aspect_ratio):
    """Compute an equivalent aspect ratio according to Cox.

    Parameters
    ----------
    aspect_ratio : float
        Aspect ratio of a cylindrical fiber.

    Returns
    -------
    float
        Equivalent aspect ratio for an ellipsoid.

    References
    ----------
    .. [1] Cox, R.G.
       'The motion of long slender bodies in a viscous fluid Part 2. Shear flow.',
       J. Fluid Mech. 1971, 45, 625-657.
       https://doi.org/10.1017/S0022112071000259

    """
    return 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))


def get_zhang_aspect_ratio(aspect_ratio):
    """Compute an equivalent aspect ratio according to Zhang.

    Parameters
    ----------
    aspect_ratio : float
        Aspect ratio of a cylindrical fiber.

    Returns
    -------
    float
        Equivalent aspect ratio for an ellipsoid.

    References
    ----------
    .. [1] Zhang, D.; Smith, D.E.; Jack, D.A.; Montgomery-Smith, S.,
       'Numerical Evaluation of Single Fiber Motion for Short-Fiber-Reinforced Composite
       Materials Processing.'
       J. Manuf. Sci. Eng. 2011, 133, 51002.
       https://doi.org/10.1115/1.4004831

    """
    return (
        0.000035 * aspect_ratio ** 3
        - 0.00467 * aspect_ratio ** 2
        + 0.764 * aspect_ratio
        + 0.404
    )
