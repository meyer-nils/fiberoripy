"""Compute equivalent aspect ratios."""
import numpy as np


def get_cox_aspect_ratio(aspect_ratio):
    u"""Jeffrey's equivalent aspect ratio.

    Approximation from
    Cox, R.G. The motion of long slender bodies in a viscous fluid. Part 2. S
    Shear flow. J. Fluid Mech. 1971, 45, 625–657.
    """
    return 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))


def get_zhang_aspect_ratio(aspect_ratio):
    """Jeffery's equivalent aspect ratio.

    Approximation from
    Zhang, D.; Smith, D.E.; Jack, D.A.; Montgomery-Smith, S. Numerical Evaluation of
    Single Fiber Motion for Short-Fiber-Reinforced Composite Materials Processing.
    J. Manuf. Sci. Eng. 2011, 133, 51002.
    """
    return (
        0.000035 * aspect_ratio ** 3
        - 0.00467 * aspect_ratio ** 2
        + 0.764 * aspect_ratio
        + 0.404
    )
