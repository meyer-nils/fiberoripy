"""Compute equivalent aspect ratios."""
import numpy as np


def get_cox_aspect_ratio(aspect_ratio):
    u"""Jeffrey's equivalent aspect ratio.

    Approximation from
    Cox et al.
    """
    return 1.24 * aspect_ratio / np.sqrt(np.log(aspect_ratio))


def get_zhang_aspect_ratio(aspect_ratio):
    """Jeffery's equivalent aspect ratio.

    Approximation from
    Zhang et al. 2011
    """
    return (
        0.000035 * aspect_ratio ** 3
        - 0.00467 * aspect_ratio ** 2
        + 0.764 * aspect_ratio
        + 0.404
    )


def get_gm_aspect_ratio(aspect_ratio):
    """Jeffery's equivalent aspect ratio.

    Approximation from
    Goldsmith and Mason
    """
    return 0.742 * aspect_ratio - 0.0017 * aspect_ratio ** 2
