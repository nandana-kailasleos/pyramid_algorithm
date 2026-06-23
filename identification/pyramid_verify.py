# ============================================================
# pyramid_verify.py
# ============================================================

import numpy as np
from identification.geometry_utils import (
    compute_edges,
    rms_error
)


def verify_candidate(
        observed_vectors,
        catalog_vectors,
        threshold=np.radians(0.01)):
    """
    Verify one candidate pyramid.

    Parameters
    ----------
    observed_vectors : list of 4 observed vectors

    catalog_vectors : list of 4 catalog vectors

    threshold : RMS error threshold (radians)

    Returns
    -------
    valid : bool
    error : float
    """

    obs_edges = compute_edges(
        observed_vectors[0],
        observed_vectors[1],
        observed_vectors[2],
        observed_vectors[3]
    )

    cat_edges = compute_edges(
        catalog_vectors[0],
        catalog_vectors[1],
        catalog_vectors[2],
        catalog_vectors[3]
    )

    error = rms_error(
        obs_edges,
        cat_edges
    )

    return error < threshold, error