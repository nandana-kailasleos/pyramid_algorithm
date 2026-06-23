# ============================================================
# geometry_utils.py
#
# Utility functions used by Pyramid algorithm.
# ============================================================

import numpy as np


def angle_between(v1, v2):
    """
    Angular separation between two unit vectors.
    Returns radians.
    """

    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)

    return np.arccos(dot)


def compute_edges(v1, v2, v3, v4):
    """
    Compute the six angular distances.
    """

    edges = np.array([
        angle_between(v1, v2),
        angle_between(v1, v3),
        angle_between(v1, v4),
        angle_between(v2, v3),
        angle_between(v2, v4),
        angle_between(v3, v4)
    ])

    return edges


def compute_invariants(v1, v2, v3, v4):
    """
    Compute Pyramid invariants.
    """

    edges = compute_edges(v1, v2, v3, v4)

    edges.sort()

    e1 = edges[0]
    e2 = edges[1]
    e6 = edges[5]

    ratio1 = e1 / e6
    ratio2 = e2 / e6

    return e6, ratio1, ratio2


def rms_error(observed_edges, catalog_edges):
    """
    RMS error between two edge sets.
    """

    observed_edges = np.sort(observed_edges)
    catalog_edges = np.sort(catalog_edges)

    return np.sqrt(
        np.mean(
            (observed_edges - catalog_edges) ** 2
        )
    )