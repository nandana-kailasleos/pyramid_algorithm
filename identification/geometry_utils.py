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
    Returns a NumPy array of six edge lengths (radians).
    """

    edges = np.array([
        angle_between(v1, v2),  # AB
        angle_between(v1, v3),  # AC
        angle_between(v1, v4),  # AD
        angle_between(v2, v3),  # BC
        angle_between(v2, v4),  # BD
        angle_between(v3, v4)   # CD
    ])

    return edges


def compute_invariants(v1, v2, v3, v4):
    """
    Compute order-independent Pyramid invariants.

    Returns
    -------
    largest_edge : float
        Largest of the six edges.

    ratio1 : float
        Second-largest edge divided by largest edge.

    ratio2 : float
        Third-largest edge divided by largest edge.
    """

    edges = compute_edges(v1, v2, v3, v4)

    # Sort edges in ascending order
    edges = np.sort(edges)

    # Largest edge
    largest_edge = edges[-1]

    # Ratios based on next two largest edges
    ratio1 = edges[-2] / largest_edge
    ratio2 = edges[-3] / largest_edge

    return largest_edge, ratio1, ratio2


def rms_error(observed_edges, catalog_edges):
    """
    Compute RMS error between two sets of six edges.

    Parameters
    ----------
    observed_edges : array-like
        Six observed edge lengths.

    catalog_edges : array-like
        Six catalog edge lengths.

    Returns
    -------
    float
        RMS error in radians.
    """

    observed_edges = np.sort(observed_edges)
    catalog_edges = np.sort(catalog_edges)

    return np.sqrt(
        np.mean(
            (observed_edges - catalog_edges) ** 2
        )
    )