# ============================================================
# pyramid.py
# ============================================================

import numpy as np
from identification.geometry_utils import compute_invariants, compute_edges, rms_error
from identification.pyramid_search import search_pyramid
from identification.pyramid_verify import verify_candidate


def cross_validate(identified_ids, catalog_vectors_all,
                   observed_extra, angle_tol=np.radians(0.1)):
    """
    After identifying 4 stars, check if at least 2 additional
    observed stars match predicted catalog positions.
    observed_extra: list of additional observed unit vectors
    Returns True if cross-validation passes.
    """
    if len(observed_extra) < 2:
        return True  # not enough extra stars to check, allow pass

    confirmed = 0
    for obs in observed_extra:
        for cat_v in catalog_vectors_all:
            if np.arccos(np.clip(np.dot(obs, cat_v), -1, 1)) < angle_tol:
                confirmed += 1
                break

    return confirmed >= 2


def identify(v1, v2, v3, v4, extra_observed=None):
    """
    Identify a 4-star pyramid.

    Parameters
    ----------
    v1..v4 : unit vectors of the 4 observed stars
    extra_observed : optional list of additional observed unit vectors
                     used for cross-validation (pass as many as detected)
    """

    observed = [v1, v2, v3, v4]

    # Load catalog
    stars = np.load("catalog/stars.npz", allow_pickle=True)
    vectors = stars["unit_vectors"]

    # Compute invariants ONCE (they are order-independent)
    edge, r1, r2 = compute_invariants(v1, v2, v3, v4)

    candidates = search_pyramid(edge, r1, r2)

    if len(candidates) == 0:
        return []

    verified = []

    for c in candidates:
        i1, i2, i3, i4 = c

        catalog_vectors = [
            vectors[i1],
            vectors[i2],
            vectors[i3],
            vectors[i4]
        ]

        ok, error, confidence = verify_candidate(observed, catalog_vectors)

        if ok:
            # Cross-validate with extra observed stars if provided
            if extra_observed:
                passes = cross_validate(
                    [i1, i2, i3, i4],
                    vectors,
                    extra_observed
                )
                if not passes:
                    continue  # reject false match

            verified.append((i1, i2, i3, i4, np.degrees(error), confidence))

    if len(verified) == 0:
        return []

    # Return best match (lowest RMS error)
    verified.sort(key=lambda x: x[4])
    return [verified[0]]