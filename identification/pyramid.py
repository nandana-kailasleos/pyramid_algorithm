# ============================================================
# pyramid.py
# ============================================================

import numpy as np

from identification.geometry_utils import compute_invariants
from identification.pyramid_search import search_pyramid
from identification.pyramid_verify import verify_candidate


def identify(v1, v2, v3, v4):

    edge, r1, r2 = compute_invariants(
        v1,
        v2,
        v3,
        v4
    )

    candidates = search_pyramid(
        edge,
        r1,
        r2
    )

    if len(candidates) == 0:
        return []

    stars = np.load(
        "catalog/stars.npz",
        allow_pickle=True
    )

    vectors = stars["unit_vectors"]

    observed = [v1, v2, v3, v4]

    verified = []

    for c in candidates:

        i1, i2, i3, i4 = c

        catalog_vectors = [
            vectors[i1],
            vectors[i2],
            vectors[i3],
            vectors[i4]
        ]

        ok, error = verify_candidate(
            observed,
            catalog_vectors
        )

        if ok:
            verified.append(
                (
                    i1,
                    i2,
                    i3,
                    i4,
                    np.degrees(error)
                )
            )

    return verified