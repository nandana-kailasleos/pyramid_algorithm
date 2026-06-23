# ============================================================
# pyramid.py
# ============================================================

import numpy as np
from itertools import permutations

from identification.geometry_utils import compute_invariants
from identification.pyramid_search import search_pyramid
from identification.pyramid_verify import verify_candidate


def identify(v1, v2, v3, v4):

    # Store observed stars
    observed = [v1, v2, v3, v4]

    # Load catalog vectors
    stars = np.load(
        "catalog/stars.npz",
        allow_pickle=True
    )

    vectors = stars["unit_vectors"]

    verified = []

    # --------------------------------------------------------
    # Try all 24 permutations of the observed stars
    # --------------------------------------------------------
    for perm in permutations(observed):

        edge, r1, r2 = compute_invariants(
            perm[0],
            perm[1],
            perm[2],
            perm[3]
        )

        candidates = search_pyramid(
            edge,
            r1,
            r2
        )

        if len(candidates) == 0:
            continue

        # ----------------------------------------------------
        # Verify candidate pyramids
        # ----------------------------------------------------
        for c in candidates:

            i1, i2, i3, i4 = c

            catalog_vectors = [
                vectors[i1],
                vectors[i2],
                vectors[i3],
                vectors[i4]
            ]

            ok, error, confidence = verify_candidate(
                perm,
                catalog_vectors
            )

            if ok:

                verified.append(
                    (
                        i1,
                        i2,
                        i3,
                        i4,
                        np.degrees(error),
                        confidence
                    )
                )

                # Stop after first verified match
                return verified

    return verified