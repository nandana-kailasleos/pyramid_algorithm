# ============================================================
# pyramid_search.py
# ============================================================

import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

db = np.load(
    BASE_DIR / "catalog" / "pyramid_db.npz"
)

largest_edge = db["largest_edge"]
ratio1 = db["ratio1"]
ratio2 = db["ratio2"]

star1 = db["star1"]
star2 = db["star2"]
star3 = db["star3"]
star4 = db["star4"]


def search_pyramid(edge,
                   r1,
                   r2,
                   edge_tol=0.001,
                   ratio_tol=0.01):

    candidates = []

    mask = (
        np.abs(largest_edge - edge) < edge_tol
    ) & (
        np.abs(ratio1 - r1) < ratio_tol
    ) & (
        np.abs(ratio2 - r2) < ratio_tol
    )

    indices = np.where(mask)[0]

    for idx in indices:

        candidates.append((
            int(star1[idx]),
            int(star2[idx]),
            int(star3[idx]),
            int(star4[idx])
        ))

    return candidates