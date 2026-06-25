# ============================================================
# build_pyramid_db.py
#
# Builds Pyramid database from brightest FOV stars.
#
# Inputs
# -------
# bright_fov.npy
# bright_fov_idx.npy
#
# Output
# -------
# pyramid_db.npz
#
# Stored arrays
# -------------
# largest_edge
# ratio1
# ratio2
# star1
# star2
# star3
# star4
#
# ============================================================

import itertools
import numpy as np
from pathlib import Path


def angle_between(v1, v2):
    """
    Angular separation between two unit vectors.
    """
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)

    return np.arccos(dot)


BASE_DIR = Path(__file__).resolve().parent

print("Loading brightest FOV stars...")

vectors = np.load(BASE_DIR / "cache_fov.npy")
indices = np.load(BASE_DIR / "cache_fov_idx.npy")

n = len(vectors)

print(f"Stars used for Pyramid: {n}")

if n < 4:
    raise ValueError(
        "Need at least four stars."
    )

num_quadruples = n * (n - 1) * (n - 2) * (n - 3) // 24

print(f"Total pyramids: {num_quadruples}")

largest_edge = []
ratio1 = []
ratio2 = []

star1 = []
star2 = []
star3 = []
star4 = []

count = 0

for i, j, k, l in itertools.combinations(range(n), 4):

    # ---- six edge lengths ----
    edges = np.array([
        angle_between(vectors[i], vectors[j]),
        angle_between(vectors[i], vectors[k]),
        angle_between(vectors[i], vectors[l]),
        angle_between(vectors[j], vectors[k]),
        angle_between(vectors[j], vectors[l]),
        angle_between(vectors[k], vectors[l])
    ], dtype=np.float32)

    edges.sort()

    e6 = edges[-1]
    e5 = edges[-2]
    e4 = edges[-3]

    largest_edge.append(e6)

    ratio1.append(e5 / e6)
    ratio2.append(e4 / e6)

    star1.append(indices[i])
    star2.append(indices[j])
    star3.append(indices[k])
    star4.append(indices[l])

    count += 1

    if count % 1000 == 0:
        print(f"Processed {count}/{num_quadruples}")

# ------------------------------------------------------------
# Convert to arrays
# ------------------------------------------------------------
largest_edge = np.array(largest_edge, dtype=np.float32)

ratio1 = np.array(ratio1, dtype=np.float32)
ratio2 = np.array(ratio2, dtype=np.float32)

star1 = np.array(star1, dtype=np.int32)
star2 = np.array(star2, dtype=np.int32)
star3 = np.array(star3, dtype=np.int32)
star4 = np.array(star4, dtype=np.int32)

# ------------------------------------------------------------
# Sort database by largest edge
# ------------------------------------------------------------
order = np.argsort(largest_edge)

largest_edge = largest_edge[order]

ratio1 = ratio1[order]
ratio2 = ratio2[order]

star1 = star1[order]
star2 = star2[order]
star3 = star3[order]
star4 = star4[order]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
print("\nSaving pyramid database...")

np.savez_compressed(
    BASE_DIR / "pyramid_db.npz",

    largest_edge=largest_edge,

    ratio1=ratio1,
    ratio2=ratio2,

    star1=star1,
    star2=star2,
    star3=star3,
    star4=star4
)

print("Done.")
print("Saved pyramid_db.npz")