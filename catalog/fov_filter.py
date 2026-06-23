# ============================================================
# fov_filter.py
#
# Filters the catalog and selects stars inside the camera FOV.
#
# Input:
#     stars.npz
#
# Output:
#     cache_fov.npy
#     cache_fov_idx.npy
#
# ============================================================

import numpy as np
from pathlib import Path


def fov_filter(vectors, center_vec, fov_deg):
    """
    Returns indices of stars inside the FOV cone.
    """

    fov_rad = np.radians(fov_deg)

    dots = vectors @ center_vec

    indices = np.where(dots >= np.cos(fov_rad))[0]

    return indices


def run_fov_filter():

    BASE_DIR = Path(__file__).resolve().parent

    stars_file = BASE_DIR / "stars.npz"

    # --------------------------------------------------------
    # Load star catalog
    # --------------------------------------------------------
    print("Loading stars.npz...")

    data = np.load(stars_file, allow_pickle=True)

    vectors = data["unit_vectors"]

    print(f"Total catalog stars: {len(vectors)}")

    # --------------------------------------------------------
    # Get boresight vector
    # --------------------------------------------------------
    print("\nEnter camera boresight vector")

    try:
        bx = float(input("bx: "))
        by = float(input("by: "))
        bz = float(input("bz: "))
    except ValueError:
        print("ERROR: Invalid input.")
        return

    center_vec = np.array([bx, by, bz], dtype=np.float64)

    norm = np.linalg.norm(center_vec)

    if norm < 1e-9:
        print("ERROR: Zero vector entered.")
        return

    center_vec = center_vec / norm

    print("Normalized boresight:", center_vec)

    # --------------------------------------------------------
    # Get FOV
    # --------------------------------------------------------
    try:
        fov = float(input("\nEnter FOV (2 to 5 degrees): "))
    except ValueError:
        print("ERROR: Invalid FOV.")
        return

    if not (2 <= fov <= 5):
        print("ERROR: FOV must be between 2° and 5°.")
        return

    # --------------------------------------------------------
    # Filter stars
    # --------------------------------------------------------
    indices = fov_filter(
        vectors,
        center_vec,
        fov
    )

    fov_vectors = vectors[indices]

    print(f"\nStars inside FOV: {len(fov_vectors)}")

    if len(fov_vectors) < 4:
        print(
            "WARNING: Pyramid requires at least 4 stars."
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    np.save(BASE_DIR / "cache_fov.npy", fov_vectors)

    np.save(BASE_DIR / "cache_fov_idx.npy", indices)

    print("\nSaved:")
    print("cache_fov.npy")
    print("cache_fov_idx.npy")

    print("\nDone.")


if __name__ == "__main__":
    run_fov_filter()