# ============================================================
# test_pyramid.py
#
# Tests the Pyramid algorithm using the first four stars
# from the FOV subset and adds realistic sensor noise.
# ============================================================

import numpy as np

from identification.pyramid import identify


# ------------------------------------------------------------
# Load FOV star vectors
# ------------------------------------------------------------
vectors = np.load(
    "catalog/cache_fov.npy"
)

print(f"\nStars available in FOV: {len(vectors)}")

if len(vectors) < 4:
    print("Need at least 4 stars.")
    exit()


# ------------------------------------------------------------
# Select four stars
# ------------------------------------------------------------
v1 = vectors[0].copy()
v2 = vectors[1].copy()
v3 = vectors[2].copy()
v4 = vectors[3].copy()


# ------------------------------------------------------------
# Add Gaussian noise (simulates sensor errors)
# ------------------------------------------------------------
noise_sigma = 1e-4

v1 += np.random.normal(0, noise_sigma, 3)
v2 += np.random.normal(0, noise_sigma, 3)
v3 += np.random.normal(0, noise_sigma, 3)
v4 += np.random.normal(0, noise_sigma, 3)


# ------------------------------------------------------------
# Renormalize vectors to unit length
# ------------------------------------------------------------
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)
v3 = v3 / np.linalg.norm(v3)
v4 = v4 / np.linalg.norm(v4)


# ------------------------------------------------------------
# Run Pyramid identification
# ------------------------------------------------------------
matches = identify(
    v1,
    v2,
    v3,
    v4
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print()
print("Verified matches:\n")

if len(matches) == 0:

    print("No match found.")

else:

    print(f"Number of matches: {len(matches)}\n")

    for m in matches:

        i1, i2, i3, i4, error, confidence = m

        print(
            f"({i1}, {i2}, {i3}, {i4})"
            f"   RMS error = {error:.8f}°"
            f"   Confidence = {confidence:.6f}"
        )