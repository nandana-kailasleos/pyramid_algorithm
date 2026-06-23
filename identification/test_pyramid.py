import numpy as np
from identification.pyramid import identify

vectors = np.load("catalog/bright_fov.npy")

v1 = vectors[0]
v2 = vectors[1]
v3 = vectors[2]
v4 = vectors[3]

matches = identify(
    v1,
    v2,
    v3,
    v4
)

print()

print("Verified matches:\n")

for m in matches:

    i1, i2, i3, i4, err = m

    print(
        f"({i1}, {i2}, {i3}, {i4})"
        f"  RMS error = {err:.6f}°"
    )