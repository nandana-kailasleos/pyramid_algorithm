# ============================================================
# main.py
#
# Entry point for Pyramid star identification
# ============================================================

import numpy as np
from identification.pyramid import identify


def main():

    print("\nPyramid Star Identification\n")

    print("Enter four observed star vectors\n")

    vectors = []

    for i in range(4):

        print(f"Star {i+1}")

        x = float(input("x : "))
        y = float(input("y : "))
        z = float(input("z : "))

        v = np.array([x, y, z], dtype=np.float64)

        norm = np.linalg.norm(v)

        if norm < 1e-9:
            print("Invalid vector.")
            return

        v = v / norm

        vectors.append(v)

        print()

    candidates = identify(
        vectors[0],
        vectors[1],
        vectors[2],
        vectors[3]
    )

    print("\nCandidate catalog matches:\n")

    if len(candidates) == 0:

        print("No match found.")

    else:

        for i, c in enumerate(candidates, 1):

            s1, s2, s3, s4, error, confidence = c

            print(
                f"{i}. "
                f"({s1}, {s2}, {s3}, {s4})"
                f"   RMS Error = {error:.8f}°"
                f"   Confidence = {confidence:.6f}"
            )


if __name__ == "__main__":
    main()