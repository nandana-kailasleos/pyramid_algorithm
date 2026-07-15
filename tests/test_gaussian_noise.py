# ============================================================
# tests/test_gaussian_noise.py
#
# T05 - Gaussian Noise Test
#
# Adding small, realistic Gaussian sensor noise to observed
# vectors should still result in a correct identification,
# with error within the verification threshold.
# ============================================================

import unittest

import numpy as np

from tests import BasePyramidTest
from identification.pyramid import identify


class TestGaussianNoise(BasePyramidTest):

    def test_small_noise_still_identifies_correctly(self):
        rng = np.random.default_rng(123)
        noise_sigma = 1e-4

        noisy_vectors = []
        for v in self.ref_vectors:
            noisy = v + rng.normal(0, noise_sigma, 3)
            noisy = noisy / np.linalg.norm(noisy)
            noisy_vectors.append(noisy)

        matches = identify(*noisy_vectors)

        self.assertEqual(len(matches), 1)

        i1, i2, i3, i4, error, confidence = matches[0]

        self.assertSetEqual({i1, i2, i3, i4}, set(self.ref_ids))
        self.assertLess(np.radians(error), np.radians(0.05))
        self.assertGreater(confidence, 0.0)

    def test_large_noise_may_fail_gracefully(self):
        # Excessive noise should not crash the pipeline, even if
        # it results in no verified match.
        rng = np.random.default_rng(7)
        noise_sigma = 0.05  # large, unrealistic noise

        noisy_vectors = []
        for v in self.ref_vectors:
            noisy = v + rng.normal(0, noise_sigma, 3)
            noisy = noisy / np.linalg.norm(noisy)
            noisy_vectors.append(noisy)

        try:
            matches = identify(*noisy_vectors)
        except Exception as e:
            self.fail(f"identify() raised an exception on noisy input: {e}")

        self.assertIsInstance(matches, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
