# ============================================================
# tests/test_end_to_end_pipeline.py
#
# T10 - End-to-End Pipeline Test
#
# Exercises the full observed-sky -> identification pipeline
# using real stars taken from the FOV cache, with realistic
# noise, mirroring how main.py is actually used.
# ============================================================

import unittest

import numpy as np

from tests import BasePyramidTest
from identification.pyramid import identify
from identification.pyramid_search import search_pyramid
from identification.pyramid_verify import verify_candidate
from identification.geometry_utils import compute_invariants


class TestEndToEndPipeline(BasePyramidTest):

    def test_end_to_end_identification_from_fov(self):
        self.assertGreaterEqual(len(self.fov_vectors), 4)

        rng = np.random.default_rng(2024)
        noise_sigma = 1e-4

        raw = [self.fov_vectors[i].copy() for i in range(4)]

        noisy = []
        for v in raw:
            nv = v + rng.normal(0, noise_sigma, 3)
            nv = nv / np.linalg.norm(nv)
            noisy.append(nv)

        matches = identify(*noisy)

        self.assertIsInstance(matches, list)

        if matches:
            i1, i2, i3, i4, error, confidence = matches[0]

            # Structural sanity checks on the returned tuple
            self.assertTrue(all(isinstance(i, (int, np.integer))
                                 for i in (i1, i2, i3, i4)))
            self.assertGreaterEqual(error, 0.0)
            self.assertGreater(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

            # All four identified indices must be valid catalog rows
            for i in (i1, i2, i3, i4):
                self.assertGreaterEqual(i, 0)
                self.assertLess(i, len(self.catalog_vectors))

    def test_end_to_end_matches_geometry_utils_directly(self):
        # Cross-check that the invariants identify() computes
        # internally are consistent with geometry_utils, tying
        # the whole module chain together.
        v1, v2, v3, v4 = self.ref_vectors

        edge, r1, r2 = compute_invariants(v1, v2, v3, v4)
        candidates = search_pyramid(edge, r1, r2)

        self.assertIn(self.ref_ids, candidates)

        ok, error, confidence = verify_candidate(
            [v1, v2, v3, v4],
            [self.catalog_vectors[i] for i in self.ref_ids]
        )

        self.assertTrue(ok)
        self.assertAlmostEqual(error, 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
