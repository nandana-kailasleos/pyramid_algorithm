# ============================================================
# test_pyramid.py
#
<<<<<<< HEAD
# Full test suite for the Pyramid star identification
# algorithm.
#
# Covers:
#   1.  Exact Match Test
#   2.  No Match Test
#   3.  Multiple Candidate Match Test
#   4.  Boundary Tolerance Test
#   5.  Gaussian Noise Test
#   6.  Missing Star Test
#   7.  Extra (False) Star Test
#   8.  Invalid Input Test
#   9.  Large Catalog Performance Test
#   10. End-to-End Pipeline Test
#
# Run with:
#   python -m identification.test_pyramid
#   (must be launched from the project root, e.g.
#    pyramid_algorithm-main/)
#
# or with pytest:
#   pytest identification/test_pyramid.py -v
# ============================================================

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Make sure relative paths ("catalog/...") resolve correctly
# no matter where this script is launched from.
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from identification.pyramid import identify, cross_validate
from identification.pyramid_search import (
    search_pyramid,
    largest_edge as DB_LARGEST_EDGE,
    ratio1 as DB_RATIO1,
    ratio2 as DB_RATIO2,
    star1 as DB_STAR1,
    star2 as DB_STAR2,
    star3 as DB_STAR3,
    star4 as DB_STAR4,
)
from identification.pyramid_verify import verify_candidate
from identification.geometry_utils import (
    angle_between,
    compute_edges,
    compute_invariants,
    rms_error,
)


class BasePyramidTest(unittest.TestCase):
    """
    Shared setup: loads the star catalog, the pyramid database,
    and the FOV cache once for all tests.
    """

    @classmethod
    def setUpClass(cls):
        stars = np.load(BASE_DIR / "catalog" / "stars.npz", allow_pickle=True)
        cls.catalog_vectors = stars["unit_vectors"]

        cls.db = np.load(BASE_DIR / "catalog" / "pyramid_db.npz")

        cls.fov_vectors = np.load(BASE_DIR / "catalog" / "cache_fov.npy")

        # A known-good pyramid taken directly from the database
        # (index 0). Used as ground truth across several tests.
        cls.ref_idx = 0
        cls.ref_ids = (
            int(cls.db["star1"][cls.ref_idx]),
            int(cls.db["star2"][cls.ref_idx]),
            int(cls.db["star3"][cls.ref_idx]),
            int(cls.db["star4"][cls.ref_idx]),
        )
        cls.ref_vectors = [cls.catalog_vectors[i] for i in cls.ref_ids]

    def random_unit_vector(self, seed=None):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)


# ============================================================
# 1. Exact Match Test
# ============================================================
class TestExactMatch(BasePyramidTest):
    """
    Feeding the algorithm four catalog star vectors with NO
    noise added must return exactly that pyramid with (near)
    zero error and maximum confidence.
    """

    def test_exact_match_returns_correct_stars(self):
        v1, v2, v3, v4 = self.ref_vectors

        matches = identify(v1, v2, v3, v4)

        self.assertEqual(len(matches), 1)

        i1, i2, i3, i4, error, confidence = matches[0]

        self.assertSetEqual(
            {i1, i2, i3, i4},
            set(self.ref_ids)
        )

        self.assertAlmostEqual(error, 0.0, places=6)
        self.assertAlmostEqual(confidence, 1.0, places=6)


# ============================================================
# 2. No Match Test
# ============================================================
class TestNoMatch(BasePyramidTest):
    """
    A geometrically impossible / synthetic pyramid that does not
    correspond to any real four-star configuration in the
    catalog must return an empty result.
    """

    def test_no_match_for_synthetic_orthogonal_pyramid(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        v4 = np.array([-1.0, 0.0, 0.0])

        matches = identify(v1, v2, v3, v4)

        self.assertEqual(matches, [])

    def test_no_match_for_random_vectors(self):
        v1 = self.random_unit_vector(1)
        v2 = self.random_unit_vector(2)
        v3 = self.random_unit_vector(3)
        v4 = self.random_unit_vector(4)

        matches = identify(v1, v2, v3, v4)

        self.assertEqual(matches, [])


# ============================================================
# 3. Multiple Candidate Match Test
# ============================================================
class TestMultipleCandidateMatch(BasePyramidTest):
    """
    With generous tolerances, more than one catalog pyramid can
    satisfy the invariant match. The search stage should return
    all of them, and identify() should still resolve to a single
    best (lowest RMS error) verified match.
    """

    def test_search_returns_multiple_candidates_with_loose_tolerance(self):
        edge = float(self.db["largest_edge"][self.ref_idx])
        r1 = float(self.db["ratio1"][self.ref_idx])
        r2 = float(self.db["ratio2"][self.ref_idx])

        candidates = search_pyramid(
            edge, r1, r2,
            edge_tol=0.01,
            ratio_tol=0.3
        )

        self.assertGreater(len(candidates), 1)
        self.assertIn(self.ref_ids, candidates)

    def test_identify_resolves_single_best_match(self):
        # identify() always picks the lowest-error verified
        # candidate, even if search_pyramid finds several.
        v1, v2, v3, v4 = self.ref_vectors

        matches = identify(v1, v2, v3, v4)

        self.assertLessEqual(len(matches), 1)


# ============================================================
# 4. Boundary Tolerance Test
# ============================================================
class TestBoundaryTolerance(BasePyramidTest):
    """
    Checks that search_pyramid's tolerance comparisons behave
    as strict "<" (exclusive) at the boundary: a query just
    outside the tolerance window is rejected, a query just
    inside is accepted.
    """

    def test_just_outside_edge_tolerance_is_excluded(self):
        edge0 = float(DB_LARGEST_EDGE[self.ref_idx])
        r10 = float(DB_RATIO1[self.ref_idx])
        r20 = float(DB_RATIO2[self.ref_idx])

        edge_tol = 0.003
        margin = 1e-4  # comfortably larger than float32 rounding noise

        edge_outside = edge0 + edge_tol + margin

        candidates = search_pyramid(
            edge_outside, r10, r20,
            edge_tol=edge_tol, ratio_tol=0.02
        )

        self.assertNotIn(self.ref_ids, candidates)

    def test_just_inside_edge_tolerance_is_included(self):
        edge0 = float(DB_LARGEST_EDGE[self.ref_idx])
        r10 = float(DB_RATIO1[self.ref_idx])
        r20 = float(DB_RATIO2[self.ref_idx])

        edge_tol = 0.003
        margin = 1e-4

        edge_inside = edge0 + edge_tol - margin

        candidates = search_pyramid(
            edge_inside, r10, r20,
            edge_tol=edge_tol, ratio_tol=0.02
        )

        self.assertIn(self.ref_ids, candidates)


# ============================================================
# 5. Gaussian Noise Test
# ============================================================
class TestGaussianNoise(BasePyramidTest):
    """
    Adding small, realistic Gaussian sensor noise to observed
    vectors should still result in a correct identification,
    with error within the verification threshold.
    """

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


# ============================================================
# 6. Missing Star Test
# ============================================================
class TestMissingStar(BasePyramidTest):
    """
    Simulates a dropped detection: one of the four "observed"
    stars is not actually part of the true pyramid (e.g. the
    real 4th star was never detected and a spurious vector was
    substituted). The algorithm must not falsely confirm a match.
    """

    def test_bogus_fourth_star_yields_no_match(self):
        v1, v2, v3, _ = self.ref_vectors

        bogus_v4 = self.random_unit_vector(99)

        matches = identify(v1, v2, v3, bogus_v4)

        self.assertEqual(matches, [])

    def test_missing_argument_raises_type_error(self):
        v1, v2, v3, _ = self.ref_vectors

        with self.assertRaises(TypeError):
            identify(v1, v2, v3)

    def test_cross_validate_passes_when_extras_insufficient(self):
        # With fewer than 2 extra observed stars, cross_validate
        # should not block an otherwise valid match.
        result = cross_validate(
            list(self.ref_ids),
            self.catalog_vectors,
            observed_extra=[self.random_unit_vector(5)]
        )
        self.assertTrue(result)


# ============================================================
# 7. Extra (False) Star Test
# ============================================================
class TestExtraFalseStar(BasePyramidTest):
    """
    When extra observed stars are supplied for cross-validation
    but do not correspond to any nearby catalog star (false
    detections / hot pixels), the match should be rejected even
    though the core 4-star geometry matches.
    """

    def test_false_extra_stars_reject_match(self):
        v1, v2, v3, v4 = self.ref_vectors

        false_extras = [
            self.random_unit_vector(11),
            self.random_unit_vector(12),
            self.random_unit_vector(13),
        ]

        matches = identify(v1, v2, v3, v4, extra_observed=false_extras)

        self.assertEqual(matches, [])

    def test_genuine_extra_stars_confirm_match(self):
        v1, v2, v3, v4 = self.ref_vectors

        # Reuse two other real catalog vectors as "genuine"
        # extra detections.
        genuine_extras = [
            self.catalog_vectors[self.ref_ids[0] + 500],
            self.catalog_vectors[self.ref_ids[0] + 501],
        ]

        matches = identify(v1, v2, v3, v4, extra_observed=genuine_extras)

        self.assertEqual(len(matches), 1)
        i1, i2, i3, i4, error, confidence = matches[0]
        self.assertSetEqual({i1, i2, i3, i4}, set(self.ref_ids))


# ============================================================
# 8. Invalid Input Test
# ============================================================
class TestInvalidInput(BasePyramidTest):
    """
    Checks that malformed input is handled predictably: either
    a clear exception, or a safe empty result -- never a silent
    incorrect match or an uncontrolled crash.
    """

    def test_wrong_shape_vector_raises(self):
        v1 = np.array([1.0, 0.0])  # only 2 components
        v2, v3, v4 = self.ref_vectors[1:]

        with self.assertRaises(ValueError):
            identify(v1, v2, v3, v4)

    def test_nan_vector_does_not_crash(self):
        v1 = np.array([np.nan, 0.0, 0.0])
        v2, v3, v4 = self.ref_vectors[1:]

        try:
            matches = identify(v1, v2, v3, v4)
        except Exception as e:
            self.fail(f"identify() raised an exception on NaN input: {e}")

        self.assertEqual(matches, [])

    def test_zero_vector_does_not_crash(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2, v3, v4 = self.ref_vectors[1:]

        try:
            matches = identify(v1, v2, v3, v4)
        except Exception as e:
            self.fail(f"identify() raised an exception on zero-vector input: {e}")

        self.assertIsInstance(matches, list)

    def test_non_numpy_list_input_is_handled(self):
        # Plain python lists instead of np.ndarray should still
        # work since numpy operations coerce them.
        v1 = list(self.ref_vectors[0])
        v2, v3, v4 = self.ref_vectors[1:]

        try:
            matches = identify(v1, v2, v3, v4)
        except Exception as e:
            self.fail(f"identify() raised an exception on list input: {e}")

        self.assertIsInstance(matches, list)


# ============================================================
# 9. Large Catalog Performance Test
# ============================================================
class TestLargeCatalogPerformance(BasePyramidTest):
    """
    Sanity-checks that the catalog is indeed large, and that
    repeated searches against it complete within a reasonable
    time budget (i.e. the search remains usable at scale).
    """

    def test_catalog_is_large(self):
        self.assertGreater(len(self.catalog_vectors), 50000)
        self.assertGreater(len(self.db["largest_edge"]), 10000)

    def test_many_searches_complete_quickly(self):
        rng = np.random.default_rng(0)
        n_queries = 300

        idxs = rng.integers(0, len(DB_LARGEST_EDGE), n_queries)

        start = time.perf_counter()

        for idx in idxs:
            search_pyramid(
                float(DB_LARGEST_EDGE[idx]),
                float(DB_RATIO1[idx]),
                float(DB_RATIO2[idx]),
            )

        elapsed = time.perf_counter() - start

        # Generous budget: should comfortably finish well under
        # this on any reasonable machine.
        self.assertLess(elapsed, 5.0)

    def test_full_identify_pipeline_is_reasonably_fast(self):
        v1, v2, v3, v4 = self.ref_vectors

        start = time.perf_counter()
        identify(v1, v2, v3, v4)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 2.0)


# ============================================================
# 10. End-to-End Pipeline Test
# ============================================================
class TestEndToEndPipeline(BasePyramidTest):
    """
    Exercises the full observed-sky -> identification pipeline
    using real stars taken from the FOV cache, with realistic
    noise, mirroring how main.py is actually used.
    """

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


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
=======
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
>>>>>>> 88f213249201b986c83d41986472fc5d00f31496
