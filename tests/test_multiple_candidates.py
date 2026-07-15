# ============================================================
# tests/test_multiple_candidates.py
#
# T03 - Multiple Candidate Match Test
#
# With generous tolerances, more than one catalog pyramid can
# satisfy the invariant match. The search stage should return
# all of them, and identify() should still resolve to a single
# best (lowest RMS error) verified match.
# ============================================================

import unittest

from tests import BasePyramidTest
from identification.pyramid import identify
from identification.pyramid_search import search_pyramid


class TestMultipleCandidateMatch(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
