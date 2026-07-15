# ============================================================
# tests/test_boundary_tolerance.py
#
# T04 - Boundary Tolerance Test
#
# Checks that search_pyramid's tolerance comparisons behave as
# strict "<" (exclusive) at the boundary: a query just outside
# the tolerance window is rejected, a query just inside is
# accepted.
# ============================================================

import unittest

from tests import BasePyramidTest
from identification.pyramid_search import (
    search_pyramid,
    largest_edge as DB_LARGEST_EDGE,
    ratio1 as DB_RATIO1,
    ratio2 as DB_RATIO2,
)


class TestBoundaryTolerance(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
