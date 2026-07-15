# ============================================================
# tests/test_large_catalog_performance.py
#
# T09 - Large Catalog Performance Test
#
# Sanity-checks that the catalog is indeed large, and that
# repeated searches against it complete within a reasonable
# time budget (i.e. the search remains usable at scale).
# ============================================================

import time
import unittest

import numpy as np

from tests import BasePyramidTest
from identification.pyramid import identify
from identification.pyramid_search import (
    search_pyramid,
    largest_edge as DB_LARGEST_EDGE,
    ratio1 as DB_RATIO1,
    ratio2 as DB_RATIO2,
)


class TestLargeCatalogPerformance(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
