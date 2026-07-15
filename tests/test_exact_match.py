# ============================================================
# tests/test_exact_match.py
#
# T01 - Exact Match Test
#
# Feeding the algorithm four catalog star vectors with NO
# noise added must return exactly that pyramid with (near)
# zero error and maximum confidence.
# ============================================================

import unittest

from tests import BasePyramidTest
from identification.pyramid import identify


class TestExactMatch(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
