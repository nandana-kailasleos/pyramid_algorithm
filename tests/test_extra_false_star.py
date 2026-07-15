# ============================================================
# tests/test_extra_false_star.py
#
# T07 - Extra (False) Star Test
#
# When extra observed stars are supplied for cross-validation
# but do not correspond to any nearby catalog star (false
# detections / hot pixels), the match should be rejected even
# though the core 4-star geometry matches.
# ============================================================

import unittest

from tests import BasePyramidTest
from identification.pyramid import identify


class TestExtraFalseStar(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
