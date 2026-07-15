# ============================================================
# tests/test_missing_star.py
#
# T06 - Missing Star Test
#
# Simulates a dropped detection: one of the four "observed"
# stars is not actually part of the true pyramid (e.g. the
# real 4th star was never detected and a spurious vector was
# substituted). The algorithm must not falsely confirm a match.
# ============================================================

import unittest

from tests import BasePyramidTest
from identification.pyramid import identify, cross_validate


class TestMissingStar(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
