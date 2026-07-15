# ============================================================
# tests/test_no_match.py
#
# T02 - No Match Test
#
# A geometrically impossible / synthetic pyramid that does not
# correspond to any real four-star configuration in the
# catalog must return an empty result.
# ============================================================

import unittest

import numpy as np

from tests import BasePyramidTest
from identification.pyramid import identify


class TestNoMatch(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
