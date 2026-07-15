# ============================================================
# tests/test_invalid_input.py
#
# T08 - Invalid Input Test
#
# Checks that malformed input is handled predictably: either
# a clear exception, or a safe empty result -- never a silent
# incorrect match or an uncontrolled crash.
# ============================================================

import unittest

import numpy as np

from tests import BasePyramidTest
from identification.pyramid import identify


class TestInvalidInput(BasePyramidTest):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
