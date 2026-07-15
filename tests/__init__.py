# ============================================================
# tests/__init__.py
#
# Shared fixtures for the Pyramid algorithm test suite.
# Every test module in this package imports BasePyramidTest
# from here so the catalog / database / FOV cache are only
# defined in one place.
# ============================================================

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Make sure relative paths ("catalog/...") resolve correctly
# no matter where the suite is launched from.
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))


class BasePyramidTest(unittest.TestCase):
    """
    Shared setup: loads the star catalog, the pyramid database,
    and the FOV cache once per test class.
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
