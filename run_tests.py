# ============================================================
# run_tests.py
#
# Runs the full tests/ suite and prints a validation report:
#   - a PASS/FAIL table for each of the 10 test cases (T01-T10)
#   - aggregate performance / stability / accuracy metrics
#
# Run from the project root:
#   python run_tests.py
# ============================================================

import io
import sys
import time
import unittest
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from tests import BasePyramidTest
from tests import (
    test_exact_match,
    test_no_match,
    test_multiple_candidates,
    test_boundary_tolerance,
    test_gaussian_noise,
    test_missing_star,
    test_extra_false_star,
    test_invalid_input,
    test_large_catalog_performance,
    test_end_to_end_pipeline,
)

from identification.pyramid import identify
from identification.pyramid_search import (
    search_pyramid,
    largest_edge as DB_LARGEST_EDGE,
    ratio1 as DB_RATIO1,
    ratio2 as DB_RATIO2,
)


TEST_CASES = [
    ("T01", "Exact Match Test", test_exact_match),
    ("T02", "No Match Test", test_no_match),
    ("T03", "Multiple Candidate Match Test", test_multiple_candidates),
    ("T04", "Boundary Tolerance Test", test_boundary_tolerance),
    ("T05", "Gaussian Noise Test", test_gaussian_noise),
    ("T06", "Missing Star Test", test_missing_star),
    ("T07", "Extra (False) Star Test", test_extra_false_star),
    ("T08", "Invalid Input Test", test_invalid_input),
    ("T09", "Large Catalog Performance Test", test_large_catalog_performance),
    ("T10", "End-to-End Pipeline Test", test_end_to_end_pipeline),
]


def run_module(module):
    """
    Run every test in a module. Returns (passed, total, remark).
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    result = runner.run(suite)

    total = result.testsRun
    problems = result.failures + result.errors
    failed = len(problems)
    passed = total - failed

    if failed == 0:
        remark = "Verified"
    else:
        last_line = problems[0][1].strip().splitlines()[-1]
        remark = f"FAILED: {last_line[:45]}"

    return passed, total, remark


def benchmark_search(n_queries=300):
    """
    Benchmarks search_pyramid() over random real catalog queries.
    Returns (average_ms, max_ms).
    """
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(DB_LARGEST_EDGE), n_queries)

    times_ms = []
    for idx in idxs:
        t0 = time.perf_counter()
        search_pyramid(
            float(DB_LARGEST_EDGE[idx]),
            float(DB_RATIO1[idx]),
            float(DB_RATIO2[idx]),
        )
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.max(times_ms))


def measure_false_rates(fixture, trials=30):
    """
    Empirically estimates false positive / false negative rates.

    False positive: identify() reports a match for pyramids that
    have no true catalog counterpart (ideally always []).

    False negative: identify() fails to recover the correct match
    for a genuine catalog pyramid under light, realistic noise
    (ideally always recovers it).
    """
    false_positives = 0
    for i in range(trials):
        rng = np.random.default_rng(1000 + i)
        vecs = [rng.normal(size=3) for _ in range(4)]
        vecs = [v / np.linalg.norm(v) for v in vecs]

        matches = identify(*vecs)
        if len(matches) > 0:
            false_positives += 1

    false_negatives = 0
    for i in range(trials):
        rng = np.random.default_rng(2000 + i)
        noisy = []
        for v in fixture.ref_vectors:
            nv = v + rng.normal(0, 1e-4, 3)
            nv = nv / np.linalg.norm(nv)
            noisy.append(nv)

        matches = identify(*noisy)
        if len(matches) == 0 or set(matches[0][:4]) != set(fixture.ref_ids):
            false_negatives += 1

    return false_positives / trials, false_negatives / trials


def print_table(rows):
    headers = ["ID", "Test Case", "Status", "Remarks"]
    widths = [4, 32, 8, 48]

    def sep():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells):
        return "| " + " | ".join(
            f"{str(c):<{w}}" for c, w in zip(cells, widths)
        ) + " |"

    print(sep())
    print(fmt_row(headers))
    print(sep())
    for r in rows:
        print(fmt_row(r))
    print(sep())


def main():
    print("\nRunning Pyramid Algorithm Validation Suite...\n")

    rows = []
    all_passed = True

    for tid, name, module in TEST_CASES:
        passed, total, remark = run_module(module)
        status = "PASSED" if passed == total else "FAILED"
        if status == "FAILED":
            all_passed = False
        rows.append((tid, name, status, remark))

    print_table(rows)

    # ---- Aggregate metrics ----
    total_cases = len(rows)
    passed_cases = sum(1 for r in rows if r[2] == "PASSED")
    failed_cases = total_cases - passed_cases
    success_rate = 100.0 * passed_cases / total_cases

    avg_ms, max_ms = benchmark_search()

    db_load_status = "Successful"
    try:
        np.load(BASE_DIR / "catalog" / "stars.npz", allow_pickle=True)
        np.load(BASE_DIR / "catalog" / "pyramid_db.npz")
        np.load(BASE_DIR / "catalog" / "cache_fov.npy")
    except Exception:
        db_load_status = "Failed"

    stability = "Stable" if all_passed else "Unstable"

    BasePyramidTest.setUpClass()
    fp_rate, fn_rate = measure_false_rates(BasePyramidTest)

    print()
    print("-" * 70)
    print("Validation Metrics")
    print("-" * 70)
    print(f"Total Test Cases Executed : {total_cases}")
    print(f"Tests Passed              : {passed_cases}")
    print(f"Tests Failed              : {failed_cases}")
    print(f"Success Rate              : {success_rate:.2f}%")
    print(f"Average Search Time       : {avg_ms:.1f} ms")
    print(f"Maximum Search Time       : {max_ms:.1f} ms")
    print(f"Database Load Status      : {db_load_status}")
    print(f"Algorithm Stability       : {stability}")
    print(f"False Positive Rate       : {fp_rate * 100:.0f}%")
    print(f"False Negative Rate       : {fn_rate * 100:.0f}%")
    print("-" * 70)
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
