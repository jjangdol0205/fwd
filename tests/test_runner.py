"""
tests/test_runner.py
Master Test Runner for Dataguide Valuation & Revision System.
Discovers and runs all test suites across Tiers 1-4.
Ensures minimum coverage targets (>=161 tests) per TEST_INFRA.md.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_tier1_features import (
    TestFeature01_EpsRevision1W,
    TestFeature02_EpsRevision1M,
    TestFeature03_EpsRevision3M,
    TestFeature04_ValueTrapDetection,
    TestFeature05_GoldenCrossDetection,
    TestFeature06_SectorMapping,
    TestFeature07_SectorMedianPE,
    TestFeature08_SectorPEPercentile,
    TestFeature09_ResearchStandardBands,
    TestFeature10_FixedMultipleBands,
    TestFeature11_Winsorization,
    TestFeature12_ScreenerTableColumns,
    TestFeature13_QuickScreenerPresets,
    TestFeature14_IntegratedDashboard,
)
from tests.test_tier2_boundaries import (
    TestBoundary01_EpsRevision1W,
    TestBoundary02_EpsRevision1M,
    TestBoundary03_EpsRevision3M,
    TestBoundary04_ValueTrapDetection,
    TestBoundary05_GoldenCrossDetection,
    TestBoundary06_SectorMapping,
    TestBoundary07_SectorMedianPE,
    TestBoundary08_SectorPEPercentile,
    TestBoundary09_ResearchStandardBands,
    TestBoundary10_FixedMultipleBands,
    TestBoundary11_Winsorization,
    TestBoundary12_ScreenerTable,
    TestBoundary13_QuickPresets,
    TestBoundary14_IntegratedDashboard,
)
from tests.test_tier3_combinations import TestTier3Combinations
from tests.test_tier4_scenarios import TestTier4Scenarios


def build_suite() -> unittest.TestSuite:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Tier 1: 14 Feature classes
    t1_classes = [
        TestFeature01_EpsRevision1W,
        TestFeature02_EpsRevision1M,
        TestFeature03_EpsRevision3M,
        TestFeature04_ValueTrapDetection,
        TestFeature05_GoldenCrossDetection,
        TestFeature06_SectorMapping,
        TestFeature07_SectorMedianPE,
        TestFeature08_SectorPEPercentile,
        TestFeature09_ResearchStandardBands,
        TestFeature10_FixedMultipleBands,
        TestFeature11_Winsorization,
        TestFeature12_ScreenerTableColumns,
        TestFeature13_QuickScreenerPresets,
        TestFeature14_IntegratedDashboard,
    ]
    for cls in t1_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Tier 2: 14 Boundary classes
    t2_classes = [
        TestBoundary01_EpsRevision1W,
        TestBoundary02_EpsRevision1M,
        TestBoundary03_EpsRevision3M,
        TestBoundary04_ValueTrapDetection,
        TestBoundary05_GoldenCrossDetection,
        TestBoundary06_SectorMapping,
        TestBoundary07_SectorMedianPE,
        TestBoundary08_SectorPEPercentile,
        TestBoundary09_ResearchStandardBands,
        TestBoundary10_FixedMultipleBands,
        TestBoundary11_Winsorization,
        TestBoundary12_ScreenerTable,
        TestBoundary13_QuickPresets,
        TestBoundary14_IntegratedDashboard,
    ]
    for cls in t2_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Tier 3 & Tier 4
    suite.addTests(loader.loadTestsFromTestCase(TestTier3Combinations))
    suite.addTests(loader.loadTestsFromTestCase(TestTier4Scenarios))

    return suite


def main():
    suite = build_suite()
    total_tests = suite.countTestCases()
    print("=" * 70)
    print("Dataguide Valuation & Earnings Momentum System - E2E Test Runner")
    print("=" * 70)
    print(f"Total Test Cases Discovered: {total_tests}")
    print("Minimum Required Target:      161 tests")
    print(f"Target Satisfied:             {'YES (>=161)' if total_tests >= 161 else 'NO'}")
    print("-" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY:")
    print(f"  Ran:       {result.testsRun}")
    print(f"  Failures:  {len(result.failures)}")
    print(f"  Errors:    {len(result.errors)}")
    print(f"  Skipped:   {len(result.skipped)}")
    print(f"  Success:   {result.wasSuccessful()}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
