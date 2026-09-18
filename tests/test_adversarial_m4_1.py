"""
tests/test_adversarial_m4_1.py
------------------------------
Milestone 4 Adversarial Stress Testing Suite:
1-Click Quick Screener Presets (app.py:apply_quick_preset vs tests/oracle.py:oracle_apply_quick_preset).
Author: Challenger M4-1 (Empirical Challenger)

Coverage Dimensions:
1. 1,000-Trial Monte Carlo Parity & Invariants:
   - 1,000 randomized synthetic valuation DataFrames comparing outputs of
     apply_quick_preset and oracle_apply_quick_preset across all 5 presets:
     ('ALL', 'TURNAROUND', 'EPS_TOP', 'VALUE', 'TRAP').
   - Invariant validation: mathematical verification that every included row meets
     all criteria and every excluded row fails at least one criterion.
   - Truncation validation: EPS_TOP returns at most 25 stocks in strictly descending order.
2. Boundary Condition Analysis:
   - TURNAROUND:
     * pe_percentile = 40.0 vs 40.0001 (<= 40.0 weak inequality)
     * sector_pe_percentile = 40.0 vs 40.0001 (<= 40.0 weak inequality)
     * eps_rev_1m = 0.0 vs 0.00001 (> 0.0 strict inequality: 0.0 MUST be excluded)
     * upside_base = 10.0 vs 9.9999 (>= 10.0 weak inequality)
   - EPS_TOP:
     * eps_rev_1m = 3.0 vs 2.9999 (>= 3.0 weak inequality)
     * current_fwd_eps = 0.0 vs 0.001 (> 0.0 strict inequality: deficit/breakeven excluded)
     * Extreme deficit grower (eps_rev_1m = +500%, current_eps = -50.0): strictly excluded
   - VALUE:
     * pe_percentile = 25.0 vs 25.0001 (<= 25.0 weak inequality)
     * eps_rev_1m = -2.0 vs -2.0001 (>= -2.0 weak inequality)
     * current_fwd_pe = 15.0 vs 15.0001 (<= 15.0 weak inequality)
     * upside_base = 15.0 vs 14.9999 (>= 15.0 weak inequality)
   - TRAP:
     * pe_percentile = 30.0 vs 30.0001 (<= 30.0 weak inequality)
     * eps_rev_1m = -3.0 vs -3.0001 (< -3.0 strict inequality: -3.0 MUST be excluded)
3. Edge Cases, Missing Columns & Robustness:
   - Empty DataFrame with schema: pd.DataFrame(columns=[...])
   - Completely empty DataFrame: pd.DataFrame() (uncovering scalar boolean KeyError: False)
   - Missing individual columns: sector_pe_percentile, upside_base, pe_percentile
   - Missing eps_rev_1m in EPS_TOP (uncovering KeyError in sort_values)
   - Missing all filter columns (uncovering scalar boolean indexing crash)
   - Floating-point NaNs and positive/negative infinities
   - Deficit stock / negative P/E anomaly in VALUE preset (uncovering current_fwd_pe <= 15.0 without > 0 guard)
   - Unknown preset keys, empty strings, and case-sensitivity behavior
   - Large-scale tie-breaking (100 identical top revision stocks)
4. Fallback Code Path Parity:
   - Monkeypatches oracle_apply_quick_preset to None in app.py module to stress test
     the standalone fallback implementation against the reference oracle.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module
from app import apply_quick_preset
from tests.oracle import oracle_apply_quick_preset


REQUIRED_COLUMNS = [
    "ticker",
    "name",
    "pe_percentile",
    "sector_pe_percentile",
    "eps_rev_1m",
    "current_fwd_eps",
    "current_fwd_pe",
    "upside_base",
]


def make_synthetic_screener_df(
    n_stocks: int,
    seed: int,
    nan_prob: float = 0.05,
    inject_boundaries: bool = True,
) -> pd.DataFrame:
    """
    Generates a randomized synthetic valuation DataFrame with realistic parameters,
    boundary condition values, and occasional NaNs.
    """
    if n_stocks == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rng = np.random.RandomState(seed)

    tickers = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    names = [f"Stock_{i:04d}" for i in range(1, n_stocks + 1)]

    # Uniform random distributions
    pe_pct = rng.uniform(0.0, 100.0, size=n_stocks)
    sec_pe_pct = rng.uniform(0.0, 100.0, size=n_stocks)
    eps_rev_1m = rng.uniform(-25.0, 35.0, size=n_stocks)
    current_eps = rng.uniform(-5000.0, 25000.0, size=n_stocks)
    current_pe = rng.uniform(-10.0, 80.0, size=n_stocks)
    upside_base = rng.uniform(-30.0, 60.0, size=n_stocks)

    # Injected boundary values on specific slots if requested
    if inject_boundaries and n_stocks >= 10:
        # Turnaround boundaries
        pe_pct[0] = 40.0
        sec_pe_pct[0] = 60.0
        eps_rev_1m[0] = 0.0001
        upside_base[0] = 10.0

        pe_pct[1] = 40.0001
        sec_pe_pct[1] = 40.0
        eps_rev_1m[1] = 0.0001
        upside_base[1] = 10.0

        pe_pct[2] = 30.0
        sec_pe_pct[2] = 30.0
        eps_rev_1m[2] = 0.0  # Boundary failure: > 0.0 is strict!
        upside_base[2] = 15.0

        # EPS_TOP boundaries
        eps_rev_1m[3] = 3.0
        current_eps[3] = 1.0

        eps_rev_1m[4] = 2.9999
        current_eps[4] = 5000.0

        eps_rev_1m[5] = 15.0
        current_eps[5] = 0.0  # Boundary failure: > 0.0 is strict!

        # VALUE boundaries
        pe_pct[6] = 25.0
        eps_rev_1m[6] = -2.0
        current_pe[6] = 15.0
        upside_base[6] = 15.0

        pe_pct[7] = 25.0001
        eps_rev_1m[7] = -1.9
        current_pe[7] = 12.0
        upside_base[7] = 16.0

        # TRAP boundaries
        pe_pct[8] = 30.0
        eps_rev_1m[8] = -3.0001

        pe_pct[9] = 30.0
        eps_rev_1m[9] = -3.0  # Boundary failure: < -3.0 is strict!

    # Inject NaNs with probability nan_prob
    if nan_prob > 0.0 and n_stocks > 10:
        mask_pe = rng.rand(n_stocks) < nan_prob
        pe_pct[mask_pe] = np.nan
        mask_sec = rng.rand(n_stocks) < nan_prob
        sec_pe_pct[mask_sec] = np.nan
        mask_rev = rng.rand(n_stocks) < nan_prob
        eps_rev_1m[mask_rev] = np.nan
        mask_eps = rng.rand(n_stocks) < nan_prob
        current_eps[mask_eps] = np.nan
        mask_pe_curr = rng.rand(n_stocks) < nan_prob
        current_pe[mask_pe_curr] = np.nan
        mask_up = rng.rand(n_stocks) < nan_prob
        upside_base[mask_up] = np.nan

    return pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "pe_percentile": pe_pct,
        "sector_pe_percentile": sec_pe_pct,
        "eps_rev_1m": eps_rev_1m,
        "current_fwd_eps": current_eps,
        "current_fwd_pe": current_pe,
        "upside_base": upside_base,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: 1,000-Trial Monte Carlo Simulation & Domain Invariants
# ─────────────────────────────────────────────────────────────────────────────
class TestAdversarialMonteCarloPresets(unittest.TestCase):
    """
    Adversarial 1,000-trial Monte Carlo stress test comparing app.py:apply_quick_preset
    against tests/oracle.py:oracle_apply_quick_preset across synthetic valuation universes.
    """

    def test_monte_carlo_1000_trials_all_presets_parity(self):
        """1,000 randomized Monte Carlo universes across all 5 preset keys."""
        rng = np.random.RandomState(20260918)
        preset_keys = ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]

        for trial in range(1000):
            # Vary universe size: 0, 1, 5, 20, 50, 100, 350
            size_mode = trial % 7
            if size_mode == 0:
                n = 0
            elif size_mode == 1:
                n = 1
            elif size_mode == 2:
                n = 5
            elif size_mode == 3:
                n = 25
            elif size_mode == 4:
                n = 60
            elif size_mode == 5:
                n = 120
            else:
                n = 350

            df = make_synthetic_screener_df(
                n_stocks=n,
                seed=trial + 1,
                nan_prob=0.08 if n > 10 else 0.0,
                inject_boundaries=True,
            )

            # Test all 5 presets on this universe
            for pkey in preset_keys:
                app_res = apply_quick_preset(df, pkey)
                oracle_res = oracle_apply_quick_preset(df, pkey)

                # 1. Structural Parity
                self.assertEqual(
                    len(app_res),
                    len(oracle_res),
                    f"Trial {trial} ({pkey}) row count mismatch: {len(app_res)} vs {len(oracle_res)}",
                )
                self.assertEqual(
                    list(app_res["ticker"]),
                    list(oracle_res["ticker"]),
                    f"Trial {trial} ({pkey}) ticker sequence mismatch",
                )

                # 2. DataFrame Equality
                pd.testing.assert_frame_equal(
                    app_res.reset_index(drop=True),
                    oracle_res.reset_index(drop=True),
                    check_dtype=False,
                )

                # 3. Domain Invariants Verification
                if pkey == "ALL":
                    self.assertEqual(len(app_res), n)

                elif pkey == "TURNAROUND":
                    for _, row in app_res.iterrows():
                        pe_ok = (row["pe_percentile"] <= 40.0) or (row["sector_pe_percentile"] <= 40.0)
                        rev_ok = row["eps_rev_1m"] > 0.0
                        up_ok = row["upside_base"] >= 10.0
                        self.assertTrue(
                            pe_ok and rev_ok and up_ok,
                            f"Trial {trial} TURNAROUND invariant violation for {row['ticker']}",
                        )

                elif pkey == "EPS_TOP":
                    self.assertLessEqual(
                        len(app_res),
                        25,
                        f"Trial {trial} EPS_TOP exceeded 25 stocks cap: {len(app_res)}",
                    )
                    revs = app_res["eps_rev_1m"].tolist()
                    for idx, row in app_res.iterrows():
                        self.assertGreaterEqual(
                            row["eps_rev_1m"],
                            3.0,
                            f"Trial {trial} EPS_TOP rev < 3.0 for {row['ticker']}",
                        )
                        self.assertGreater(
                            row["current_fwd_eps"],
                            0.0,
                            f"Trial {trial} EPS_TOP non-positive EPS for {row['ticker']}",
                        )
                    # Verify monotonic descending sort
                    self.assertTrue(
                        all(revs[i] >= revs[i + 1] for i in range(len(revs) - 1)),
                        f"Trial {trial} EPS_TOP sort order not descending",
                    )

                elif pkey == "VALUE":
                    for _, row in app_res.iterrows():
                        pct_ok = row["pe_percentile"] <= 25.0
                        rev_ok = row["eps_rev_1m"] >= -2.0
                        pe_ok = row["current_fwd_pe"] <= 15.0
                        up_ok = row["upside_base"] >= 15.0
                        self.assertTrue(
                            pct_ok and rev_ok and pe_ok and up_ok,
                            f"Trial {trial} VALUE invariant violation for {row['ticker']}",
                        )

                elif pkey == "TRAP":
                    for _, row in app_res.iterrows():
                        pct_ok = row["pe_percentile"] <= 30.0
                        rev_ok = row["eps_rev_1m"] < -3.0
                        self.assertTrue(
                            pct_ok and rev_ok,
                            f"Trial {trial} TRAP invariant violation for {row['ticker']}",
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: High-Precision Boundary Condition Stress Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAdversarialBoundaryConditions(unittest.TestCase):
    """
    Adversarial boundary value analysis targeting exact mathematical thresholds:
    - TURNAROUND: pe_pct <= 40.0 | sector_pe_pct <= 40.0, rev_1m > 0.0, upside >= 10.0
    - EPS_TOP: rev_1m >= 3.0, eps > 0.0
    - VALUE: pe_pct <= 25.0, rev_1m >= -2.0, fwd_pe <= 15.0, upside >= 15.0
    - TRAP: pe_pct <= 30.0, rev_1m < -3.0
    """

    def test_turnaround_exact_boundary_pe_percentile_40(self):
        """pe_percentile = 40.0 is included (<= 40.0), 40.0001 is excluded."""
        df = pd.DataFrame([
            {"ticker": "IN_40", "pe_percentile": 40.0, "sector_pe_percentile": 50.0, "eps_rev_1m": 2.0, "upside_base": 12.0},
            {"ticker": "OUT_40", "pe_percentile": 40.0001, "sector_pe_percentile": 50.0, "eps_rev_1m": 2.0, "upside_base": 12.0},
        ])
        res = apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(list(res["ticker"]), ["IN_40"])

    def test_turnaround_exact_boundary_sector_pe_percentile_40(self):
        """sector_pe_percentile = 40.0 is included (<= 40.0), 40.0001 is excluded."""
        df = pd.DataFrame([
            {"ticker": "IN_SEC40", "pe_percentile": 60.0, "sector_pe_percentile": 40.0, "eps_rev_1m": 1.5, "upside_base": 15.0},
            {"ticker": "OUT_SEC40", "pe_percentile": 60.0, "sector_pe_percentile": 40.0001, "eps_rev_1m": 1.5, "upside_base": 15.0},
        ])
        res = apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(list(res["ticker"]), ["IN_SEC40"])

    def test_turnaround_exact_boundary_eps_rev_1m_zero(self):
        """eps_rev_1m = 0.0 MUST BE EXCLUDED (condition is strictly > 0.0), 0.0001 is included."""
        df = pd.DataFrame([
            {"ticker": "ZERO_REV", "pe_percentile": 30.0, "sector_pe_percentile": 30.0, "eps_rev_1m": 0.0, "upside_base": 15.0},
            {"ticker": "POS_REV", "pe_percentile": 30.0, "sector_pe_percentile": 30.0, "eps_rev_1m": 0.0001, "upside_base": 15.0},
            {"ticker": "NEG_REV", "pe_percentile": 30.0, "sector_pe_percentile": 30.0, "eps_rev_1m": -0.0001, "upside_base": 15.0},
        ])
        res = apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(list(res["ticker"]), ["POS_REV"])

    def test_turnaround_exact_boundary_upside_base_10(self):
        """upside_base = 10.0 is included (>= 10.0), 9.9999 is excluded."""
        df = pd.DataFrame([
            {"ticker": "UP_10", "pe_percentile": 20.0, "sector_pe_percentile": 20.0, "eps_rev_1m": 3.0, "upside_base": 10.0},
            {"ticker": "UP_9_999", "pe_percentile": 20.0, "sector_pe_percentile": 20.0, "eps_rev_1m": 3.0, "upside_base": 9.9999},
        ])
        res = apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(list(res["ticker"]), ["UP_10"])

    def test_eps_top_exact_boundary_rev_3_0(self):
        """eps_rev_1m = 3.0 is included (>= 3.0), 2.9999 is excluded."""
        df = pd.DataFrame([
            {"ticker": "IN_3_0", "eps_rev_1m": 3.0, "current_fwd_eps": 5000.0},
            {"ticker": "OUT_2_9999", "eps_rev_1m": 2.9999, "current_fwd_eps": 5000.0},
        ])
        res = apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(list(res["ticker"]), ["IN_3_0"])

    def test_eps_top_exact_boundary_current_eps_zero(self):
        """current_fwd_eps = 0.0 MUST BE EXCLUDED (strictly > 0.0), 0.0001 is included."""
        df = pd.DataFrame([
            {"ticker": "ZERO_EPS", "eps_rev_1m": 15.0, "current_fwd_eps": 0.0},
            {"ticker": "POS_EPS", "eps_rev_1m": 15.0, "current_fwd_eps": 0.0001},
            {"ticker": "NEG_EPS", "eps_rev_1m": 50.0, "current_fwd_eps": -100.0},
        ])
        res = apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(list(res["ticker"]), ["POS_EPS"])

    def test_value_exact_boundaries(self):
        """VALUE boundaries: pe_pct <= 25.0, rev >= -2.0, fwd_pe <= 15.0, upside >= 15.0."""
        # Exact on all 4 boundaries -> INCLUDED
        df_exact = pd.DataFrame([{
            "ticker": "ALL_EXACT",
            "pe_percentile": 25.0,
            "eps_rev_1m": -2.0,
            "current_fwd_pe": 15.0,
            "upside_base": 15.0,
        }])
        self.assertEqual(len(apply_quick_preset(df_exact, "VALUE")), 1)

        # Single boundary failures
        df_fail_pe_pct = pd.DataFrame([{
            "ticker": "FAIL_PCT", "pe_percentile": 25.0001, "eps_rev_1m": -2.0, "current_fwd_pe": 15.0, "upside_base": 15.0,
        }])
        self.assertEqual(len(apply_quick_preset(df_fail_pe_pct, "VALUE")), 0)

        df_fail_rev = pd.DataFrame([{
            "ticker": "FAIL_REV", "pe_percentile": 25.0, "eps_rev_1m": -2.0001, "current_fwd_pe": 15.0, "upside_base": 15.0,
        }])
        self.assertEqual(len(apply_quick_preset(df_fail_rev, "VALUE")), 0)

        df_fail_pe = pd.DataFrame([{
            "ticker": "FAIL_PE", "pe_percentile": 25.0, "eps_rev_1m": -2.0, "current_fwd_pe": 15.0001, "upside_base": 15.0,
        }])
        self.assertEqual(len(apply_quick_preset(df_fail_pe, "VALUE")), 0)

        df_fail_up = pd.DataFrame([{
            "ticker": "FAIL_UP", "pe_percentile": 25.0, "eps_rev_1m": -2.0, "current_fwd_pe": 15.0, "upside_base": 14.9999,
        }])
        self.assertEqual(len(apply_quick_preset(df_fail_up, "VALUE")), 0)

    def test_trap_exact_boundaries(self):
        """TRAP boundaries: pe_pct <= 30.0, rev < -3.0 (strictly less than -3.0)."""
        df = pd.DataFrame([
            {"ticker": "IN_TRAP", "pe_percentile": 30.0, "eps_rev_1m": -3.0001},
            {"ticker": "FAIL_EXACT_3", "pe_percentile": 30.0, "eps_rev_1m": -3.0},  # Must be excluded!
            {"ticker": "FAIL_PCT_30_0001", "pe_percentile": 30.0001, "eps_rev_1m": -5.0},
        ])
        res = apply_quick_preset(df, "TRAP")
        self.assertEqual(list(res["ticker"]), ["IN_TRAP"])


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: Edge Cases, Vulnerabilities & Robustness Challenges
# ─────────────────────────────────────────────────────────────────────────────
class TestAdversarialEdgeCasesAndVulnerabilities(unittest.TestCase):
    """
    Adversarial edge-case discovery:
    - Empty DataFrames with and without schema
    - Missing columns behavior
    - Deficit stock anomaly in VALUE preset
    - Truncation & tie-breaking behavior (>25 stocks in EPS_TOP)
    - Non-finite numbers (NaN, Inf)
    - Unknown or lowercase preset keys
    """

    def test_empty_dataframe_with_schema(self):
        """Empty DataFrame with valid schema must return empty DataFrame across all presets."""
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        for pkey in ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]:
            res = apply_quick_preset(empty_df, pkey)
            self.assertEqual(len(res), 0, f"Preset {pkey} failed on empty DataFrame with schema")

    def test_completely_empty_dataframe_without_columns(self):
        """
        Hardened Behavior: pd.DataFrame() without columns returns empty DataFrame safely without raising KeyError.
        """
        blank_df = pd.DataFrame()
        for pkey in ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]:
            res = apply_quick_preset(blank_df, pkey)
            self.assertEqual(len(res), 0)

    def test_missing_single_column_defaults(self):
        """
        Missing sector_pe_percentile defaults to 100, so stocks qualify only if pe_percentile <= 40.0.
        """
        df = pd.DataFrame([
            {"ticker": "T1", "pe_percentile": 35.0, "eps_rev_1m": 2.0, "upside_base": 12.0},
            {"ticker": "T2", "pe_percentile": 45.0, "eps_rev_1m": 2.0, "upside_base": 12.0},
        ])
        # T1 has pe_percentile <= 40; T2 fails since sector_pe_percentile is missing (defaults to 100)
        res = apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(list(res["ticker"]), ["T1"])

    def test_missing_eps_rev_1m_in_eps_top_raises_keyerror(self):
        """
        Hardened Behavior: If eps_rev_1m is missing from df, EPS_TOP safely excludes rows without raising KeyError.
        """
        df = pd.DataFrame([
            {"ticker": "NO_REV", "current_fwd_eps": 5000.0},
        ])
        res = apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(len(res), 0)

    def test_all_nans_safely_excluded(self):
        """Rows with all NaN values are excluded from strategy presets and preserved in ALL."""
        nan_df = pd.DataFrame([
            {"ticker": "NAN_1", "pe_percentile": np.nan, "sector_pe_percentile": np.nan, "eps_rev_1m": np.nan, "current_fwd_eps": np.nan, "current_fwd_pe": np.nan, "upside_base": np.nan},
            {"ticker": "NAN_2", "pe_percentile": np.nan, "sector_pe_percentile": np.nan, "eps_rev_1m": np.nan, "current_fwd_eps": np.nan, "current_fwd_pe": np.nan, "upside_base": np.nan},
        ])
        self.assertEqual(len(apply_quick_preset(nan_df, "ALL")), 2)
        self.assertEqual(len(apply_quick_preset(nan_df, "TURNAROUND")), 0)
        self.assertEqual(len(apply_quick_preset(nan_df, "EPS_TOP")), 0)
        self.assertEqual(len(apply_quick_preset(nan_df, "VALUE")), 0)
        self.assertEqual(len(apply_quick_preset(nan_df, "TRAP")), 0)

    def test_deficit_stock_negative_pe_anomaly_in_value(self):
        """
        Hardened Behavior: Deficit stocks with negative P/E (e.g. -5.0) are strictly excluded from VALUE preset.
        Condition: (0.0 < current_fwd_pe <= 15.0) prevents negative P/E loss-makers.
        """
        df = pd.DataFrame([{
            "ticker": "DEFICIT_VALUE",
            "pe_percentile": 10.0,
            "eps_rev_1m": 1.0,
            "current_fwd_pe": -5.0,  # Negative P/E (loss-maker) must be excluded!
            "upside_base": 20.0,
        }])
        res = apply_quick_preset(df, "VALUE")
        self.assertEqual(len(res), 0)

    def test_eps_top_truncation_to_top_25(self):
        """When 100 stocks qualify for EPS_TOP, exactly the top 25 highest revision stocks are returned."""
        n_qualifying = 100
        df = pd.DataFrame({
            "ticker": [f"STK_{i:03d}" for i in range(n_qualifying)],
            "eps_rev_1m": [float(i + 3.0) for i in range(n_qualifying)],  # 3.0 to 102.0
            "current_fwd_eps": [1000.0] * n_qualifying,
        })
        res = apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(len(res), 25)
        # Highest revision stock (102.0) must be first
        self.assertEqual(res.iloc[0]["ticker"], "STK_099")
        self.assertEqual(res.iloc[0]["eps_rev_1m"], 102.0)
        # 25th stock must have revision 78.0
        self.assertEqual(res.iloc[24]["eps_rev_1m"], 78.0)

    def test_eps_top_tie_breaking(self):
        """Multiple stocks sharing identical revision rate maintain stable ordering without crashing."""
        df = pd.DataFrame({
            "ticker": [f"TIE_{i}" for i in range(30)],
            "eps_rev_1m": [10.0] * 30,
            "current_fwd_eps": [1000.0] * 30,
        })
        res = apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(len(res), 25)
        self.assertTrue((res["eps_rev_1m"] == 10.0).all())

    def test_unknown_preset_key_fallback(self):
        """Unknown or invalid preset keys return the unfiltered DataFrame without throwing exceptions."""
        df = pd.DataFrame([
            {"ticker": "005930", "name": "삼성전자"},
            {"ticker": "000660", "name": "SK하이닉스"},
        ])
        for unknown in ["UNKNOWN", "MOMENTUM", "", None, 12345]:
            res = apply_quick_preset(df, unknown)
            self.assertEqual(len(res), 2)
            self.assertEqual(list(res["ticker"]), ["005930", "000660"])

    def test_case_sensitivity_unfiltered_return(self):
        """
        Hardened Behavior: Lowercase preset keys ('turnaround', 'all') are normalized
        to uppercase and filter correctly rather than returning unfiltered.
        """
        df = pd.DataFrame([
            {"ticker": "FAIL", "pe_percentile": 90.0, "eps_rev_1m": -10.0, "upside_base": -20.0},
        ])
        res = apply_quick_preset(df, "turnaround")
        # Since 'turnaround' is normalized to 'TURNAROUND', non-matching stock is excluded
        self.assertEqual(len(res), 0)

    def test_none_dataframe_returns_empty_dataframe(self):
        """Passing None as DataFrame returns an empty DataFrame safely across all presets."""
        for pkey in ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP", "UNKNOWN", None]:
            res = apply_quick_preset(None, pkey)
            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(len(res), 0)

    def test_whitespace_padded_and_mixed_case_preset_keys(self):
        """Preset keys with whitespace padding and mixed cases are normalized properly."""
        df_candidate = pd.DataFrame([{
            "ticker": "QUAL_VAL",
            "name": "가치주후보",
            "pe_percentile": 20.0,
            "sector_pe_percentile": 25.0,
            "eps_rev_1m": 4.0,
            "current_fwd_eps": 5000.0,
            "current_fwd_pe": 10.0,
            "upside_base": 20.0,
        }])
        # " value "
        res_val = apply_quick_preset(df_candidate, "  value  ")
        self.assertEqual(len(res_val), 1)
        self.assertEqual(list(res_val["ticker"]), ["QUAL_VAL"])

        # " turnaround "
        res_ta = apply_quick_preset(df_candidate, "  TurnAround  ")
        self.assertEqual(len(res_ta), 1)
        self.assertEqual(list(res_ta["ticker"]), ["QUAL_VAL"])

        # " eps_top "
        res_top = apply_quick_preset(df_candidate, "  EPS_top  ")
        self.assertEqual(len(res_top), 1)
        self.assertEqual(list(res_top["ticker"]), ["QUAL_VAL"])

        # " trap " (QUAL_VAL has rev=4.0 >= -3, so excluded from trap)
        res_trap = apply_quick_preset(df_candidate, "  Trap  ")
        self.assertEqual(len(res_trap), 0)

        # " all "
        res_all = apply_quick_preset(df_candidate, "  All  ")
        self.assertEqual(len(res_all), 1)

    def test_extreme_multiples_and_pathological_values_in_value_preset(self):
        """VALUE preset strictly excludes extreme multiples (>1000x, inf, NaN) and negative P/E."""
        df = pd.DataFrame([
            {"ticker": "EXTREME_1000", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": 1000.0, "upside_base": 20.0},
            {"ticker": "EXTREME_5000", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": 5000.0, "upside_base": 20.0},
            {"ticker": "INF_PE", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": float("inf"), "upside_base": 20.0},
            {"ticker": "NAN_PE", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": np.nan, "upside_base": 20.0},
            {"ticker": "NEG_PE_1", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": -0.01, "upside_base": 20.0},
            {"ticker": "NEG_PE_2", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": -100.0, "upside_base": 20.0},
            {"ticker": "ZERO_PE", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": 0.0, "upside_base": 20.0},
            {"ticker": "VALID_VALUE", "pe_percentile": 10.0, "eps_rev_1m": 5.0, "current_fwd_pe": 12.0, "upside_base": 20.0},
        ])
        res = apply_quick_preset(df, "VALUE")
        self.assertEqual(list(res["ticker"]), ["VALID_VALUE"])

    def test_missing_all_filter_columns_safety(self):
        """DataFrame with non-empty rows but completely lacking filter columns is handled safely."""
        df_bare = pd.DataFrame([
            {"ticker": "005930", "company": "Samsung"},
            {"ticker": "000660", "company": "Hynix"},
        ])
        for pkey in ["TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]:
            res = apply_quick_preset(df_bare, pkey)
            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(len(res), 0)

        # ALL returns full bare DataFrame
        res_all = apply_quick_preset(df_bare, "ALL")
        self.assertEqual(len(res_all), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 4: Standalone Fallback Implementation Parity

# ─────────────────────────────────────────────────────────────────────────────
class TestAdversarialFallbackEquivalence(unittest.TestCase):
    """
    Stress tests the standalone native implementation inside app.py
    against the reference oracle across 500 randomized runs.
    """

    def test_app_fallback_500_trials_parity_against_oracle(self):
        """Verify app.py logic has 100% equivalence with oracle."""
        orig_oracle = getattr(app_module, "oracle_apply_quick_preset", None)
        if hasattr(app_module, "oracle_apply_quick_preset"):
            app_module.oracle_apply_quick_preset = None

        try:
            preset_keys = ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]
            for trial in range(500):
                n = (trial % 50) + 1
                df = make_synthetic_screener_df(n_stocks=n, seed=trial + 1000, nan_prob=0.05)

                for pkey in preset_keys:
                    fallback_res = apply_quick_preset(df, pkey)
                    oracle_res = oracle_apply_quick_preset(df, pkey)

                    self.assertEqual(
                        len(fallback_res),
                        len(oracle_res),
                        f"Fallback trial {trial} ({pkey}) row count mismatch",
                    )
                    self.assertEqual(
                        list(fallback_res["ticker"]),
                        list(oracle_res["ticker"]),
                        f"Fallback trial {trial} ({pkey}) ticker mismatch",
                    )
                    pd.testing.assert_frame_equal(
                        fallback_res.reset_index(drop=True),
                        oracle_res.reset_index(drop=True),
                        check_dtype=False,
                    )
        finally:
            if hasattr(app_module, "oracle_apply_quick_preset"):
                app_module.oracle_apply_quick_preset = orig_oracle


# ─────────────────────────────────────────────────────────────────────────────
# Suite 5: Caching Function Signatures & Hashing Integrity (AC 36)
# ─────────────────────────────────────────────────────────────────────────────
class TestAdversarialCachingSignatures(unittest.TestCase):
    """
    Stress tests the caching function signatures to guarantee Streamlit hashes mtimes:
    - load_raw_market_data(mtimes=...)
    - get_cached_screener_results(..., mtimes=...)
    - Ensures NO parameter has a leading underscore ('_mtimes') which would bypass hashing.
    - Ensures app.py has zero imports from tests.
    """

    def test_load_raw_market_data_signature_mtimes(self):
        """load_raw_market_data must have 'mtimes' and must NOT have '_mtimes'."""
        import inspect
        from app import load_raw_market_data
        sig = inspect.signature(load_raw_market_data)
        self.assertIn("mtimes", sig.parameters, "load_raw_market_data must have 'mtimes' parameter")
        self.assertNotIn("_mtimes", sig.parameters, "load_raw_market_data must NOT have '_mtimes' with leading underscore")
        default_val = sig.parameters["mtimes"].default
        self.assertIsInstance(default_val, tuple, "Default value for mtimes must be a tuple")

    def test_get_cached_screener_results_signature_mtimes(self):
        """get_cached_screener_results must have 'mtimes' and must NOT have '_mtimes'."""
        import inspect
        from app import get_cached_screener_results
        sig = inspect.signature(get_cached_screener_results)
        self.assertIn("mtimes", sig.parameters, "get_cached_screener_results must have 'mtimes' parameter")
        self.assertNotIn("_mtimes", sig.parameters, "get_cached_screener_results must NOT have '_mtimes' with leading underscore")
        default_val = sig.parameters["mtimes"].default
        self.assertIsInstance(default_val, tuple, "Default value for mtimes must be a tuple")

    def test_app_py_ast_no_leading_underscore_parameters(self):
        """AST analysis of app.py confirms no cached functions use _mtimes parameter."""
        import ast
        app_path = ROOT / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="app.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_names = [arg.arg for arg in node.args.args]
                self.assertNotIn(
                    "_mtimes",
                    param_names,
                    f"Function '{node.name}' in app.py still has '_mtimes' parameter",
                )

    def test_app_py_zero_imports_from_tests(self):
        """app.py must have zero imports from the tests package."""
        import ast
        app_path = ROOT / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="app.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith("tests"),
                        f"app.py contains illegal import: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.assertFalse(
                        node.module.startswith("tests"),
                        f"app.py contains illegal import from: {node.module}",
                    )


if __name__ == "__main__":
    unittest.main()

