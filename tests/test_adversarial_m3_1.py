"""
tests/test_adversarial_m3_1.py
------------------------------
Milestone 3 Adversarial Stress Testing Suite:
Quantitative Formulas, Outlier Winsorization, Research Standard Bands & Fixed Multiples.
Author: Challenger M3-1 (Empirical Challenger)

Coverage Dimensions:
1. winsorize_pe (Feature 11 / AC 33):
   - All identical values (small sample <10 vs large sample >=10)
   - Massive outliers (10^6, 10^12, -10^6, -10^12) vs normal P/E distribution
   - Pre-filtering bounds: [min_pe=1.0, max_pe=150.0]
   - Exact boundary thresholds: 0.99999 vs 1.0, 150.0 vs 150.0001
   - NaNs, positive/negative infinities, None values
   - Small sample guards (<10 samples return copy without distortion, 0 samples return empty series)
   - Custom quantiles (lower_q, upper_q) and custom bounds
   - Index and timestamp alignment retention
   - 1,000-trial Monte Carlo equivalence against oracle_winsorize
2. calc_mean_sd_bands (Feature 9 / AC 33):
   - Cyclical downturn collapse: high-volatility series where (mean - 2*std < 0)
   - Floor rule verification: floor_val = max(floor_pe, min_obs * 0.8) strictly prevents negative/zero multiples
   - Dynamic floor scaling when min_obs is elevated
   - Small sample guards: <4 samples return {} without uncaught exceptions
   - Bessel's sample standard deviation (ddof=1)
   - Monotonic band ordering: m2sd <= m1sd <= mean <= p1sd <= p2sd
   - Constant series with zero variance
   - Winsorization toggle impact on volatility and band tightness
   - 1,000-trial Monte Carlo equivalence against oracle_mean_sd_bands
3. calc_fixed_multiple_bands (Feature 10 / AC 33):
   - Negative EPS (target strictly 0.0, upside -100.0%)
   - Zero EPS (target strictly 0.0, upside -100.0%)
   - Zero stock price (upside strictly 0.0, no division by zero)
   - Negative stock price (upside strictly 0.0)
   - Both zero EPS and zero price simultaneously
   - Tiny fractional EPS (1e-6, 1e-4) without underflow
   - Extreme multiples (0.1x to 5,000x)
   - Empty multiples list handling
   - Non-integer and fractional multiples (7.5x, 12.25x)
   - 1,000-trial Monte Carlo equivalence against oracle_fixed_multiple_bands
4. Valuation Engine Integration (Features 9, 10, 11 / AC 33, 38):
   - PEBandResult dataclass integrity across all band models (percentile, mean_sd, fixed)
   - Complete time-series band DataFrame generation (band_series_pct, band_series_sd, band_series_fixed)
   - Negative EPS masking (eps <= 0 masked to NaN in time-series bands)
   - Dynamic scenario targets (bear, base, bull) and upside percentages per model
   - Insufficient data (<4 samples) graceful None return
   - Positional parameter backward compatibility for legacy callers
"""

import unittest
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

from core.calculator import (
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    calc_pe_band,
    calc_fwd_pe_series,
    PEBandResult,
    _clean_ticker,
)
from tests.oracle import (
    oracle_winsorize,
    oracle_mean_sd_bands,
    oracle_fixed_multiple_bands,
)


class TestAdversarialWinsorizePE(unittest.TestCase):
    """Adversarial stress testing of winsorize_pe (Feature 11 / AC 33)."""

    def test_all_identical_values_large_n(self):
        """Large sample (N=50) of all identical values must preserve value without division error."""
        s = pd.Series([15.0] * 50)
        win = winsorize_pe(s)
        self.assertEqual(len(win), 50)
        self.assertTrue((win == 15.0).all())
        self.assertAlmostEqual(win.mean(), 15.0)
        self.assertAlmostEqual(win.std(), 0.0)

    def test_all_identical_values_small_n(self):
        """Small sample (N=5 < 10) of all identical values returns unmodified copy."""
        s = pd.Series([20.0] * 5)
        win = winsorize_pe(s)
        self.assertEqual(len(win), 5)
        self.assertTrue((win == 20.0).all())

    def test_massive_outliers_positive_and_negative(self):
        """Massive outliers (+-1e6, +-1e12, -999.0) must be purged by pre-filter [1.0, 150.0]."""
        dirty_values = [
            -1e12, -1e6, -999.0, -10.0, 0.0, 0.5, 0.999,
            10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0,
            150.001, 200.0, 500.0, 1e6, 1e12
        ]
        s = pd.Series(dirty_values)
        win = winsorize_pe(s)
        # All returned values must be strictly in [1.0, 150.0]
        self.assertTrue((win >= 1.0).all())
        self.assertTrue((win <= 150.0).all())
        # The 10 valid elements are 10.0..35.0
        self.assertGreaterEqual(len(win), 10)
        self.assertNotIn(-1e12, win.values)
        self.assertNotIn(1e12, win.values)

    def test_nan_and_inf_mixes(self):
        """Infs and NaNs interspersed in the series must be completely purged."""
        mixed = [10.0, np.nan, 12.0, np.inf, -np.inf, float("nan"), 14.0, 16.0, 18.0,
                 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        s = pd.Series(mixed)
        win = winsorize_pe(s)
        self.assertFalse(win.isna().any())
        self.assertFalse(np.isinf(win).any())
        self.assertTrue((win >= 1.0).all())
        self.assertTrue((win <= 150.0).all())

    def test_small_sample_guards_exact_thresholds(self):
        """Samples with len < 10 must bypass clipping and return valid unmodified copy."""
        # 0 elements (empty)
        empty_s = pd.Series([], dtype=float)
        win_empty = winsorize_pe(empty_s)
        self.assertEqual(len(win_empty), 0)

        # 1 element
        s1 = pd.Series([12.0])
        self.assertEqual(list(winsorize_pe(s1)), [12.0])

        # 9 elements (just below 10)
        s9 = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        win9 = winsorize_pe(s9)
        self.assertEqual(len(win9), 9)
        self.assertEqual(list(win9), list(s9))

        # Exactly 10 elements (threshold for clipping)
        s10 = pd.Series([1.5] + [10.0] * 8 + [145.0])
        win10 = winsorize_pe(s10)
        self.assertEqual(len(win10), 10)
        # Should have clipped the extreme spike
        self.assertLess(win10.max(), 145.0)

    def test_exact_boundary_min_pe_1_0(self):
        """Exact boundary tests at min_pe = 1.0."""
        s = pd.Series([0.99999, 1.0, 1.00001, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
        win = winsorize_pe(s)
        # 0.99999 must be dropped
        self.assertNotIn(0.99999, win.values)
        # 1.0 and 1.00001 must be preserved
        self.assertIn(1.0, win.values)
        self.assertIn(1.00001, win.values)

    def test_exact_boundary_max_pe_150_0(self):
        """Exact boundary tests at max_pe = 150.0."""
        s = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 149.9999, 150.0, 150.0001])
        win = winsorize_pe(s)
        # 150.0001 must be dropped
        self.assertNotIn(150.0001, win.values)
        # 149.9999 and 150.0 must be preserved
        self.assertIn(150.0, win.values)

    def test_all_outliers_returns_empty_series(self):
        """If all elements are outside [min_pe, max_pe], return empty series without crash."""
        s = pd.Series([0.1, 0.5, 0.9, -5.0, 150.1, 200.0, 999.0])
        win = winsorize_pe(s)
        self.assertEqual(len(win), 0)

    def test_custom_bounds_and_quantiles(self):
        """Supports custom lower_q, upper_q, min_pe, max_pe."""
        s = pd.Series(list(range(1, 101)), dtype=float)
        win = winsorize_pe(s, lower_q=0.10, upper_q=0.90, min_pe=5.0, max_pe=80.0)
        self.assertGreaterEqual(win.min(), 5.0)
        self.assertLessEqual(win.max(), 80.0)
        self.assertEqual(len(win), 76)  # 5 through 80 = 76 elements

    def test_index_and_timestamp_retention(self):
        """Winsorized series retains original Series index for retained elements."""
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        vals = [10.0, -5.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 300.0]
        s = pd.Series(vals, index=dates)
        win = winsorize_pe(s)
        self.assertNotIn(dates[1], win.index)   # -5.0 dropped
        self.assertNotIn(dates[11], win.index)  # 300.0 dropped
        self.assertIn(dates[0], win.index)
        self.assertIn(dates[10], win.index)

    def test_monte_carlo_1000_trials_winsorize_parity(self):
        """1,000-trial randomized Monte Carlo stress test against oracle_winsorize."""
        rng = np.random.RandomState(42)
        for trial in range(1000):
            # Vary sample sizes from 0 to 300
            n = rng.randint(0, 301)
            dist_type = trial % 5

            if n == 0:
                raw = np.array([], dtype=float)
            elif dist_type == 0:
                # Normal distribution with extreme outliers
                raw = rng.normal(loc=15.0, scale=8.0, size=n)
                if n > 5:
                    raw[rng.choice(n, size=min(3, n), replace=False)] = [-1000.0, 0.5, 999.0][:min(3, n)]
            elif dist_type == 1:
                # Uniform distribution spanning negative to huge positive
                raw = rng.uniform(low=-20.0, high=200.0, size=n)
            elif dist_type == 2:
                # Heavy-tailed Cauchy distribution
                raw = rng.standard_cauchy(size=n) * 10.0 + 15.0
            elif dist_type == 3:
                # All identical or near-identical
                raw = np.full(n, 12.5) + rng.choice([0.0, 1e-4, -1e-4], size=n)
            else:
                # Lognormal distribution
                raw = rng.lognormal(mean=2.5, sigma=1.0, size=n)

            # Randomly inject NaNs and Infs
            if n > 10:
                mask_nan = rng.rand(n) < 0.05
                raw[mask_nan] = np.nan
                mask_inf = rng.rand(n) < 0.02
                raw[mask_inf] = np.inf

            series = pd.Series(raw)
            res_worker = winsorize_pe(series)
            res_oracle = oracle_winsorize(series)

            self.assertEqual(len(res_worker), len(res_oracle), f"Length mismatch on trial {trial}")
            if len(res_worker) > 0:
                np.testing.assert_allclose(
                    res_worker.values,
                    res_oracle.values,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch on trial {trial}",
                )


class TestAdversarialMeanSdBands(unittest.TestCase):
    """Adversarial stress testing of calc_mean_sd_bands (Feature 9 / AC 33)."""

    def test_cyclical_collapse_negative_sd_prevention(self):
        """Cyclical downturn with extreme volatility where (mean - 2*std < 0).
        Floor rule max(floor_pe, min_obs * 0.8) MUST prevent negative/zero multiples.
        """
        # Volatile series with severe cyclical earnings collapse
        s = pd.Series([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 120.0])
        bands = calc_mean_sd_bands(s, floor_pe=1.0, winsorize=False)

        raw_mean = float(s.mean())
        raw_std = float(s.std(ddof=1))
        # Verify that unfloored raw calculation would have been negative
        self.assertLess(raw_mean - 2.0 * raw_std, 0.0, "Prerequisite: raw mean - 2*std must be negative")

        min_obs = 1.2
        expected_floor = max(1.0, min_obs * 0.8)  # max(1.0, 0.96) = 1.0
        self.assertGreaterEqual(bands["m2sd"], expected_floor)
        self.assertGreaterEqual(bands["m1sd"], expected_floor)
        self.assertGreater(bands["m2sd"], 0.0, "m2sd must be strictly positive")
        self.assertGreater(bands["m1sd"], 0.0, "m1sd must be strictly positive")

    def test_floor_rule_dynamic_scaling_with_min_obs(self):
        """When min_obs is large (e.g. 50.0), min_obs * 0.8 = 40.0 dominates floor_pe = 1.0."""
        # High-multiple sector (e.g. bio/tech) with min_obs = 50.0
        s = pd.Series([50.0, 50.0, 50.0, 140.0, 140.0])
        bands = calc_mean_sd_bands(s, floor_pe=1.0, winsorize=False)
        expected_floor = max(1.0, 50.0 * 0.8)  # 40.0
        self.assertAlmostEqual(expected_floor, 40.0)
        self.assertGreaterEqual(bands["m1sd"], 40.0)
        self.assertGreaterEqual(bands["m2sd"], 40.0)

    def test_small_sample_guards_under_4(self):
        """Series with < 4 valid observations must return {} without raising exceptions."""
        for count in [0, 1, 2, 3]:
            s = pd.Series([10.0 + i for i in range(count)])
            res = calc_mean_sd_bands(s)
            self.assertEqual(res, {}, f"Expected {{}} for sample count {count}, got {res}")

        # Exactly 4 observations should successfully calculate
        s4 = pd.Series([10.0, 12.0, 14.0, 16.0])
        res4 = calc_mean_sd_bands(s4, winsorize=False)
        self.assertNotEqual(res4, {})
        self.assertIn("mean", res4)
        self.assertIn("m2sd", res4)

    def test_monotonic_band_ordering(self):
        """Monotonic order m2sd <= m1sd <= mean <= p1sd <= p2sd must hold across all test series."""
        test_series_list = [
            pd.Series([10.0, 12.0, 15.0, 18.0, 22.0]),
            pd.Series([5.0, 5.0, 5.0, 50.0]),
            pd.Series([1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 100.0]),
            pd.Series([30.0] * 20),
        ]
        for s in test_series_list:
            bands = calc_mean_sd_bands(s, floor_pe=1.0, winsorize=True)
            if bands:
                self.assertLessEqual(bands["m2sd"], bands["m1sd"])
                self.assertLessEqual(bands["m1sd"], bands["mean"])
                self.assertLessEqual(bands["mean"], bands["p1sd"])
                self.assertLessEqual(bands["p1sd"], bands["p2sd"])

    def test_constant_series_zero_std(self):
        """Constant series has std = 0.0 and all bands equal to the constant value."""
        s = pd.Series([25.0] * 30)
        bands = calc_mean_sd_bands(s, winsorize=False)
        self.assertAlmostEqual(bands["mean"], 25.0)
        self.assertAlmostEqual(bands["std"], 0.0)
        self.assertAlmostEqual(bands["p2sd"], 25.0)
        self.assertAlmostEqual(bands["p1sd"], 25.0)
        self.assertAlmostEqual(bands["m1sd"], 25.0)
        self.assertAlmostEqual(bands["m2sd"], 25.0)

    def test_sample_std_uses_ddof_1(self):
        """Standard deviation must strictly use Bessel's correction ddof=1."""
        s = pd.Series([10.0, 15.0, 20.0, 25.0, 30.0])
        bands = calc_mean_sd_bands(s, winsorize=False)
        expected_std = float(s.std(ddof=1))
        self.assertAlmostEqual(bands["std"], expected_std, places=6)

    def test_winsorize_toggle_impact(self):
        """Winsorize=True must effectively insulate Mean ± SD from extreme outlier spikes."""
        s = pd.Series([10.0] * 20 + [12.0] * 20 + [145.0])
        bands_raw = calc_mean_sd_bands(s, winsorize=False)
        bands_win = calc_mean_sd_bands(s, winsorize=True)
        # Outlier inflates raw std; winsorized std must be strictly lower
        self.assertLess(bands_win["std"], bands_raw["std"])
        self.assertLess(bands_win["mean"], bands_raw["mean"])
        self.assertLess(bands_win["p2sd"], bands_raw["p2sd"])

    def test_monte_carlo_1000_trials_mean_sd_parity(self):
        """1,000-trial randomized Monte Carlo stress test against oracle_mean_sd_bands."""
        rng = np.random.RandomState(101)
        for trial in range(1000):
            n = rng.randint(0, 150)
            if n < 4:
                raw = rng.uniform(5.0, 30.0, size=n)
            else:
                dist_choice = trial % 4
                if dist_choice == 0:
                    raw = rng.normal(loc=18.0, scale=12.0, size=n)
                elif dist_choice == 1:
                    raw = rng.uniform(low=0.5, high=160.0, size=n)
                elif dist_choice == 2:
                    raw = rng.exponential(scale=10.0, size=n) + 1.0
                else:
                    raw = np.full(n, 15.0)

            # Randomly pick floor_pe and winsorize toggle
            floor_pe = rng.choice([0.5, 1.0, 2.0, 5.0])
            winsorize_flag = bool(rng.choice([True, False]))

            series = pd.Series(raw)
            res_worker = calc_mean_sd_bands(series, floor_pe=floor_pe, winsorize=winsorize_flag)
            res_oracle = oracle_mean_sd_bands(series, floor_pe=floor_pe, winsorize=winsorize_flag)

            self.assertEqual(list(res_worker.keys()), list(res_oracle.keys()), f"Keys mismatch on trial {trial}")
            for k in res_worker:
                self.assertAlmostEqual(
                    res_worker[k],
                    res_oracle[k],
                    places=5,
                    msg=f"Value mismatch on key '{k}' in trial {trial}",
                )


class TestAdversarialFixedMultipleBands(unittest.TestCase):
    """Adversarial stress testing of calc_fixed_multiple_bands (Feature 10 / AC 33)."""

    def test_negative_current_eps(self):
        """Negative EPS must yield target = 0.0 and upside = -100.0% (when price > 0)."""
        multiples = [8.0, 10.0, 12.0, 15.0]
        res = calc_fixed_multiple_bands(multiples, current_eps=-500.0, current_price=50000.0)
        for m in multiples:
            self.assertEqual(res["targets"][m], 0.0)
            self.assertEqual(res["upsides"][m], -100.0)

    def test_zero_current_eps(self):
        """Zero EPS must yield target = 0.0 and upside = -100.0% (when price > 0)."""
        multiples = [8.0, 10.0]
        res = calc_fixed_multiple_bands(multiples, current_eps=0.0, current_price=20000.0)
        for m in multiples:
            self.assertEqual(res["targets"][m], 0.0)
            self.assertEqual(res["upsides"][m], -100.0)

    def test_zero_stock_price(self):
        """Zero stock price must yield upside = 0.0 to prevent ZeroDivisionError."""
        multiples = [10.0, 12.0]
        res = calc_fixed_multiple_bands(multiples, current_eps=3000.0, current_price=0.0)
        self.assertEqual(res["targets"][10.0], 30000.0)
        self.assertEqual(res["upsides"][10.0], 0.0)
        self.assertEqual(res["upsides"][12.0], 0.0)

    def test_negative_stock_price(self):
        """Negative stock price (anomaly) must yield upside = 0.0."""
        res = calc_fixed_multiple_bands([10.0], current_eps=2000.0, current_price=-1000.0)
        self.assertEqual(res["upsides"][10.0], 0.0)

    def test_both_zero_eps_and_zero_price(self):
        """Simultaneous zero EPS and zero price."""
        res = calc_fixed_multiple_bands([8.0, 10.0], current_eps=0.0, current_price=0.0)
        for m in [8.0, 10.0]:
            self.assertEqual(res["targets"][m], 0.0)
            self.assertEqual(res["upsides"][m], 0.0)

    def test_tiny_fractional_eps(self):
        """Tiny EPS (e.g. 0.0001 KRW) scales targets without numerical underflow."""
        res = calc_fixed_multiple_bands([10.0, 20.0], current_eps=0.0001, current_price=0.001)
        self.assertAlmostEqual(res["targets"][10.0], 0.001, places=6)
        self.assertAlmostEqual(res["targets"][20.0], 0.002, places=6)
        self.assertAlmostEqual(res["upsides"][10.0], 0.0, places=4)
        self.assertAlmostEqual(res["upsides"][20.0], 100.0, places=4)

    def test_extreme_multiples_list(self):
        """Extreme multiples from 0.1x to 5,000x."""
        extreme_m = [0.1, 0.5, 1.0, 50.0, 500.0, 5000.0]
        res = calc_fixed_multiple_bands(extreme_m, current_eps=1000.0, current_price=10000.0)
        self.assertAlmostEqual(res["targets"][0.1], 100.0)
        self.assertAlmostEqual(res["targets"][5000.0], 5000000.0)
        self.assertAlmostEqual(res["upsides"][0.1], -99.0)
        self.assertAlmostEqual(res["upsides"][5000.0], 49900.0)

    def test_empty_multiples_list(self):
        """Empty multiples list returns empty targets and upsides dicts."""
        res = calc_fixed_multiple_bands([], current_eps=5000.0, current_price=50000.0)
        self.assertEqual(res, {"targets": {}, "upsides": {}})

    def test_unordered_and_fractional_multiples(self):
        """Unordered and fractional multiples preserve dict keys and exact mappings."""
        multiples = [12.25, 7.5, 15.0, 8.0]
        res = calc_fixed_multiple_bands(multiples, current_eps=4000.0, current_price=40000.0)
        self.assertEqual(set(res["targets"].keys()), set(multiples))
        self.assertAlmostEqual(res["targets"][7.5], 30000.0)
        self.assertAlmostEqual(res["targets"][12.25], 49000.0)
        self.assertAlmostEqual(res["upsides"][7.5], -25.0)
        self.assertAlmostEqual(res["upsides"][12.25], 22.5)

    def test_monte_carlo_1000_trials_fixed_multiples_parity(self):
        """1,000-trial randomized Monte Carlo stress test against oracle_fixed_multiple_bands."""
        rng = np.random.RandomState(202)
        for trial in range(1000):
            num_m = rng.randint(0, 8)
            multiples = list(rng.uniform(1.0, 100.0, size=num_m))
            # 20% probability of negative or zero EPS
            if rng.rand() < 0.1:
                eps = float(rng.uniform(-1000.0, 0.0))
            elif rng.rand() < 0.1:
                eps = 0.0
            else:
                eps = float(rng.uniform(10.0, 50000.0))

            # 10% probability of zero or negative price
            if rng.rand() < 0.05:
                price = 0.0
            elif rng.rand() < 0.05:
                price = -100.0
            else:
                price = float(rng.uniform(1000.0, 500000.0))

            res_worker = calc_fixed_multiple_bands(multiples, current_eps=eps, current_price=price)
            res_oracle = oracle_fixed_multiple_bands(multiples, current_eps=eps, current_price=price)

            self.assertEqual(res_worker, res_oracle, f"Mismatch on trial {trial}")


class TestAdversarialBandIntegration(unittest.TestCase):
    """Adversarial stress testing of calc_pe_band with M3 extensions and time series generation."""

    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        self.dates = dates
        self.p_series = pd.Series([50000.0 + i * 800 for i in range(36)], index=dates)
        self.e_series = pd.Series([4000.0 + i * 50 for i in range(36)], index=dates)

    def test_all_three_band_series_generated(self):
        """calc_pe_band must generate band_series_pct, band_series_sd, and band_series_fixed."""
        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.p_series,
            eps_series=self.e_series,
            band_model="percentile",
            winsorize=True,
        )
        self.assertIsNotNone(res)
        # (1) Percentile DataFrame
        self.assertIsNotNone(res.band_series_pct)
        self.assertEqual(list(res.band_series_pct.columns), ["p10", "p25", "p50", "p75", "p90"])
        self.assertEqual(len(res.band_series_pct), 36)

        # (2) Mean +- SD DataFrame
        self.assertIsNotNone(res.band_series_sd)
        self.assertEqual(list(res.band_series_sd.columns), ["-2SD", "-1SD", "Mean", "+1SD", "+2SD"])
        self.assertEqual(len(res.band_series_sd), 36)

        # (3) Fixed Multiples DataFrame
        self.assertIsNotNone(res.band_series_fixed)
        self.assertEqual(list(res.band_series_fixed.columns), ["8x", "10x", "12x", "15x"])
        self.assertEqual(len(res.band_series_fixed), 36)

    def test_negative_eps_masking_in_time_series_bands(self):
        """Negative EPS values in historical series must be masked to NaN in band series."""
        eps_with_neg = self.e_series.copy()
        eps_with_neg.iloc[5] = -200.0
        eps_with_neg.iloc[6] = 0.0

        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.p_series,
            eps_series=eps_with_neg,
        )
        self.assertIsNotNone(res)
        # Check that rows with eps <= 0 produce NaN in band series
        self.assertTrue(np.isnan(res.band_series_pct.iloc[5]["p50"]))
        self.assertTrue(np.isnan(res.band_series_pct.iloc[6]["p50"]))
        self.assertTrue(np.isnan(res.band_series_sd.iloc[5]["Mean"]))
        self.assertTrue(np.isnan(res.band_series_fixed.iloc[5]["10x"]))
        # Valid positive row must remain non-NaN
        self.assertFalse(np.isnan(res.band_series_pct.iloc[7]["p50"]))

    def test_dynamic_targets_per_band_model(self):
        """Targets and upsides must dynamically adjust based on band_model setting."""
        # Mean +- SD mode
        res_sd = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.p_series,
            eps_series=self.e_series,
            band_model="mean_sd",
        )
        self.assertIsNotNone(res_sd)
        cur_eps = res_sd.current_fwd_eps
        cur_p = res_sd.current_price
        m_metrics = res_sd.mean_sd_metrics
        self.assertAlmostEqual(res_sd.target_bear, cur_eps * m_metrics["m1sd"])
        self.assertAlmostEqual(res_sd.target_base, cur_eps * m_metrics["mean"])
        self.assertAlmostEqual(res_sd.target_bull, cur_eps * m_metrics["p1sd"])
        self.assertAlmostEqual(res_sd.upside_base, ((res_sd.target_base / cur_p) - 1.0) * 100.0)

        # Fixed multiples mode
        res_fixed = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.p_series,
            eps_series=self.e_series,
            band_model="fixed",
            fixed_multiples=[8.0, 10.0, 12.0, 15.0],
        )
        self.assertIsNotNone(res_fixed)
        self.assertAlmostEqual(res_fixed.target_bear, cur_eps * 8.0)
        self.assertAlmostEqual(res_fixed.target_base, cur_eps * 10.0)
        self.assertAlmostEqual(res_fixed.target_bull, cur_eps * 15.0)

    def test_insufficient_data_returns_none(self):
        """Fewer than 4 valid observations returns None without crash."""
        short_p = self.p_series.iloc[:3]
        short_e = self.e_series.iloc[:3]
        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=short_p,
            eps_series=short_e,
        )
        self.assertIsNone(res)

    def test_backward_compatibility_sector_positional_arg(self):
        """Legacy callers passing sector as 6th positional argument must maintain compatibility."""
        res = calc_pe_band(
            "005930",
            "삼성전자",
            self.p_series,
            self.e_series,
            5,
            "반도체",  # 6th arg as sector string
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.sector, "반도체")
        self.assertEqual(res.band_model, "percentile")


if __name__ == "__main__":
    unittest.main(verbosity=2)
