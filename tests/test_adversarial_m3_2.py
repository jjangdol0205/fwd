"""
tests/test_adversarial_m3_2.py
------------------------------
Adversarial Stress-Testing Suite for Milestone 3 (Research Standard Bands & Outlier Handling).
Focus: PEBandResult, calc_pe_band time-series integration, dynamic model switching,
and M2 polish regression verification.

Author: Challenger M3-2 (Empirical Challenger)

Coverage Dimensions:
1. Time-Series Band Generation Across Diverse Stock Profiles & Regimes:
   - Profile A: Normal/Mature Stock (stable valuation, steady growth)
   - Profile B: High Growth / High Multiple Stock (explosive EPS and P/E expansion)
   - Profile C: Cyclical Turnaround with Negative Historical EPS (recession to recovery)
   - Profile D: Flatline Price / Zero Variance Stock (constant price & EPS, std=0)
   - Profile E: Extreme Outlier Spike Stock (insulating bands via floor protection)
   - Negative EPS date behavior: strictly NaN bands on negative EPS dates, positive on positive dates
   - Current loss-maker guard: calc_pe_band returns None when latest EPS <= 0
   - Historical depth guard: calc_pe_band returns None when valid historical P/E < 4
2. Time-Series DataFrame Integrity & Invariants:
   - Exact DatetimeIndex equality across price, eps, and all 3 band DataFrames
   - Preserving time-series alignment across Monthly Price vs Daily EPS (ffill reindexing)
   - Non-null and finite value guarantees on all valid positive dates (np.isfinite)
   - Multiples column naming and formatting ('8x', '6.5x', etc.)
   - Monotonic column ordering:
     * band_series_pct: p10 <= p25 <= p50 <= p75 <= p90
     * band_series_sd: -2SD <= -1SD <= Mean <= +1SD <= +2SD
     * band_series_fixed: 8x <= 10x <= 12x <= 15x
3. Dynamic Model Switching & Winsorization Toggle:
   - Dynamic target assignment: "percentile", "mean_sd", "fixed"
   - Winsorization toggle: winsorize=True vs winsorize=False variance reduction & outlier insulation
   - Fixed multiples scenario targets: odd count, even count, 2 multiples, 1 multiple, empty list
   - Zero current price guard: upside values 0.0 without ZeroDivisionError
   - Backward compatibility: 6th positional sector argument mapping
4. M2 Polish Items Regression & Edge Shapes:
   - SectorMappingDict.__contains__: clean, A-prefixed, a-prefixed, whitespace-padded, integer, non-existent, None, NaN
   - _calc_sector_relative_df: empty DataFrame, columnless DataFrame, single-row DataFrame,
     ticker in index vs column vs first column, loss-making constituents
5. Reference Oracle Parity:
   - Strict numerical equality with oracle_winsorize, oracle_mean_sd_bands, oracle_fixed_multiple_bands
"""

import sys
import math
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calculator import (
    calc_pe_band,
    calc_fwd_pe_series,
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    _calc_sector_relative_df,
    _clean_ticker,
    PEBandResult,
    SectorMappingDict,
    load_sector_mapping,
)
from tests.oracle import (
    oracle_winsorize,
    oracle_mean_sd_bands,
    oracle_fixed_multiple_bands,
    clean_ticker,
)


class TestTimeSeriesBandGenerationAdversarial(unittest.TestCase):
    """Adversarial stress-testing of time-series band generation across diverse market regimes."""

    def test_profile_normal_mature_stock(self):
        """Profile A: Mature stock with steady earnings growth and stable P/E multiples."""
        dates = pd.date_range("2021-01-31", periods=36, freq="M")
        # Steady EPS growing from 4,000 to 5,400 KRW
        eps_vals = [4000.0 + i * 40.0 for i in range(36)]
        # Price fluctuating around 12x-15x P/E (50,000 to 80,000 KRW)
        price_vals = [eps_vals[i] * (12.0 + (i % 5) * 0.7) for i in range(36)]
        eps_s = pd.Series(eps_vals, index=dates)
        price_s = pd.Series(price_vals, index=dates)

        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=price_s,
            eps_series=eps_s,
            band_years=5,
            band_model="percentile",
            winsorize=True,
            fixed_multiples=[8.0, 10.0, 12.0, 15.0],
        )

        self.assertIsNotNone(res)
        self.assertIsInstance(res.band_series_pct, pd.DataFrame)
        self.assertIsInstance(res.band_series_sd, pd.DataFrame)
        self.assertIsInstance(res.band_series_fixed, pd.DataFrame)

        # 1. Verify exact columns
        self.assertEqual(list(res.band_series_pct.columns), ["p10", "p25", "p50", "p75", "p90"])
        self.assertEqual(list(res.band_series_sd.columns), ["-2SD", "-1SD", "Mean", "+1SD", "+2SD"])
        self.assertEqual(list(res.band_series_fixed.columns), ["8x", "10x", "12x", "15x"])

        # 2. Row-by-row strict monotonicity across all dates
        for dt in dates:
            row_pct = res.band_series_pct.loc[dt]
            self.assertTrue(row_pct["p10"] <= row_pct["p25"] <= row_pct["p50"] <= row_pct["p75"] <= row_pct["p90"])

            row_sd = res.band_series_sd.loc[dt]
            self.assertTrue(row_sd["-2SD"] <= row_sd["-1SD"] <= row_sd["Mean"] <= row_sd["+1SD"] <= row_sd["+2SD"])

            row_fx = res.band_series_fixed.loc[dt]
            self.assertTrue(row_fx["8x"] <= row_fx["10x"] <= row_fx["12x"] <= row_fx["15x"])

        # 3. Mathematical relation checks
        cur_eps = eps_vals[-1]
        self.assertAlmostEqual(res.band_series_fixed.loc[dates[-1], "10x"], cur_eps * 10.0, places=4)
        self.assertAlmostEqual(res.band_series_sd.loc[dates[-1], "Mean"], cur_eps * res.mean_sd_metrics["mean"], places=4)
        self.assertAlmostEqual(res.band_series_pct.loc[dates[-1], "p50"], cur_eps * res.pe_median, places=4)

    def test_profile_high_growth_expanding_multiple_stock(self):
        """Profile B: Explosive growth stock (EPS 5x, P/E expanding from 20x to 80x)."""
        dates = pd.date_range("2020-01-31", periods=48, freq="M")
        eps_vals = [500.0 * (1.05 ** i) for i in range(48)]  # 500 -> ~5100
        pe_multiples = [20.0 + i * 1.2 for i in range(48)]    # 20x -> ~76x
        price_vals = [e * m for e, m in zip(eps_vals, pe_multiples)]
        eps_s = pd.Series(eps_vals, index=dates)
        price_s = pd.Series(price_vals, index=dates)

        res = calc_pe_band(
            ticker="035720",
            name="카카오",
            price_series=price_s,
            eps_series=eps_s,
            band_years=5,
            band_model="mean_sd",
            winsorize=True,
        )

        self.assertIsNotNone(res)
        # Verify no NaN or Inf in all band series
        self.assertTrue(np.isfinite(res.band_series_pct.values).all())
        self.assertTrue(np.isfinite(res.band_series_sd.values).all())
        self.assertTrue(np.isfinite(res.band_series_fixed.values).all())

        # Verify bands scale smoothly up to top date
        self.assertGreater(res.band_series_sd.loc[dates[-1], "+2SD"], res.band_series_sd.loc[dates[0], "+2SD"])
        self.assertGreater(res.target_bull, res.target_base)
        self.assertGreater(res.target_base, res.target_bear)

    def test_profile_cyclical_turnaround_with_negative_historical_eps(self):
        """
        Profile C: Cyclical stock with negative historical EPS during an industry recession.
        Months 0-11: positive EPS (3,000 KRW)
        Months 12-23: negative EPS (-1,500 KRW) deep cyclical loss
        Months 24-35: turnaround recovery (1,000 -> 4,000 KRW)
        Latest EPS is positive (4,000 KRW).
        """
        dates = pd.date_range("2021-01-31", periods=36, freq="M")
        eps_vals = []
        for i in range(36):
            if i < 12:
                eps_vals.append(3000.0)
            elif i < 24:
                eps_vals.append(-1500.0 + (i - 12) * 50.0)  # All negative: -1500 to -950
            else:
                eps_vals.append(1000.0 + (i - 24) * 250.0)  # Positive recovery: 1000 to 3750

        # Stock price remains positive throughout (15,000 to 60,000)
        price_vals = [30000.0 + (i - 18) * 1000.0 for i in range(36)]
        eps_s = pd.Series(eps_vals, index=dates)
        price_s = pd.Series(price_vals, index=dates)

        res = calc_pe_band(
            ticker="000660",
            name="SK하이닉스",
            price_series=price_s,
            eps_series=eps_s,
            band_years=5,
            band_model="percentile",
            winsorize=True,
        )

        self.assertIsNotNone(res)

        # 1. Timeline integrity: full 36 months preserved
        self.assertEqual(len(res.band_series_pct), 36)
        self.assertEqual(len(res.band_series_sd), 36)
        self.assertEqual(len(res.band_series_fixed), 36)

        # 2. Critical Adversarial Check:
        # On negative EPS months (12 to 23), band values MUST BE strictly NaN
        # (Stock price bands cannot be negative or zero for unprofitable periods)
        for i in range(12, 24):
            dt = dates[i]
            pct_row = res.band_series_pct.loc[dt]
            sd_row = res.band_series_sd.loc[dt]
            fixed_row = res.band_series_fixed.loc[dt]

            self.assertTrue(pct_row.isna().all(), f"pct bands at negative EPS date {dt} must be NaN")
            self.assertTrue(sd_row.isna().all(), f"sd bands at negative EPS date {dt} must be NaN")
            self.assertTrue(fixed_row.isna().all(), f"fixed bands at negative EPS date {dt} must be NaN")

        # 3. On positive EPS months (0 to 11 and 24 to 35), band values MUST BE finite and positive
        valid_indices = list(range(0, 12)) + list(range(24, 36))
        for i in valid_indices:
            dt = dates[i]
            pct_row = res.band_series_pct.loc[dt]
            sd_row = res.band_series_sd.loc[dt]
            fixed_row = res.band_series_fixed.loc[dt]

            self.assertTrue(np.isfinite(pct_row.values).all(), f"pct bands at valid date {dt} must be finite")
            self.assertTrue((pct_row > 0).all(), f"pct bands at valid date {dt} must be strictly positive")
            self.assertTrue(np.isfinite(sd_row.values).all(), f"sd bands at valid date {dt} must be finite")
            self.assertTrue((sd_row > 0).all(), f"sd bands at valid date {dt} must be strictly positive")
            self.assertTrue(np.isfinite(fixed_row.values).all(), f"fixed bands at valid date {dt} must be finite")
            self.assertTrue((fixed_row > 0).all(), f"fixed bands at valid date {dt} must be strictly positive")

    def test_cyclical_current_deficit_returns_none(self):
        """Current loss-maker (latest EPS <= 0) must return None regardless of historical profits."""
        dates = pd.date_range("2021-01-31", periods=36, freq="M")
        eps_vals = [3000.0] * 35 + [-500.0]  # Latest is loss-making
        price_vals = [50000.0] * 36
        res = calc_pe_band("000660", "SK하이닉스", pd.Series(price_vals, index=dates), pd.Series(eps_vals, index=dates))
        self.assertIsNone(res, "Latest EPS <= 0 must return None")

        # Zero current EPS
        eps_zero = [3000.0] * 35 + [0.0]
        res_zero = calc_pe_band("000660", "SK하이닉스", pd.Series(price_vals, index=dates), pd.Series(eps_zero, index=dates))
        self.assertIsNone(res_zero, "Latest EPS == 0 must return None")

    def test_insufficient_historical_pe_data_returns_none(self):
        """Fewer than 4 valid historical P/E observations must return None."""
        dates = pd.date_range("2021-01-31", periods=20, freq="M")
        # Only 2 positive EPS months out of 20
        eps_vals = [-1000.0] * 18 + [2000.0, 2500.0]
        price_vals = [30000.0] * 20
        res = calc_pe_band("000000", "테스트", pd.Series(price_vals, index=dates), pd.Series(eps_vals, index=dates))
        self.assertIsNone(res, "Historical positive P/E count < 4 must return None")

    def test_profile_flatline_price_zero_variance_stock(self):
        """
        Profile D: Stock with perfectly constant price and EPS across 36 months (std == 0).
        Evaluates floor rule and zero-variance stability without division by zero.
        """
        dates = pd.date_range("2021-01-31", periods=36, freq="M")
        price_s = pd.Series([20000.0] * 36, index=dates)
        eps_s = pd.Series([2000.0] * 36, index=dates)  # P/E is constant 10.0

        res = calc_pe_band(
            ticker="000020",
            name="동화약품",
            price_series=price_s,
            eps_series=eps_s,
            band_years=5,
            band_model="mean_sd",
            winsorize=True,
        )

        self.assertIsNotNone(res)
        self.assertAlmostEqual(res.current_fwd_pe, 10.0, places=4)
        self.assertAlmostEqual(res.mean_sd_metrics["std"], 0.0, places=4)
        self.assertAlmostEqual(res.mean_sd_metrics["mean"], 10.0, places=4)

        # Floor rule: min_observed = 10.0, floor_val = max(1.0, 10.0 * 0.8) = 8.0
        # m1sd = max(10 - 0, 8.0) = 10.0? Wait: 10 - 0 = 10.0, floor_val = 8.0 -> max(10.0, 8.0) = 10.0!
        self.assertAlmostEqual(res.mean_sd_metrics["m1sd"], 10.0, places=4)
        self.assertAlmostEqual(res.mean_sd_metrics["m2sd"], 10.0, places=4)
        self.assertAlmostEqual(res.mean_sd_metrics["p1sd"], 10.0, places=4)
        self.assertAlmostEqual(res.mean_sd_metrics["p2sd"], 10.0, places=4)

        # Monotonicity check
        row_sd = res.band_series_sd.iloc[-1]
        self.assertTrue(row_sd["-2SD"] <= row_sd["-1SD"] <= row_sd["Mean"] <= row_sd["+1SD"] <= row_sd["+2SD"])

    def test_profile_extreme_outlier_spikes_and_floor_protection(self):
        """
        Profile E: Highly volatile P/E series where std is larger than mean.
        Tests that m2sd is strictly protected by floor_val = max(floor_pe, min_observed * 0.8).
        """
        dates = pd.date_range("2021-01-31", periods=12, freq="M")
        # Volatile P/E: 1.5, 2.0, 3.0, 14.0, 15.0...
        pe_vals = [1.5, 2.0, 2.5, 3.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0]
        eps_vals = [1000.0] * 12
        price_vals = [1000.0 * p for p in pe_vals]
        eps_s = pd.Series(eps_vals, index=dates)
        price_s = pd.Series(price_vals, index=dates)

        res = calc_pe_band(
            ticker="000100",
            name="유한양행",
            price_series=price_s,
            eps_series=eps_s,
            band_years=5,
            band_model="mean_sd",
            winsorize=False,  # Raw to observe full dispersion
        )

        self.assertIsNotNone(res)
        metrics = res.mean_sd_metrics
        min_observed = float(np.min(pe_vals))
        expected_floor = max(1.0, min_observed * 0.8)
        self.assertGreaterEqual(metrics["m1sd"], expected_floor)
        self.assertGreaterEqual(metrics["m2sd"], expected_floor)
        self.assertTrue(res.band_series_sd["-2SD"].min() >= expected_floor * 1000.0)


class TestTimeSeriesDataFrameIntegrityAdversarial(unittest.TestCase):
    """Adversarial testing of DatetimeIndex equality, alignment, and non-null guarantees."""

    def test_datetime_index_strict_equality(self):
        """All 3 band series DataFrames must share identical DatetimeIndex with price and EPS."""
        dates = pd.date_range("2020-01-31", periods=48, freq="M")
        eps_s = pd.Series([3000.0 + i * 20.0 for i in range(48)], index=dates)
        price_s = pd.Series([40000.0 + i * 300.0 for i in range(48)], index=dates)

        res = calc_pe_band("005930", "삼성전자", price_s, eps_s)
        self.assertIsNotNone(res)

        self.assertTrue(res.band_series_pct.index.equals(dates))
        self.assertTrue(res.band_series_sd.index.equals(dates))
        self.assertTrue(res.band_series_fixed.index.equals(dates))
        self.assertIsInstance(res.band_series_pct.index, pd.DatetimeIndex)
        self.assertTrue(res.band_series_pct.index.is_monotonic_increasing)

    def test_monthly_price_vs_daily_eps_ffill_alignment(self):
        """
        Adversarial test for date alignment:
        Price is monthly (36 end-of-month dates), EPS is daily (750 daily dates).
        Their direct intersection is tiny (< 6).
        calc_pe_band must ffill EPS to monthly dates and produce 36 aligned rows.
        """
        monthly_dates = pd.date_range("2021-01-31", periods=36, freq="M")
        daily_dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")

        price_s = pd.Series([50000.0 + i * 500.0 for i in range(len(monthly_dates))], index=monthly_dates)
        # Daily EPS with step changes
        eps_s = pd.Series([4000.0 + (i // 30) * 100.0 for i in range(len(daily_dates))], index=daily_dates)

        res = calc_pe_band("005930", "삼성전자", price_s, eps_s)
        self.assertIsNotNone(res)
        self.assertEqual(len(res.band_series_pct), 36)
        self.assertEqual(len(res.band_series_sd), 36)
        self.assertEqual(len(res.band_series_fixed), 36)
        self.assertTrue(res.band_series_pct.index.equals(monthly_dates))

    def test_finite_non_null_guarantee_on_valid_dates(self):
        """On all dates where EPS > 0, every band column must be finite, non-null, and positive."""
        np.random.seed(42)
        dates = pd.date_range("2019-12-31", periods=60, freq="M")
        eps_vals = np.random.uniform(2000.0, 5000.0, size=60)
        price_vals = eps_vals * np.random.uniform(8.0, 25.0, size=60)
        eps_s = pd.Series(eps_vals, index=dates)
        price_s = pd.Series(price_vals, index=dates)

        res = calc_pe_band("005930", "삼성전자", price_s, eps_s, winsorize=True)
        self.assertIsNotNone(res)

        for df_name, df in [
            ("pct", res.band_series_pct),
            ("sd", res.band_series_sd),
            ("fixed", res.band_series_fixed),
        ]:
            self.assertFalse(df.isna().any().any(), f"{df_name} contains unexpected NaN")
            self.assertTrue(np.isfinite(df.values).all(), f"{df_name} contains inf or -inf")
            self.assertTrue((df.values > 0).all(), f"{df_name} contains non-positive values")

    def test_custom_fixed_multiples_formatting(self):
        """Verify column names for fractional, integer, and unsorted fixed multiples."""
        dates = pd.date_range("2021-01-31", periods=24, freq="M")
        eps_s = pd.Series([3000.0] * 24, index=dates)
        price_s = pd.Series([30000.0] * 24, index=dates)

        custom_m = [6.5, 8.0, 11.25, 15.0]
        res = calc_pe_band(
            "005930", "삼성전자", price_s, eps_s,
            band_model="fixed",
            fixed_multiples=custom_m,
        )
        self.assertIsNotNone(res)
        expected_cols = ["6.5x", "8x", "11.25x", "15x"]
        self.assertEqual(list(res.band_series_fixed.columns), expected_cols)
        self.assertAlmostEqual(res.band_series_fixed.loc[dates[-1], "6.5x"], 3000.0 * 6.5, places=4)
        self.assertAlmostEqual(res.band_series_fixed.loc[dates[-1], "11.25x"], 3000.0 * 11.25, places=4)

    def test_empty_fixed_multiples_list(self):
        """Empty fixed multiples list produces empty band_series_fixed DataFrame with index preserved."""
        dates = pd.date_range("2021-01-31", periods=24, freq="M")
        eps_s = pd.Series([3000.0] * 24, index=dates)
        price_s = pd.Series([30000.0] * 24, index=dates)

        res = calc_pe_band("005930", "삼성전자", price_s, eps_s, fixed_multiples=[])
        self.assertIsNotNone(res)
        self.assertEqual(len(res.band_series_fixed.columns), 0)
        self.assertTrue(res.band_series_fixed.index.equals(dates))
        self.assertEqual(res.fixed_targets, {})
        self.assertEqual(res.fixed_upsides, {})


class TestDynamicModelSwitchingAdversarial(unittest.TestCase):
    """Adversarial stress-testing of dynamic model switching and Winsorization toggle."""

    def setUp(self):
        self.dates = pd.date_range("2021-01-31", periods=36, freq="M")
        self.eps_s = pd.Series([4000.0 + i * 50.0 for i in range(36)], index=self.dates)
        # Introduce a large outlier spike at month 20
        pe_vals = [10.0 + (i % 6) * 1.5 for i in range(36)]
        pe_vals[20] = 145.0  # extreme spike
        self.price_s = pd.Series([self.eps_s.iloc[i] * pe_vals[i] for i in range(36)], index=self.dates)
        self.cur_eps = float(self.eps_s.iloc[-1])
        self.cur_price = float(self.price_s.iloc[-1])

    def test_percentile_model_switching(self):
        """Test band_model='percentile' targets and upsides align with p25, median, p75."""
        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.price_s,
            eps_series=self.eps_s,
            band_model="percentile",
            winsorize=True,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.band_model, "percentile")

        # Bear = 25th percentile, Base = median, Bull = 75th percentile
        self.assertAlmostEqual(res.target_bear, self.cur_eps * res.pe_p25, places=4)
        self.assertAlmostEqual(res.target_base, self.cur_eps * res.pe_median, places=4)
        self.assertAlmostEqual(res.target_bull, self.cur_eps * res.pe_p75, places=4)

        expected_up_base = ((res.target_base / self.cur_price) - 1.0) * 100.0
        self.assertAlmostEqual(res.upside_base, expected_up_base, places=4)

    def test_mean_sd_model_switching(self):
        """Test band_model='mean_sd' targets and upsides align with m1sd, mean, p1sd."""
        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=self.price_s,
            eps_series=self.eps_s,
            band_model="mean_sd",
            winsorize=True,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.band_model, "mean_sd")

        # Bear = m1sd, Base = mean, Bull = p1sd
        m = res.mean_sd_metrics
        self.assertAlmostEqual(res.target_bear, self.cur_eps * m["m1sd"], places=4)
        self.assertAlmostEqual(res.target_base, self.cur_eps * m["mean"], places=4)
        self.assertAlmostEqual(res.target_bull, self.cur_eps * m["p1sd"], places=4)

        expected_up_bull = ((res.target_bull / self.cur_price) - 1.0) * 100.0
        self.assertAlmostEqual(res.upside_bull, expected_up_bull, places=4)

    def test_fixed_model_scenario_targets(self):
        """Test fixed multiple scenario target derivation across varied constituent counts."""
        # 1. 4 multiples: [8.0, 10.0, 12.0, 15.0] -> bear=8, base=10, bull=15
        r4 = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="fixed", fixed_multiples=[8.0, 10.0, 12.0, 15.0])
        self.assertAlmostEqual(r4.target_bear, self.cur_eps * 8.0, places=4)
        self.assertAlmostEqual(r4.target_base, self.cur_eps * 10.0, places=4)
        self.assertAlmostEqual(r4.target_bull, self.cur_eps * 15.0, places=4)

        # 2. 5 multiples: [6.0, 8.0, 10.0, 12.0, 15.0] -> len=5, base=index 2 (10.0)
        r5 = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="fixed", fixed_multiples=[6.0, 8.0, 10.0, 12.0, 15.0])
        self.assertAlmostEqual(r5.target_bear, self.cur_eps * 6.0, places=4)
        self.assertAlmostEqual(r5.target_base, self.cur_eps * 10.0, places=4)
        self.assertAlmostEqual(r5.target_bull, self.cur_eps * 15.0, places=4)

        # 3. 2 multiples: [10.0, 20.0] -> bear=10, base=15.0, bull=20
        r2 = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="fixed", fixed_multiples=[10.0, 20.0])
        self.assertAlmostEqual(r2.target_bear, self.cur_eps * 10.0, places=4)
        self.assertAlmostEqual(r2.target_base, self.cur_eps * 15.0, places=4)
        self.assertAlmostEqual(r2.target_bull, self.cur_eps * 20.0, places=4)

        # 4. 1 multiple: [12.0] -> bear=base=bull=12.0
        r1 = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="fixed", fixed_multiples=[12.0])
        self.assertAlmostEqual(r1.target_bear, self.cur_eps * 12.0, places=4)
        self.assertAlmostEqual(r1.target_base, self.cur_eps * 12.0, places=4)
        self.assertAlmostEqual(r1.target_bull, self.cur_eps * 12.0, places=4)

        # 5. Unsorted multiples [15.0, 8.0, 12.0, 10.0] must sort correctly
        runs = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="fixed", fixed_multiples=[15.0, 8.0, 12.0, 10.0])
        self.assertAlmostEqual(runs.target_bear, self.cur_eps * 8.0, places=4)
        self.assertAlmostEqual(runs.target_base, self.cur_eps * 10.0, places=4)
        self.assertAlmostEqual(runs.target_bull, self.cur_eps * 15.0, places=4)

    def test_winsorize_toggle_impact(self):
        """Toggling winsorize=True vs False effectively shields Mean ± SD bands from extreme distortion."""
        res_win = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="mean_sd", winsorize=True)
        res_raw = calc_pe_band("005930", "삼성전자", self.price_s, self.eps_s, band_model="mean_sd", winsorize=False)

        # Winsorized std and mean must be strictly lower due to capping the 145.0 spike
        self.assertLess(res_win.mean_sd_metrics["std"], res_raw.mean_sd_metrics["std"])
        self.assertLess(res_win.mean_sd_metrics["mean"], res_raw.mean_sd_metrics["mean"])
        self.assertLess(res_win.pe_max, res_raw.pe_max)

        # Band values under +2SD must be lower in winsorized mode (less distorted upward)
        self.assertLess(res_win.band_series_sd.loc[self.dates[-1], "+2SD"], res_raw.band_series_sd.loc[self.dates[-1], "+2SD"])

    def test_zero_current_price_guard(self):
        """Current price == 0.0 must yield 0.0 upsides without throwing ZeroDivisionError."""
        zero_price_s = self.price_s.copy()
        zero_price_s.iloc[-1] = 0.0

        res = calc_pe_band("005930", "삼성전자", zero_price_s, self.eps_s)
        self.assertIsNotNone(res)
        self.assertEqual(res.upside_bear, 0.0)
        self.assertEqual(res.upside_base, 0.0)
        self.assertEqual(res.upside_bull, 0.0)

    def test_legacy_positional_sector_signature_backward_compat(self):
        """6th positional argument being a sector string preserves sector and defaults band_model."""
        res = calc_pe_band(
            "005930",
            "삼성전자",
            self.price_s,
            self.eps_s,
            5,
            "반도체",  # legacy 6th positional arg was sector in M2
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.sector, "반도체")
        self.assertEqual(res.band_model, "percentile")


class TestM2PolishRegressionsAdversarial(unittest.TestCase):
    """Adversarial testing of M2 polish items: SectorMappingDict and _calc_sector_relative_df."""

    def test_sector_mapping_dict_contains_adversarial(self):
        """
        Adversarial checks for SectorMappingDict.__contains__:
        Tests clean, A-prefixed, a-prefixed, whitespace-padded, integer, and non-existent keys.
        """
        sec_map = load_sector_mapping()
        self.assertIsInstance(sec_map, SectorMappingDict)

        # Valid stocks in universe
        clean_codes = ["005930", "000660", "035420", "051910"]
        for c in clean_codes:
            # 1. Clean 6-digit
            self.assertIn(c, sec_map)
            # 2. A-prefixed
            self.assertIn(f"A{c}", sec_map)
            # 3. a-prefixed
            self.assertIn(f"a{c}", sec_map)
            # 4. Whitespace padded
            self.assertIn(f"  {c}  ", sec_map)
            self.assertIn(f"\tA{c}\n", sec_map)
            # 5. Integer code
            self.assertIn(int(c), sec_map)

            # Verification of dictionary access parity
            self.assertEqual(sec_map[c], sec_map[f"A{c}"])
            self.assertEqual(sec_map.get(c), sec_map.get(f"A{c}"))

        # Non-existent / invalid tickers must evaluate to False
        invalids = [
            "999999",
            "A999999",
            "UNKNOWN",
            "",
            "   ",
            None,
            float("nan"),
            np.nan,
            "XYZ123",
            -1,
        ]
        for inv in invalids:
            self.assertNotIn(inv, sec_map, f"Invalid key {inv!r} must NOT be in SectorMappingDict")
            self.assertEqual(sec_map.get(inv, "DEFAULT"), "DEFAULT")

    def test_calc_sector_relative_df_empty_and_degenerate_shapes(self):
        """Empty and columnless DataFrames must return copies without IndexError or crash."""
        sec_map = load_sector_mapping()

        # 1. Completely empty DataFrame
        df_empty = pd.DataFrame()
        res_empty = _calc_sector_relative_df(df_empty, sec_map)
        self.assertTrue(res_empty.empty)

        # 2. Columnless DataFrame with index
        df_col_empty = pd.DataFrame(index=[0, 1, 2, 3])
        res_col_empty = _calc_sector_relative_df(df_col_empty, sec_map)
        self.assertEqual(len(res_col_empty.columns), 0)
        self.assertEqual(len(res_col_empty), 4)

        # 3. DataFrame with columns but 0 rows (df.empty is True -> returns df.copy())
        df_zero_rows = pd.DataFrame(columns=["ticker", "current_fwd_pe"])
        res_zero_rows = _calc_sector_relative_df(df_zero_rows, sec_map)
        self.assertEqual(len(res_zero_rows), 0)
        self.assertTrue(res_zero_rows.empty)
        self.assertEqual(list(res_zero_rows.columns), ["ticker", "current_fwd_pe"])

    def test_calc_sector_relative_df_single_row_ticker_column(self):
        """Single-row DataFrame with 'ticker' column."""
        sec_map = load_sector_mapping()
        df = pd.DataFrame([{"ticker": "005930", "current_fwd_pe": 12.0}])
        res = _calc_sector_relative_df(df, sec_map)

        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["sector"], "반도체")
        self.assertAlmostEqual(res.iloc[0]["sector_median_pe"], 12.0, places=4)
        self.assertAlmostEqual(res.iloc[0]["sector_relative_pe"], 1.0, places=4)
        self.assertAlmostEqual(res.iloc[0]["sector_pe_percentile"], 50.0, places=4)

    def test_calc_sector_relative_df_single_row_ticker_in_index(self):
        """Single-row DataFrame with ticker in index named 'ticker'."""
        sec_map = load_sector_mapping()
        df = pd.DataFrame({"current_fwd_pe": [15.0]}, index=pd.Index(["000660"], name="ticker"))
        res = _calc_sector_relative_df(df, sec_map)

        self.assertEqual(res.iloc[0]["sector"], "반도체")
        self.assertAlmostEqual(res.iloc[0]["sector_relative_pe"], 1.0, places=4)

    def test_calc_sector_relative_df_single_row_ticker_unnamed_first_column(self):
        """Single-row DataFrame with ticker as first column (not named 'ticker')."""
        sec_map = load_sector_mapping()
        df = pd.DataFrame({"code": ["035420"], "current_fwd_pe": [20.0]})
        res = _calc_sector_relative_df(df, sec_map)

        self.assertEqual(res.iloc[0]["sector"], "IT플랫폼/소프트웨어")
        self.assertAlmostEqual(res.iloc[0]["sector_relative_pe"], 1.0, places=4)

    def test_calc_sector_relative_df_loss_making_single_row(self):
        """Single-row DataFrame with negative or NaN P/E."""
        sec_map = load_sector_mapping()
        df_neg = pd.DataFrame([{"ticker": "005930", "current_fwd_pe": -5.0}])
        res_neg = _calc_sector_relative_df(df_neg, sec_map)

        self.assertEqual(res_neg.iloc[0]["sector"], "반도체")
        self.assertTrue(pd.isna(res_neg.iloc[0]["sector_median_pe"]))
        self.assertTrue(pd.isna(res_neg.iloc[0]["sector_relative_pe"]))
        self.assertTrue(pd.isna(res_neg.iloc[0]["sector_pe_percentile"]))


class TestOracleParityAdversarial(unittest.TestCase):
    """Stress-testing M3 calculator algorithms against reference oracle definitions."""

    def test_winsorize_pe_oracle_parity(self):
        """winsorize_pe must match oracle_winsorize across varied sample sizes and distributions."""
        np.random.seed(123)
        # 1. Standard lognormal distribution with spikes
        samples = np.exp(np.random.normal(2.5, 0.6, size=50))
        samples[0] = 0.5   # below min_pe
        samples[1] = 160.0 # above max_pe
        samples[2] = 145.0 # extreme spike within [1, 150]
        s = pd.Series(samples)

        win_res = winsorize_pe(s)
        ora_res = oracle_winsorize(s)
        self.assertTrue(np.allclose(win_res.values, ora_res.values))

        # 2. Short series (< 10 valid)
        short_s = pd.Series([10.0, 15.0, 20.0])
        self.assertTrue(np.allclose(winsorize_pe(short_s).values, oracle_winsorize(short_s).values))

    def test_calc_mean_sd_bands_oracle_parity(self):
        """calc_mean_sd_bands must achieve 100% parity with oracle_mean_sd_bands."""
        pe_s = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        bands = calc_mean_sd_bands(pe_s, floor_pe=1.0, winsorize=False)
        ora_bands = oracle_mean_sd_bands(pe_s, floor_pe=1.0, winsorize=False)

        for k in ["mean", "std", "p1sd", "p2sd", "m1sd", "m2sd"]:
            self.assertAlmostEqual(bands[k], ora_bands[k], places=5)

    def test_calc_fixed_multiple_bands_oracle_parity(self):
        """calc_fixed_multiple_bands must match oracle_fixed_multiple_bands exactly."""
        multiples = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
        fm = calc_fixed_multiple_bands(multiples, current_eps=4500.0, current_price=54000.0)
        ora_fm = oracle_fixed_multiple_bands(multiples, current_eps=4500.0, current_price=54000.0)

        self.assertEqual(fm, ora_fm)


if __name__ == "__main__":
    unittest.main()
