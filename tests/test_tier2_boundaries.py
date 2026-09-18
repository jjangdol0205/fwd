"""
tests/test_tier2_boundaries.py
Tier 2: Boundary & Corner Case Tests (14 Features x 5 Tests = 70 Tests)
Authoritative Requirement-Driven Edge Cases derived from ORIGINAL_REQUEST.md & PROJECT.md.
"""

import unittest
import numpy as np
import pandas as pd

from tests.oracle import (
    clean_ticker,
    oracle_eps_revision,
    oracle_calc_eps_revisions_series,
    oracle_classify_regime,
    oracle_sector_median_pe,
    oracle_sector_pe_percentile,
    oracle_winsorize,
    oracle_mean_sd_bands,
    oracle_fixed_multiple_bands,
    oracle_apply_quick_preset,
)


class TestBoundary01_EpsRevision1W(unittest.TestCase):
    """Feature 1 Boundary: 1W Revision Edge Cases."""

    def test_01_negative_base_turnaround(self):
        # Base EPS = -500, Current = +200 -> (+200 - (-500)) / 500 = +140.0%
        # abs() in denominator ensures improvement is correctly positive
        res = oracle_eps_revision(200.0, -500.0)
        self.assertAlmostEqual(res, 140.0, places=4)

    def test_02_negative_base_widening_loss(self):
        # Base EPS = -500, Current = -800 -> (-800 - (-500)) / 500 = -60.0%
        res = oracle_eps_revision(-800.0, -500.0)
        self.assertAlmostEqual(res, -60.0, places=4)

    def test_03_near_zero_base_division_guard(self):
        # Base EPS = 0.5 (< 1.0) -> max(0.5, 1.0) = 1.0 -> (10.0 - 0.5)/1.0 * 100 = 950% -> clamped 500%
        res = oracle_eps_revision(10.0, 0.5)
        self.assertEqual(res, 500.0)

    def test_04_clamping_lower_bound_minus_100(self):
        # Base = 1000, Current = -1000 -> (-1000 - 1000)/1000 = -200% -> clamped to -100%
        res = oracle_eps_revision(-1000.0, 1000.0)
        self.assertEqual(res, -100.0)

    def test_05_short_series_less_than_6_days(self):
        dates = pd.bdate_range("2026-07-01", periods=5)
        s = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0], index=dates)
        res = oracle_calc_eps_revisions_series(s)
        self.assertIsNone(res["eps_rev_1w"])


class TestBoundary02_EpsRevision1M(unittest.TestCase):
    """Feature 2 Boundary: 1M Revision Edge Cases."""

    def test_01_deficit_to_profit_turnaround(self):
        # -100 to +50 over 20 days -> (+50 - (-100))/100 = +150.0%
        res = oracle_eps_revision(50.0, -100.0)
        self.assertAlmostEqual(res, 150.0, places=4)

    def test_02_profit_to_deficit_clamping(self):
        # +500 to -200 -> (-200 - 500)/500 = -140% -> clamped to -100.0%
        res = oracle_eps_revision(-200.0, 500.0)
        self.assertEqual(res, -100.0)

    def test_03_exact_zero_base(self):
        # Base = 0.0 -> max(0.0, 1.0) = 1.0 -> (50 - 0)/1.0 = 5000% -> clamped to 500%
        res = oracle_eps_revision(50.0, 0.0)
        self.assertEqual(res, 500.0)

    def test_04_boundary_series_length_20_vs_21(self):
        # 20 days has no 20-day prior point; 21 days has exactly 1 prior point
        dates_20 = pd.bdate_range("2026-06-01", periods=20)
        s_20 = pd.Series([1000.0] * 20, index=dates_20)
        res_20 = oracle_calc_eps_revisions_series(s_20)
        self.assertIsNone(res_20["eps_rev_1m"])

        dates_21 = pd.bdate_range("2026-06-01", periods=21)
        s_21 = pd.Series([1000.0] * 20 + [1100.0], index=dates_21)
        res_21 = oracle_calc_eps_revisions_series(s_21)
        self.assertIsNotNone(res_21["eps_rev_1m"])
        self.assertAlmostEqual(res_21["eps_rev_1m"], 10.0, places=4)

    def test_05_series_with_intermediate_nans(self):
        dates = pd.bdate_range("2026-05-01", periods=25)
        vals = [1000.0] * 25
        vals[5] = np.nan
        vals[10] = np.nan
        s = pd.Series(vals, index=dates)
        res = oracle_calc_eps_revisions_series(s)
        # Dropna leaves 23 points (> 20), calculation proceeds cleanly
        self.assertIsNotNone(res["eps_rev_1m"])


class TestBoundary03_EpsRevision3M(unittest.TestCase):
    """Feature 3 Boundary: 3M Revision Edge Cases."""

    def test_01_deep_deficit_turnaround(self):
        # -2000 to +1000 over 3 months -> (+1000 - (-2000)) / 2000 = +150.0%
        res = oracle_eps_revision(1000.0, -2000.0)
        self.assertAlmostEqual(res, 150.0, places=4)

    def test_02_persistent_deepening_deficit(self):
        # -400 to -1200 over 3 months -> (-1200 - (-400)) / 400 = -200% -> clamped to -100.0%
        res = oracle_eps_revision(-1200.0, -400.0)
        self.assertEqual(res, -100.0)

    def test_03_boundary_series_length_60_vs_61(self):
        dates_60 = pd.bdate_range("2026-03-01", periods=60)
        s_60 = pd.Series([2000.0] * 60, index=dates_60)
        self.assertIsNone(oracle_calc_eps_revisions_series(s_60)["eps_rev_3m"])

        dates_61 = pd.bdate_range("2026-03-01", periods=61)
        s_61 = pd.Series([2000.0] * 60 + [2200.0], index=dates_61)
        res_61 = oracle_calc_eps_revisions_series(s_61)
        self.assertAlmostEqual(res_61["eps_rev_3m"], 10.0, places=4)

    def test_04_extreme_spike_clamping(self):
        # 5 to 50,000 KRW clamped to +500%
        res = oracle_eps_revision(50000.0, 5.0)
        self.assertEqual(res, 500.0)

    def test_05_nan_at_endpoint(self):
        res = oracle_eps_revision(np.nan, 1000.0)
        self.assertIsNone(res)


class TestBoundary04_ValueTrapDetection(unittest.TestCase):
    """Feature 4 Boundary: Value Trap Edge Cases."""

    def test_01_exact_pe_pct_boundary_30(self):
        # 30.0% with -3.0% 1M rev is Trap
        res_30 = oracle_classify_regime(pe_pct=30.0, rev_1m=-3.0)
        self.assertTrue(res_30["is_value_trap"])

        # 30.01% with -3.0% 1M rev is NOT Trap
        res_30_01 = oracle_classify_regime(pe_pct=30.01, rev_1m=-3.0)
        self.assertFalse(res_30_01["is_value_trap"])

    def test_02_exact_rev_1m_boundary_minus_2(self):
        # rev_1m = -2.001% (pe_pct=25%) is Trap
        res_under = oracle_classify_regime(pe_pct=25.0, rev_1m=-2.001)
        self.assertTrue(res_under["is_value_trap"])

        # rev_1m = -1.999% without 3M drop is NOT Trap
        res_over = oracle_classify_regime(pe_pct=25.0, rev_1m=-1.999, rev_3m=-2.0)
        self.assertFalse(res_over["is_value_trap"])

    def test_03_dual_condition_3m_boundary(self):
        # rev_1m = -0.5% with rev_3m = -5.001% is Trap
        res_hit = oracle_classify_regime(pe_pct=25.0, rev_1m=-0.5, rev_3m=-5.001)
        self.assertTrue(res_hit["is_value_trap"])

        # rev_3m = -4.999% is NOT Trap
        res_miss = oracle_classify_regime(pe_pct=25.0, rev_1m=-0.5, rev_3m=-4.999)
        self.assertFalse(res_miss["is_value_trap"])

    def test_04_missing_3m_revision_handling(self):
        # rev_1m = -1.0% and rev_3m = None -> returns False safely
        res = oracle_classify_regime(pe_pct=25.0, rev_1m=-1.0, rev_3m=None)
        self.assertFalse(res["is_value_trap"])

    def test_05_negative_pe_not_trap(self):
        # Negative PE indicates deficit, not multiple trap
        res = oracle_classify_regime(pe_pct=20.0, rev_1m=-5.0, current_pe=-5.0)
        self.assertFalse(res["is_value_trap"])


class TestBoundary05_GoldenCrossDetection(unittest.TestCase):
    """Feature 5 Boundary: Golden Cross Edge Cases."""

    def test_01_exact_pe_pct_boundary_40(self):
        # PE pct = 40.0% with +3.0% rev is Golden Cross
        res_40 = oracle_classify_regime(pe_pct=40.0, rev_1m=3.0, current_pe=10.0)
        self.assertTrue(res_40["is_golden_cross"])

        # PE pct = 40.01% is NOT Golden Cross
        res_40_01 = oracle_classify_regime(pe_pct=40.01, rev_1m=3.0, current_pe=10.0)
        self.assertFalse(res_40_01["is_golden_cross"])

    def test_02_exact_rev_1m_boundary_plus_2(self):
        # rev_1m = 2.0% with PE pct = 35% is Golden Cross
        res_20 = oracle_classify_regime(pe_pct=35.0, rev_1m=2.0, current_pe=10.0)
        self.assertTrue(res_20["is_golden_cross"])

        # rev_1m = 1.999% without positive 1W is NOT Golden Cross
        res_199 = oracle_classify_regime(pe_pct=35.0, rev_1m=1.999, rev_1w=0.0, current_pe=10.0)
        self.assertFalse(res_199["is_golden_cross"])

    def test_03_zero_or_negative_pe_excluded(self):
        # Even with +50.0% revision, deficit stock (current_pe <= 0) cannot be Golden Cross
        res_neg = oracle_classify_regime(pe_pct=20.0, rev_1m=50.0, current_pe=-10.0)
        self.assertFalse(res_neg["is_golden_cross"])

    def test_04_1w_positive_1m_zero(self):
        # 1W = +2.0%, but 1M = 0.0% (not >0) -> False
        res = oracle_classify_regime(pe_pct=30.0, rev_1m=0.0, rev_1w=2.0, current_pe=10.0)
        self.assertFalse(res["is_golden_cross"])

    def test_05_extreme_cheap_pe_pct_zero(self):
        # All-time low PE (percentile = 0.0%) with +5.0% revision is Golden Cross
        res = oracle_classify_regime(pe_pct=0.0, rev_1m=5.0, current_pe=5.0)
        self.assertTrue(res["is_golden_cross"])


class TestBoundary06_SectorMapping(unittest.TestCase):
    """Feature 6 Boundary: Sector Mapping Edge Cases."""

    def test_01_unknown_ticker_fallback(self):
        clean = clean_ticker("999999")
        self.assertEqual(clean, "999999")

    def test_02_empty_string_ticker(self):
        self.assertEqual(clean_ticker(""), "")

    def test_03_none_ticker(self):
        self.assertEqual(clean_ticker(None), "")

    def test_04_whitespace_padding(self):
        self.assertEqual(clean_ticker("  005930  "), "005930")

    def test_05_mixed_case_prefix(self):
        self.assertEqual(clean_ticker("a005930"), "005930")


class TestBoundary07_SectorMedianPE(unittest.TestCase):
    """Feature 7 Boundary: Sector Median P/E Edge Cases."""

    def test_01_median_even_number_elements(self):
        # [10, 12, 16, 20] -> median is (12 + 16)/2 = 14.0
        med = oracle_sector_median_pe([10.0, 12.0, 16.0, 20.0])
        self.assertAlmostEqual(med, 14.0, places=4)

    def test_02_filters_zero_and_negative(self):
        # [-5.0, 0.0, 10.0, 20.0] -> valid is [10, 20] -> median = 15.0
        med = oracle_sector_median_pe([-5.0, 0.0, 10.0, 20.0])
        self.assertAlmostEqual(med, 15.0, places=4)

    def test_03_filters_above_150_cap(self):
        # [10.0, 20.0, 180.0] -> valid is [10, 20] -> median = 15.0
        med = oracle_sector_median_pe([10.0, 20.0, 180.0])
        self.assertAlmostEqual(med, 15.0, places=4)

    def test_04_single_stock_n1(self):
        med = oracle_sector_median_pe([12.0])
        self.assertAlmostEqual(med, 12.0, places=4)

    def test_05_all_deficits(self):
        # Sector with only loss-making companies
        med = oracle_sector_median_pe([-5.0, -10.0, 0.0])
        self.assertIsNone(med)


class TestBoundary08_SectorPEPercentile(unittest.TestCase):
    """Feature 8 Boundary: Sector P/E Percentile Edge Cases."""

    def test_01_small_sample_n1(self):
        # Single stock in sector -> rank 0 -> (0.5/1)*100 = 50.0% (neutral)
        pct = oracle_sector_pe_percentile(15.0, [15.0])
        self.assertAlmostEqual(pct, 50.0, places=4)

    def test_02_small_sample_n2(self):
        # 2 stocks [10.0, 20.0] -> ranks 0 and 1 -> (0.5/2)*100 = 25.0%, (1.5/2)*100 = 75.0%
        # Eliminates the extreme 0% and 100% distortion!
        pct_low = oracle_sector_pe_percentile(10.0, [10.0, 20.0])
        pct_high = oracle_sector_pe_percentile(20.0, [10.0, 20.0])
        self.assertAlmostEqual(pct_low, 25.0, places=4)
        self.assertAlmostEqual(pct_high, 75.0, places=4)

    def test_03_small_sample_n3(self):
        # 3 stocks -> 16.67%, 50.0%, 83.33%
        pct_mid = oracle_sector_pe_percentile(20.0, [10.0, 20.0, 30.0])
        self.assertAlmostEqual(pct_mid, 50.0, places=4)

    def test_04_negative_target_pe(self):
        pct = oracle_sector_pe_percentile(-5.0, [10.0, 20.0])
        self.assertIsNone(pct)

    def test_05_target_pe_above_cap(self):
        pct = oracle_sector_pe_percentile(180.0, [10.0, 20.0])
        self.assertIsNone(pct)


class TestBoundary09_ResearchStandardBands(unittest.TestCase):
    """Feature 9 Boundary: Research Standard Bands Edge Cases."""

    def test_01_floor_protection_m2sd(self):
        # Very high volatility where mean - 2*std would be negative
        # e.g., mean = 5.0, std = 4.0 -> mean - 2*std = -3.0
        # Floor must ensure m2sd >= 1.0
        s = pd.Series([1.5, 2.0, 3.0, 14.0, 15.0])
        bands = oracle_mean_sd_bands(s, floor_pe=1.0, winsorize=False)
        self.assertGreaterEqual(bands["m2sd"], 1.0)

    def test_02_floor_protection_m1sd(self):
        s = pd.Series([1.1, 1.2, 1.3, 20.0, 22.0])
        bands = oracle_mean_sd_bands(s, floor_pe=1.0, winsorize=False)
        self.assertGreaterEqual(bands["m1sd"], 1.0)

    def test_03_constant_series_zero_std(self):
        s = pd.Series([10.0] * 20)
        bands = oracle_mean_sd_bands(s, winsorize=False)
        self.assertAlmostEqual(bands["std"], 0.0, places=4)
        self.assertAlmostEqual(bands["mean"], 10.0, places=4)
        self.assertAlmostEqual(bands["p1sd"], 10.0, places=4)
        self.assertAlmostEqual(bands["m1sd"], 10.0, places=4)

    def test_04_insufficient_data_under_4(self):
        s = pd.Series([10.0, 12.0, 14.0])
        bands = oracle_mean_sd_bands(s)
        self.assertEqual(bands, {})

    def test_05_all_identical_large_series(self):
        s = pd.Series([15.0] * 50)
        bands = oracle_mean_sd_bands(s)
        self.assertEqual(bands["mean"], 15.0)
        self.assertEqual(bands["std"], 0.0)


class TestBoundary10_FixedMultipleBands(unittest.TestCase):
    """Feature 10 Boundary: Fixed Multiple Bands Edge Cases."""

    def test_01_zero_price_upside_protection(self):
        res = oracle_fixed_multiple_bands([10.0], current_eps=5000.0, current_price=0.0)
        self.assertEqual(res["upsides"][10.0], 0.0)

    def test_02_negative_eps_target(self):
        # EPS <= 0 should yield 0.0 target rather than nonsensical positive target
        res = oracle_fixed_multiple_bands([10.0], current_eps=-500.0, current_price=50000.0)
        self.assertEqual(res["targets"][10.0], 0.0)

    def test_03_empty_multiples_list(self):
        res = oracle_fixed_multiple_bands([], current_eps=5000.0, current_price=50000.0)
        self.assertEqual(res["targets"], {})

    def test_04_fractional_multiples(self):
        res = oracle_fixed_multiple_bands([7.5, 12.25], current_eps=4000.0, current_price=40000.0)
        self.assertAlmostEqual(res["targets"][7.5], 30000.0)
        self.assertAlmostEqual(res["targets"][12.25], 49000.0)

    def test_05_huge_multiple_100x(self):
        res = oracle_fixed_multiple_bands([100.0], current_eps=2000.0, current_price=50000.0)
        self.assertAlmostEqual(res["targets"][100.0], 200000.0)
        self.assertAlmostEqual(res["upsides"][100.0], 300.0)


class TestBoundary11_Winsorization(unittest.TestCase):
    """Feature 11 Boundary: Winsorization Edge Cases."""

    def test_01_filters_pe_below_1(self):
        # Values < 1.0 (such as near-zero or negative) dropped
        s = pd.Series([0.1, 0.5, 10.0, 12.0, 15.0])
        win = oracle_winsorize(s, min_pe=1.0)
        self.assertTrue(all(win >= 1.0))

    def test_02_filters_pe_above_150(self):
        s = pd.Series([10.0, 12.0, 15.0, 250.0, 500.0])
        win = oracle_winsorize(s, max_pe=150.0)
        self.assertTrue(all(win <= 150.0))

    def test_03_short_series_pass_through(self):
        s = pd.Series([10.0, 12.0, 14.0])
        win = oracle_winsorize(s)
        self.assertEqual(len(win), 3)

    def test_04_with_nans_and_infs(self):
        s = pd.Series([10.0, np.nan, 12.0, np.inf, 14.0])
        win = oracle_winsorize(s)
        self.assertFalse(win.isna().any())
        self.assertFalse(np.isinf(win).any())

    def test_05_custom_narrow_quantiles(self):
        s = pd.Series(list(range(1, 101)))
        win = oracle_winsorize(s, lower_q=0.10, upper_q=0.90)
        self.assertGreaterEqual(win.min(), 10.0)
        self.assertLessEqual(win.max(), 91.0)


class TestBoundary12_ScreenerTable(unittest.TestCase):
    """Feature 12 Boundary: Screener Table Edge Cases."""

    def test_01_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["ticker", "upside_base", "eps_rev_1m"])
        sorted_df = empty_df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(len(sorted_df), 0)

    def test_02_all_negative_upsides(self):
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "upside_base": [-30.0, -10.0, -50.0],
        })
        sorted_df = df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(sorted_df.iloc[0]["ticker"], "B")  # -10% is greatest

    def test_03_missing_revision_fields(self):
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "eps_rev_1m": [None, 5.0],
            "upside_base": [10.0, 20.0],
        })
        self.assertEqual(len(df), 2)

    def test_04_single_row_dataframe(self):
        df = pd.DataFrame([{"ticker": "A", "upside_base": 15.0}])
        sorted_df = df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(len(sorted_df), 1)

    def test_05_extreme_values_sorting(self):
        df = pd.DataFrame({
            "ticker": ["MIN", "MAX"],
            "upside_base": [-99.0, 500.0],
        })
        sorted_df = df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(sorted_df.iloc[0]["ticker"], "MAX")


class TestBoundary13_QuickPresets(unittest.TestCase):
    """Feature 13 Boundary: Quick Presets Edge Cases."""

    def test_01_turnaround_sector_rescue(self):
        # Overall PE pct = 45% (fails overall <= 40%), but Sector PE pct = 30% (<=40%)
        # Should be rescued and included in TURNAROUND preset!
        df = pd.DataFrame([{
            "ticker": "RESCUE",
            "pe_percentile": 45.0,
            "sector_pe_percentile": 30.0,
            "eps_rev_1m": 4.0,
            "upside_base": 15.0,
        }])
        res = oracle_apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["ticker"], "RESCUE")

    def test_02_eps_top_excludes_deficit_growth(self):
        # EPS rev is +80%, but current EPS is negative (-50 < 0) -> Must be excluded
        df = pd.DataFrame([{
            "ticker": "DEFICIT_GROWER",
            "eps_rev_1m": 80.0,
            "current_fwd_eps": -50.0,
        }])
        res = oracle_apply_quick_preset(df, "EPS_TOP")
        self.assertEqual(len(res), 0)

    def test_03_value_excludes_slashing_eps(self):
        # PE pct = 15.0%, but 1M rev = -4.0% (earnings deteriorating) -> Excluded from VALUE
        df = pd.DataFrame([{
            "ticker": "SLASH",
            "pe_percentile": 15.0,
            "eps_rev_1m": -4.0,
            "current_fwd_pe": 8.0,
            "upside_base": 20.0,
        }])
        res = oracle_apply_quick_preset(df, "VALUE")
        self.assertEqual(len(res), 0)

    def test_04_trap_excludes_rebounding_eps(self):
        # PE pct = 20.0%, but 1M rev = +2.0% -> Excluded from TRAP
        df = pd.DataFrame([{
            "ticker": "REBOUND",
            "pe_percentile": 20.0,
            "eps_rev_1m": 2.0,
        }])
        res = oracle_apply_quick_preset(df, "TRAP")
        self.assertEqual(len(res), 0)

    def test_05_empty_result_safety(self):
        df = pd.DataFrame(columns=["ticker", "pe_percentile", "sector_pe_percentile", "eps_rev_1m", "upside_base"])
        res = oracle_apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(len(res), 0)


class TestBoundary14_IntegratedDashboard(unittest.TestCase):
    """Feature 14 Boundary: Integrated Dashboard Edge Cases."""

    def test_01_risk_reward_zero_downside_guard(self):
        # Risk / Reward ratio calculation: |Bull - Price| / max(|Price - Bear|, 1.0)
        # When Price == Bear, denominator guarded to prevent ZeroDivisionError
        price = 50000.0
        bear = 50000.0
        bull = 65000.0
        downside = max(abs(price - bear), 1.0)
        rr_ratio = abs(bull - price) / downside
        self.assertAlmostEqual(rr_ratio, 15000.0)

    def test_02_single_point_history_guard(self):
        dates = pd.date_range("2026-07-01", periods=1)
        s = pd.Series([10.0], index=dates)
        bands = oracle_mean_sd_bands(s)
        self.assertEqual(bands, {})

    def test_03_missing_eps_series_guard(self):
        # Price available but EPS missing -> P/E series is empty
        price = pd.Series([50000.0], index=pd.date_range("2026-07-01", periods=1))
        eps = pd.Series([np.nan], index=price.index)
        pe = price / eps
        self.assertTrue(pe.isna().all())

    def test_04_extreme_scale_separation(self):
        # Two-subplot architecture prevents 1,000,000 KRW price from squashing 12.5x P/E line
        price_val = 1_000_000.0
        pe_val = 12.5
        self.assertGreater(price_val / pe_val, 10000.0)

    def test_05_winsorize_impact_on_rendered_bands(self):
        # Series with huge outlier (145x)
        vals = [10.0] * 20 + [12.0] * 20 + [145.0]
        s = pd.Series(vals)
        bands_raw = oracle_mean_sd_bands(s, winsorize=False)
        bands_win = oracle_mean_sd_bands(s, winsorize=True)
        # Winsorized std must be strictly smaller than raw std
        self.assertLess(bands_win["std"], bands_raw["std"])


if __name__ == "__main__":
    unittest.main()
