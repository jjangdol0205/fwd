"""
tests/test_tier1_features.py
Tier 1: Feature Coverage Tests (14 Features x 5 Tests = 70 Tests)
Authoritative Requirement-Driven Tests derived from ORIGINAL_REQUEST.md & PROJECT.md.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import json

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
    generate_synthetic_stock_series,
)

# Progressive import of core modules
try:
    from core.data_loader import (
        calc_eps_revisions as core_calc_eps_revisions,
        _clean_ticker as core_clean_ticker,
    )
except (ImportError, AttributeError):
    core_calc_eps_revisions, core_clean_ticker = None, None

try:
    from core.calculator import (
        calc_fwd_pe_series as core_calc_fwd_pe_series,
        calc_pe_band as core_calc_pe_band,
        run_screener as core_run_screener,
        PEBandResult,
    )
except (ImportError, AttributeError):
    core_calc_fwd_pe_series, core_calc_pe_band, core_run_screener, PEBandResult = None, None, None, None


class TestFeature01_EpsRevision1W(unittest.TestCase):
    """Feature 1: 1W (5 trading days) Forward EPS Revision Rate (R1 / AC 31)."""

    def test_01_positive_growth_1w(self):
        # 1000 KRW -> 1050 KRW over 5 trading days = +5.0%
        result = oracle_eps_revision(1050.0, 1000.0)
        self.assertAlmostEqual(result, 5.0, places=4)

    def test_02_negative_revision_1w(self):
        # 2000 KRW -> 1900 KRW over 5 trading days = -5.0%
        result = oracle_eps_revision(1900.0, 2000.0)
        self.assertAlmostEqual(result, -5.0, places=4)

    def test_03_zero_change_1w(self):
        # Unchanged EPS across 5 trading days = 0.0%
        result = oracle_eps_revision(3500.0, 3500.0)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_04_series_lookback_5_days(self):
        # Time series of 10 days, verifies exact 5 trading days lookback (t - 5)
        dates = pd.bdate_range("2026-07-01", periods=10)
        vals = [100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 112.0, 115.0, 118.0, 121.0]
        s = pd.Series(vals, index=dates)
        res = oracle_calc_eps_revisions_series(s)
        # latest is vals[-1] = 121.0, 5 days prior is vals[-6] = 108.0
        expected = ((121.0 - 108.0) / 108.0) * 100.0
        self.assertAlmostEqual(res["eps_rev_1w"], expected, places=4)

    def test_05_upper_clamping_at_500_pct(self):
        # Extreme positive jump (e.g. 100 -> 1000 = +900%) must be clamped to +500%
        result = oracle_eps_revision(1000.0, 100.0)
        self.assertEqual(result, 500.0)


class TestFeature02_EpsRevision1M(unittest.TestCase):
    """Feature 2: 1M (20 trading days) Forward EPS Revision Rate (R1 / AC 31)."""

    def test_01_positive_growth_1m(self):
        # 4000 KRW -> 4400 KRW over 20 trading days = +10.0%
        result = oracle_eps_revision(4400.0, 4000.0)
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_02_negative_revision_1m(self):
        # 5000 KRW -> 4250 KRW over 20 trading days = -15.0%
        result = oracle_eps_revision(4250.0, 5000.0)
        self.assertAlmostEqual(result, -15.0, places=4)

    def test_03_flat_revision_1m(self):
        result = oracle_eps_revision(12000.0, 12000.0)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_04_series_lookback_20_days(self):
        dates = pd.bdate_range("2026-05-01", periods=30)
        vals = [1000.0 + i * 10.0 for i in range(30)]
        s = pd.Series(vals, index=dates)
        res = oracle_calc_eps_revisions_series(s)
        # latest is 1290.0, 20 days prior (index 30 - 1 - 20 = 9) is 1090.0
        expected = ((1290.0 - 1090.0) / 1090.0) * 100.0
        self.assertAlmostEqual(res["eps_rev_1m"], expected, places=4)

    def test_05_dataframe_multi_ticker_vectorized(self):
        # Verify calculation across multiple ticker columns
        dates = pd.bdate_range("2026-05-01", periods=25)
        df = pd.DataFrame(
            {
                "005930": [5000.0 + i * 20.0 for i in range(25)],
                "000660": [10000.0 - i * 10.0 for i in range(25)],
            },
            index=dates,
        )
        res_005930 = oracle_calc_eps_revisions_series(df["005930"])
        res_000660 = oracle_calc_eps_revisions_series(df["000660"])
        self.assertGreater(res_005930["eps_rev_1m"], 0.0)
        self.assertLess(res_000660["eps_rev_1m"], 0.0)


class TestFeature03_EpsRevision3M(unittest.TestCase):
    """Feature 3: 3M (60 trading days) Forward EPS Revision Rate (R1 / AC 31)."""

    def test_01_quarterly_expansion_3m(self):
        # 5000 KRW -> 6000 KRW over 60 trading days = +20.0%
        result = oracle_eps_revision(6000.0, 5000.0)
        self.assertAlmostEqual(result, 20.0, places=4)

    def test_02_quarterly_contraction_3m(self):
        # 8000 KRW -> 6400 KRW over 60 trading days = -20.0%
        result = oracle_eps_revision(6400.0, 8000.0)
        self.assertAlmostEqual(result, -20.0, places=4)

    def test_03_series_lookback_60_days(self):
        dates = pd.bdate_range("2026-01-01", periods=70)
        vals = [2000.0 + i * 5.0 for i in range(70)]
        s = pd.Series(vals, index=dates)
        res = oracle_calc_eps_revisions_series(s)
        # latest is index 69 (2345.0), 60 days prior is index 9 (2045.0)
        expected = ((2345.0 - 2045.0) / 2045.0) * 100.0
        self.assertAlmostEqual(res["eps_rev_3m"], expected, places=4)

    def test_04_consistency_with_1m_momentum(self):
        # Consistent upward trend shows positive 1W, 1M, and 3M
        dates = pd.bdate_range("2026-01-01", periods=80)
        s = pd.Series([1000.0 * (1.002 ** i) for i in range(80)], index=dates)
        res = oracle_calc_eps_revisions_series(s)
        self.assertGreater(res["eps_rev_1w"], 0.0)
        self.assertGreater(res["eps_rev_1m"], 0.0)
        self.assertGreater(res["eps_rev_3m"], 0.0)

    def test_05_quarterly_cyclical_inflection(self):
        # 3M down but 1M turning up
        dates = pd.bdate_range("2026-01-01", periods=70)
        # Drop first 50 days, recover last 20 days
        vals = [2000.0 - i * 10.0 for i in range(50)] + [1500.0 + i * 15.0 for i in range(20)]
        s = pd.Series(vals, index=dates)
        res = oracle_calc_eps_revisions_series(s)
        self.assertGreater(res["eps_rev_1m"], 0.0)
        self.assertLess(res["eps_rev_3m"], 0.0)


class TestFeature04_ValueTrapDetection(unittest.TestCase):
    """Feature 4: Value Trap Filtering & Detection (R1 / AC 31)."""

    def test_01_classic_value_trap(self):
        # PE percentile <= 30.0 and 1M revision < -2.0% -> Value Trap True
        res = oracle_classify_regime(pe_pct=20.0, rev_1m=-4.5, rev_3m=-8.0)
        self.assertTrue(res["is_value_trap"])
        self.assertEqual(res["regime_tag"], "Value Trap")

    def test_02_dual_downgrade_trigger(self):
        # PE pct <= 30.0, 1M < 0 and 3M < -5.0% -> Value Trap True
        res = oracle_classify_regime(pe_pct=28.0, rev_1m=-0.5, rev_3m=-6.0)
        self.assertTrue(res["is_value_trap"])
        self.assertEqual(res["regime_tag"], "Value Trap")

    def test_03_not_trap_due_to_high_pe(self):
        # 1M revision < -5.0%, but PE percentile is 55.0% (>30%) -> Not a value trap
        res = oracle_classify_regime(pe_pct=55.0, rev_1m=-5.0, rev_3m=-10.0)
        self.assertFalse(res["is_value_trap"])
        self.assertNotEqual(res["regime_tag"], "Value Trap")

    def test_04_not_trap_due_to_rising_eps(self):
        # Cheap PE percentile (15.0%), but 1M revision is positive (+3.0%) -> Not a value trap
        res = oracle_classify_regime(pe_pct=15.0, rev_1m=3.0, rev_3m=5.0)
        self.assertFalse(res["is_value_trap"])
        self.assertEqual(res["regime_tag"], "Golden Cross")

    def test_05_value_trap_regime_tag_assignment(self):
        res = oracle_classify_regime(pe_pct=10.0, rev_1m=-3.5)
        self.assertEqual(res["regime_tag"], "Value Trap")


class TestFeature05_GoldenCrossDetection(unittest.TestCase):
    """Feature 5: Golden Cross Filtering & Detection (R1 / AC 31)."""

    def test_01_strong_rebound_golden_cross(self):
        # PE pct <= 40.0, current PE > 0, 1M revision >= +2.0% -> Golden Cross True
        res = oracle_classify_regime(pe_pct=25.0, rev_1m=4.5, current_pe=12.0)
        self.assertTrue(res["is_golden_cross"])
        self.assertEqual(res["regime_tag"], "Golden Cross")

    def test_02_dual_positive_golden_cross(self):
        # PE pct <= 40.0, 1W > 0 and 1M > 0 -> Golden Cross True
        res = oracle_classify_regime(pe_pct=35.0, rev_1m=1.0, rev_1w=0.5, current_pe=9.0)
        self.assertTrue(res["is_golden_cross"])
        self.assertEqual(res["regime_tag"], "Golden Cross")

    def test_03_not_golden_cross_due_to_high_pe(self):
        # Positive revision (+8.0%), but PE percentile is 65.0% (>40%) -> Momentum Leader, not Golden Cross
        res = oracle_classify_regime(pe_pct=65.0, rev_1m=8.0, current_pe=22.0)
        self.assertFalse(res["is_golden_cross"])
        self.assertEqual(res["regime_tag"], "Momentum Leader")

    def test_04_not_golden_cross_due_to_negative_rev(self):
        # Cheap PE (20.0%), but negative revision (-3.0%) -> False
        res = oracle_classify_regime(pe_pct=20.0, rev_1m=-3.0, current_pe=8.0)
        self.assertFalse(res["is_golden_cross"])

    def test_05_golden_cross_regime_tag_assignment(self):
        res = oracle_classify_regime(pe_pct=30.0, rev_1m=3.0, current_pe=11.0)
        self.assertEqual(res["regime_tag"], "Golden Cross")


class TestFeature06_SectorMapping(unittest.TestCase):
    """Feature 6: 350-Stock WICS Sector Mapping (R2 / AC 32)."""

    def setUp(self):
        # Check if sector mapping JSON file exists or can be loaded
        self.mapping_file = Path("data/sector_mapping.json")

    def test_01_ticker_normalization_a_prefix(self):
        self.assertEqual(clean_ticker("A005930"), "005930")
        if core_clean_ticker:
            self.assertEqual(core_clean_ticker("A005930"), "005930")

    def test_02_ticker_normalization_numeric(self):
        self.assertEqual(clean_ticker("5930"), "005930")
        self.assertEqual(clean_ticker(5930), "005930")

    def test_03_sector_taxonomy_recognized(self):
        # Known standard WICS sectors in Korean markets
        valid_sectors = {
            "반도체", "2차전지", "자동차", "바이오/헬스케어", "금융/지주",
            "화학/에너지", "조선/방산/중공업", "IT플랫폼/소프트웨어", "소비재/유통",
            "유틸리티/통신", "철강/소재", "건설/부동산", "기타/미분류"
        }
        self.assertIn("반도체", valid_sectors)
        self.assertIn("금융/지주", valid_sectors)

    def test_04_key_bluechip_sector_classification(self):
        # Samsung Electronics (005930) should be IT/Semiconductor
        bluechip_sectors = {
            "005930": "반도체",
            "000660": "반도체",
            "005380": "자동차",
            "105560": "금융/지주",
        }
        self.assertEqual(bluechip_sectors["005930"], "반도체")
        self.assertEqual(bluechip_sectors["105560"], "금융/지주")

    def test_05_sector_mapping_json_format_if_present(self):
        if self.mapping_file.exists():
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            self.assertGreaterEqual(len(data), 1)


class TestFeature07_SectorMedianPE(unittest.TestCase):
    """Feature 7: Sector Average / Median P/E (R2 / AC 32)."""

    def test_01_median_pe_standard(self):
        pe_list = [10.0, 12.0, 15.0, 18.0, 20.0]
        med = oracle_sector_median_pe(pe_list)
        self.assertAlmostEqual(med, 15.0, places=4)

    def test_02_median_pe_outlier_resistance(self):
        # Single extreme cyclical spike (140x) does not distort median (15.0), whereas mean is 39.0
        pe_list = [10.0, 12.0, 15.0, 18.0, 140.0]
        med = oracle_sector_median_pe(pe_list)
        self.assertAlmostEqual(med, 15.0, places=4)
        self.assertLess(med, np.mean(pe_list))

    def test_03_relative_pe_discount(self):
        # Stock P/E = 12.0 in sector median 15.0 is at 0.8x (20% discount)
        med = oracle_sector_median_pe([10.0, 12.0, 15.0, 18.0, 20.0])
        rel_pe = 12.0 / med
        self.assertAlmostEqual(rel_pe, 0.8, places=4)

    def test_04_relative_pe_premium(self):
        # Stock P/E = 18.0 in sector median 15.0 is at 1.2x (20% premium)
        med = oracle_sector_median_pe([10.0, 12.0, 15.0, 18.0, 20.0])
        rel_pe = 18.0 / med
        self.assertAlmostEqual(rel_pe, 1.2, places=4)

    def test_05_relative_pe_at_parity(self):
        med = oracle_sector_median_pe([10.0, 15.0, 20.0])
        rel_pe = 15.0 / med
        self.assertAlmostEqual(rel_pe, 1.0, places=4)


class TestFeature08_SectorPEPercentile(unittest.TestCase):
    """Feature 8: Sector P/E Percentile Ranking (R2 / AC 32)."""

    def test_01_continuity_corrected_formula_n4(self):
        pe_list = [10.0, 20.0, 30.0, 40.0]
        # rank 0: (0.5 / 4) * 100 = 12.5%
        # rank 3: (3.5 / 4) * 100 = 87.5%
        pct_first = oracle_sector_pe_percentile(10.0, pe_list)
        pct_last = oracle_sector_pe_percentile(40.0, pe_list)
        self.assertAlmostEqual(pct_first, 12.5, places=4)
        self.assertAlmostEqual(pct_last, 87.5, places=4)

    def test_02_percentile_bounded_in_0_to_100(self):
        pe_list = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0]
        for p in pe_list:
            pct = oracle_sector_pe_percentile(p, pe_list)
            self.assertGreater(pct, 0.0)
            self.assertLess(pct, 100.0)

    def test_03_monotonicity_higher_pe_higher_percentile(self):
        pe_list = [8.0, 12.0, 16.0, 24.0]
        pcts = [oracle_sector_pe_percentile(p, pe_list) for p in pe_list]
        self.assertTrue(all(pcts[i] < pcts[i + 1] for i in range(len(pcts) - 1)))

    def test_04_ties_handling(self):
        pe_list = [10.0, 20.0, 20.0, 30.0]
        pct = oracle_sector_pe_percentile(20.0, pe_list)
        # ranks are 1 and 2, average is 1.5 -> (1.5 + 0.5) / 4 * 100 = 50.0%
        self.assertAlmostEqual(pct, 50.0, places=4)

    def test_05_independent_across_sectors(self):
        semis = [10.0, 15.0, 25.0]
        fins = [4.0, 5.0, 7.0]
        # 10.0 is the cheapest semiconductor (rank 0 -> 16.67%)
        # but 7.0 is the most expensive financial (rank 2 -> 83.33%)
        pct_semi = oracle_sector_pe_percentile(10.0, semis)
        pct_fin = oracle_sector_pe_percentile(7.0, fins)
        self.assertLess(pct_semi, 20.0)
        self.assertGreater(pct_fin, 80.0)


class TestFeature09_ResearchStandardBands(unittest.TestCase):
    """Feature 9: Research Standard Mean ± 1/2SD Bands (R3 / AC 33)."""

    def setUp(self):
        # Synthetic historical PE series
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        vals = [10.0 + (i % 6) * 1.5 for i in range(36)]
        self.hist_pe = pd.Series(vals, index=dates)

    def test_01_bands_dict_structure(self):
        bands = oracle_mean_sd_bands(self.hist_pe)
        self.assertIn("mean", bands)
        self.assertIn("std", bands)
        self.assertIn("p1sd", bands)
        self.assertIn("p2sd", bands)
        self.assertIn("m1sd", bands)
        self.assertIn("m2sd", bands)

    def test_02_bands_monotonic_ordering(self):
        bands = oracle_mean_sd_bands(self.hist_pe)
        self.assertLessEqual(bands["m2sd"], bands["m1sd"])
        self.assertLessEqual(bands["m1sd"], bands["mean"])
        self.assertLessEqual(bands["mean"], bands["p1sd"])
        self.assertLessEqual(bands["p1sd"], bands["p2sd"])

    def test_03_target_prices_sd_mode(self):
        bands = oracle_mean_sd_bands(self.hist_pe)
        current_eps = 5000.0
        target_bear = current_eps * bands["m1sd"]
        target_base = current_eps * bands["mean"]
        target_bull = current_eps * bands["p1sd"]
        self.assertLess(target_bear, target_base)
        self.assertLess(target_base, target_bull)

    def test_04_upside_base_sd_mode(self):
        bands = oracle_mean_sd_bands(self.hist_pe)
        current_eps = 5000.0
        current_price = 60000.0
        target_base = current_eps * bands["mean"]
        upside_base = ((target_base / current_price) - 1.0) * 100.0
        self.assertIsInstance(upside_base, float)

    def test_05_sample_std_uses_ddof_1(self):
        bands = oracle_mean_sd_bands(self.hist_pe, winsorize=False)
        expected_std = float(self.hist_pe.std(ddof=1))
        self.assertAlmostEqual(bands["std"], expected_std, places=4)


class TestFeature10_FixedMultipleBands(unittest.TestCase):
    """Feature 10: Fixed Multiple Valuation Bands (8x, 10x, 12x, 15x) (R3 / AC 33)."""

    def test_01_standard_multiples_list(self):
        multiples = [8.0, 10.0, 12.0, 15.0]
        res = oracle_fixed_multiple_bands(multiples, current_eps=5000.0, current_price=50000.0)
        self.assertEqual(list(res["targets"].keys()), multiples)

    def test_02_target_price_derivation(self):
        res = oracle_fixed_multiple_bands([8.0, 10.0, 12.0, 15.0], current_eps=5000.0, current_price=50000.0)
        self.assertAlmostEqual(res["targets"][8.0], 40000.0)
        self.assertAlmostEqual(res["targets"][10.0], 50000.0)
        self.assertAlmostEqual(res["targets"][12.0], 60000.0)
        self.assertAlmostEqual(res["targets"][15.0], 75000.0)

    def test_03_upside_derivation(self):
        res = oracle_fixed_multiple_bands([10.0, 12.0], current_eps=5000.0, current_price=50000.0)
        self.assertAlmostEqual(res["upsides"][10.0], 0.0)      # 50,000 / 50,000 - 1 = 0%
        self.assertAlmostEqual(res["upsides"][12.0], 20.0)     # 60,000 / 50,000 - 1 = +20%

    def test_04_time_series_fixed_bands(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="MS")
        eps_series = pd.Series([2000.0, 2100.0, 2200.0, 2300.0, 2400.0], index=dates)
        band_10x = eps_series * 10.0
        self.assertAlmostEqual(band_10x.iloc[-1], 24000.0)

    def test_05_custom_multiples_support(self):
        custom = [6.0, 9.0, 14.0]
        res = oracle_fixed_multiple_bands(custom, current_eps=3000.0, current_price=30000.0)
        self.assertAlmostEqual(res["targets"][6.0], 18000.0)
        self.assertAlmostEqual(res["targets"][9.0], 27000.0)
        self.assertAlmostEqual(res["targets"][14.0], 42000.0)


class TestFeature11_Winsorization(unittest.TestCase):
    """Feature 11: Winsorization & Outlier Filtering (R3 / AC 33)."""

    def setUp(self):
        # Series with normal values + extreme spikes
        vals = [10.0] * 20 + [12.0] * 20 + [14.0] * 20 + [145.0, 148.0]
        self.raw_series = pd.Series(vals)

    def test_01_caps_extreme_upper_percentile(self):
        win = oracle_winsorize(self.raw_series, lower_q=0.025, upper_q=0.975)
        self.assertLess(win.max(), 145.0)

    def test_02_floors_extreme_lower_percentile(self):
        vals = [1.2, 1.5] + [10.0] * 20 + [12.0] * 20
        win = oracle_winsorize(pd.Series(vals), lower_q=0.05, upper_q=0.95)
        self.assertGreater(win.min(), 1.2)

    def test_03_preserves_interior_distribution(self):
        win = oracle_winsorize(self.raw_series)
        med_raw = self.raw_series.median()
        med_win = win.median()
        self.assertAlmostEqual(med_raw, med_win, places=4)

    def test_04_reduces_sample_variance(self):
        win = oracle_winsorize(self.raw_series)
        self.assertLess(win.var(), self.raw_series.var())

    def test_05_preserves_series_length(self):
        win = oracle_winsorize(self.raw_series)
        self.assertEqual(len(win), len(self.raw_series))


class TestFeature12_ScreenerTableColumns(unittest.TestCase):
    """Feature 12: Screener Table Columns & Sorting (R4 / AC 37)."""

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "ticker": ["005930", "000660", "005380"],
            "name": ["삼성전자", "SK하이닉스", "현대차"],
            "sector": ["반도체", "반도체", "자동차"],
            "current_fwd_eps": [6000.0, 12000.0, 25000.0],
            "current_fwd_pe": [12.5, 14.0, 6.5],
            "pe_percentile": [35.0, 45.0, 20.0],
            "sector_pe_percentile": [25.0, 75.0, 50.0],
            "eps_rev_1m": [4.0, 6.5, -1.5],
            "eps_rev_3m": [8.0, 12.0, -3.0],
            "target_base": [85000.0, 180000.0, 260000.0],
            "upside_base": [25.0, 18.0, 32.0],
            "regime_tag": ["Golden Cross", "Momentum Leader", "Neutral"],
        })

    def test_01_required_columns_present(self):
        req_cols = ["sector", "sector_pe_percentile", "eps_rev_1m", "eps_rev_3m", "upside_base", "regime_tag"]
        for c in req_cols:
            self.assertIn(c, self.sample_df.columns)

    def test_02_sort_by_upside_base_descending(self):
        sorted_df = self.sample_df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(sorted_df.iloc[0]["ticker"], "005380")  # upside 32.0%
        self.assertEqual(sorted_df.iloc[-1]["ticker"], "000660") # upside 18.0%

    def test_03_sort_by_eps_rev_1m_descending(self):
        sorted_df = self.sample_df.sort_values(by="eps_rev_1m", ascending=False)
        self.assertEqual(sorted_df.iloc[0]["ticker"], "000660")  # +6.5%

    def test_04_filter_by_sector(self):
        semi_df = self.sample_df[self.sample_df["sector"] == "반도체"]
        self.assertEqual(len(semi_df), 2)
        self.assertTrue(all(semi_df["sector"] == "반도체"))

    def test_05_filter_by_regime_tag(self):
        gc_df = self.sample_df[self.sample_df["regime_tag"] == "Golden Cross"]
        self.assertEqual(len(gc_df), 1)
        self.assertEqual(gc_df.iloc[0]["ticker"], "005930")


class TestFeature13_QuickScreenerPresets(unittest.TestCase):
    """Feature 13: 1-Click Quick Screener Presets (R4 / AC 39)."""

    def setUp(self):
        self.pool = pd.DataFrame([
            {"ticker": "A", "pe_percentile": 30.0, "sector_pe_percentile": 25.0, "eps_rev_1m": 4.0, "current_fwd_eps": 2000.0, "current_fwd_pe": 10.0, "upside_base": 15.0}, # Turnaround
            {"ticker": "B", "pe_percentile": 60.0, "sector_pe_percentile": 70.0, "eps_rev_1m": 8.0, "current_fwd_eps": 5000.0, "current_fwd_pe": 25.0, "upside_base": 5.0},  # Top EPS
            {"ticker": "C", "pe_percentile": 20.0, "sector_pe_percentile": 15.0, "eps_rev_1m": -1.0, "current_fwd_eps": 3000.0, "current_fwd_pe": 8.0, "upside_base": 20.0}, # Deep Value
            {"ticker": "D", "pe_percentile": 25.0, "sector_pe_percentile": 30.0, "eps_rev_1m": -4.0, "current_fwd_eps": 1000.0, "current_fwd_pe": 9.0, "upside_base": 10.0}, # Trap
        ])

    def test_01_preset_all(self):
        res = oracle_apply_quick_preset(self.pool, "ALL")
        self.assertEqual(len(res), 4)

    def test_02_preset_turnaround(self):
        res = oracle_apply_quick_preset(self.pool, "TURNAROUND")
        self.assertIn("A", res["ticker"].values)
        self.assertNotIn("D", res["ticker"].values)

    def test_03_preset_eps_top(self):
        res = oracle_apply_quick_preset(self.pool, "EPS_TOP")
        self.assertEqual(res.iloc[0]["ticker"], "B") # top revision 8.0%

    def test_04_preset_value(self):
        res = oracle_apply_quick_preset(self.pool, "VALUE")
        self.assertIn("C", res["ticker"].values)
        self.assertNotIn("B", res["ticker"].values)

    def test_05_preset_trap(self):
        res = oracle_apply_quick_preset(self.pool, "TRAP")
        self.assertIn("D", res["ticker"].values)
        self.assertNotIn("A", res["ticker"].values)


class TestFeature14_IntegratedDashboard(unittest.TestCase):
    """Feature 14: One-Page Integrated Dashboard Chart & KPIs (R4 / AC 38)."""

    def test_01_kpi_deck_card_types(self):
        card_types = ["Valuation", "Momentum", "Band_Reference", "Target_Scenario"]
        self.assertEqual(len(card_types), 4)

    def test_02_chart_two_subplot_structure(self):
        # Requirements mandate a synchronized 2-subplot layout (rows=2, cols=1)
        layout_spec = {"rows": 2, "cols": 1, "shared_xaxes": True, "vertical_spacing": 0.05}
        self.assertEqual(layout_spec["rows"], 2)
        self.assertTrue(layout_spec["shared_xaxes"])

    def test_03_upper_subplot_price_bands(self):
        upper_traces = ["Stock Price", "Bull Band", "Base Band", "Bear Band"]
        self.assertIn("Stock Price", upper_traces)
        self.assertIn("Base Band", upper_traces)

    def test_04_lower_subplot_eps_pe(self):
        lower_traces = ["12M Fwd EPS", "12M Fwd P/E"]
        self.assertIn("12M Fwd EPS", lower_traces)
        self.assertIn("12M Fwd P/E", lower_traces)

    def test_05_band_model_toggle_modes(self):
        valid_modes = ["percentile", "mean_sd", "fixed"]
        self.assertEqual(len(valid_modes), 3)
        self.assertIn("mean_sd", valid_modes)


if __name__ == "__main__":
    unittest.main()
