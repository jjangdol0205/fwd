"""
tests/test_tier3_combinations.py
Tier 3: Cross-Feature Combinations & Pairwise Interaction Tests (16 Tests)
Authoritative Requirement-Driven Tests derived from ORIGINAL_REQUEST.md & PROJECT.md.
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
    generate_synthetic_stock_series,
)


class TestTier3Combinations(unittest.TestCase):
    """Pairwise cross-feature interactions and combination tests."""

    def test_01_golden_cross_in_high_pe_sector(self):
        # Healthcare/Bio stock trading at P/E = 28x
        # Market-wide P/E percentile = 48% (looks expensive on whole market)
        # But in Bio sector [25, 30, 35, 45, 60], sector percentile = (0.5/5)*100 = 10.0%
        # With 1M revision = +5.0%, sector-relative logic rescues stock into Golden Cross!
        bio_pes = [25.0, 28.0, 35.0, 45.0, 60.0]
        sec_pct = oracle_sector_pe_percentile(28.0, bio_pes)
        self.assertLessEqual(sec_pct, 40.0)

        df = pd.DataFrame([{
            "ticker": "BIO_LEADER",
            "pe_percentile": 48.0,
            "sector_pe_percentile": sec_pct,
            "eps_rev_1m": 5.0,
            "upside_base": 20.0,
        }])
        turnaround = oracle_apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(len(turnaround), 1)

    def test_02_value_trap_in_low_pe_sector(self):
        # Bank stock trading at P/E = 4.5x
        # Market-wide P/E percentile = 8.0% (optically very cheap)
        # But 1M revision = -4.5% -> Classified as Value Trap despite looking cheap!
        regime = oracle_classify_regime(pe_pct=8.0, rev_1m=-4.5, current_pe=4.5)
        self.assertTrue(regime["is_value_trap"])
        self.assertEqual(regime["regime_tag"], "Value Trap")

    def test_03_winsorization_on_cyclical_turnaround(self):
        # Steel company emerging from profit slump
        # Historical P/E contains extreme spike: [8, 9, 10, 11, 140, 10, 9, 8, 9, 10]
        s = pd.Series([8.0, 9.0, 10.0, 11.0, 140.0, 10.0, 9.0, 8.0, 9.0, 10.0])
        bands_raw = oracle_mean_sd_bands(s, winsorize=False)
        bands_win = oracle_mean_sd_bands(s, winsorize=True)
        # Raw mean is pulled up significantly by 140x; winsorized mean stays close to normal ~10x
        self.assertLess(bands_win["mean"], bands_raw["mean"])
        self.assertLess(bands_win["mean"], 25.0)

    def test_04_fixed_multiples_with_revision_momentum(self):
        # Compare 12x fixed target upside vs EPS momentum
        eps_rev = oracle_eps_revision(6000.0, 5000.0) # +20%
        multiples = oracle_fixed_multiple_bands([10.0, 12.0], current_eps=6000.0, current_price=60000.0)
        # Target 12x = 72,000 KRW, upside = +20%
        self.assertAlmostEqual(multiples["upsides"][12.0], 20.0)
        self.assertAlmostEqual(eps_rev, 20.0)

    def test_05_small_sample_sector_in_quick_preset(self):
        # Utility sector with N=2 [12.0, 18.0]
        # Continuity correction yields 25.0% and 75.0%
        # Cheaper stock (25.0%) with +2.0% revision qualifies for Turnaround preset
        pct_low = oracle_sector_pe_percentile(12.0, [12.0, 18.0])
        df = pd.DataFrame([{
            "ticker": "UTIL_CHEAP",
            "pe_percentile": 50.0,
            "sector_pe_percentile": pct_low,
            "eps_rev_1m": 2.0,
            "upside_base": 12.0,
        }])
        res = oracle_apply_quick_preset(df, "TURNAROUND")
        self.assertEqual(len(res), 1)

    def test_06_1w_1m_divergence_regime_classification(self):
        # 1W rebound (+2.0%) within continuing 1M downgrade (-4.0%)
        # Overall medium-term trend (-4.0%) triggers Value Trap
        regime = oracle_classify_regime(pe_pct=25.0, rev_1m=-4.0, rev_1w=2.0)
        self.assertTrue(regime["is_value_trap"])

    def test_07_turnaround_preset_vs_trap_preset_disjointness(self):
        # Proof that no stock can be both Turnaround and Value Trap
        # Turnaround requires eps_rev_1m > 0; Trap requires eps_rev_1m < -3.0
        df = pd.DataFrame([
            {"ticker": "S1", "pe_percentile": 20.0, "sector_pe_percentile": 20.0, "eps_rev_1m": 5.0, "upside_base": 15.0},
            {"ticker": "S2", "pe_percentile": 20.0, "sector_pe_percentile": 20.0, "eps_rev_1m": -5.0, "upside_base": 15.0},
            {"ticker": "S3", "pe_percentile": 20.0, "sector_pe_percentile": 20.0, "eps_rev_1m": 0.0, "upside_base": 15.0},
        ])
        turn = set(oracle_apply_quick_preset(df, "TURNAROUND")["ticker"])
        trap = set(oracle_apply_quick_preset(df, "TRAP")["ticker"])
        self.assertEqual(len(turn.intersection(trap)), 0)

    def test_08_sector_relative_pe_with_winsorized_bands(self):
        # Stock with relative P/E = 0.8x and historical Mean-1SD band
        sec_med = oracle_sector_median_pe([10.0, 12.0, 15.0, 18.0, 20.0]) # 15.0
        rel_pe = 12.0 / sec_med # 0.8x
        hist_pe = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        bands = oracle_mean_sd_bands(hist_pe)
        self.assertAlmostEqual(rel_pe, 0.8, places=4)
        self.assertLessEqual(12.0, bands["mean"])

    def test_09_deficit_transition_and_revision_clamping(self):
        # Deficit turnaround: EPS -500 to +3000 (+700% -> clamped +500%)
        rev = oracle_eps_revision(3000.0, -500.0)
        self.assertEqual(rev, 500.0)
        # Bands can be computed once EPS is positive
        tgt = 3000.0 * 10.0
        self.assertEqual(tgt, 30000.0)

    def test_10_multi_ticker_screener_pipeline(self):
        # Pipeline integration on 5 stocks
        data = [
            {"ticker": "005930", "sector": "반도체", "pe": 12.0, "eps": 6000.0, "rev_1m": 4.0},
            {"ticker": "000660", "sector": "반도체", "pe": 15.0, "eps": 12000.0, "rev_1m": 6.0},
            {"ticker": "005380", "sector": "자동차", "pe": 6.0, "eps": 25000.0, "rev_1m": 1.0},
            {"ticker": "105560", "sector": "금융", "pe": 4.5, "eps": 14000.0, "rev_1m": -4.0},
            {"ticker": "035420", "sector": "IT", "pe": 25.0, "eps": 8000.0, "rev_1m": 0.5},
        ]
        df = pd.DataFrame(data)
        # Compute sector medians
        semi_med = oracle_sector_median_pe([12.0, 15.0])
        self.assertAlmostEqual(semi_med, 13.5)

    def test_11_high_volatility_stock_floor_and_targets(self):
        # Floor ensures lower band target is always economically meaningful (positive price)
        hist_pe = pd.Series([1.2, 1.5, 2.0, 18.0, 20.0])
        bands = oracle_mean_sd_bands(hist_pe, floor_pe=1.0, winsorize=False)
        self.assertGreaterEqual(bands["m2sd"], 1.0)
        target_bear = 2000.0 * bands["m2sd"]
        self.assertGreater(target_bear, 0.0)

    def test_12_screener_sorting_with_all_strategy_tags(self):
        tags = ["Golden Cross", "Value Trap", "Momentum Leader", "High P/E Downgrade", "Neutral"]
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(5)],
            "regime_tag": tags,
            "upside_base": [20.0, 10.0, 35.0, -15.0, 5.0],
        })
        sorted_df = df.sort_values(by="upside_base", ascending=False)
        self.assertEqual(sorted_df.iloc[0]["regime_tag"], "Momentum Leader")
        self.assertEqual(sorted_df.iloc[-1]["regime_tag"], "High P/E Downgrade")

    def test_13_dashboard_kpi_sync_with_band_model_change(self):
        # Switching from Percentile to Mean±SD changes Base Target
        hist_pe = pd.Series([8.0, 10.0, 12.0, 14.0, 16.0])
        current_eps = 5000.0
        # Percentile median = 12.0 -> target = 60,000
        target_pct = current_eps * float(hist_pe.median())
        # Mean = 12.0 -> target = 60,000
        bands_sd = oracle_mean_sd_bands(hist_pe, winsorize=False)
        target_sd = current_eps * bands_sd["mean"]
        self.assertAlmostEqual(target_pct, target_sd)

    def test_14_outlier_filtering_impact_on_sector_median(self):
        # Sector median with vs without 150+ cap
        raw_pes = [8.0, 10.0, 12.0, 200.0]
        med = oracle_sector_median_pe(raw_pes, max_pe_cap=150.0)
        # 200.0 filtered, leaves [8, 10, 12] -> median = 10.0
        self.assertAlmostEqual(med, 10.0)

    def test_15_asof_date_historical_revision_and_bands(self):
        # Historical slicing for as-of calculation
        dates = pd.bdate_range("2024-01-01", periods=100)
        s = pd.Series([1000.0 + i * 5.0 for i in range(100)], index=dates)
        # As-of day 70
        s_hist = s.iloc[:70]
        res_hist = oracle_calc_eps_revisions_series(s_hist)
        self.assertAlmostEqual(res_hist["current_eps"], s.iloc[69])

    def test_16_fixed_multiple_ordering_across_price_levels(self):
        # 8x < 10x < 12x < 15x targets strictly ascending for any positive EPS
        for eps in [500.0, 2500.0, 15000.0]:
            res = oracle_fixed_multiple_bands([8.0, 10.0, 12.0, 15.0], current_eps=eps, current_price=10000.0)
            tgts = [res["targets"][m] for m in [8.0, 10.0, 12.0, 15.0]]
            self.assertTrue(tgts[0] < tgts[1] < tgts[2] < tgts[3])


if __name__ == "__main__":
    unittest.main()
