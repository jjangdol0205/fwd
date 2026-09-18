"""
tests/test_tier4_scenarios.py
Tier 4: Realistic End-to-End Financial Analyst Workflow Scenarios (10 Tests)
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


class TestTier4Scenarios(unittest.TestCase):
    """Tier 4: Real-world analytical scenarios modeling equity research analyst workflows."""

    def test_scenario_01_semiconductor_cyclical_inflection(self):
        """Scenario 1: Analyst detects semiconductor cyclical bottom turnaround."""
        # 1. 70 days of daily EPS data showing bottom inflection
        dates = pd.bdate_range("2026-03-01", periods=70)
        # Drops from 5000 to 4500 over first 45 days, rebounds from 4500 to 4850 over last 25 days
        vals = list(np.linspace(5000, 4500, 45)) + list(np.linspace(4500, 4850, 25))
        eps_series = pd.Series(vals, index=dates)

        # 2. Compute revisions
        revs = oracle_calc_eps_revisions_series(eps_series)
        self.assertGreater(revs["eps_rev_1m"], 0.0) # 1M is positive
        self.assertGreater(revs["eps_rev_1w"], 0.0) # 1W is accelerating

        # 3. Stock P/E is 12.0x, historical percentile is 32% (discounted)
        current_price = 58200.0
        current_pe = current_price / revs["current_eps"]
        regime = oracle_classify_regime(pe_pct=32.0, rev_1m=revs["eps_rev_1m"], rev_1w=revs["eps_rev_1w"], current_pe=current_pe)
        self.assertTrue(regime["is_golden_cross"])
        self.assertEqual(regime["regime_tag"], "Golden Cross")

        # 4. Check Mean + 1SD bull target
        hist_pe = pd.Series([10.0, 11.5, 12.5, 14.0, 15.5, 17.0])
        bands = oracle_mean_sd_bands(hist_pe)
        target_bull = revs["current_eps"] * bands["p1sd"]
        upside_bull = ((target_bull / current_price) - 1.0) * 100.0
        self.assertGreater(upside_bull, 15.0)

    def test_scenario_02_bank_value_trap_avoidance(self):
        """Scenario 2: Analyst screens banking sector and avoids optical Value Trap."""
        # Bank stock trades at P/E = 4.2x (10th percentile - looks tempting)
        # But net interest margin cut leads to -4.8% 1M EPS revision and -9.0% 3M revision
        regime = oracle_classify_regime(pe_pct=10.0, rev_1m=-4.8, rev_3m=-9.0, current_pe=4.2)
        self.assertTrue(regime["is_value_trap"])
        self.assertEqual(regime["regime_tag"], "Value Trap")

        # Preset filter: TRAP preset catches it, TURNAROUND preset rejects it
        df = pd.DataFrame([{
            "ticker": "055550",
            "pe_percentile": 10.0,
            "sector_pe_percentile": 12.0,
            "eps_rev_1m": -4.8,
            "upside_base": 25.0,
        }])
        turn = oracle_apply_quick_preset(df, "TURNAROUND")
        trap = oracle_apply_quick_preset(df, "TRAP")
        self.assertEqual(len(turn), 0)
        self.assertEqual(len(trap), 1)

    def test_scenario_03_high_growth_battery_downgrade(self):
        """Scenario 3: High multiple battery stock faces earnings downgrade."""
        # P/E = 48x, historical 85th percentile. 1M revision = -6.5%
        regime = oracle_classify_regime(pe_pct=85.0, rev_1m=-6.5, current_pe=48.0)
        self.assertEqual(regime["regime_tag"], "High P/E Downgrade")
        self.assertFalse(regime["is_value_trap"]) # Not cheap, so not value trap
        self.assertFalse(regime["is_golden_cross"])

    def test_scenario_04_analyst_1click_turnaround_workflow(self):
        """Scenario 4: Analyst runs 1-click turnaround preset workflow."""
        universe = pd.DataFrame([
            {"ticker": "005930", "pe_percentile": 35.0, "sector_pe_percentile": 20.0, "eps_rev_1m": 4.5, "upside_base": 25.0},
            {"ticker": "000660", "pe_percentile": 42.0, "sector_pe_percentile": 38.0, "eps_rev_1m": 6.0, "upside_base": 30.0},
            {"ticker": "105560", "pe_percentile": 15.0, "sector_pe_percentile": 25.0, "eps_rev_1m": -3.5, "upside_base": 20.0},
            {"ticker": "035420", "pe_percentile": 60.0, "sector_pe_percentile": 55.0, "eps_rev_1m": 1.0, "upside_base": 5.0},
        ])

        filtered = oracle_apply_quick_preset(universe, "TURNAROUND")
        # 005930 (pe_pct=35 <= 40, rev>0, up>=10) and 000660 (sec_pct=38 <= 40, rev>0, up>=10) qualify
        self.assertEqual(len(filtered), 2)
        self.assertIn("005930", filtered["ticker"].values)
        self.assertIn("000660", filtered["ticker"].values)
        self.assertNotIn("105560", filtered["ticker"].values) # Trap, rev < 0

    def test_scenario_05_analyst_model_comparison(self):
        """Scenario 5: Analyst compares Percentile vs Mean±SD vs Fixed Multiples models."""
        hist_pe = pd.Series([8.0, 9.5, 11.0, 12.5, 14.0, 16.0])
        current_eps = 4500.0
        current_price = 50000.0

        # 1. Percentile Mode
        target_bear_pct = current_eps * float(hist_pe.quantile(0.25))
        target_base_pct = current_eps * float(hist_pe.median())
        target_bull_pct = current_eps * float(hist_pe.quantile(0.75))

        # 2. Mean ± SD Mode
        bands_sd = oracle_mean_sd_bands(hist_pe, winsorize=False)
        target_bear_sd = current_eps * bands_sd["m1sd"]
        target_base_sd = current_eps * bands_sd["mean"]
        target_bull_sd = current_eps * bands_sd["p1sd"]

        # 3. Fixed Multiples Mode (10x, 12x)
        fixed = oracle_fixed_multiple_bands([10.0, 12.0], current_eps, current_price)

        # Confirm all target sets are monotonic
        self.assertLess(target_bear_pct, target_base_pct)
        self.assertLess(target_base_pct, target_bull_pct)
        self.assertLess(target_bear_sd, target_base_sd)
        self.assertLess(target_base_sd, target_bull_sd)
        self.assertLess(fixed["targets"][10.0], fixed["targets"][12.0])

    def test_scenario_06_small_sector_utility_analyst_check(self):
        """Scenario 6: Analyst checks small-sample utility sector percentile ranks."""
        # 2 utility stocks with P/E = 8.0 and 12.0
        pes = [8.0, 12.0]
        rank_0 = oracle_sector_pe_percentile(8.0, pes)
        rank_1 = oracle_sector_pe_percentile(12.0, pes)
        # Continuity correction provides 25% and 75%
        self.assertAlmostEqual(rank_0, 25.0)
        self.assertAlmostEqual(rank_1, 75.0)
        # Avoids hazard 0% and 100%
        self.assertNotEqual(rank_0, 0.0)
        self.assertNotEqual(rank_1, 100.0)

    def test_scenario_07_quarterly_earnings_season_bulk_revision(self):
        """Scenario 7: Full quarterly update across universe identifying top revisions."""
        rng = np.random.default_rng(100)
        tickers = [f"T{i:03d}" for i in range(25)]
        dates = pd.bdate_range("2026-03-01", periods=65)

        records = []
        for t in tickers:
            base = rng.uniform(2000, 8000)
            growth = rng.normal(0.001, 0.01, 65)
            s = pd.Series(base * np.cumprod(1.0 + growth), index=dates)
            revs = oracle_calc_eps_revisions_series(s)
            records.append({
                "ticker": t,
                "current_fwd_eps": revs["current_eps"],
                "eps_rev_1m": revs["eps_rev_1m"],
            })

        df = pd.DataFrame(records)
        top_eps = oracle_apply_quick_preset(df, "EPS_TOP")
        self.assertLessEqual(len(top_eps), 25)
        # Confirms sorted descending
        for i in range(len(top_eps) - 1):
            self.assertGreaterEqual(top_eps.iloc[i]["eps_rev_1m"], top_eps.iloc[i + 1]["eps_rev_1m"])

    def test_scenario_08_recession_stress_test_winsorization(self):
        """Scenario 8: Severe market recession stress test with 30% deficit stocks."""
        # Mixture of normal P/Es, deficits (<= 0), and near-zero cyclical spikes (140x)
        pe_pool = [8.0, 9.0, 10.0, 12.0, 14.0, 15.0, 18.0, 145.0, -10.0, 0.0, -50.0]
        # Calculator filters <= 0 and applies Winsorization to [8, 9, 10, 12, 14, 15, 18, 145]
        s = pd.Series(pe_pool)
        win = oracle_winsorize(s)
        self.assertTrue(all(win >= 1.0))
        self.assertTrue(all(win <= 150.0))
        bands = oracle_mean_sd_bands(win)
        self.assertGreater(bands["mean"], 0.0)
        self.assertFalse(np.isnan(bands["std"]))

    def test_scenario_09_analyst_valuation_sensitivity_matrix(self):
        """Scenario 9: Valuation sensitivity matrix: EPS shift x Multiple."""
        base_eps = 5000.0
        eps_shifts = [-0.10, -0.05, 0.0, 0.05, 0.10]
        multiples = [8.0, 10.0, 12.0]

        matrix = {}
        for shift in eps_shifts:
            sim_eps = base_eps * (1.0 + shift)
            matrix[shift] = {m: sim_eps * m for m in multiples}

        # Monotonicity check across EPS shifts
        for m in multiples:
            for i in range(len(eps_shifts) - 1):
                self.assertLess(matrix[eps_shifts[i]][m], matrix[eps_shifts[i + 1]][m])

        # Monotonicity check across Multiples
        for shift in eps_shifts:
            self.assertLess(matrix[shift][8.0], matrix[shift][10.0])
            self.assertLess(matrix[shift][10.0], matrix[shift][12.0])

    def test_scenario_10_end_to_end_screener_to_integrated_chart(self):
        """Scenario 10: Complete pipeline from daily time series to chart data structure."""
        dates = pd.bdate_range("2025-01-01", periods=120)
        price_series = pd.Series(np.linspace(50000, 65000, 120), index=dates)
        eps_series = pd.Series(np.linspace(4000, 5200, 120), index=dates)

        # 1. Revisions
        revs = oracle_calc_eps_revisions_series(eps_series)
        self.assertIsNotNone(revs["eps_rev_1m"])

        # 2. P/E Series
        pe_series = price_series / eps_series
        self.assertTrue(all(pe_series > 0))

        # 3. Bands
        bands = oracle_mean_sd_bands(pe_series)
        self.assertIn("mean", bands)

        # 4. KPI Deck
        current_price = float(price_series.iloc[-1])
        current_eps = float(eps_series.iloc[-1])
        target_base = current_eps * bands["mean"]
        upside_base = ((target_base / current_price) - 1.0) * 100.0

        kpi_deck = {
            "current_price": current_price,
            "current_fwd_eps": current_eps,
            "current_fwd_pe": current_price / current_eps,
            "eps_rev_1m": revs["eps_rev_1m"],
            "target_base": target_base,
            "upside_base": upside_base,
        }
        self.assertGreater(kpi_deck["upside_base"], -100.0)


if __name__ == "__main__":
    unittest.main()
