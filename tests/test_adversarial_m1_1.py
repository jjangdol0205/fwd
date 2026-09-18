"""
tests/test_adversarial_m1_1.py
------------------------------
Adversarial Stress-Testing Suite for Milestone 1 (Quantitative Formulas & Engine).
Author: Challenger M1-1 (Empirical Challenger)

Coverage Dimensions:
1. Rapid oscillations in EPS (+1000 -> -500 -> +200)
2. Zero-crossing EPS (deficit to surplus, loss widening/narrowing)
3. Tiny EPS values (0.001 KRW to 0.005 KRW, near-zero floor protection)
4. Stale/constant EPS (0% revision, invariance)
5. Missing data / NaNs interspersed in time series
6. Boundary conditions for pe_pct (30.0, 30.1, 40.0, 40.1, 70.0)
7. Boundary conditions for revisions (-2.0, -2.001, +2.0, +1.999)
8. Conflicting signals (1M down / 1W up, 1M up / 1W down)
9. Strict disjointness / mutual exclusivity of Value Trap vs Golden Cross
10. Ticker sanitization edge cases
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from core.data_loader import (
    calc_eps_revision_rate,
    calc_eps_revisions_series,
    calc_eps_revisions,
    classify_regime,
    classify_valuation_momentum,
    clean_ticker,
    _clean_ticker,
    RegimeResult,
)
from tests.oracle import (
    oracle_eps_revision,
    oracle_calc_eps_revisions_series,
    oracle_classify_regime,
)


class TestEPSRevisionRapidOscillations(unittest.TestCase):
    """Stress testing rapid sign flips and extreme oscillations in EPS."""

    def test_rapid_oscillation_sequence(self):
        """Test +1000 -> -500 -> +200 sequence."""
        # 1. +1000 -> -500: Deterioration into loss
        # Formula: (-500 - 1000) / 1000 * 100 = -150% -> clamped to -100%
        rev_crash = calc_eps_revision_rate(-500.0, 1000.0)
        self.assertEqual(rev_crash, -100.0)

        # 2. -500 -> +200: Turnaround from deficit to profit
        # Absolute denominator: (+200 - (-500)) / max(|-500|, 1) * 100 = +700 / 500 * 100 = +140.0%
        # CRITICAL ADVERSARIAL CHECK: Non-inverting positive sign!
        rev_turnaround = calc_eps_revision_rate(200.0, -500.0)
        self.assertAlmostEqual(rev_turnaround, 140.0, places=4)
        self.assertGreater(rev_turnaround, 0.0, "Turnaround from deficit to surplus MUST be positive!")

        # 3. Direct comparison +1000 -> +200:
        # (+200 - 1000) / 1000 * 100 = -80.0%
        rev_net = calc_eps_revision_rate(200.0, 1000.0)
        self.assertAlmostEqual(rev_net, -80.0, places=4)

    def test_violent_ping_pong_oscillations(self):
        """Test alternating sign flips [+100, -100, +100, -100]."""
        # +100 -> -100: (-100 - 100) / 100 * 100 = -200% -> clamped to -100%
        self.assertEqual(calc_eps_revision_rate(-100.0, 100.0), -100.0)
        # -100 -> +100: (+100 - (-100)) / 100 * 100 = +200%
        self.assertEqual(calc_eps_revision_rate(100.0, -100.0), 200.0)
        # -100 -> -200: (-200 - (-100)) / 100 * 100 = -100%
        self.assertEqual(calc_eps_revision_rate(-200.0, -100.0), -100.0)

    def test_series_with_rapid_oscillations(self):
        """Test full time series undergoing rapid cyclical oscillation."""
        dates = pd.bdate_range("2026-01-01", periods=100)
        # Create a series where:
        # t-60 is +1000
        # t-20 is -500
        # t-5 is +100
        # t-0 is +200
        vals = np.ones(100) * 500.0
        vals[-61] = 1000.0  # 3M ago
        vals[-21] = -500.0  # 1M ago
        vals[-6] = 100.0   # 1W ago
        vals[-1] = 200.0   # Current
        s = pd.Series(vals, index=dates, name="009990")

        rev = calc_eps_revisions_series(s)
        self.assertEqual(rev["current_eps"], 200.0)
        self.assertEqual(rev["eps_1w_ago"], 100.0)
        self.assertEqual(rev["eps_1m_ago"], -500.0)
        self.assertEqual(rev["eps_3m_ago"], 1000.0)
        self.assertAlmostEqual(rev["eps_rev_1w"], 100.0, places=3)
        self.assertAlmostEqual(rev["eps_rev_1m"], 140.0, places=3)
        self.assertAlmostEqual(rev["eps_rev_3m"], -80.0, places=3)
        self.assertTrue(rev["is_turnaround"])
        self.assertFalse(rev["is_deficit"])


class TestEPSRevisionZeroCrossing(unittest.TestCase):
    """Stress testing transitions across zero EPS."""

    def test_deficit_to_surplus_crossover(self):
        """Deficit to surplus produces positive revision and sets is_turnaround."""
        # -200 -> +300: (300 - (-200)) / 200 * 100 = +250.0%
        rev = calc_eps_revision_rate(300.0, -200.0)
        self.assertEqual(rev, 250.0)

    def test_surplus_to_deficit_crossover(self):
        """Surplus to deficit produces negative revision and sets is_deficit."""
        # +300 -> -200: (-200 - 300) / 300 * 100 = -166.67% -> clamped to -100.0%
        rev = calc_eps_revision_rate(-200.0, 300.0)
        self.assertEqual(rev, -100.0)

    def test_loss_widening(self):
        """Loss widening from -100 to -300."""
        # (-300 - (-100)) / 100 * 100 = -200% -> clamped -100%
        self.assertEqual(calc_eps_revision_rate(-300.0, -100.0), -100.0)

    def test_loss_narrowing(self):
        """Loss narrowing from -400 to -100."""
        # (-100 - (-400)) / 400 * 100 = +300 / 400 * 100 = +75.0%
        self.assertEqual(calc_eps_revision_rate(-100.0, -400.0), 75.0)

    def test_exact_zero_base(self):
        """Base is exactly 0.0 (division protection by max(|0|, 1.0))."""
        # 0.0 -> 50.0: (50 - 0) / 1.0 * 100 = 5000% -> clamped 500%
        self.assertEqual(calc_eps_revision_rate(50.0, 0.0), 500.0)
        # 0.0 -> -50.0: (-50 - 0) / 1.0 * 100 = -5000% -> clamped -100%
        self.assertEqual(calc_eps_revision_rate(-50.0, 0.0), -100.0)
        # 0.0 -> 0.0: 0.0%
        self.assertEqual(calc_eps_revision_rate(0.0, 0.0), 0.0)

    def test_exact_zero_current(self):
        """Current is exactly 0.0."""
        # 100.0 -> 0.0: (0 - 100) / 100 * 100 = -100.0%
        self.assertEqual(calc_eps_revision_rate(0.0, 100.0), -100.0)
        # -100.0 -> 0.0: (0 - (-100)) / 100 * 100 = +100.0%
        self.assertEqual(calc_eps_revision_rate(0.0, -100.0), 100.0)

    def test_turnaround_and_deficit_flags(self):
        """Verify is_turnaround and is_deficit flags in calc_eps_revisions."""
        dates = pd.bdate_range("2026-01-01", periods=100)
        # Stock 1: Turnaround (-50 -> +100)
        s1 = pd.Series(np.linspace(-50, 100, 100), index=dates, name="001000")
        # Stock 2: Ongoing deficit (-100 -> -20)
        s2 = pd.Series(np.linspace(-100, -20, 100), index=dates, name="002000")
        # Stock 3: Stable profit (1000 -> 1100)
        s3 = pd.Series(np.linspace(1000, 1100, 100), index=dates, name="003000")
        # Stock 4: Falling into deficit (500 -> -50)
        s4 = pd.Series(np.linspace(500, -50, 100), index=dates, name="004000")

        df = pd.DataFrame({"001000": s1, "002000": s2, "003000": s3, "004000": s4})
        rev_df = calc_eps_revisions(df)

        self.assertTrue(rev_df.loc["001000", "is_turnaround"])
        self.assertFalse(rev_df.loc["001000", "is_deficit"])

        self.assertFalse(rev_df.loc["002000", "is_turnaround"])
        self.assertTrue(rev_df.loc["002000", "is_deficit"])

        self.assertFalse(rev_df.loc["003000", "is_turnaround"])
        self.assertFalse(rev_df.loc["003000", "is_deficit"])

        self.assertFalse(rev_df.loc["004000", "is_turnaround"])
        self.assertTrue(rev_df.loc["004000", "is_deficit"])


class TestEPSRevisionTinyValues(unittest.TestCase):
    """Stress testing tiny EPS values (near-zero floor protection)."""

    def test_sub_won_eps_values(self):
        """0.001 KRW to 0.005 KRW: denominator floor at 1.0 prevents explosive distortion."""
        # (0.005 - 0.001) / max(0.001, 1.0) * 100 = 0.004 / 1.0 * 100 = 0.4%
        rev = calc_eps_revision_rate(0.005, 0.001)
        self.assertAlmostEqual(rev, 0.4, places=5)

    def test_symmetric_negative_sub_won(self):
        """-0.005 KRW to -0.001 KRW."""
        # (-0.001 - (-0.005)) / 1.0 * 100 = +0.4%
        rev = calc_eps_revision_rate(-0.001, -0.005)
        self.assertAlmostEqual(rev, 0.4, places=5)

    def test_fractional_crossover(self):
        """-0.5 KRW to +0.5 KRW."""
        # (0.5 - (-0.5)) / max(|-0.5|, 1.0) * 100 = 1.0 / 1.0 * 100 = +100.0%
        rev = calc_eps_revision_rate(0.5, -0.5)
        self.assertAlmostEqual(rev, 100.0, places=5)

    def test_machine_epsilon(self):
        """1e-12 to 2e-12: no crash, no division by zero."""
        rev = calc_eps_revision_rate(2e-12, 1e-12)
        self.assertAlmostEqual(rev, 1e-10, places=8)


class TestEPSRevisionStaleAndInvariance(unittest.TestCase):
    """Stress testing stale/constant EPS (0% revision)."""

    def test_constant_positive(self):
        self.assertEqual(calc_eps_revision_rate(5000.0, 5000.0), 0.0)

    def test_constant_negative(self):
        self.assertEqual(calc_eps_revision_rate(-500.0, -500.0), 0.0)

    def test_flat_series(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        s = pd.Series(5000.0, index=dates, name="005930")
        rev = calc_eps_revisions_series(s)
        self.assertEqual(rev["eps_rev_1w"], 0.0)
        self.assertEqual(rev["eps_rev_1m"], 0.0)
        self.assertEqual(rev["eps_rev_3m"], 0.0)
        self.assertFalse(rev["is_turnaround"])
        self.assertFalse(rev["is_deficit"])


class TestEPSRevisionMissingAndNaNHandling(unittest.TestCase):
    """Stress testing NaNs and missing observations."""

    def test_all_nan_series(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        s = pd.Series(np.nan, index=dates, name="000001")
        rev = calc_eps_revisions_series(s)
        self.assertIsNone(rev["current_eps"])
        self.assertIsNone(rev["eps_rev_1w"])
        self.assertIsNone(rev["eps_rev_1m"])
        self.assertIsNone(rev["eps_rev_3m"])
        self.assertFalse(rev["is_turnaround"])
        self.assertFalse(rev["is_deficit"])

    def test_short_series_boundaries(self):
        """Test boundary lengths: 5 (no 1W), 6 (1W only), 20 (1W only), 21 (1W+1M), 60, 61."""
        dates = pd.bdate_range("2026-01-01", periods=100)

        # Length 5: all revisions None
        s5 = pd.Series(np.arange(1, 6) * 100.0, index=dates[:5], name="000005")
        r5 = calc_eps_revisions_series(s5)
        self.assertIsNone(r5["eps_rev_1w"])
        self.assertIsNone(r5["eps_rev_1m"])
        self.assertIsNone(r5["eps_rev_3m"])

        # Length 6: 1W available, 1M None, 3M None
        s6 = pd.Series(np.arange(1, 7) * 100.0, index=dates[:6], name="000006")
        r6 = calc_eps_revisions_series(s6)
        self.assertIsNotNone(r6["eps_rev_1w"])
        self.assertIsNone(r6["eps_rev_1m"])
        self.assertIsNone(r6["eps_rev_3m"])

        # Length 21: 1W available, 1M available, 3M None
        s21 = pd.Series(np.arange(1, 22) * 100.0, index=dates[:21], name="000021")
        r21 = calc_eps_revisions_series(s21)
        self.assertIsNotNone(r21["eps_rev_1w"])
        self.assertIsNotNone(r21["eps_rev_1m"])
        self.assertIsNone(r21["eps_rev_3m"])

        # Length 61: all three available
        s61 = pd.Series(np.arange(1, 62) * 100.0, index=dates[:61], name="000061")
        r61 = calc_eps_revisions_series(s61)
        self.assertIsNotNone(r61["eps_rev_1w"])
        self.assertIsNotNone(r61["eps_rev_1m"])
        self.assertIsNotNone(r61["eps_rev_3m"])

    def test_interspersed_nans(self):
        """Test time series with interspersed NaNs."""
        dates = pd.bdate_range("2026-01-01", periods=100)
        vals = np.linspace(1000, 2000, 100)
        # Interlink NaNs every 3rd index
        vals[::3] = np.nan
        s = pd.Series(vals, index=dates, name="005930")
        rev = calc_eps_revisions_series(s)
        # Should not crash, and should drop NaNs cleanly
        self.assertIsNotNone(rev["current_eps"])
        self.assertIsNotNone(rev["eps_rev_1w"])

    def test_as_of_date_filtering(self):
        """Test calc_eps_revisions with as_of_date."""
        dates = pd.bdate_range("2025-01-01", periods=120)
        df = pd.DataFrame({
            "005930": np.linspace(1000, 2000, 120),
            "000660": np.linspace(500, 1500, 120),
        }, index=dates)

        cutoff = dates[50]
        rev_cutoff = calc_eps_revisions(df, as_of_date=cutoff)
        self.assertEqual(len(rev_cutoff), 2)
        # Latest EPS should match cutoff date value
        self.assertAlmostEqual(rev_cutoff.loc["005930", "current_eps"], df.loc[cutoff, "005930"], places=4)


class TestClassifyRegimeBoundaries(unittest.TestCase):
    """Stress testing 5-regime classification boundaries."""

    def test_pe_pct_30_value_trap_boundary(self):
        """pe_pct <= 30.0 boundary tests."""
        # 1. pe_pct = 30.0 and rev_1m = -2.001 -> Value Trap
        res1 = classify_regime(pe_pct=30.0, eps_rev_1m=-2.001)
        self.assertTrue(res1.is_value_trap)
        self.assertEqual(res1.regime_tag, "Value Trap")

        # 2. pe_pct = 30.0 and rev_1m = -2.000 -> Neutral (strict inequality rev < -2.0)
        res2 = classify_regime(pe_pct=30.0, eps_rev_1m=-2.000)
        self.assertFalse(res2.is_value_trap)
        self.assertEqual(res2.regime_tag, "Neutral")

        # 3. pe_pct = 30.1 and rev_1m = -2.001 -> Neutral (pe_pct > 30.0)
        res3 = classify_regime(pe_pct=30.1, eps_rev_1m=-2.001)
        self.assertFalse(res3.is_value_trap)
        self.assertEqual(res3.regime_tag, "Neutral")

        # 4. pe_pct = 29.9 and rev_1m = -2.001 -> Value Trap
        res4 = classify_regime(pe_pct=29.9, eps_rev_1m=-2.001)
        self.assertTrue(res4.is_value_trap)
        self.assertEqual(res4.regime_tag, "Value Trap")

        # 5. pe_pct = 30.0, rev_1m = -0.001, rev_3m = -5.001 -> Value Trap (dual negative)
        res5 = classify_regime(pe_pct=30.0, eps_rev_1m=-0.001, eps_rev_3m=-5.001)
        self.assertTrue(res5.is_value_trap)
        self.assertEqual(res5.regime_tag, "Value Trap")

        # 6. pe_pct = 30.0, rev_1m = -0.001, rev_3m = -5.000 -> Neutral (rev_3m < -5.0 is strict)
        res6 = classify_regime(pe_pct=30.0, eps_rev_1m=-0.001, eps_rev_3m=-5.000)
        self.assertFalse(res6.is_value_trap)
        self.assertEqual(res6.regime_tag, "Neutral")

    def test_pe_pct_40_golden_cross_boundary(self):
        """pe_pct <= 40.0 boundary tests."""
        # 1. pe_pct = 40.0 and rev_1m = +2.000 -> Golden Cross
        res1 = classify_regime(pe_pct=40.0, eps_rev_1m=2.000, current_pe=10.0)
        self.assertTrue(res1.is_golden_cross)
        self.assertEqual(res1.regime_tag, "Golden Cross")

        # 2. pe_pct = 40.0 and rev_1m = +1.999 (without 1W) -> Neutral
        res2 = classify_regime(pe_pct=40.0, eps_rev_1m=1.999, current_pe=10.0)
        self.assertFalse(res2.is_golden_cross)
        self.assertEqual(res2.regime_tag, "Neutral")

        # 3. pe_pct = 40.0 and rev_1m = +1.999 with rev_1w = +0.001 -> Golden Cross
        res3 = classify_regime(pe_pct=40.0, eps_rev_1m=1.999, eps_rev_1w=0.001, current_pe=10.0)
        self.assertTrue(res3.is_golden_cross)
        self.assertEqual(res3.regime_tag, "Golden Cross")

        # 4. pe_pct = 40.1 and rev_1m = +2.000 -> Neutral (pe_pct > 40.0 and rev < 3.0)
        res4 = classify_regime(pe_pct=40.1, eps_rev_1m=2.000, current_pe=10.0)
        self.assertFalse(res4.is_golden_cross)
        self.assertEqual(res4.regime_tag, "Neutral")

        # 5. pe_pct = 40.1 and rev_1m = +3.000 -> Momentum Leader
        res5 = classify_regime(pe_pct=40.1, eps_rev_1m=3.000, current_pe=10.0)
        self.assertFalse(res5.is_golden_cross)
        self.assertEqual(res5.regime_tag, "Momentum Leader")

        # 6. pe_pct = 40.0 and rev_1m = +3.000 -> Golden Cross (Golden Cross takes precedence at <= 40)
        res6 = classify_regime(pe_pct=40.0, eps_rev_1m=3.000, current_pe=10.0)
        self.assertTrue(res6.is_golden_cross)
        self.assertEqual(res6.regime_tag, "Golden Cross")

    def test_pe_pct_70_downgrade_boundary(self):
        """pe_pct >= 70.0 boundary tests."""
        # 1. pe_pct = 70.0 and rev_1m = -0.001 -> High P/E Downgrade
        res1 = classify_regime(pe_pct=70.0, eps_rev_1m=-0.001, current_pe=25.0)
        self.assertEqual(res1.regime_tag, "High P/E Downgrade")

        # 2. pe_pct = 70.0 and rev_1m = 0.000 -> Neutral
        res2 = classify_regime(pe_pct=70.0, eps_rev_1m=0.000, current_pe=25.0)
        self.assertEqual(res2.regime_tag, "Neutral")

        # 3. pe_pct = 69.9 and rev_1m = -1.000 -> Neutral (pe_pct < 70)
        res3 = classify_regime(pe_pct=69.9, eps_rev_1m=-1.000, current_pe=25.0)
        self.assertEqual(res3.regime_tag, "Neutral")

        # 4. pe_pct = 75.0 and rev_1m = +3.5 -> Momentum Leader (growth leader precedence)
        res4 = classify_regime(pe_pct=75.0, eps_rev_1m=3.5, current_pe=30.0)
        self.assertEqual(res4.regime_tag, "Momentum Leader")

    def test_deficit_stocks_excluded(self):
        """Deficit companies (current_pe <= 0) must NEVER be Golden Cross or Value Trap."""
        res_zero = classify_regime(pe_pct=10.0, eps_rev_1m=5.0, current_pe=0.0)
        self.assertFalse(res_zero.is_golden_cross)
        self.assertFalse(res_zero.is_value_trap)
        self.assertEqual(res_zero.regime_tag, "Neutral")

        res_neg = classify_regime(pe_pct=10.0, eps_rev_1m=5.0, current_pe=-5.0)
        self.assertFalse(res_neg.is_golden_cross)
        self.assertFalse(res_neg.is_value_trap)
        self.assertEqual(res_neg.regime_tag, "Neutral")

    def test_conflicting_signals(self):
        """Conflicting signals between 1M and 1W revisions."""
        # 1. 1M down (-3.0%) but 1W up (+5.0%) at pe_pct = 20.0
        # Should be Value Trap, NOT Golden Cross
        res1 = classify_regime(pe_pct=20.0, eps_rev_1m=-3.0, eps_rev_1w=5.0, current_pe=10.0)
        self.assertTrue(res1.is_value_trap)
        self.assertFalse(res1.is_golden_cross)
        self.assertEqual(res1.regime_tag, "Value Trap")

        # 2. 1M up (+0.5%) but 1W down (-1.0%) at pe_pct = 25.0
        # Does not meet rev_1m >= 2.0, and 1W is negative -> Neutral
        res2 = classify_regime(pe_pct=25.0, eps_rev_1m=0.5, eps_rev_1w=-1.0, current_pe=10.0)
        self.assertFalse(res2.is_golden_cross)
        self.assertFalse(res2.is_value_trap)
        self.assertEqual(res2.regime_tag, "Neutral")

        # 3. 1M up (+2.0%) but 1W down (-1.0%) at pe_pct = 25.0
        # Meets rev_1m >= 2.0 -> Golden Cross
        res3 = classify_regime(pe_pct=25.0, eps_rev_1m=2.0, eps_rev_1w=-1.0, current_pe=10.0)
        self.assertTrue(res3.is_golden_cross)
        self.assertEqual(res3.regime_tag, "Golden Cross")

    def test_strict_disjointness_property(self):
        """Test across 1,000 parameter combinations that is_value_trap and is_golden_cross are disjoint."""
        rng = np.random.default_rng(42)
        pe_pcts = rng.uniform(0.0, 100.0, 1000)
        rev_1ms = rng.uniform(-10.0, 10.0, 1000)
        rev_3ms = rng.uniform(-15.0, 15.0, 1000)
        rev_1ws = rng.uniform(-5.0, 5.0, 1000)
        current_pes = rng.uniform(-5.0, 50.0, 1000)

        for p_pct, r1m, r3m, r1w, c_pe in zip(pe_pcts, rev_1ms, rev_3ms, rev_1ws, current_pes):
            res = classify_regime(pe_pct=p_pct, eps_rev_1m=r1m, eps_rev_3m=r3m, eps_rev_1w=r1w, current_pe=c_pe)
            # Property: Cannot be both
            self.assertFalse(
                res.is_value_trap and res.is_golden_cross,
                f"Disjointness violated for pe_pct={p_pct}, r1m={r1m}, r3m={r3m}, r1w={r1w}, pe={c_pe}"
            )
            # Tag must match flags
            if res.is_golden_cross:
                self.assertEqual(res.regime_tag, "Golden Cross")
            elif res.is_value_trap:
                self.assertEqual(res.regime_tag, "Value Trap")


class TestTickerSanitization(unittest.TestCase):
    """Stress testing ticker normalization edge cases."""

    def test_ticker_patterns(self):
        self.assertEqual(clean_ticker("A005930"), "005930")
        self.assertEqual(clean_ticker("a005930"), "005930")
        self.assertEqual(clean_ticker("5930"), "005930")
        self.assertEqual(clean_ticker(5930), "005930")
        self.assertEqual(clean_ticker("005930"), "005930")
        self.assertEqual(clean_ticker("  000660  "), "000660")
        self.assertEqual(clean_ticker(""), "")
        self.assertEqual(clean_ticker(None), "")
        self.assertEqual(clean_ticker("nan"), "")
        self.assertEqual(clean_ticker("NaN"), "")
        self.assertEqual(clean_ticker("none"), "")
        self.assertEqual(clean_ticker("None"), "")
        self.assertEqual(clean_ticker("00593K"), "00593K")
        self.assertEqual(clean_ticker("A00593K"), "00593K")


if __name__ == "__main__":
    unittest.main()
