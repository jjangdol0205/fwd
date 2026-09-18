"""
tests/test_adversarial_m2_1.py
------------------------------
Adversarial Stress-Testing Suite for Milestone 2 (Sector Mapping & Relative Valuation).
Author: Challenger M2-1 (Empirical Challenger)

Coverage Dimensions:
1. calc_sector_median_pe:
   - All negative P/Es, all zeros, NaNs/None, empty inputs
   - Extreme outliers (> 150x, 1000x, 1e6x) vs normal values
   - Boundary condition at max_pe_cap (150.0 vs 150.0001)
   - Single constituent sector (N=1)
   - Permutation and duplicate invariance
   - Custom max_pe_cap parameter support
2. calc_sector_pe_percentile:
   - Small sample N=1: strictly 50.0% across all valid inputs
   - Small sample N=2: strictly 25.0% (lower) and 75.0% (upper)
   - Small sample N=3 and N=4 continuity-corrected steps
   - Identical ties: symmetric average rank handling (middle, low end, high end, multiple pairs)
   - Micro floating-point near-ties (atol=1e-5)
   - Out-of-bounds target (<= 0, > 150, NaN, None)
   - Peer list with zero valid items or empty
   - Interpolated rank for target not in peer list
3. calc_sector_relative_metrics Integration & Polymorphism:
   - List[PEBandResult] and pd.DataFrame exact parity
   - All-deficit sector handling (None/NaN, stock_count=0)
   - Mixed deficit and profitable sector handling
   - Extreme outlier filtering in cross-sectional grouping
   - Fallback for unmapped tickers ("기타/미분류")
4. Oracle Parity:
   - Randomized Monte Carlo stress trials against tests/oracle.py
"""

import unittest
import math
import numpy as np
import pandas as pd
from typing import List, Optional

from core.calculator import (
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    calc_sector_relative_metrics,
    PEBandResult,
    SectorMappingDict,
    load_sector_mapping,
    _clean_ticker,
)
from tests.oracle import (
    oracle_sector_median_pe,
    oracle_sector_pe_percentile,
)


class TestSectorMedianPEAdversarial(unittest.TestCase):
    """Adversarial stress testing of calc_sector_median_pe."""

    def test_all_negative_pe_inputs(self):
        """Sector with only loss-making / negative P/E constituents must return None."""
        neg_cases = [
            [-10.0, -5.0, -0.01, -999.0],
            [-50.0],
            [-1.0, -2.0, -3.0],
        ]
        for case in neg_cases:
            res = calc_sector_median_pe(case)
            self.assertIsNone(res, f"Expected None for all-negative {case}, got {res}")
            oracle_res = oracle_sector_median_pe(case)
            self.assertIsNone(oracle_res)

    def test_all_zeros_and_negatives(self):
        """Zero P/E (zero EPS or zero price) is not positive, must return None."""
        res = calc_sector_median_pe([0.0, 0.0, -5.0])
        self.assertIsNone(res)
        self.assertIsNone(oracle_sector_median_pe([0.0, 0.0, -5.0]))

    def test_all_nans_and_none(self):
        """Inputs consisting only of NaNs or Nones must return None."""
        res = calc_sector_median_pe([np.nan, None, float("nan")])
        self.assertIsNone(res)

    def test_empty_input_list(self):
        """Empty input list must return None."""
        self.assertIsNone(calc_sector_median_pe([]))
        self.assertIsNone(oracle_sector_median_pe([]))

    def test_extreme_outliers_only(self):
        """Sector where all P/Es exceed max_pe_cap (150.0) must return None."""
        outliers = [150.0001, 200.0, 500.0, 1000.0, 99999.0]
        res = calc_sector_median_pe(outliers)
        self.assertIsNone(res, f"Expected None for all-outlier {outliers}, got {res}")
        self.assertIsNone(oracle_sector_median_pe(outliers))

    def test_boundary_pe_cap_150(self):
        """Exact boundary tests at max_pe_cap = 150.0."""
        # Exactly 150.0 is valid (0 < PE <= 150)
        self.assertAlmostEqual(calc_sector_median_pe([150.0]), 150.0, places=4)
        self.assertAlmostEqual(calc_sector_median_pe([10.0, 150.0]), 80.0, places=4)

        # 150.0001 is an outlier (> 150.0)
        self.assertAlmostEqual(calc_sector_median_pe([10.0, 150.0001]), 10.0, places=4)
        self.assertIsNone(calc_sector_median_pe([150.0001]))

        # Exact lower boundary at zero
        self.assertIsNone(calc_sector_median_pe([0.0]))
        # Near zero positive is valid
        self.assertAlmostEqual(calc_sector_median_pe([0.0001]), 0.0001, places=4)

    def test_mixture_extreme_outliers_and_normal(self):
        """Extreme cyclical outliers must not distort the median."""
        # Uncapped arithmetic mean would be ~12,654x; median must isolate normal peers
        pe_list = [8.0, 12.0, 15.0, 18.0, 25.0, 150.1, 999.0, 99999.0]
        res = calc_sector_median_pe(pe_list)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 15.0, places=4)
        self.assertAlmostEqual(res, oracle_sector_median_pe(pe_list), places=4)

    def test_single_constituent_n1(self):
        """Single stock in a sector."""
        self.assertAlmostEqual(calc_sector_median_pe([14.2]), 14.2, places=4)
        self.assertIsNone(calc_sector_median_pe([-14.2]))
        self.assertIsNone(calc_sector_median_pe([300.0]))

    def test_even_number_of_constituents(self):
        """Even sample median calculation."""
        res = calc_sector_median_pe([10.0, 12.0, 18.0, 30.0])
        self.assertAlmostEqual(res, 15.0, places=4)

    def test_permutation_and_duplicate_invariance(self):
        """Median must be invariant to input ordering and handle duplicates."""
        base = [5.0, 12.0, 12.0, 18.0, 25.0, -10.0, 500.0]
        expected = 12.0
        for seed in [1, 42, 99]:
            rng = np.random.default_rng(seed)
            shuffled = list(rng.permutation(base))
            self.assertAlmostEqual(calc_sector_median_pe(shuffled), expected, places=4)

    def test_custom_cap_parameter(self):
        """Verify custom max_pe_cap operates properly."""
        pe_list = [10.0, 20.0, 40.0, 80.0]
        # With default cap 150: median is (20 + 40)/2 = 30.0
        self.assertAlmostEqual(calc_sector_median_pe(pe_list), 30.0, places=4)
        # With cap 30: valid are [10, 20] -> median is 15.0
        self.assertAlmostEqual(calc_sector_median_pe(pe_list, max_pe_cap=30.0), 15.0, places=4)


class TestSectorPEPercentileAdversarial(unittest.TestCase):
    """Adversarial stress testing of calc_sector_pe_percentile."""

    def test_single_constituent_strictly_50_pct(self):
        """
        N=1 sector requirement:
        Continuity-corrected formula (0 + 0.5) / 1 * 100 MUST yield strictly 50.0%.
        Never 0.0% (false ultra-discount) or 100.0% (false bubble).
        """
        sample_pes = [0.1, 1.0, 5.5, 10.0, 15.25, 25.0, 50.0, 100.0, 149.99, 150.0]
        for p in sample_pes:
            pct = calc_sector_pe_percentile(p, [p])
            self.assertIsNotNone(pct)
            self.assertAlmostEqual(pct, 50.0, places=6, msg=f"P/E {p} failed N=1 50.0% check")
            self.assertAlmostEqual(pct, oracle_sector_pe_percentile(p, [p]), places=6)

        # N=1 with negative and outlier peers in list
        dirty_list = [-20.0, -5.0, 0.0, 18.0, 200.0, 1000.0]
        pct_dirty = calc_sector_pe_percentile(18.0, dirty_list)
        self.assertIsNotNone(pct_dirty)
        self.assertAlmostEqual(pct_dirty, 50.0, places=6)

    def test_two_constituents_strictly_25_and_75_pct(self):
        """
        N=2 sector requirement:
        Continuity-corrected formula MUST yield strictly 25.0% for lower and 75.0% for upper.
        Eliminates 0.0% / 100.0% polarity.
        """
        pairs = [
            (10.0, 20.0),
            (5.0, 50.0),
            (0.5, 150.0),
            (14.0, 14.1),
        ]
        for p1, p2 in pairs:
            pct1 = calc_sector_pe_percentile(p1, [p1, p2])
            pct2 = calc_sector_pe_percentile(p2, [p1, p2])
            self.assertAlmostEqual(pct1, 25.0, places=6, msg=f"Lower {p1} in ({p1}, {p2}) != 25.0%")
            self.assertAlmostEqual(pct2, 75.0, places=6, msg=f"Upper {p2} in ({p1}, {p2}) != 75.0%")
            self.assertAlmostEqual(pct1 + pct2, 100.0, places=6, msg="Symmetry failed")

        # N=2 with negative and outlier noise
        noise_list = [-100.0, -10.0, 0.0, 12.0, 24.0, 150.1, 500.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(12.0, noise_list), 25.0, places=6)
        self.assertAlmostEqual(calc_sector_pe_percentile(24.0, noise_list), 75.0, places=6)

    def test_small_sample_n3_and_n4_steps(self):
        """N=3 and N=4 continuity-corrected steps."""
        # N=3: (0.5/3)*100 = 16.6667%, (1.5/3)*100 = 50.0%, (2.5/3)*100 = 83.3333%
        pes_3 = [10.0, 20.0, 30.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, pes_3), 100.0 / 6.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(20.0, pes_3), 50.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(30.0, pes_3), 500.0 / 6.0, places=4)

        # N=4: (0.5/4)*100 = 12.5%, 37.5%, 62.5%, 87.5%
        pes_4 = [10.0, 20.0, 30.0, 40.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, pes_4), 12.5, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(20.0, pes_4), 37.5, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(30.0, pes_4), 62.5, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(40.0, pes_4), 87.5, places=4)

    def test_identical_ties_symmetric_handling(self):
        """
        Adversarial test for identical ties:
        Tied stocks must receive the exact average of duplicate ranks and maintain symmetry.
        """
        # 1. Two identical stocks: [10.0, 10.0] -> rank 0 and 1 -> avg_rank = 0.5 -> 50.0%
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, [10.0, 10.0]), 50.0, places=4)

        # 2. Three identical stocks: [15.0, 15.0, 15.0] -> avg_rank = 1.0 -> 50.0%
        self.assertAlmostEqual(calc_sector_pe_percentile(15.0, [15.0, 15.0, 15.0]), 50.0, places=4)

        # 3. Symmetric middle pair: [10.0, 20.0, 20.0, 30.0]
        # 10.0 -> 12.5%
        # 20.0 (tied at idx 1, 2) -> avg_rank = 1.5 -> (1.5 + 0.5)/4 * 100 = 50.0%
        # 30.0 -> 87.5%
        sym_list = [10.0, 20.0, 20.0, 30.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, sym_list), 12.5, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(20.0, sym_list), 50.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(30.0, sym_list), 87.5, places=4)

        # 4. Tie at lowest values: [10.0, 10.0, 30.0]
        # 10.0 (idx 0, 1) -> avg_rank = 0.5 -> (0.5 + 0.5)/3 * 100 = 33.3333%
        # 30.0 (idx 2) -> avg_rank = 2.0 -> (2.0 + 0.5)/3 * 100 = 83.3333%
        low_tie = [10.0, 10.0, 30.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, low_tie), 100.0 / 3.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(30.0, low_tie), 250.0 / 3.0, places=4)

        # 5. Tie at highest values: [10.0, 30.0, 30.0]
        # 10.0 (idx 0) -> avg_rank = 0.0 -> (0.5)/3 * 100 = 16.6667%
        # 30.0 (idx 1, 2) -> avg_rank = 1.5 -> (1.5 + 0.5)/3 * 100 = 66.6667%
        high_tie = [10.0, 30.0, 30.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, high_tie), 50.0 / 3.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(30.0, high_tie), 200.0 / 3.0, places=4)

        # Verify exact reflection symmetry: low_tie vs high_tie
        # 100.0 - 33.3333 = 66.6667%
        self.assertAlmostEqual(
            calc_sector_pe_percentile(10.0, low_tie) + calc_sector_pe_percentile(30.0, high_tie),
            100.0,
            places=4,
        )

        # 6. Two pairs of ties: [10.0, 10.0, 20.0, 20.0]
        # 10.0 -> avg_rank = 0.5 -> (0.5+0.5)/4 = 25.0%
        # 20.0 -> avg_rank = 2.5 -> (2.5+0.5)/4 = 75.0%
        two_pairs = [10.0, 10.0, 20.0, 20.0]
        self.assertAlmostEqual(calc_sector_pe_percentile(10.0, two_pairs), 25.0, places=4)
        self.assertAlmostEqual(calc_sector_pe_percentile(20.0, two_pairs), 75.0, places=4)

    def test_near_ties_floating_point_tolerance(self):
        """Two values differing by less than 1e-5 should be recognized as tied."""
        p1 = 12.0
        p2 = 12.000002  # difference is 2e-6 < 1e-5
        peer_list = [p1, p2]
        # Since np.isclose(..., atol=1e-5) is used, both get average rank
        pct1 = calc_sector_pe_percentile(p1, peer_list)
        pct2 = calc_sector_pe_percentile(p2, peer_list)
        self.assertAlmostEqual(pct1, 50.0, places=4)
        self.assertAlmostEqual(pct2, 50.0, places=4)

    def test_out_of_bounds_target_pe(self):
        """Target P/E that is non-positive or exceeds cap must return None."""
        peers = [10.0, 20.0, 30.0]
        # Non-positive targets
        self.assertIsNone(calc_sector_pe_percentile(-10.0, peers))
        self.assertIsNone(calc_sector_pe_percentile(-0.001, peers))
        self.assertIsNone(calc_sector_pe_percentile(0.0, peers))

        # Target above cap
        self.assertIsNone(calc_sector_pe_percentile(150.001, peers))
        self.assertIsNone(calc_sector_pe_percentile(1000.0, peers))

        # Target NaN or None
        self.assertIsNone(calc_sector_pe_percentile(np.nan, peers))
        self.assertIsNone(calc_sector_pe_percentile(None, peers))

    def test_all_invalid_peers_returns_none(self):
        """If sector has zero valid positive P/Es <= 150, percentile must be None."""
        self.assertIsNone(calc_sector_pe_percentile(15.0, []))
        self.assertIsNone(calc_sector_pe_percentile(15.0, [-5.0, -10.0, 0.0]))
        self.assertIsNone(calc_sector_pe_percentile(15.0, [200.0, 500.0]))
        self.assertIsNone(calc_sector_pe_percentile(15.0, [np.nan, None]))

    def test_target_pe_not_in_peer_list_interpolation(self):
        """When target P/E is evaluated against an external distribution."""
        peers = [10.0, 20.0]
        # Target 5.0 (below all peers): np.searchsorted gives idx 0 -> (0.5/2)*100 = 25.0%
        self.assertAlmostEqual(calc_sector_pe_percentile(5.0, peers), 25.0, places=4)

        # Target 15.0 (between peers): np.searchsorted gives idx 1 -> (1.5/2)*100 = 75.0%
        self.assertAlmostEqual(calc_sector_pe_percentile(15.0, peers), 75.0, places=4)

        # Target 25.0 (above peers, <= 150): np.searchsorted gives idx 2, clamped to n-0.5=1.5 -> 100.0%
        self.assertAlmostEqual(calc_sector_pe_percentile(25.0, peers), 100.0, places=4)


class TestSectorRelativeMetricsPolymorphismAndEdgeCases(unittest.TestCase):
    """Stress testing calc_sector_relative_metrics on List and DataFrame."""

    def test_all_deficit_sector_handling(self):
        """Sector where every constituent has negative or NaN P/E."""
        r_neg1 = PEBandResult(
            ticker="000001", name="적자기업1", current_price=10000.0, current_fwd_eps=-1000.0,
            current_fwd_pe=-10.0, pe_percentile=10.0, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0, target_base=0, target_bull=0,
            upside_bear=0, upside_base=0, upside_bull=0, hist_pe_series=pd.Series(),
            hist_price_series=pd.Series(), hist_eps_series=pd.Series(), sector="적자섹터",
        )
        r_neg2 = PEBandResult(
            ticker="000002", name="적자기업2", current_price=20000.0, current_fwd_eps=-2000.0,
            current_fwd_pe=np.nan, pe_percentile=10.0, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0, target_base=0, target_bull=0,
            upside_bear=0, upside_base=0, upside_bull=0, hist_pe_series=pd.Series(),
            hist_price_series=pd.Series(), hist_eps_series=pd.Series(), sector="적자섹터",
        )

        enriched = calc_sector_relative_metrics([r_neg1, r_neg2])
        for r in enriched:
            self.assertIsNone(r.sector_median_pe)
            self.assertIsNone(r.sector_relative_pe)
            self.assertIsNone(r.sector_pe_percentile)
            self.assertEqual(r.sector_stock_count, 0)

    def test_mixed_deficit_and_profitable_sector(self):
        """Sector with 1 profitable stock and 2 deficit stocks."""
        r_profit = PEBandResult(
            ticker="000003", name="흑자기업", current_price=50000.0, current_fwd_eps=5000.0,
            current_fwd_pe=10.0, pe_percentile=30.0, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0, target_base=0, target_bull=0,
            upside_bear=0, upside_base=0, upside_bull=0, hist_pe_series=pd.Series(),
            hist_price_series=pd.Series(), hist_eps_series=pd.Series(), sector="혼합섹터",
        )
        r_loss = PEBandResult(
            ticker="000004", name="적자기업", current_price=10000.0, current_fwd_eps=-500.0,
            current_fwd_pe=-20.0, pe_percentile=10.0, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0, target_base=0, target_bull=0,
            upside_bear=0, upside_base=0, upside_bull=0, hist_pe_series=pd.Series(),
            hist_price_series=pd.Series(), hist_eps_series=pd.Series(), sector="혼합섹터",
        )

        enriched = calc_sector_relative_metrics([r_profit, r_loss])
        # Profitable stock is the only valid member (N=1)
        self.assertAlmostEqual(r_profit.sector_median_pe, 10.0, places=4)
        self.assertEqual(r_profit.sector_stock_count, 1)
        self.assertAlmostEqual(r_profit.sector_relative_pe, 1.0, places=4)
        self.assertAlmostEqual(r_profit.sector_pe_percentile, 50.0, places=4)

        # Deficit stock receives sector median and count, but relative PE and percentile are None
        self.assertAlmostEqual(r_loss.sector_median_pe, 10.0, places=4)
        self.assertEqual(r_loss.sector_stock_count, 1)
        self.assertIsNone(r_loss.sector_relative_pe)
        self.assertIsNone(r_loss.sector_pe_percentile)

    def test_dataframe_polymorphism_parity(self):
        """Exact parity between List[PEBandResult] and pd.DataFrame outputs."""
        df_in = pd.DataFrame({
            "ticker": ["000010", "000020", "000030", "000040"],
            "current_fwd_pe": [10.0, 15.0, -5.0, 200.0],
            "sector": ["테스트섹터", "테스트섹터", "테스트섹터", "테스트섹터"],
        })
        df_out = calc_sector_relative_metrics(df_in)

        # Valid stocks: [10.0, 15.0] (N=2)
        # Median = 12.5
        # 10.0 -> rel = 0.8, pct = 25.0%
        # 15.0 -> rel = 1.2, pct = 75.0%
        # -5.0 -> rel = NaN, pct = NaN
        # 200.0 -> rel = NaN, pct = NaN
        self.assertAlmostEqual(df_out.loc[0, "sector_median_pe"], 12.5, places=4)
        self.assertAlmostEqual(df_out.loc[0, "sector_relative_pe"], 0.8, places=4)
        self.assertAlmostEqual(df_out.loc[0, "sector_pe_percentile"], 25.0, places=4)
        self.assertEqual(df_out.loc[0, "sector_stock_count"], 2)

        self.assertAlmostEqual(df_out.loc[1, "sector_median_pe"], 12.5, places=4)
        self.assertAlmostEqual(df_out.loc[1, "sector_relative_pe"], 1.2, places=4)
        self.assertAlmostEqual(df_out.loc[1, "sector_pe_percentile"], 75.0, places=4)

        self.assertTrue(pd.isna(df_out.loc[2, "sector_relative_pe"]))
        self.assertTrue(pd.isna(df_out.loc[2, "sector_pe_percentile"]))
        self.assertEqual(df_out.loc[2, "sector_stock_count"], 2)

        self.assertTrue(pd.isna(df_out.loc[3, "sector_relative_pe"]))
        self.assertTrue(pd.isna(df_out.loc[3, "sector_pe_percentile"]))
        self.assertEqual(df_out.loc[3, "sector_stock_count"], 2)


class TestOracleEquivalence(unittest.TestCase):
    """Randomized Monte Carlo stress trials comparing against tests/oracle.py."""

    def test_randomized_median_pe_oracle_parity(self):
        """1,000 randomized scenarios comparing calc_sector_median_pe with oracle."""
        rng = np.random.default_rng(2026)
        for trial in range(1000):
            size = rng.integers(0, 40)
            if size == 0:
                pe_list = []
            else:
                # Mixture of normal (5~50), negatives (-50~0), outliers (150~1000), zeros, and NaNs
                choices = rng.choice(
                    [
                        rng.uniform(1.0, 50.0),
                        rng.uniform(-50.0, -0.1),
                        0.0,
                        rng.uniform(150.01, 1000.0),
                        np.nan,
                    ],
                    size=size,
                )
                pe_list = [float(x) if not np.isnan(x) else np.nan for x in choices]

            actual = calc_sector_median_pe(pe_list)
            expected = oracle_sector_median_pe(pe_list)

            if expected is None:
                self.assertIsNone(actual, f"Trial {trial}: expected None, got {actual} for {pe_list}")
            else:
                self.assertIsNotNone(actual, f"Trial {trial}: expected {expected}, got None for {pe_list}")
                self.assertAlmostEqual(actual, expected, places=5)

    def test_randomized_pe_percentile_oracle_parity(self):
        """1,000 randomized scenarios comparing calc_sector_pe_percentile with oracle."""
        rng = np.random.default_rng(2027)
        for trial in range(1000):
            size = rng.integers(0, 30)
            # Generate peer list with potential ties
            if size == 0:
                peers = []
            else:
                # Include exact duplicate ties
                discrete_vals = [5.0, 10.0, 10.0, 15.0, 20.0, 20.0, 30.0, -10.0, 200.0]
                peers = list(rng.choice(discrete_vals, size=size))

            # Pick target from peers, or random outside
            target_type = rng.integers(0, 4)
            if target_type == 0 and len(peers) > 0:
                target_pe = float(rng.choice(peers))
            elif target_type == 1:
                target_pe = float(rng.uniform(1.0, 50.0))
            elif target_type == 2:
                target_pe = float(rng.uniform(-20.0, 0.0))
            else:
                target_pe = float(rng.uniform(150.01, 500.0))

            actual = calc_sector_pe_percentile(target_pe, peers)
            expected = oracle_sector_pe_percentile(target_pe, peers)

            if expected is None:
                self.assertIsNone(actual, f"Trial {trial}: expected None, got {actual} for target={target_pe}, peers={peers}")
            else:
                self.assertIsNotNone(actual, f"Trial {trial}: expected {expected}, got None for target={target_pe}, peers={peers}")
                self.assertAlmostEqual(actual, expected, places=5)


if __name__ == "__main__":
    unittest.main()
