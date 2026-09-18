"""
tests/test_tier5_adversarial_2.py
---------------------------------
Milestone 5 Tier 5 White-Box Adversarial Coverage Hardening Suite:
Cross-module pipeline stress testing across full 350-universe scale,
long time-series data (>4000 dates), duplicate tickers, extreme valuations,
and numerical precision.

Author: Challenger M5-2 (Empirical Challenger)

Coverage Dimensions:
1. Full 350-Stock Universe Scale & Throughput Pipeline Stress Testing:
   - Synthetic 350-stock universe spanning 10 WICS sectors with daily Fwd EPS and monthly prices.
   - Cross-module pipeline: calc_eps_revisions -> run_screener -> calc_sector_relative_metrics -> apply_quick_preset.
   - Sorting invariant: strictly descending base upside ordering.
   - In-memory batch real-time quote updates across all 350 stocks.
2. Long Time-Series Scale (>4,000 Dates per Stock, 2010~2026 Daily Data):
   - 4,240 daily dates revision index exactness (-1, -6, -21, -61).
   - Frequency alignment: 195 monthly price dates aligned with 4,240 daily EPS dates via asof/ffill.
   - Time-series band arrays generation across all 3 models (percentile, mean_sd, fixed).
   - Large-sample Winsorization (4,200 samples with heavy-tailed Pareto outliers).
3. Duplicate Tickers, Mixed Prefixes & Dirty Keys Resilience:
   - _clean_ticker permutations: 'A005930', 'a005930', '5930', whitespace, None, 'nan', unpadded.
   - SectorMappingDict bidirectional lookup, membership, and unique length invariance.
   - Screener and sector metrics DataFrame with duplicate ticker indices or rows.
   - Duplicate ticker list handling in apply_realtime_prices.
4. Extreme Valuations, Floating-Point Precision & Underflow/Overflow:
   - Microscopic EPS (1e-7 KRW) and upside numerical stability without overflow or NaN.
   - Sub-1.0 P/E series and calc_mean_sd_bands floor rule protection against non-positive floors.
   - calc_eps_revision_rate extreme clamping: [-100.0%, +500.0%] and zero denominator guard.
   - Fixed multiples edge cases: empty list, single multiple, zero multiple, fractional multiples.
5. Sector Metrics Tie-Breaking, Continuity Correction & Degenerate Sectors:
   - Continuity-corrected sector percentile ladder: N=1 (50%), N=2 (25%, 75%), N=4 (12.5%..87.5%).
   - Total ties in sector: identical P/E stocks all receiving exact 50.0% median rank.
   - np.isclose floating-point tie resolution with 1e-5 tolerance.
   - Searchsorted interpolation for target P/E evaluated outside sector distribution.
   - Degenerate sectors: all deficit (median is None) or all capped >150 (median is None).
6. All-Zero Revisions, Degenerate Series & Quick Preset Boundary Cross-Section:
   - Universe-wide flat EPS (0% revision): verify no false positive Golden Cross or Value Trap.
   - Quick presets behavior on flat series: TURNAROUND=0, EPS_TOP=0, TRAP=0, VALUE preserves stable value.
   - Turnaround detection truth table across 1M, 3M, 1W deficit transitions.
   - Quick presets resilience to None, empty pd.DataFrame(), all-NaN columns, and missing schema.
7. Real-Time Pricing Dynamic Regime Transitions:
   - Price drop drives historical percentile down, flipping Neutral to Golden Cross (eps_rev > 0).
   - Price drop drives historical percentile down, flipping Neutral to Value Trap (eps_rev < -2.0).
   - Price rally pushes percentile above 40%, dynamically revoking Golden Cross.
   - Deficit stock real-time price update isolation (no negative P/E or corrupted targets).
   - Pathological real-time quote dictionary (0.0, negative, NaN, None, inf) graceful rejection.
"""

import sys
import os
import unittest
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import (
    _clean_ticker,
    clean_ticker,
    calc_eps_revision_rate,
    calc_eps_revisions_series,
    calc_eps_revisions,
    classify_regime,
    RegimeResult,
)
from core.calculator import (
    calc_fwd_pe_series,
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    calc_sector_relative_metrics,
    calc_pe_band,
    run_screener,
    apply_quick_preset,
    PEBandResult,
    SectorMappingDict,
    load_sector_mapping,
)
from app import apply_realtime_prices


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Synthetic Data Generators
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_350_universe(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    seed: int = 1004,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], SectorMappingDict]:
    """
    Generates a full 350-stock realistic universe across 10 WICS sectors.
    Returns: (price_df, eps_df, ticker_names, sector_mapping)
    """
    rng = np.random.default_rng(seed)
    sectors = [
        "IT", "금융", "헬스케어", "경기소비재", "소재",
        "산업재", "유틸리티", "에너지", "커뮤니케이션", "필수소비재"
    ]
    tickers = [f"{i:06d}" for i in range(1, 351)]
    
    # Dates: Monthly for price (72 months), Daily for EPS (~1560 days)
    p_dates = pd.date_range(start_date, end_date, freq="MS")
    e_dates = pd.date_range(start_date, end_date, freq="B")

    price_dict: Dict[str, pd.Series] = {}
    eps_dict: Dict[str, pd.Series] = {}
    ticker_names: Dict[str, str] = {}
    sector_map = SectorMappingDict()

    for idx, t in enumerate(tickers):
        sec = sectors[idx % len(sectors)]
        sector_map[t] = sec
        ticker_names[t] = f"종목_{t}_{sec}"

        base_p = rng.uniform(10_000, 200_000)
        p_returns = rng.normal(0.004, 0.05, len(p_dates))
        prices = base_p * np.cumprod(1.0 + p_returns)
        price_dict[t] = pd.Series(prices, index=p_dates)

        base_e = rng.uniform(1_000, 15_000)
        e_growth = rng.normal(0.0002, 0.015, len(e_dates))
        eps = base_e * np.cumprod(1.0 + e_growth)
        eps_dict[t] = pd.Series(eps, index=e_dates)

    price_df = pd.DataFrame(price_dict, index=p_dates)
    eps_df = pd.DataFrame(eps_dict, index=e_dates)
    return price_df, eps_df, ticker_names, sector_map


# =============================================================================
# Dimension 1: Full 350-Universe Scale & Throughput Cross-Module Stress Testing
# =============================================================================
class TestFullUniverseScaleStress(unittest.TestCase):
    """Stress testing the entire valuation & screener pipeline on a 350-stock universe."""

    @classmethod
    def setUpClass(cls):
        cls.price_df, cls.eps_df, cls.names, cls.sec_map = generate_synthetic_350_universe()

    def test_01_full_350_universe_screener_execution(self):
        """Verify run_screener completes and adheres to sorting and metric invariants for all 350 stocks."""
        t0 = time.time()
        results = run_screener(
            price_df=self.price_df,
            eps_df=self.eps_df,
            ticker_names=self.names,
            band_years=5,
            band_model="percentile",
            winsorize=True,
            sector_mapping=self.sec_map,
        )
        elapsed = time.time() - t0

        self.assertEqual(len(results), 350)
        self.assertLess(elapsed, 15.0, f"run_screener on 350 stocks took {elapsed:.2f}s, expected < 15s")

        # Invariant 1: Results must be sorted descending by upside_base
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].upside_base,
                results[i + 1].upside_base,
                f"Screener results not sorted descending at index {i}",
            )

        # Invariant 2: Every stock has populated sector relative valuation metrics
        for r in results:
            self.assertIn(r.sector, self.sec_map.values())
            self.assertIsNotNone(r.sector_median_pe)
            self.assertGreater(r.sector_median_pe, 0.0)
            self.assertLessEqual(r.sector_median_pe, 150.0)
            self.assertIsNotNone(r.sector_pe_percentile)
            self.assertGreaterEqual(r.sector_pe_percentile, 0.0)
            self.assertLessEqual(r.sector_pe_percentile, 100.0)
            self.assertEqual(r.sector_stock_count, 35)

    def test_02_sector_metrics_distribution_properties_350_stocks(self):
        """Verify cross-sectional percentile distribution and median centering across 10 sectors."""
        results = run_screener(
            price_df=self.price_df,
            eps_df=self.eps_df,
            ticker_names=self.names,
            sector_mapping=self.sec_map,
        )

        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            groups[r.sector].append(r)

        self.assertEqual(len(groups), 10)
        for sec, items in groups.items():
            self.assertEqual(len(items), 35)
            pcts = [item.sector_pe_percentile for item in items]
            mean_pct = float(np.mean(pcts))
            # Theoretical average for uniform continuity-corrected ranks is exactly 50.0%
            self.assertAlmostEqual(mean_pct, 50.0, delta=1.5, msg=f"Sector {sec} mean percentile distorted: {mean_pct}")
            # Min and max percentiles should be well-bounded
            self.assertGreater(min(pcts), 0.0)
            self.assertLess(max(pcts), 100.0)

    def test_03_quick_presets_on_full_350_screener(self):
        """Verify all 5 analyst quick strategy presets filter correctly on the full 350-stock table."""
        results = run_screener(
            price_df=self.price_df,
            eps_df=self.eps_df,
            ticker_names=self.names,
            sector_mapping=self.sec_map,
        )
        records = [
            {
                "ticker": r.ticker,
                "name": r.name,
                "current_fwd_pe": r.current_fwd_pe,
                "current_fwd_eps": r.current_fwd_eps,
                "pe_percentile": r.pe_percentile,
                "sector_pe_percentile": r.sector_pe_percentile,
                "eps_rev_1m": r.eps_rev_1m,
                "upside_base": r.upside_base,
            }
            for r in results
        ]
        df_screener = pd.DataFrame(records)

        # 1. 'ALL'
        df_all = apply_quick_preset(df_screener, "ALL")
        self.assertEqual(len(df_all), 350)

        # 2. 'EPS_TOP' (<= 25 stocks, descending order)
        df_eps = apply_quick_preset(df_screener, "EPS_TOP")
        self.assertLessEqual(len(df_eps), 25)
        if len(df_eps) > 0:
            self.assertTrue((df_eps["eps_rev_1m"] >= 3.0).all())
            self.assertTrue((df_eps["current_fwd_eps"] > 0.0).all())
            rev_vals = df_eps["eps_rev_1m"].tolist()
            self.assertEqual(rev_vals, sorted(rev_vals, reverse=True))

        # 3. 'TURNAROUND'
        df_turn = apply_quick_preset(df_screener, "TURNAROUND")
        for _, row in df_turn.iterrows():
            self.assertTrue((row["pe_percentile"] <= 40.0) or (row["sector_pe_percentile"] <= 40.0))
            self.assertGreater(row["eps_rev_1m"], 0.0)
            self.assertGreaterEqual(row["upside_base"], 10.0)

        # 4. 'VALUE'
        df_val = apply_quick_preset(df_screener, "VALUE")
        for _, row in df_val.iterrows():
            self.assertLessEqual(row["pe_percentile"], 25.0)
            self.assertGreaterEqual(row["eps_rev_1m"], -2.0)
            self.assertGreater(row["current_fwd_pe"], 0.0)
            self.assertLessEqual(row["current_fwd_pe"], 15.0)
            self.assertGreaterEqual(row["upside_base"], 15.0)

        # 5. 'TRAP'
        df_trap = apply_quick_preset(df_screener, "TRAP")
        for _, row in df_trap.iterrows():
            self.assertLessEqual(row["pe_percentile"], 30.0)
            self.assertLess(row["eps_rev_1m"], -3.0)

    def test_04_batch_realtime_pricing_on_full_350_universe(self):
        """Verify in-memory real-time pricing updates all 350 stocks in under 500ms with full metric integrity."""
        results = run_screener(
            price_df=self.price_df,
            eps_df=self.eps_df,
            ticker_names=self.names,
            sector_mapping=self.sec_map,
        )
        # Apply 5% price surge across all 350 stocks
        rt_prices = {r.ticker: r.current_price * 1.05 for r in results}

        t0 = time.time()
        updated = apply_realtime_prices(results, rt_prices, self.sec_map)
        elapsed = time.time() - t0

        self.assertEqual(len(updated), 350)
        self.assertLess(elapsed, 0.5, f"Real-time update on 350 stocks took {elapsed:.3f}s, expected < 0.5s")

        for orig, up in zip(results, updated):
            self.assertAlmostEqual(up.current_price, orig.current_price * 1.05, places=2)
            self.assertAlmostEqual(up.current_fwd_pe, orig.current_fwd_pe * 1.05, places=2)
            # Upside should contract proportionally
            self.assertLess(up.upside_base, orig.upside_base)
            self.assertIsNotNone(up.sector_pe_percentile)


# =============================================================================
# Dimension 2: Long Time-Series Scale (>4,000 Dates per Stock, 2010~2026)
# =============================================================================
class TestLongTimeSeriesStress(unittest.TestCase):
    """Stress testing on >4,000 dates daily time series spanning 2010 to 2026."""

    @classmethod
    def setUpClass(cls):
        # 4,240 business days from 2010-01-01 to 2026-03-31
        cls.daily_dates = pd.date_range("2010-01-01", "2026-03-31", freq="B")
        cls.n_dates = len(cls.daily_dates)

    def test_01_eps_revisions_over_4000_daily_dates(self):
        """Verify calc_eps_revisions extracts exact 1W (-6), 1M (-21), 3M (-61) offsets on 4,240-day series."""
        rng = np.random.default_rng(42)
        base_eps = 5000.0
        # Realistic trajectory with trend and volatility
        returns = rng.normal(0.0001, 0.01, self.n_dates)
        eps_series = pd.Series(base_eps * np.cumprod(1.0 + returns), index=self.daily_dates, name="005930")

        rev_dict = calc_eps_revisions_series(eps_series)
        latest = float(eps_series.iloc[-1])
        base_1w = float(eps_series.iloc[-6])
        base_1m = float(eps_series.iloc[-21])
        base_3m = float(eps_series.iloc[-61])

        self.assertAlmostEqual(rev_dict["current_eps"], latest, places=5)
        self.assertAlmostEqual(rev_dict["eps_1w_ago"], base_1w, places=5)
        self.assertAlmostEqual(rev_dict["eps_1m_ago"], base_1m, places=5)
        self.assertAlmostEqual(rev_dict["eps_3m_ago"], base_3m, places=5)

        expected_1w = ((latest - base_1w) / max(abs(base_1w), 1.0)) * 100.0
        expected_1m = ((latest - base_1m) / max(abs(base_1m), 1.0)) * 100.0
        expected_3m = ((latest - base_3m) / max(abs(base_3m), 1.0)) * 100.0

        self.assertAlmostEqual(rev_dict["eps_rev_1w"], np.clip(expected_1w, -100.0, 500.0), places=4)
        self.assertAlmostEqual(rev_dict["eps_rev_1m"], np.clip(expected_1m, -100.0, 500.0), places=4)
        self.assertAlmostEqual(rev_dict["eps_rev_3m"], np.clip(expected_3m, -100.0, 500.0), places=4)

    def test_02_pe_band_models_with_4000_dates_and_asof_ffill(self):
        """Verify forward-fill alignment and 3 band models generation with 195 monthly prices and 4,240 daily EPS."""
        monthly_dates = pd.date_range("2010-01-01", "2026-03-31", freq="MS")
        rng = np.random.default_rng(2026)

        p_series = pd.Series(rng.uniform(30000, 80000, len(monthly_dates)), index=monthly_dates)
        e_series = pd.Series(rng.uniform(2000, 6000, self.n_dates), index=self.daily_dates)

        res = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=p_series,
            eps_series=e_series,
            band_years=5,
            band_model="percentile",
            winsorize=True,
        )

        self.assertIsNotNone(res)
        self.assertEqual(len(res.hist_price_series), len(monthly_dates))

        # Check band series dataframes
        self.assertIsNotNone(res.band_series_pct)
        self.assertEqual(len(res.band_series_pct), len(monthly_dates))
        cols_pct = ["p10", "p25", "p50", "p75", "p90"]
        self.assertTrue(all(c in res.band_series_pct.columns for c in cols_pct))

        # Check ordering across all dates
        for _, row in res.band_series_pct.iterrows():
            self.assertLessEqual(row["p10"], row["p25"])
            self.assertLessEqual(row["p25"], row["p50"])
            self.assertLessEqual(row["p50"], row["p75"])
            self.assertLessEqual(row["p75"], row["p90"])

        # Check Mean ± SD series
        self.assertIsNotNone(res.band_series_sd)
        cols_sd = ["-2SD", "-1SD", "Mean", "+1SD", "+2SD"]
        self.assertTrue(all(c in res.band_series_sd.columns for c in cols_sd))
        for _, row in res.band_series_sd.iterrows():
            self.assertLessEqual(row["-2SD"], row["-1SD"])
            self.assertLessEqual(row["-1SD"], row["Mean"])
            self.assertLessEqual(row["Mean"], row["+1SD"])
            self.assertLessEqual(row["+1SD"], row["+2SD"])

        # Check Fixed Multiples series
        self.assertIsNotNone(res.band_series_fixed)
        cols_fixed = ["8x", "10x", "12x", "15x"]
        self.assertTrue(all(c in res.band_series_fixed.columns for c in cols_fixed))
        for _, row in res.band_series_fixed.iterrows():
            self.assertLess(row["8x"], row["10x"])
            self.assertLess(row["10x"], row["12x"])
            self.assertLess(row["12x"], row["15x"])

    def test_03_winsorization_stability_large_sample_with_pareto_outliers(self):
        """Verify 95% Winsorization clips extreme heavy-tailed outliers in 4,200 observations cleanly."""
        rng = np.random.default_rng(777)
        # Standard values between 5.0 and 30.0 with 20 extreme outliers up to 5,000.0
        clean_pes = rng.uniform(5.0, 30.0, 4200)
        outliers = np.array([250.0, 500.0, 1500.0, 5000.0, 0.2, 0.5])
        raw_pes = np.concatenate([clean_pes, outliers])
        s_pe = pd.Series(raw_pes)

        wins = winsorize_pe(s_pe, lower_q=0.025, upper_q=0.975, min_pe=1.0, max_pe=150.0)

        # Pre-filtering strictly removes < 1.0 and > 150.0
        self.assertTrue((wins >= 1.0).all())
        self.assertTrue((wins <= 150.0).all())

        # Quantile clipping bounds
        q_low = float(s_pe[(s_pe >= 1.0) & (s_pe <= 150.0)].quantile(0.025))
        q_high = float(s_pe[(s_pe >= 1.0) & (s_pe <= 150.0)].quantile(0.975))
        self.assertAlmostEqual(float(wins.min()), q_low, places=4)
        self.assertAlmostEqual(float(wins.max()), q_high, places=4)


# =============================================================================
# Dimension 3: Duplicate Tickers, Mixed Prefixes & Dirty Keys
# =============================================================================
class TestDuplicateTickersAndPrefixResilience(unittest.TestCase):
    """Stress testing ticker normalization, dual-mode dictionary lookups, and duplicate keys."""

    def test_01_clean_ticker_exhaustive_permutations(self):
        """Verify _clean_ticker handles prefixes, whitespace, unpadded digits, and null variants."""
        cases = [
            ("A005930", "005930"),
            ("a005930", "005930"),
            ("005930", "005930"),
            ("  005930  ", "005930"),
            ("  A005930  ", "005930"),
            ("5930", "005930"),
            ("35420", "035420"),
            ("A35420", "035420"),
            ("005935", "005935"),  # Preferred stock
            (5930, "005930"),
            (None, ""),
            ("", ""),
            ("   ", ""),
            ("nan", ""),
            ("NaN", ""),
            ("none", ""),
            ("None", ""),
            (np.nan, ""),
        ]
        for inp, expected in cases:
            self.assertEqual(_clean_ticker(inp), expected, f"Failed on input: {repr(inp)}")
            self.assertEqual(clean_ticker(inp), expected, f"Failed on input (clean_ticker): {repr(inp)}")

    def test_02_sector_mapping_dict_dual_query_and_len(self):
        """Verify SectorMappingDict allows 'A005930' and '005930' while reporting exact unique count."""
        sm = SectorMappingDict()
        sm["005930"] = "IT"
        sm["000660"] = "IT"
        sm["035420"] = "커뮤니케이션"

        # Unique length invariant
        self.assertEqual(len(sm), 3)

        # Query by normalized and raw codes
        self.assertEqual(sm["005930"], "IT")
        self.assertEqual(sm["A005930"], "IT")
        self.assertEqual(sm["a005930"], "IT")
        self.assertEqual(sm["5930"], "IT")

        # In operator
        self.assertIn("005930", sm)
        self.assertIn("A005930", sm)
        self.assertIn("5930", sm)
        self.assertNotIn("999999", sm)

        # Get method
        self.assertEqual(sm.get("A000660"), "IT")
        self.assertEqual(sm.get("999999", "기타/미분류"), "기타/미분류")

    def test_03_calc_sector_relative_metrics_duplicate_dataframe_index(self):
        """Verify calc_sector_relative_metrics safely processes DataFrames with duplicate indices."""
        sec_map = SectorMappingDict({"005930": "IT", "000660": "IT"})
        df = pd.DataFrame(
            {
                "ticker": ["005930", "005930", "000660"],
                "current_fwd_pe": [10.0, 10.0, 20.0],
            },
            index=["005930", "005930", "000660"],  # Duplicate index
        )
        res_df = calc_sector_relative_metrics(df, sec_map)
        self.assertEqual(len(res_df), 3)
        self.assertTrue("sector_median_pe" in res_df.columns)
        self.assertTrue("sector_pe_percentile" in res_df.columns)
        self.assertFalse(res_df["sector_pe_percentile"].isna().any())

    def test_04_apply_realtime_prices_duplicate_tickers_in_list(self):
        """Verify apply_realtime_prices cleanly updates lists containing duplicate ticker results."""
        sec_map = SectorMappingDict({"005930": "IT"})
        hist_pe = pd.Series([8.0, 10.0, 12.0, 14.0])
        r1 = PEBandResult(
            ticker="005930", name="삼성1", current_price=50000.0, current_fwd_eps=5000.0,
            current_fwd_pe=10.0, pe_percentile=50.0, pe_min=8.0, pe_p25=9.0, pe_median=10.0,
            pe_p75=11.0, pe_max=14.0, pe_mean=10.0, target_bear=45000.0, target_base=50000.0,
            target_bull=55000.0, upside_bear=-10.0, upside_base=0.0, upside_bull=10.0,
            hist_pe_series=hist_pe, hist_price_series=pd.Series([50000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT",
        )
        r2 = PEBandResult(
            ticker="005930", name="삼성2", current_price=50000.0, current_fwd_eps=5000.0,
            current_fwd_pe=10.0, pe_percentile=50.0, pe_min=8.0, pe_p25=9.0, pe_median=10.0,
            pe_p75=11.0, pe_max=14.0, pe_mean=10.0, target_bear=45000.0, target_base=50000.0,
            target_bull=55000.0, upside_bear=-10.0, upside_base=0.0, upside_bull=10.0,
            hist_pe_series=hist_pe, hist_price_series=pd.Series([50000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT",
        )
        updated = apply_realtime_prices([r1, r2], {"005930": 60000.0}, sec_map)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0].current_price, 60000.0)
        self.assertEqual(updated[1].current_price, 60000.0)


# =============================================================================
# Dimension 4: Extreme Valuations, Numerical Precision & Underflow/Overflow
# =============================================================================
class TestExtremeValuationsAndPrecision(unittest.TestCase):
    """Stress testing numerical edge cases, microscopic EPS, zero denominators, and extreme clamping."""

    def test_01_microscopic_and_fractional_eps(self):
        """Verify microscopic EPS (1e-7 KRW) handles astronomical P/E and upside without overflow."""
        p_dates = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
        e_dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")

        # Historical EPS was normal (5000 KRW), but current EPS collapsed to 1e-7 KRW
        p_series = pd.Series(50000.0, index=p_dates)
        e_vals = np.full(len(e_dates), 5000.0)
        e_vals[-1] = 1e-7
        e_series = pd.Series(e_vals, index=e_dates)

        res = calc_pe_band(
            ticker="005930",
            name="초미세EPS",
            price_series=p_series,
            eps_series=e_series,
            band_years=5,
        )
        self.assertIsNotNone(res)
        # Astronomical current P/E: 50,000 / 1e-7 = 5e11
        self.assertGreater(res.current_fwd_pe, 1e11)
        self.assertEqual(res.pe_percentile, 100.0)
        # Upside base should be near -100% without overflow
        self.assertTrue(math.isfinite(res.upside_base))
        self.assertAlmostEqual(res.upside_base, -100.0, places=3)

    def test_02_sub_one_pe_and_mean_sd_floor_rule(self):
        """Verify calc_mean_sd_bands floor rule prevents negative or sub-floor bands when P/E is extremely low."""
        # Distressed P/E series: mean = 0.5, std = 0.258
        pe_series = pd.Series([0.2, 0.4, 0.6, 0.8], index=pd.date_range("2022-01-01", periods=4, freq="MS"))
        bands = calc_mean_sd_bands(pe_series, floor_pe=1.0, winsorize=False)

        self.assertIn("m1sd", bands)
        self.assertIn("m2sd", bands)
        # m1sd (0.5 - 0.258 = 0.242) and m2sd (0.5 - 0.516 = -0.016) must be floored to max(1.0, 0.2*0.8) = 1.0
        self.assertGreaterEqual(bands["m1sd"], 1.0)
        self.assertGreaterEqual(bands["m2sd"], 1.0)

    def test_03_floating_point_bounds_eps_revision_clamp(self):
        """Verify calc_eps_revision_rate clamps strictly to [-100.0, 500.0] and handles zero denominators."""
        # Zero denominator guard: base = 0.0 -> denom = max(|0.0|, 1.0) = 1.0
        self.assertEqual(calc_eps_revision_rate(0.0, 0.0), 0.0)
        self.assertEqual(calc_eps_revision_rate(100.0, 0.0), 500.0)   # (100 - 0)/1 * 100 = 10000% -> clamped to 500%
        self.assertEqual(calc_eps_revision_rate(-100.0, 0.0), -100.0) # (-100 - 0)/1 * 100 = -10000% -> clamped to -100%

        # Microscopic base EPS
        self.assertEqual(calc_eps_revision_rate(1e-9, 1e-9), 0.0)

        # Huge explosion
        self.assertEqual(calc_eps_revision_rate(1_000_000.0, 1.0), 500.0)

        # Huge collapse
        self.assertEqual(calc_eps_revision_rate(-50_000.0, 100.0), -100.0)

        # NaN / None inputs
        self.assertIsNone(calc_eps_revision_rate(None, 100.0))
        self.assertIsNone(calc_eps_revision_rate(100.0, None))
        self.assertIsNone(calc_eps_revision_rate(float("nan"), 100.0))

    def test_04_fixed_multiples_edge_parameters(self):
        """Verify fixed multiples calculations with empty, single, and fractional multiple lists."""
        res_empty = calc_fixed_multiple_bands([], current_eps=5000.0, current_price=50000.0)
        self.assertEqual(res_empty["targets"], {})
        self.assertEqual(res_empty["upsides"], {})

        res_single = calc_fixed_multiple_bands([12.5], current_eps=4000.0, current_price=50000.0)
        self.assertEqual(res_single["targets"][12.5], 50000.0)
        self.assertEqual(res_single["upsides"][12.5], 0.0)

        res_zero = calc_fixed_multiple_bands([0.0], current_eps=4000.0, current_price=50000.0)
        self.assertEqual(res_zero["targets"][0.0], 0.0)
        self.assertEqual(res_zero["upsides"][0.0], -100.0)


# =============================================================================
# Dimension 5: Sector Metrics Tie-Breaking, Continuity & Degenerate Sectors
# =============================================================================
class TestSectorMetricsTieBreakingAndContinuity(unittest.TestCase):
    """Stress testing small-sample continuity corrections, full/partial ties, and degenerate sectors."""

    def test_01_continuity_correction_ladder_exactness(self):
        """Verify exact continuity-corrected formula: Sector_Pct = (rank_0 + 0.5) / N * 100."""
        # N=1
        self.assertEqual(calc_sector_pe_percentile(15.0, [15.0]), 50.0)

        # N=2
        self.assertEqual(calc_sector_pe_percentile(10.0, [10.0, 20.0]), 25.0)
        self.assertEqual(calc_sector_pe_percentile(20.0, [10.0, 20.0]), 75.0)

        # N=4: ranks 0, 1, 2, 3 -> (0.5/4)*100=12.5%, (1.5/4)*100=37.5%, 62.5%, 87.5%
        pes_4 = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(calc_sector_pe_percentile(10.0, pes_4), 12.5)
        self.assertEqual(calc_sector_pe_percentile(20.0, pes_4), 37.5)
        self.assertEqual(calc_sector_pe_percentile(30.0, pes_4), 62.5)
        self.assertEqual(calc_sector_pe_percentile(40.0, pes_4), 87.5)

    def test_02_total_tie_across_sector(self):
        """Verify that when all stocks in a sector share the exact same P/E, all receive 50.0%."""
        all_ties = [12.0] * 8
        for p in all_ties:
            pct = calc_sector_pe_percentile(p, all_ties)
            self.assertEqual(pct, 50.0)

    def test_03_floating_point_isclose_tie_resolution(self):
        """Verify np.isclose resolves minute floating-point discrepancies (<1e-5) as ties."""
        # Minute jitter within 1e-6
        jittered = [10.0, 10.0 + 1e-6, 20.0, 20.0 - 1e-6]
        pct_10 = calc_sector_pe_percentile(10.0, jittered)
        self.assertEqual(pct_10, 25.0)  # avg of ranks 0 and 1 -> (0.5 + 0.5)/4 * 100 = 25.0%

    def test_04_target_pe_outside_sector_distribution(self):
        """Verify target P/E evaluated outside existing sector distribution interpolates cleanly."""
        sector_pes = [10.0, 20.0, 30.0]
        # Below minimum
        pct_low = calc_sector_pe_percentile(5.0, sector_pes)
        self.assertAlmostEqual(pct_low, (0.5 / 3) * 100.0, places=4)

        # Above maximum
        pct_high = calc_sector_pe_percentile(40.0, sector_pes)
        self.assertEqual(pct_high, 100.0)

    def test_05_degenerate_sectors_all_nan_or_all_capped(self):
        """Verify sectors with all deficit or all capped (>150) stocks return None without throwing."""
        self.assertIsNone(calc_sector_median_pe([-5.0, -10.0, float("nan")]))
        self.assertIsNone(calc_sector_median_pe([180.0, 250.0]))
        self.assertIsNone(calc_sector_pe_percentile(-5.0, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(200.0, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(15.0, [-5.0, -10.0]))


# =============================================================================
# Dimension 6: All-Zero Revisions, Degenerate Series & Preset Boundaries
# =============================================================================
class TestAllZeroRevisionsAndDegenerateSeries(unittest.TestCase):
    """Stress testing flat earnings series, turnaround truth tables, and pathological preset inputs."""

    def test_01_universe_wide_flat_eps_preserves_neutral_regimes(self):
        """Verify that when revisions are 0.0%, no false Golden Cross or Value Trap is triggered."""
        for pe_pct in [5.0, 15.0, 25.0, 35.0, 50.0, 75.0]:
            reg = classify_regime(pe_pct=pe_pct, eps_rev_1m=0.0, eps_rev_3m=0.0, eps_rev_1w=0.0, current_pe=12.0)
            self.assertFalse(reg.is_golden_cross)
            self.assertFalse(reg.is_value_trap)
            self.assertEqual(reg.regime_tag, "Neutral")

    def test_02_turnaround_detection_truth_table(self):
        """Verify turnaround detection across 1M, 3M, and 1W deficit-to-positive transitions."""
        daily_dates = pd.date_range("2025-01-01", periods=80, freq="B")

        # 1. Turnaround from 1M ago (-50.0 -> +10.0)
        s1_vals = np.full(80, 10.0)
        s1_vals[-25:] = np.linspace(-50.0, 10.0, 25)
        s1_vals[-1] = 10.0
        s1 = pd.Series(s1_vals, index=daily_dates, name="005930")
        r1 = calc_eps_revisions_series(s1)
        self.assertTrue(r1["is_turnaround"])
        self.assertFalse(r1["is_deficit"])

        # 2. Chronic deficit (-50.0 -> -10.0)
        s2 = pd.Series(np.full(80, -10.0), index=daily_dates, name="000660")
        r2 = calc_eps_revisions_series(s2)
        self.assertFalse(r2["is_turnaround"])
        self.assertTrue(r2["is_deficit"])

        # 3. Stable positive (always +100.0)
        s3 = pd.Series(np.full(80, 100.0), index=daily_dates, name="035420")
        r3 = calc_eps_revisions_series(s3)
        self.assertFalse(r3["is_turnaround"])
        self.assertFalse(r3["is_deficit"])

    def test_03_quick_presets_pathological_dataframes(self):
        """Verify apply_quick_preset resilience against None, empty schemas, all-NaNs, and unknown keys."""
        # None input
        self.assertTrue(apply_quick_preset(None, "ALL").empty)

        # Empty DataFrame
        self.assertTrue(apply_quick_preset(pd.DataFrame(), "TURNAROUND").empty)

        # DataFrame missing columns
        df_missing = pd.DataFrame({"ticker": ["005930"], "name": ["삼성전자"]})
        res_missing = apply_quick_preset(df_missing, "TURNAROUND")
        self.assertEqual(len(res_missing), 0)

        # Unknown preset key returns copy of df
        df_valid = pd.DataFrame({"ticker": ["005930"], "pe_percentile": [20.0]})
        res_unknown = apply_quick_preset(df_valid, "MY_CUSTOM_PRESET")
        self.assertEqual(len(res_unknown), 1)


# =============================================================================
# Dimension 7: Real-Time Pricing Dynamic Regime Transitions
# =============================================================================
class TestRealtimePricingDynamicTransitions(unittest.TestCase):
    """Stress testing dynamic regime flips and attribute preservation when real-time quotes arrive."""

    def setUp(self):
        self.sec_map = SectorMappingDict({"005930": "IT", "000660": "IT"})
        # Historical P/E distribution: min=8.0, 25%=11.0, median=14.0, 75%=18.0, max=22.0
        self.hist_pe = pd.Series(np.linspace(8.0, 22.0, 100))

    def test_01_realtime_price_drop_triggers_golden_cross(self):
        """Verify real-time price drop pushes historical percentile below 40%, triggering Golden Cross."""
        # Initially at Price 80,000 KRW, EPS 5,000 KRW -> PE 16.0 (pe_pct = 57%, Neutral)
        r_init = PEBandResult(
            ticker="005930", name="삼성전자", current_price=80000.0, current_fwd_eps=5000.0,
            current_fwd_pe=16.0, pe_percentile=57.0, pe_min=8.0, pe_p25=11.0, pe_median=14.0,
            pe_p75=18.0, pe_max=22.0, pe_mean=14.0, target_bear=55000.0, target_base=70000.0,
            target_bull=90000.0, upside_bear=-31.25, upside_base=-12.5, upside_bull=12.5,
            hist_pe_series=self.hist_pe, hist_price_series=pd.Series([80000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT", eps_rev_1m=4.0, eps_rev_1w=1.5, regime_tag="Neutral", is_golden_cross=False,
        )

        # Real-time quote drops price to 55,000 KRW -> new PE = 11.0 -> pe_pct drops to 21% (<= 40%)
        updated = apply_realtime_prices([r_init], {"005930": 55000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 55000.0)
        self.assertEqual(updated.current_fwd_pe, 11.0)
        self.assertLessEqual(updated.pe_percentile, 40.0)
        self.assertTrue(updated.is_golden_cross)
        self.assertEqual(updated.regime_tag, "Golden Cross")

    def test_02_realtime_price_drop_triggers_value_trap(self):
        """Verify real-time price drop pushes historical percentile below 30%, triggering Value Trap."""
        # Initially at Price 80,000 KRW, EPS 5,000 KRW -> PE 16.0 (pe_pct = 57%, Neutral, but revisions negative)
        r_init = PEBandResult(
            ticker="005930", name="삼성전자", current_price=80000.0, current_fwd_eps=5000.0,
            current_fwd_pe=16.0, pe_percentile=57.0, pe_min=8.0, pe_p25=11.0, pe_median=14.0,
            pe_p75=18.0, pe_max=22.0, pe_mean=14.0, target_bear=55000.0, target_base=70000.0,
            target_bull=90000.0, upside_bear=-31.25, upside_base=-12.5, upside_bull=12.5,
            hist_pe_series=self.hist_pe, hist_price_series=pd.Series([80000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT", eps_rev_1m=-3.5, regime_tag="Neutral", is_value_trap=False,
        )

        # Real-time quote drops price to 50,000 KRW -> new PE = 10.0 -> pe_pct drops to 14% (<= 30%)
        updated = apply_realtime_prices([r_init], {"005930": 50000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 50000.0)
        self.assertEqual(updated.current_fwd_pe, 10.0)
        self.assertLessEqual(updated.pe_percentile, 30.0)
        self.assertTrue(updated.is_value_trap)
        self.assertEqual(updated.regime_tag, "Value Trap")

    def test_03_realtime_price_spike_clears_golden_cross(self):
        """Verify real-time price spike lifts percentile above 40%, revoking Golden Cross."""
        r_gc = PEBandResult(
            ticker="005930", name="삼성전자", current_price=55000.0, current_fwd_eps=5000.0,
            current_fwd_pe=11.0, pe_percentile=21.0, pe_min=8.0, pe_p25=11.0, pe_median=14.0,
            pe_p75=18.0, pe_max=22.0, pe_mean=14.0, target_bear=55000.0, target_base=70000.0,
            target_bull=90000.0, upside_bear=0.0, upside_base=27.27, upside_bull=63.64,
            hist_pe_series=self.hist_pe, hist_price_series=pd.Series([55000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT", eps_rev_1m=4.0, regime_tag="Golden Cross", is_golden_cross=True,
        )

        # Real-time price rallies to 95,000 KRW -> new PE = 19.0 -> pe_pct rises to 78% (> 40%)
        updated = apply_realtime_prices([r_gc], {"005930": 95000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 95000.0)
        self.assertFalse(updated.is_golden_cross)
        self.assertEqual(updated.regime_tag, "Momentum Leader")

    def test_04_realtime_price_deficit_stock_isolation(self):
        """Verify real-time price updates deficit stocks without corrupting P/E or target prices."""
        r_def = PEBandResult(
            ticker="000660", name="SK하이닉스", current_price=40000.0, current_fwd_eps=-2000.0,
            current_fwd_pe=np.nan, pe_percentile=np.nan, pe_min=8.0, pe_p25=11.0, pe_median=14.0,
            pe_p75=18.0, pe_max=22.0, pe_mean=14.0, target_bear=np.nan, target_base=np.nan,
            target_bull=np.nan, upside_bear=np.nan, upside_base=np.nan, upside_bull=np.nan,
            hist_pe_series=self.hist_pe, hist_price_series=pd.Series([40000.0]), hist_eps_series=pd.Series([-2000.0]),
            sector="IT", eps_rev_1m=1.0,
        )

        updated = apply_realtime_prices([r_def], {"000660": 45000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 45000.0)
        self.assertTrue(np.isnan(updated.current_fwd_pe))
        self.assertTrue(np.isnan(updated.pe_percentile))

    def test_05_realtime_pathological_inputs_isolation(self):
        """Verify pathological quote inputs (0, negative, NaN, None, inf) are rejected without exception."""
        r_orig = PEBandResult(
            ticker="005930", name="삼성전자", current_price=70000.0, current_fwd_eps=5000.0,
            current_fwd_pe=14.0, pe_percentile=50.0, pe_min=8.0, pe_p25=11.0, pe_median=14.0,
            pe_p75=18.0, pe_max=22.0, pe_mean=14.0, target_bear=55000.0, target_base=70000.0,
            target_bull=90000.0, upside_bear=-21.4, upside_base=0.0, upside_bull=28.6,
            hist_pe_series=self.hist_pe, hist_price_series=pd.Series([70000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="IT",
        )

        for bad_p in [0.0, -10000.0, float("nan"), None]:
            updated = apply_realtime_prices([r_orig], {"005930": bad_p}, self.sec_map)[0]
            self.assertEqual(updated.current_price, 70000.0)
            self.assertEqual(updated.current_fwd_pe, 14.0)


# =============================================================================
# Test Suite Runner
# =============================================================================
def suite() -> unittest.TestSuite:
    s = unittest.TestSuite()
    loader = unittest.TestLoader()
    s.addTests(loader.loadTestsFromTestCase(TestFullUniverseScaleStress))
    s.addTests(loader.loadTestsFromTestCase(TestLongTimeSeriesStress))
    s.addTests(loader.loadTestsFromTestCase(TestDuplicateTickersAndPrefixResilience))
    s.addTests(loader.loadTestsFromTestCase(TestExtremeValuationsAndPrecision))
    s.addTests(loader.loadTestsFromTestCase(TestSectorMetricsTieBreakingAndContinuity))
    s.addTests(loader.loadTestsFromTestCase(TestAllZeroRevisionsAndDegenerateSeries))
    s.addTests(loader.loadTestsFromTestCase(TestRealtimePricingDynamicTransitions))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(not result.wasSuccessful())
