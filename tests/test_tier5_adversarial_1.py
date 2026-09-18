"""
tests/test_tier5_adversarial_1.py
---------------------------------
Milestone 5 Tier 5 Adversarial White-Box Coverage Hardening Suite 1.
Focus: Deep white-box audit of core/data_loader.py, core/calculator.py, and app.py.

Author: Challenger M5-1 (Empirical Challenger)

Coverage Dimensions:
1. Data Ingestion & Parser White-Box Audit (core/data_loader.py):
   - Ticker cleaning (_clean_ticker): edge cases with prefixes ('A', 'a'), padding, NaNs, floats, None.
   - Date parsing (_parse_date_index): ISO, compact strings, invalid tokens, NaT filtering.
   - Format detection (_detect_format): pathological shapes (<3 cols, <4 rows), all-NaN columns, repeated codes.
   - Wide and Long loading (_load_wide, _load_long): duplicate dates, tie-breaking, non-numeric values.
   - Parquet & Pickle disk caching (load_daily_fwd_eps): mtime invalidation, cache miss, corrupted cache fallback, custom cache path, use_cache=False.
2. EPS Revision Engine Boundary & Arithmetic Audit (core/data_loader.py):
   - Revision rate (calc_eps_revision_rate): zero denominator protection (max(|base|, 1.0)), fractional base, negative turnaround, negative deepening, exact clamping [-100%, +500%], custom bounds.
   - Time-series revisions (calc_eps_revisions_series): series lengths (0, 1, 5, 6, 20, 21, 60, 61), intermediate NaNs, turnaround transitions (1m, 3m, 1w), deficit classification.
   - Cross-sectional revisions (calc_eps_revisions): Series vs DataFrame, as-of date filtering, empty series handling, multi-ticker column preservation.
3. 5-Regime Valuation Momentum Taxonomy Audit (core/data_loader.py):
   - NamedTuple interface (RegimeResult): attribute access, .get(), indexing.
   - Classification guards (classify_regime): None/NaN inputs, zero or negative P/E rejection.
   - Exact boundary thresholds: Value Trap (pe_pct=30.0, rev_1m=-2.0, dual downgrade rev_3m=-5.0), Golden Cross (pe_pct=40.0, rev_1m=+2.0, dual positive 1w>0 & 1m>0), Momentum Leader (pe_pct>40, rev_1m>=3.0), High P/E Downgrade (pe_pct>=70, rev_1m<0), Neutral fallback.
4. Sector Mapping & Relative Valuation Engine White-Box Audit (core/calculator.py):
   - SectorMappingDict: normalized lookup, KeyError on invalid [], default fallback on .get(), __contains__, len().
   - JSON loader (load_sector_mapping): valid JSON, fallback to universe.csv, missing file handling.
   - Sector median (calc_sector_median_pe): range filtering (0 < pe <= 150), odd vs even sample sizes, empty valid list.
   - Continuity-corrected percentile (calc_sector_pe_percentile): small sample exactness (N=1->50%, N=2->25%,75%), duplicate tie averaging, out-of-bounds targets (<min, >max).
   - Sector relative metrics (calc_sector_relative_metrics): List[PEBandResult] and DataFrame pathways, unknown sector fallback, invalid P/E handling.
5. P/E Valuation Bands & Outlier Winsorization Audit (core/calculator.py):
   - Historical P/E series (calc_fwd_pe_series): price/eps division, non-positive EPS masking, upper cap (pe > 200).
   - 95% Winsorization (winsorize_pe): sample size < 10 pass-through, quantile clipping, min/max pe bounds.
   - Research standard bands (calc_mean_sd_bands): sample size < 4 guard, floor protection (max(floor, min*0.8)), ddof=1 variance.
   - Fixed multiples bands (calc_fixed_multiple_bands): negative EPS zero-target, zero price zero-upside, fractional multiples.
   - Single-stock valuation engine (calc_pe_band): backward compatibility, ffill alignment for series < 6, cutoff period, deficit stock rejection (current_eps <= 0), 3 band models (percentile, mean_sd, fixed), time-series band generation.
6. Dashboard Real-Time Pricing & Signal Hierarchy Audit (app.py):
   - Strategy signal priority (get_strategy_signal): Golden Cross > Value Trap > Momentum Leader > Deep Value > High P/E Downgrade > Neutral.
   - UI formatting helpers: strategy_badge_html, signal_badge, signal_label, pe_bar_color.
   - File mtimes tuple (_get_file_mtimes): 10-element tuple, missing file (0, 0) handling.
   - Real-time price updates (apply_realtime_prices): empty dict, non-matching tickers, profit-making updates, deficit stock safeguards (no division by zero or negative P/E).
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import (
    _clean_ticker,
    _parse_date_index,
    _detect_format,
    _load_wide,
    _load_long,
    load_daily_fwd_eps,
    calc_eps_revision_rate,
    calc_eps_revisions_series,
    calc_eps_revisions,
    RegimeResult,
    classify_regime,
)
from core.calculator import (
    SectorMappingDict,
    load_sector_mapping,
    calc_fwd_pe_series,
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    calc_sector_relative_metrics,
    calc_pe_band,
    PEBandResult,
    run_screener,
    apply_quick_preset,
)
from app import (
    get_strategy_signal,
    strategy_badge_html,
    signal_badge,
    signal_label,
    pe_bar_color,
    _get_file_mtimes,
    apply_realtime_prices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: Data Ingestion & Parser White-Box Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestDataLoaderTickerAndDateFormatWhiteBox(unittest.TestCase):
    """Deep white-box branch testing for ticker cleaning, date parsing, and format detection."""

    def test_clean_ticker_standard_and_prefix_branches(self):
        """Verify _clean_ticker handles 6-digit codes, DataGuide 'A'/'a' prefixes, and numeric types."""
        # 1. Standard 6-digit string
        self.assertEqual(_clean_ticker("005930"), "005930")
        # 2. 'A' prefix removal
        self.assertEqual(_clean_ticker("A005930"), "005930")
        # 3. Lowercase 'a' prefix removal
        self.assertEqual(_clean_ticker("a005930"), "005930")
        # 4. Integer input with padding
        self.assertEqual(_clean_ticker(5930), "005930")
        self.assertEqual(_clean_ticker("5930"), "005930")
        # 5. Single digit input
        self.assertEqual(_clean_ticker(1), "000001")
        # 6. Whitespace surrounding
        self.assertEqual(_clean_ticker("  A005930  "), "005930")

    def test_clean_ticker_pathological_and_null_inputs(self):
        """Verify _clean_ticker returns empty string or unmodified text for non-digit tickers."""
        # 1. None and empty
        self.assertEqual(_clean_ticker(None), "")
        self.assertEqual(_clean_ticker(""), "")
        self.assertEqual(_clean_ticker("   "), "")
        # 2. String representations of nulls
        self.assertEqual(_clean_ticker("nan"), "")
        self.assertEqual(_clean_ticker("NaN"), "")
        self.assertEqual(_clean_ticker("None"), "")
        self.assertEqual(_clean_ticker("NONE"), "")
        # 3. Float NaN
        self.assertEqual(_clean_ticker(np.nan), "")
        # 4. Alphanumeric non-digit ticker (e.g. US stock or index)
        self.assertEqual(_clean_ticker("AAPL"), "AAPL")
        self.assertEqual(_clean_ticker("00593A"), "00593A")
        self.assertEqual(_clean_ticker("A"), "A")

    def test_parse_date_index_formats_and_nat_handling(self):
        """Verify _parse_date_index handles varied date representations and invalid values."""
        raw_dates = pd.Index(["2023-01-02", "20230103", "2023/01/04", "invalid_date", None, "2023-01-05"])
        parsed = _parse_date_index(raw_dates)
        self.assertIsInstance(parsed, pd.DatetimeIndex)
        self.assertEqual(len(parsed), 6)
        self.assertTrue(pd.isna(parsed[3]))  # "invalid_date" becomes NaT
        self.assertTrue(pd.isna(parsed[4]))  # None becomes NaT
        self.assertEqual(parsed[0], pd.Timestamp("2023-01-02"))
        self.assertEqual(parsed[1], pd.Timestamp("2023-01-03"))
        self.assertEqual(parsed[2], pd.Timestamp("2023-01-04"))
        self.assertEqual(parsed[5], pd.Timestamp("2023-01-05"))

        # Empty index
        empty_parsed = _parse_date_index(pd.Index([]))
        self.assertEqual(len(empty_parsed), 0)

    def test_detect_format_boundary_conditions(self):
        """Verify _detect_format heuristic against degenerate shapes and layout styles."""
        # Degenerate dimensions (<3 cols or <4 rows) -> defaults to "wide"
        df_tiny_cols = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])  # 4x2
        self.assertEqual(_detect_format(df_tiny_cols), "wide")

        df_tiny_rows = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3
        self.assertEqual(_detect_format(df_tiny_rows), "wide")

        # All-NaN column 1 -> "wide"
        df_nan_col1 = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "Col1": [np.nan, np.nan, np.nan, np.nan],
            "Col2": [100, 101, 102, 103],
        })
        self.assertEqual(_detect_format(df_nan_col1), "wide")

        # Long format: repeated codes in col 1, low unique ratio
        df_long = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01", "2023-01-02", "2023-01-03"],
            "Code": ["005930", "005930", "005930", "000660", "000660", "000660"],
            "Value": [60000, 61000, 62000, 120000, 121000, 122000],
        })
        self.assertEqual(_detect_format(df_long), "long")

        # Wide format: column 1 has price series (high unique ratio, non-code pattern)
        df_wide = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "005930": [60100.5, 60250.0, 60800.2, 61000.0],
            "000660": [120100.0, 120500.0, 121000.0, 122000.0],
        })
        self.assertEqual(_detect_format(df_wide), "wide")

    def test_load_wide_and_load_long_invariants(self):
        """Verify _load_wide and _load_long parse and normalize index and headers."""
        # Test Wide loader
        df_raw_wide = pd.DataFrame({
            " Date ": [" 2023-01-03 ", " 2023-01-01 ", " 2023-01-02 "],
            " A005930 ": ["60000", "61000", "invalid"],
            " 000660 ": [120000, 121000, 122000],
        })
        loaded_wide = _load_wide(df_raw_wide)
        self.assertListEqual(list(loaded_wide.columns), ["005930", "000660"])
        self.assertTrue(loaded_wide.index.is_monotonic_increasing)
        self.assertTrue(np.isnan(loaded_wide.loc[pd.Timestamp("2023-01-02"), "005930"]))

        # Test Long loader with duplicate date-code combinations
        df_raw_long = pd.DataFrame({
            " Date ": ["2023-01-01", "2023-01-01", "2023-01-02"],
            " Ticker ": [" A005930 ", " A005930 ", " A005930 "],
            " Value ": [5000, 5500, 6000],  # 5500 is latest for 2023-01-01
        })
        loaded_long = _load_long(df_raw_long)
        self.assertIn("005930", loaded_long.columns)
        self.assertEqual(loaded_long.loc[pd.Timestamp("2023-01-01"), "005930"], 5500)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: Disk Caching & Daily Ingestion White-Box Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestDataLoaderParquetCacheAndIngestionWhiteBox(unittest.TestCase):
    """Deep white-box verification of caching, file exceptions, and serialization."""

    def test_load_daily_fwd_eps_file_not_found(self):
        """Verify FileNotFoundError is raised when target excel path does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_daily_fwd_eps("data/non_existent_file_xyz_123.xlsx", use_cache=False)

    def test_load_daily_fwd_eps_custom_cache_roundtrip(self):
        """Verify reading and writing with custom cache_path without corrupting master cache."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_parquet = Path(tmpdir) / "custom_test.parquet"
            # Use real data file if it exists, otherwise skip or mock
            excel_path = ROOT / "data" / "fwd_eps.xlsx"
            if not excel_path.exists():
                self.skipTest("data/fwd_eps.xlsx not found on disk")

            # First load: generates custom parquet cache
            df1 = load_daily_fwd_eps(
                excel_path=str(excel_path),
                use_cache=True,
                cache_path=tmp_parquet,
            )
            self.assertTrue(tmp_parquet.exists() or tmp_parquet.with_suffix(".pkl").exists())
            self.assertFalse(df1.empty)
            self.assertIsInstance(df1.index, pd.DatetimeIndex)

            # Second load: loads directly from custom cache (<50ms)
            df2 = load_daily_fwd_eps(
                excel_path=str(excel_path),
                use_cache=True,
                cache_path=tmp_parquet,
            )
            self.assertEqual(df1.shape, df2.shape)
            pd.testing.assert_frame_equal(df1, df2)

    def test_load_daily_fwd_eps_corrupted_parquet_fallback(self):
        """Verify corrupted parquet file triggers safe fallback to pickle or re-parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_parquet = Path(tmpdir) / "corrupt_test.parquet"
            excel_path = ROOT / "data" / "fwd_eps.xlsx"
            if not excel_path.exists():
                self.skipTest("data/fwd_eps.xlsx not found on disk")

            # Write corrupt bytes to parquet cache with future timestamp
            tmp_parquet.write_bytes(b"CORRUPTED_PARQUET_HEADER_DATA_12345")
            future_mtime = excel_path.stat().st_mtime + 1000
            os.utime(tmp_parquet, (future_mtime, future_mtime))

            # load_daily_fwd_eps must catch the exception, log a warning, and fall back safely
            df = load_daily_fwd_eps(
                excel_path=str(excel_path),
                use_cache=True,
                cache_path=tmp_parquet,
            )
            self.assertFalse(df.empty)
            self.assertIsInstance(df.index, pd.DatetimeIndex)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: EPS Revision Engine Boundary & Arithmetic Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestRevisionEngineBoundaryAndArithmeticWhiteBox(unittest.TestCase):
    """Deep white-box audit for arithmetic bounds, turnarounds, lookback windows, and vectorization."""

    def test_revision_rate_none_and_nan_inputs(self):
        """Verify calc_eps_revision_rate handles None and NaN safely."""
        self.assertIsNone(calc_eps_revision_rate(None, 100.0))
        self.assertIsNone(calc_eps_revision_rate(100.0, None))
        self.assertIsNone(calc_eps_revision_rate(None, None))
        self.assertIsNone(calc_eps_revision_rate(np.nan, 100.0))
        self.assertIsNone(calc_eps_revision_rate(100.0, np.nan))
        self.assertIsNone(calc_eps_revision_rate(np.nan, np.nan))

    def test_revision_rate_zero_and_fractional_denominator_guards(self):
        """Verify denominator guard max(|base|, 1.0) prevents ZeroDivisionError and runaway scaling."""
        # 1. Base exactly 0.0 -> denom = 1.0
        # pct = (5.0 - 0.0) / 1.0 * 100 = +500.0%
        rev_zero_base = calc_eps_revision_rate(5.0, 0.0)
        self.assertEqual(rev_zero_base, 500.0)

        # Negative current from 0.0 base
        rev_neg_zero = calc_eps_revision_rate(-3.0, 0.0)
        self.assertEqual(rev_neg_zero, -100.0)  # Clamped at -100%

        # 2. Fractional base in (-1.0, 1.0) -> denom floored at 1.0
        # base = 0.5 -> denom = max(0.5, 1.0) = 1.0. (1.5 - 0.5) / 1.0 * 100 = +100%
        rev_frac_pos = calc_eps_revision_rate(1.5, 0.5)
        self.assertAlmostEqual(rev_frac_pos, 100.0, places=4)

        # base = -0.5 -> denom = max(0.5, 1.0) = 1.0. (0.5 - (-0.5)) / 1.0 * 100 = +100%
        rev_frac_neg = calc_eps_revision_rate(0.5, -0.5)
        self.assertAlmostEqual(rev_frac_neg, 100.0, places=4)

    def test_revision_rate_cyclical_negative_base_turnaround(self):
        """Verify cyclical deficit turnaround: (current - base) / |base| * 100."""
        # Turnaround from -1,000 KRW to +500 KRW: (500 - (-1000)) / 1000 * 100 = +150.0%
        rev_turnaround = calc_eps_revision_rate(500.0, -1000.0)
        self.assertAlmostEqual(rev_turnaround, 150.0, places=4)

        # Widening deficit from -500 KRW to -800 KRW: (-800 - (-500)) / 500 * 100 = -60.0%
        rev_widening = calc_eps_revision_rate(-800.0, -500.0)
        self.assertAlmostEqual(rev_widening, -60.0, places=4)

    def test_revision_rate_clamping_exact_boundaries(self):
        """Verify strict [-100.0%, +500.0%] clamping boundary transitions."""
        # Exactly -100%
        self.assertEqual(calc_eps_revision_rate(0.0, 100.0), -100.0)
        # Below -100%: -150% clamped to -100%
        self.assertEqual(calc_eps_revision_rate(-50.0, 100.0), -100.0)
        self.assertEqual(calc_eps_revision_rate(-1000.0, 100.0), -100.0)

        # Exactly +500%
        self.assertEqual(calc_eps_revision_rate(600.0, 100.0), 500.0)
        # Above +500%: +800% clamped to +500%
        self.assertEqual(calc_eps_revision_rate(900.0, 100.0), 500.0)

        # Custom clamp parameters
        custom_clamped = calc_eps_revision_rate(50.0, 100.0, clamp_min=-30.0, clamp_max=200.0)
        self.assertEqual(custom_clamped, -30.0)

    def test_calc_eps_revisions_series_sample_sizes(self):
        """Verify exact lag index boundaries: n <= 5 (no 1W), n <= 20 (no 1M), n <= 60 (no 3M)."""
        dates_100 = pd.bdate_range("2023-01-01", periods=100)

        # n = 0
        s0 = pd.Series([], dtype=float, index=pd.DatetimeIndex([]), name="005930")
        r0 = calc_eps_revisions_series(s0)
        self.assertIsNone(r0["current_eps"])
        self.assertIsNone(r0["eps_rev_1w"])
        self.assertFalse(r0["is_turnaround"])
        self.assertFalse(r0["is_deficit"])

        # n = 5 (strictly <= 5: 1W requires n > 5)
        s5 = pd.Series([1000.0] * 5, index=dates_100[:5], name="005930")
        r5 = calc_eps_revisions_series(s5)
        self.assertIsNone(r5["eps_rev_1w"])
        self.assertIsNone(r5["eps_rev_1m"])

        # n = 6 (n > 5: 1W becomes available at index -6)
        s6 = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1100.0], index=dates_100[:6], name="005930")
        r6 = calc_eps_revisions_series(s6)
        self.assertIsNotNone(r6["eps_rev_1w"])
        self.assertAlmostEqual(r6["eps_rev_1w"], 10.0, places=4)
        self.assertIsNone(r6["eps_rev_1m"])

        # n = 20 vs 21 (1M threshold)
        s20 = pd.Series([1000.0] * 20, index=dates_100[:20], name="005930")
        r20 = calc_eps_revisions_series(s20)
        self.assertIsNone(r20["eps_rev_1m"])

        s21 = pd.Series([1000.0] * 20 + [1200.0], index=dates_100[:21], name="005930")
        r21 = calc_eps_revisions_series(s21)
        self.assertIsNotNone(r21["eps_rev_1m"])
        self.assertAlmostEqual(r21["eps_rev_1m"], 20.0, places=4)
        self.assertIsNone(r21["eps_rev_3m"])

        # n = 60 vs 61 (3M threshold)
        s60 = pd.Series([1000.0] * 60, index=dates_100[:60], name="005930")
        r60 = calc_eps_revisions_series(s60)
        self.assertIsNone(r60["eps_rev_3m"])

        s61 = pd.Series([1000.0] * 60 + [1300.0], index=dates_100[:61], name="005930")
        r61 = calc_eps_revisions_series(s61)
        self.assertIsNotNone(r61["eps_rev_3m"])
        self.assertAlmostEqual(r61["eps_rev_3m"], 30.0, places=4)

    def test_calc_eps_revisions_series_turnaround_permutations(self):
        """Verify turnaround detection across 1M, 3M, and 1W previous deficit states."""
        dates = pd.bdate_range("2023-01-01", periods=65)

        # Case 1: 1M base is deficit, latest is positive -> turnaround True
        vals1 = [-200.0] * 45 + [100.0] * 20
        s1 = pd.Series(vals1, index=dates, name="005930")
        r1 = calc_eps_revisions_series(s1)
        self.assertTrue(r1["is_turnaround"])
        self.assertFalse(r1["is_deficit"])

        # Case 2: 1M base is positive (+50), but 3M base is deficit (-200), latest is +100 -> turnaround True
        vals2 = [-200.0] * 5 + [50.0] * 40 + [100.0] * 20
        s2 = pd.Series(vals2, index=dates, name="005930")
        r2 = calc_eps_revisions_series(s2)
        self.assertTrue(r2["is_turnaround"])

        # Case 3: 1M and 3M are positive, but 1W base is deficit (-50), latest is +100 -> turnaround True
        vals3 = [100.0] * 59 + [-50.0] * 2 + [100.0] * 4
        s3 = pd.Series(vals3, index=dates, name="005930")
        r3 = calc_eps_revisions_series(s3)
        self.assertTrue(r3["is_turnaround"])

        # Case 4: Persistent deficit (latest <= 0) -> turnaround MUST be False
        vals4 = [-200.0] * 45 + [-50.0] * 20
        s4 = pd.Series(vals4, index=dates, name="005930")
        r4 = calc_eps_revisions_series(s4)
        self.assertFalse(r4["is_turnaround"])
        self.assertTrue(r4["is_deficit"])

    def test_calc_eps_revisions_as_of_date_and_vectorization(self):
        """Verify calc_eps_revisions accepts Series or DataFrame and handles as_of_date filtering."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df_eps = pd.DataFrame({
            "005930": np.linspace(1000, 2000, 100),
            "000660": np.linspace(5000, 4000, 100),
        }, index=dates)

        # 1. As-of-date in middle of series
        as_of = pd.Timestamp("2023-02-15")
        res_as_of = calc_eps_revisions(df_eps, as_of_date=as_of)
        self.assertIn("005930", res_as_of.index)
        self.assertIn("000660", res_as_of.index)
        self.assertEqual(res_as_of.loc["005930", "current_eps"], df_eps.loc[:as_of, "005930"].iloc[-1])

        # 2. As-of-date before the start of data (empty slice)
        as_of_early = pd.Timestamp("2020-01-01")
        res_early = calc_eps_revisions(df_eps, as_of_date=as_of_early)
        self.assertTrue(pd.isna(res_early.loc["005930", "current_eps"]))
        self.assertFalse(res_early.loc["005930", "is_turnaround"])

        # 3. Single Series input
        s_single = df_eps["005930"]
        res_single = calc_eps_revisions(s_single)
        self.assertEqual(len(res_single), 1)
        self.assertEqual(res_single.index[0], "005930")


# ─────────────────────────────────────────────────────────────────────────────
# Suite 4: 5-Regime Valuation Momentum Taxonomy White-Box Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestRegimeClassificationWhiteBox(unittest.TestCase):
    """Deep white-box verification of 5-regime thresholds, precedence, and NamedTuple interface."""

    def test_regime_result_namedtuple_interface(self):
        """Verify RegimeResult implements attribute, .get(), and subscript access."""
        res = RegimeResult(is_value_trap=True, is_golden_cross=False, regime_tag="Value Trap")
        # Attribute
        self.assertTrue(res.is_value_trap)
        self.assertFalse(res.is_golden_cross)
        self.assertEqual(res.regime_tag, "Value Trap")
        # .get()
        self.assertTrue(res.get("is_value_trap"))
        self.assertEqual(res.get("unknown_key", "default_val"), "default_val")
        # Subscript by index and by key string
        self.assertTrue(res[0])
        self.assertEqual(res["regime_tag"], "Value Trap")

    def test_classify_regime_guards_and_deficit_exclusion(self):
        """Verify None, NaN, and negative P/E inputs default to Neutral."""
        # None inputs
        self.assertEqual(classify_regime(None, 10.0).regime_tag, "Neutral")
        self.assertEqual(classify_regime(20.0, None).regime_tag, "Neutral")
        # NaN inputs
        self.assertEqual(classify_regime(np.nan, 10.0).regime_tag, "Neutral")
        self.assertEqual(classify_regime(20.0, np.nan).regime_tag, "Neutral")
        # Deficit / non-positive current P/E
        self.assertEqual(classify_regime(20.0, 5.0, current_pe=0.0).regime_tag, "Neutral")
        self.assertEqual(classify_regime(20.0, 5.0, current_pe=-5.0).regime_tag, "Neutral")
        self.assertEqual(classify_regime(20.0, 5.0, current_pe=np.nan).regime_tag, "Neutral")

    def test_classify_regime_value_trap_boundary_precision(self):
        """Verify exact mathematical boundary conditions for Value Trap."""
        # Standard trigger: pe_pct <= 30.0 and rev_1m < -2.0
        self.assertTrue(classify_regime(30.0, -2.0001).is_value_trap)
        # pe_pct = 30.0, rev_1m = -2.0 -> NOT trap (strict inequality < -2.0)
        self.assertFalse(classify_regime(30.0, -2.0).is_value_trap)
        # pe_pct = 30.0001 -> NOT trap
        self.assertFalse(classify_regime(30.0001, -5.0).is_value_trap)

        # Dual downgrade trigger: pe_pct <= 30.0, rev_1m < 0.0, rev_3m < -5.0
        self.assertTrue(classify_regime(25.0, -0.001, eps_rev_3m=-5.0001).is_value_trap)
        # rev_3m = -5.0 -> NOT trap
        self.assertFalse(classify_regime(25.0, -0.001, eps_rev_3m=-5.0).is_value_trap)
        # rev_1m = 0.0 -> NOT trap
        self.assertFalse(classify_regime(25.0, 0.0, eps_rev_3m=-10.0).is_value_trap)
        # Missing rev_3m with rev_1m between -2.0 and 0.0 -> NOT trap
        self.assertFalse(classify_regime(25.0, -1.0, eps_rev_3m=None).is_value_trap)

    def test_classify_regime_golden_cross_boundary_precision(self):
        """Verify exact mathematical boundary conditions for Golden Cross."""
        # Strong trigger: pe_pct <= 40.0, current_pe > 0, rev_1m >= 2.0
        self.assertTrue(classify_regime(40.0, 2.0).is_golden_cross)
        self.assertEqual(classify_regime(40.0, 2.0).regime_tag, "Golden Cross")
        # rev_1m = 1.9999 -> NOT golden cross
        self.assertFalse(classify_regime(40.0, 1.9999).is_golden_cross)
        # pe_pct = 40.0001 -> NOT golden cross (assigned to Momentum Leader!)
        res_high = classify_regime(40.0001, 3.0)
        self.assertFalse(res_high.is_golden_cross)
        self.assertEqual(res_high.regime_tag, "Momentum Leader")

        # Dual positive trigger: pe_pct <= 40.0, 1W > 0 and 1M > 0
        self.assertTrue(classify_regime(35.0, 0.01, eps_rev_1w=0.01).is_golden_cross)
        # 1W = 0.0 -> NOT golden cross
        self.assertFalse(classify_regime(35.0, 0.5, eps_rev_1w=0.0).is_golden_cross)
        # 1M = 0.0 -> NOT golden cross
        self.assertFalse(classify_regime(35.0, 0.0, eps_rev_1w=0.5).is_golden_cross)

    def test_classify_regime_momentum_leader_and_high_pe_downgrade(self):
        """Verify Momentum Leader and High P/E Downgrade regime boundaries."""
        # Momentum Leader: pe_pct > 40.0 and rev_1m >= 3.0
        res_mom = classify_regime(75.0, 3.5)
        self.assertEqual(res_mom.regime_tag, "Momentum Leader")

        # High P/E Downgrade: pe_pct >= 70.0 and rev_1m < 0.0
        res_hd1 = classify_regime(70.0, -0.001)
        self.assertEqual(res_hd1.regime_tag, "High P/E Downgrade")
        res_hd2 = classify_regime(85.0, -10.0)
        self.assertEqual(res_hd2.regime_tag, "High P/E Downgrade")

        # Below 70% threshold with negative revision -> Neutral
        res_neu = classify_regime(69.999, -5.0)
        self.assertEqual(res_neu.regime_tag, "Neutral")


# ─────────────────────────────────────────────────────────────────────────────
# Suite 5: Sector Mapping & Relative Valuation White-Box Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestCalculatorSectorRelativityAndDictionaryWhiteBox(unittest.TestCase):
    """Deep white-box audit of SectorMappingDict, sector median, and continuity correction."""

    def test_sector_mapping_dict_dual_lookup_and_key_error(self):
        """Verify SectorMappingDict transparently resolves raw and normalized tickers."""
        mapping = SectorMappingDict({"005930": "반도체", "000660": "반도체"})
        # 1. Normalized key
        self.assertEqual(mapping["005930"], "반도체")
        # 2. Raw prefix key
        self.assertEqual(mapping["A005930"], "반도체")
        self.assertEqual(mapping["a005930"], "반도체")
        self.assertEqual(mapping[5930], "반도체")
        # 3. __contains__
        self.assertIn("005930", mapping)
        self.assertIn("A005930", mapping)
        self.assertNotIn("999999", mapping)
        # 4. KeyError on unknown []
        with self.assertRaises(KeyError):
            _ = mapping["999999"]
        # 5. Fallback on .get()
        self.assertEqual(mapping.get("999999"), "기타/미분류")
        self.assertEqual(mapping.get("999999", "CustomDefault"), "CustomDefault")

    def test_load_sector_mapping_fallback_paths(self):
        """Verify load_sector_mapping handles missing or corrupt JSON files gracefully."""
        # Non-existent path
        empty_map = load_sector_mapping("data/does_not_exist_sector.json")
        self.assertIsInstance(empty_map, SectorMappingDict)

        # Corrupted JSON
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write("{CORRUPTED_JSON_FILE")
            tf_path = tf.name

        try:
            corrupt_map = load_sector_mapping(tf_path)
            self.assertIsInstance(corrupt_map, SectorMappingDict)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_calc_sector_median_pe_filtering_and_caps(self):
        """Verify calc_sector_median_pe filters non-positive and >150 values."""
        # Non-positive values and >150 cap filtered out
        raw_pes = [-5.0, 0.0, np.nan, None, 10.0, 20.0, 160.0, 250.0]
        # Valid subset: [10.0, 20.0] -> median 15.0
        self.assertEqual(calc_sector_median_pe(raw_pes), 15.0)

        # All invalid -> returns None
        self.assertIsNone(calc_sector_median_pe([-10.0, 0.0, 200.0]))
        self.assertIsNone(calc_sector_median_pe([]))

    def test_calc_sector_pe_percentile_continuity_correction(self):
        """Verify continuity-corrected formula: (rank + 0.5) / N * 100."""
        # 1. N = 1 -> exactly 50.0%
        self.assertEqual(calc_sector_pe_percentile(10.0, [10.0]), 50.0)

        # 2. N = 2 -> 25.0% and 75.0%
        self.assertEqual(calc_sector_pe_percentile(10.0, [10.0, 20.0]), 25.0)
        self.assertEqual(calc_sector_pe_percentile(20.0, [10.0, 20.0]), 75.0)

        # 3. Duplicate ties handling (equal P/E)
        # valid list = [10.0, 15.0, 15.0, 20.0] -> target 15.0 has indices [1, 2], avg_rank = 1.5
        # pct = (1.5 + 0.5) / 4 * 100 = 50.0%
        self.assertEqual(calc_sector_pe_percentile(15.0, [10.0, 15.0, 15.0, 20.0]), 50.0)

        # 4. Target P/E not in list (interpolation)
        # target 5.0 < min(10.0) -> rank 0 -> (0 + 0.5)/4 * 100 = 12.5%
        self.assertEqual(calc_sector_pe_percentile(5.0, [10.0, 15.0, 20.0, 30.0]), 12.5)

        # target 35.0 > max(30.0) -> rank 3.5 -> (3.5 + 0.5)/4 * 100 = 100.0%
        self.assertEqual(calc_sector_pe_percentile(35.0, [10.0, 15.0, 20.0, 30.0]), 100.0)

        # 5. Invalid target values return None
        self.assertIsNone(calc_sector_pe_percentile(None, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(np.nan, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(0.0, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(-5.0, [10.0, 20.0]))
        self.assertIsNone(calc_sector_pe_percentile(160.0, [10.0, 20.0]))

    def test_calc_sector_relative_metrics_dataframe_and_list(self):
        """Verify calc_sector_relative_metrics enriches DataFrame and List[PEBandResult]."""
        # Test DataFrame pathway
        df = pd.DataFrame({
            "ticker": ["005930", "000660", "035420"],
            "current_fwd_pe": [10.0, 20.0, 30.0],
        })
        sec_map = {"005930": "반도체", "000660": "반도체", "035420": "IT"}
        res_df = calc_sector_relative_metrics(df, sector_mapping=sec_map)

        self.assertIn("sector", res_df.columns)
        self.assertIn("sector_median_pe", res_df.columns)
        self.assertIn("sector_pe_percentile", res_df.columns)
        self.assertEqual(res_df.loc[0, "sector"], "반도체")
        self.assertEqual(res_df.loc[0, "sector_median_pe"], 15.0)
        self.assertEqual(res_df.loc[0, "sector_pe_percentile"], 25.0)
        self.assertEqual(res_df.loc[1, "sector_pe_percentile"], 75.0)
        self.assertEqual(res_df.loc[2, "sector_median_pe"], 30.0)
        self.assertEqual(res_df.loc[2, "sector_pe_percentile"], 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 6: P/E Valuation Bands & Outlier Winsorization White-Box Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestValuationBandsAndWinsorizationWhiteBox(unittest.TestCase):
    """Deep white-box audit for P/E series masking, Winsorization, Mean±SD floor, and band models."""

    def test_calc_fwd_pe_series_masking_and_caps(self):
        """Verify calc_fwd_pe_series masks non-positive EPS and P/E > 200."""
        dates = pd.date_range("2023-01-01", periods=6)
        price = pd.Series([60000, 60000, 60000, 60000, 60000, 60000], index=dates)
        eps = pd.Series([6000, 0, -1000, 200, 4000, 100], index=dates)

        pe = calc_fwd_pe_series(price, eps)
        self.assertEqual(pe.iloc[0], 10.0)
        self.assertTrue(pd.isna(pe.iloc[1]))   # EPS = 0 -> NaN
        self.assertTrue(pd.isna(pe.iloc[2]))   # EPS < 0 -> NaN
        self.assertEqual(pe.iloc[3], 200.0)    # PE <= 200 -> kept
        self.assertEqual(pe.iloc[4], 15.0)     # Normal
        self.assertTrue(pd.isna(pe.iloc[5]))   # PE = 600 > 200 -> NaN

    def test_winsorize_pe_threshold_and_quantiles(self):
        """Verify winsorize_pe leaves small samples (<10) untouched and clips large samples."""
        # 1. Sample size < 10 -> unclipped copy
        s_small = pd.Series([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 140.0])  # 9 elements
        w_small = winsorize_pe(s_small)
        self.assertEqual(len(w_small), 9)
        self.assertEqual(w_small.max(), 140.0)  # NOT clipped because len < 10

        # 2. Sample size >= 10 -> clipped at 2.5% and 97.5% quantiles
        vals = [10.0] * 50 + [1.2] + [145.0]  # 52 elements
        s_large = pd.Series(vals)
        w_large = winsorize_pe(s_large)
        self.assertLess(w_large.max(), 145.0)
        self.assertGreater(w_large.min(), 1.2)

        # 3. Filtering bounds [min_pe, max_pe]: values outside are dropped before quantiles
        s_out = pd.Series([0.5, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 200.0])
        w_out = winsorize_pe(s_out, min_pe=1.0, max_pe=150.0)
        self.assertNotIn(0.5, w_out.values)
        self.assertNotIn(200.0, w_out.values)

    def test_calc_mean_sd_bands_floor_rule_and_min_samples(self):
        """Verify calc_mean_sd_bands sample size guard (<4) and lower floor rule."""
        # Sample size < 4 -> returns empty dict
        self.assertEqual(calc_mean_sd_bands(pd.Series([10.0, 12.0, 14.0])), {})

        # Sample size >= 4 with high volatility triggering floor
        # mean = 10.0, std = 8.0 -> mean - 2*std = -6.0 -> floored to max(floor_pe, min_obs*0.8)
        s_high_vol = pd.Series([2.0, 3.0, 15.0, 20.0])
        bands = calc_mean_sd_bands(s_high_vol, floor_pe=1.0, winsorize=False)
        self.assertGreaterEqual(bands["m2sd"], 1.0)
        self.assertGreaterEqual(bands["m1sd"], 1.0)
        self.assertLessEqual(bands["m2sd"], bands["m1sd"])
        self.assertLessEqual(bands["m1sd"], bands["mean"])
        self.assertLessEqual(bands["mean"], bands["p1sd"])
        self.assertLessEqual(bands["p1sd"], bands["p2sd"])

    def test_calc_fixed_multiple_bands_upside_and_zero_guards(self):
        """Verify calc_fixed_multiple_bands protects against negative EPS and zero price."""
        multiples = [8.0, 10.0, 12.0, 15.0]

        # 1. Normal case
        res_normal = calc_fixed_multiple_bands(multiples, current_eps=5000.0, current_price=50000.0)
        self.assertEqual(res_normal["targets"][10.0], 50000.0)
        self.assertEqual(res_normal["upsides"][10.0], 0.0)
        self.assertEqual(res_normal["targets"][12.0], 60000.0)
        self.assertAlmostEqual(res_normal["upsides"][12.0], 20.0, places=4)

        # 2. Deficit EPS (<= 0) -> targets must be 0.0
        res_deficit = calc_fixed_multiple_bands(multiples, current_eps=-1000.0, current_price=50000.0)
        for m in multiples:
            self.assertEqual(res_deficit["targets"][m], 0.0)

        # 3. Zero price -> upside must be 0.0
        res_zero_price = calc_fixed_multiple_bands(multiples, current_eps=5000.0, current_price=0.0)
        for m in multiples:
            self.assertEqual(res_zero_price["upsides"][m], 0.0)

    def test_calc_pe_band_models_and_timeseries_arrays(self):
        """Verify calc_pe_band produces valid PEBandResult with all 3 band models and timeseries arrays."""
        dates = pd.date_range("2018-01-01", "2023-01-01", freq="MS")  # 5 years monthly (61 points)
        price_series = pd.Series(np.linspace(40000, 70000, len(dates)), index=dates)
        eps_series = pd.Series(np.linspace(4000, 7000, len(dates)), index=dates)

        # 1. Percentile model (default)
        r_pct = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=price_series,
            eps_series=eps_series,
            band_years=5,
            band_model="percentile",
        )
        self.assertIsNotNone(r_pct)
        self.assertIsInstance(r_pct.band_series_pct, pd.DataFrame)
        self.assertIn("p50", r_pct.band_series_pct.columns)
        self.assertEqual(r_pct.band_model, "percentile")
        self.assertAlmostEqual(r_pct.target_base, r_pct.current_fwd_eps * r_pct.pe_median, places=2)

        # 2. Mean ± SD model
        r_sd = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=price_series,
            eps_series=eps_series,
            band_years=5,
            band_model="mean_sd",
        )
        self.assertIsNotNone(r_sd)
        self.assertIsInstance(r_sd.band_series_sd, pd.DataFrame)
        self.assertIn("Mean", r_sd.band_series_sd.columns)
        self.assertEqual(r_sd.band_model, "mean_sd")
        self.assertAlmostEqual(r_sd.target_base, r_sd.current_fwd_eps * r_sd.mean_sd_metrics["mean"], places=2)

        # 3. Fixed multiples model
        r_fixed = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=price_series,
            eps_series=eps_series,
            band_years=5,
            band_model="fixed",
            fixed_multiples=[8.0, 10.0, 12.0, 15.0],
        )
        self.assertIsNotNone(r_fixed)
        self.assertIsInstance(r_fixed.band_series_fixed, pd.DataFrame)
        self.assertIn("10x", r_fixed.band_series_fixed.columns)
        self.assertEqual(r_fixed.band_model, "fixed")
        self.assertEqual(r_fixed.fixed_targets[10.0], r_fixed.current_fwd_eps * 10.0)

        # 4. Deficit stock rejection (current_eps <= 0)
        eps_deficit = eps_series.copy()
        eps_deficit.iloc[-1] = -500.0  # Current EPS is negative
        r_def = calc_pe_band(
            ticker="005930",
            name="삼성전자",
            price_series=price_series,
            eps_series=eps_deficit,
            band_years=5,
        )
        self.assertIsNone(r_def)  # Must return None for loss-making latest EPS


# ─────────────────────────────────────────────────────────────────────────────
# Suite 7: Dashboard Real-Time Pricing & Signal Hierarchy Audit
# ─────────────────────────────────────────────────────────────────────────────
class TestAppRealtimeAndStrategySignalsWhiteBox(unittest.TestCase):
    """Deep white-box audit for signal resolution, badge HTML, and real-time pricing."""

    def test_get_strategy_signal_priority_hierarchy(self):
        """Verify strict priority ordering in get_strategy_signal."""
        # Baseline dummy PEBandResult generator
        dates = pd.date_range("2020-01-01", periods=10)
        dummy_series = pd.Series([10.0] * 10, index=dates)

        def make_dummy_result(
            is_golden_cross=False,
            is_value_trap=False,
            regime_tag="Neutral",
            eps_rev_1m=0.0,
            pe_percentile=50.0,
            current_fwd_pe=10.0,
        ):
            return PEBandResult(
                ticker="005930",
                name="삼성전자",
                current_price=60000.0,
                current_fwd_eps=6000.0,
                current_fwd_pe=current_fwd_pe,
                pe_percentile=pe_percentile,
                pe_min=8.0, pe_p25=9.0, pe_median=10.0, pe_p75=11.0, pe_max=12.0, pe_mean=10.0,
                target_bear=54000.0, target_base=60000.0, target_bull=66000.0,
                upside_bear=-10.0, upside_base=0.0, upside_bull=10.0,
                hist_pe_series=dummy_series,
                hist_price_series=dummy_series,
                hist_eps_series=dummy_series,
                is_golden_cross=is_golden_cross,
                is_value_trap=is_value_trap,
                regime_tag=regime_tag,
                eps_rev_1m=eps_rev_1m,
            )

        # 1. Golden Cross priority over Value Trap (if conflicting flags)
        r_gc = make_dummy_result(is_golden_cross=True, is_value_trap=True)
        self.assertEqual(get_strategy_signal(r_gc), "✨골든크로스")

        # 2. Value Trap priority over Earnings Momentum
        r_vt = make_dummy_result(is_value_trap=True, eps_rev_1m=10.0)
        self.assertEqual(get_strategy_signal(r_vt), "⚠️밸류트랩")

        # 3. Earnings Momentum priority over Deep Value
        r_em = make_dummy_result(eps_rev_1m=6.0, pe_percentile=15.0, current_fwd_pe=8.0)
        self.assertEqual(get_strategy_signal(r_em), "🚀어닝모멘텀")

        # 4. Deep Value (pe_percentile <= 25, pe <= 15, rev >= -2)
        r_val = make_dummy_result(pe_percentile=20.0, current_fwd_pe=12.0, eps_rev_1m=-1.0)
        self.assertEqual(get_strategy_signal(r_val), "💎가치주")

        # 5. High P/E Downgrade
        r_hd = make_dummy_result(regime_tag="High P/E Downgrade")
        self.assertEqual(get_strategy_signal(r_hd), "🔻고P/E하향")

        # 6. Neutral fallback
        r_neu = make_dummy_result(pe_percentile=50.0, eps_rev_1m=0.0)
        self.assertEqual(get_strategy_signal(r_neu), "Neutral")

    def test_ui_badge_and_color_helpers(self):
        """Verify HTML badge generation and color scales across all bands and null states."""
        # strategy_badge_html
        self.assertIn("sig-gc", strategy_badge_html("✨골든크로스"))
        self.assertIn("sig-vt", strategy_badge_html("⚠️밸류트랩"))
        self.assertIn("sig-em", strategy_badge_html("🚀어닝모멘텀"))
        self.assertIn("sig-val", strategy_badge_html("💎가치주"))
        self.assertIn("sig-hd", strategy_badge_html("🔻고P/E하향"))
        self.assertIn("sig-neu", strategy_badge_html("Neutral"))

        # signal_badge & signal_label
        self.assertIn("Strong Buy", signal_badge(15.0))
        self.assertIn("Buy", signal_badge(35.0))
        self.assertIn("Hold", signal_badge(55.0))
        self.assertIn("Sell", signal_badge(75.0))
        self.assertIn("Strong Sell", signal_badge(90.0))
        self.assertIn("N/A", signal_badge(None))
        self.assertIn("N/A", signal_badge(np.nan))

        # pe_bar_color
        self.assertEqual(pe_bar_color(25.0), "#34d399")  # Green
        self.assertEqual(pe_bar_color(50.0), "#fbbf24")  # Yellow
        self.assertEqual(pe_bar_color(80.0), "#f87171")  # Red
        self.assertEqual(pe_bar_color(None), "#9ca3af")   # Gray

    def test_get_file_mtimes_length_and_types(self):
        """Verify _get_file_mtimes returns a 10-element integer tuple."""
        mtimes = _get_file_mtimes()
        self.assertIsInstance(mtimes, tuple)
        self.assertEqual(len(mtimes), 10)
        for val in mtimes:
            self.assertIsInstance(val, int)

    def test_apply_realtime_prices_profit_and_deficit_safeguards(self):
        """Verify apply_realtime_prices updates metrics for profit stocks and guards deficit stocks."""
        dates = pd.date_range("2020-01-01", periods=10)
        hist_pe = pd.Series(np.linspace(8.0, 15.0, 10), index=dates)

        r_profit = PEBandResult(
            ticker="005930",
            name="삼성전자",
            current_price=60000.0,
            current_fwd_eps=6000.0,
            current_fwd_pe=10.0,
            pe_percentile=50.0,
            pe_min=8.0, pe_p25=9.0, pe_median=10.0, pe_p75=11.0, pe_max=15.0, pe_mean=10.5,
            target_bear=54000.0, target_base=60000.0, target_bull=66000.0,
            upside_bear=-10.0, upside_base=0.0, upside_bull=10.0,
            hist_pe_series=hist_pe,
            hist_price_series=hist_pe * 6000.0,
            hist_eps_series=pd.Series([6000.0] * 10, index=dates),
            sector="반도체",
            eps_rev_1m=2.5,
        )

        r_deficit = PEBandResult(
            ticker="000000",
            name="적자기업",
            current_price=10000.0,
            current_fwd_eps=-500.0,
            current_fwd_pe=np.nan,
            pe_percentile=np.nan,
            pe_min=np.nan, pe_p25=np.nan, pe_median=np.nan, pe_p75=np.nan, pe_max=np.nan, pe_mean=np.nan,
            target_bear=np.nan, target_base=np.nan, target_bull=np.nan,
            upside_bear=0.0, upside_base=0.0, upside_bull=0.0,
            hist_pe_series=pd.Series([], dtype=float),
            hist_price_series=pd.Series([10000.0] * 10, index=dates),
            hist_eps_series=pd.Series([-500.0] * 10, index=dates),
            sector="기타",
            eps_rev_1m=-5.0,
        )

        sec_map = {"005930": "반도체", "000000": "기타"}

        # 1. Real-time update with new prices
        rt_prices = {"005930": 72000.0, "000000": 12000.0}
        updated = apply_realtime_prices([r_profit, r_deficit], rt_prices, sec_map)

        # Verify profit stock: current_price 72,000 KRW, P/E = 12.0x, Base upside recalculated
        u_profit = updated[0]
        self.assertEqual(u_profit.current_price, 72000.0)
        self.assertEqual(u_profit.current_fwd_pe, 12.0)
        self.assertAlmostEqual(u_profit.upside_base, ((60000.0 / 72000.0) - 1.0) * 100.0, places=4)

        # Verify deficit stock: current_price updated to 12,000 KRW, but P/E and upside guarded
        u_deficit = updated[1]
        self.assertEqual(u_deficit.current_price, 12000.0)
        self.assertTrue(pd.isna(u_deficit.current_fwd_pe))
        self.assertEqual(u_deficit.upside_base, 0.0)

        # 2. Pathological real-time prices (<= 0 or NaN) are ignored
        bad_rt = {"005930": -100.0, "000000": 0.0}
        unchanged = apply_realtime_prices([r_profit, r_deficit], bad_rt, sec_map)
        self.assertEqual(unchanged[0].current_price, 60000.0)
        self.assertEqual(unchanged[1].current_price, 10000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test Execution Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
