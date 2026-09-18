"""
tests/test_adversarial_m2_2.py
------------------------------
Adversarial Stress-Testing Suite for Milestone 2:
Universe Mapping Completeness & Full Screener Pipeline Integration.
Author: Challenger M2-2 (Empirical Challenger)

Target Scope:
1. Universe Completeness & Mapping Stress:
   - Check all 350 stocks in data/universe.csv against data/sector_mapping.json.
   - Bijective 100% mapping: zero missing keys, zero extra keys, zero unmapped/기타.
   - Sector taxonomy validation against the 12 recognized WICS Korean equity sectors.
   - Sector constituent count distributions (all >= 5).
2. Ticker Normalization Stress Across 350 Universe Constituents:
   - 6-digit, A-prefix, a-prefix, whitespace padding, alphanumeric codes (0126Z0, 0009K0, 00680K).
   - SectorMappingDict behavior, protocol consistency, and adversarial boundary checks.
3. Full-Universe Screener Pipeline Simulation & calc_sector_relative_metrics:
   - Full 350-constituent synthetic and realistic valuation datasets.
   - Null safety: non-positive, NaN, and >150 outlier handling.
   - Ratio validity: sector_relative_pe > 0 for all positive P/Es.
   - Bounds validity: sector_pe_percentile strictly in [0.0, 100.0] and (0.0, 100.0).
   - Monotonic consistency within sectors.
   - List[PEBandResult] and pd.DataFrame polymorphism parity across 350 stocks.
   - Adversarial edge cases: all-deficit sectors, single survivor in 68-stock sector, massive ties.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
import unittest
import numpy as np
import pandas as pd

from core.calculator import (
    load_sector_mapping,
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    calc_sector_relative_metrics,
    _clean_ticker,
    PEBandResult,
    SectorMappingDict,
)
from tests.oracle import (
    clean_ticker,
    oracle_sector_median_pe,
    oracle_sector_pe_percentile,
)

ROOT = Path(__file__).resolve().parent.parent

VALID_WICS_12_SECTORS = {
    "반도체",
    "2차전지",
    "자동차",
    "바이오/헬스케어",
    "금융/지주",
    "화학/에너지",
    "조선/방산/중공업",
    "IT플랫폼/소프트웨어",
    "소비재/유통",
    "유틸리티/통신",
    "철강/소재",
    "건설/부동산",
}


class TestUniverseMappingCompletenessAdversarial(unittest.TestCase):
    """Adversarial stress-testing of data/universe.csv and data/sector_mapping.json."""

    def setUp(self):
        self.univ_path = ROOT / "data" / "universe.csv"
        self.map_path = ROOT / "data" / "sector_mapping.json"
        self.assertTrue(self.univ_path.exists(), "data/universe.csv is missing!")
        self.assertTrue(self.map_path.exists(), "data/sector_mapping.json is missing!")

        self.univ_df = pd.read_csv(self.univ_path)
        with open(self.map_path, "r", encoding="utf-8") as f:
            self.sector_map = json.load(f)

    def test_01_universe_exact_constituent_count(self):
        """Universe must contain exactly 350 stocks (200 KOSPI200, 150 KOSDAQ150)."""
        self.assertEqual(len(self.univ_df), 350)
        market_counts = self.univ_df["market"].value_counts().to_dict()
        self.assertEqual(market_counts.get("KOSPI200"), 200)
        self.assertEqual(market_counts.get("KOSDAQ150"), 150)

    def test_02_universe_ticker_uniqueness(self):
        """Every ticker in data/universe.csv must be strictly unique."""
        tickers = self.univ_df["ticker"].astype(str).str.strip().tolist()
        self.assertEqual(len(tickers), 350)
        self.assertEqual(len(set(tickers)), 350, "Duplicate tickers detected in data/universe.csv!")

    def test_03_sector_mapping_exact_count(self):
        """data/sector_mapping.json must contain exactly 350 mapped keys."""
        self.assertEqual(len(self.sector_map), 350)

    def test_04_bijective_zero_omissions_and_zero_extras(self):
        """Exact 1-to-1 match between data/universe.csv and data/sector_mapping.json."""
        univ_tickers = {clean_ticker(t) for t in self.univ_df["ticker"]}
        json_tickers = {clean_ticker(t) for t in self.sector_map.keys()}

        missing_in_json = univ_tickers - json_tickers
        extra_in_json = json_tickers - univ_tickers

        self.assertEqual(len(missing_in_json), 0, f"Unmapped universe tickers: {missing_in_json}")
        self.assertEqual(len(extra_in_json), 0, f"Spurious keys in sector_mapping: {extra_in_json}")
        self.assertEqual(univ_tickers, json_tickers)

    def test_05_all_sectors_conform_to_12_wics_taxonomy(self):
        """Every single ticker must be classified into one of the 12 recognized WICS sectors."""
        invalid_sectors = {}
        for ticker, sector in self.sector_map.items():
            if sector not in VALID_WICS_12_SECTORS:
                invalid_sectors[ticker] = sector

        self.assertEqual(len(invalid_sectors), 0, f"Invalid or unrecognized sectors: {invalid_sectors}")

        # Explicitly verify no placeholder, unclassified, or empty tags
        forbidden_tags = {"기타", "미분류", "기타/미분류", "None", "nan", ""}
        found_forbidden = {t: s for t, s in self.sector_map.items() if s in forbidden_tags}
        self.assertEqual(len(found_forbidden), 0, f"Forbidden sector placeholders found: {found_forbidden}")

    def test_06_sector_distribution_robustness(self):
        """Verify all 12 sectors are populated with realistic sample sizes (minimum >= 5)."""
        sector_counts = pd.Series(list(self.sector_map.values())).value_counts().to_dict()
        self.assertEqual(len(sector_counts), 12, "Not all 12 WICS sectors are represented!")

        for sec in VALID_WICS_12_SECTORS:
            count = sector_counts.get(sec, 0)
            self.assertGreaterEqual(
                count, 5, f"Sector {sec} has insufficient sample size ({count} < 5)!"
            )

        # Expected key benchmark distributions
        self.assertEqual(sector_counts["반도체"], 68)
        self.assertEqual(sector_counts["금융/지주"], 52)
        self.assertEqual(sector_counts["조선/방산/중공업"], 48)
        self.assertEqual(sector_counts["바이오/헬스케어"], 46)
        self.assertEqual(sector_counts["소비재/유통"], 43)

    def test_07_special_ticker_encodings(self):
        """Verify handling of tickers with letters (preferred stocks, special codes)."""
        special_tickers = {
            "0126Z0": "바이오/헬스케어",  # 삼성에피스홀딩스
            "0009K0": "바이오/헬스케어",  # 에임드바이오
            "00680K": "금융/지주",       # 미래에셋증권2우B
            "005387": "자동차",          # 현대차2우B
            "005385": "자동차",          # 현대차우
            "000155": "금융/지주",       # 두산우
            "009155": "반도체",          # 삼성전기우
        }
        for code, expected_sec in special_tickers.items():
            self.assertIn(code, self.sector_map)
            self.assertEqual(self.sector_map[code], expected_sec)


class TestTickerNormalizationAndSectorMappingDict(unittest.TestCase):
    """Adversarial stress-testing of ticker normalization and SectorMappingDict."""

    def setUp(self):
        self.sec_map = load_sector_mapping()

    def test_01_all_350_clean_lookups(self):
        """All 350 clean tickers must resolve correctly via __getitem__ and get()."""
        for ticker in self.sec_map.keys():
            self.assertEqual(self.sec_map[ticker], self.sec_map.get(ticker))
            self.assertIn(self.sec_map[ticker], VALID_WICS_12_SECTORS)

    def test_02_all_350_a_prefix_lookups(self):
        """All 350 tickers prefixed with uppercase 'A' must resolve identically."""
        for ticker in list(self.sec_map.keys()):
            a_ticker = f"A{ticker}"
            self.assertEqual(self.sec_map[a_ticker], self.sec_map[ticker])
            self.assertEqual(self.sec_map.get(a_ticker), self.sec_map[ticker])

    def test_03_all_350_lowercase_a_prefix_lookups(self):
        """All 350 tickers prefixed with lowercase 'a' must resolve identically."""
        for ticker in list(self.sec_map.keys()):
            a_ticker = f"a{ticker}"
            self.assertEqual(self.sec_map[a_ticker], self.sec_map[ticker])
            self.assertEqual(self.sec_map.get(a_ticker), self.sec_map[ticker])

    def test_04_whitespace_padding_lookups(self):
        """Tickers with leading/trailing whitespace must resolve cleanly."""
        self.assertEqual(self.sec_map["  005930  "], "반도체")
        self.assertEqual(self.sec_map.get("  A005930  "), "반도체")
        self.assertEqual(self.sec_map.get("\t005380\n"), "자동차")

    def test_05_integer_ticker_lookups(self):
        """Numeric integer inputs (e.g. 5930 -> 005930) must resolve cleanly."""
        self.assertEqual(self.sec_map[5930], "반도체")
        self.assertEqual(self.sec_map.get(5380), "자동차")
        self.assertEqual(self.sec_map[660], "반도체")

    def test_06_unmapped_ticker_handling(self):
        """Non-existent tickers must raise KeyError on [] and return default on get()."""
        with self.assertRaises(KeyError):
            _ = self.sec_map["999999"]

        self.assertEqual(self.sec_map.get("999999"), "기타/미분류")
        self.assertEqual(self.sec_map.get("UNKNOWN", "DEFAULT_TAG"), "DEFAULT_TAG")

    def test_07_adversarial_contains_protocol_check(self):
        """
        Adversarial Boundary Check:
        Document behavior of 'in' operator on SectorMappingDict.
        Note: clean code '005930' is True; 'A005930' will query dict.__contains__ without normalization
        unless __contains__ is overridden. We verify clean code presence.
        """
        self.assertTrue("005930" in self.sec_map)
        self.assertTrue("000660" in self.sec_map)
        self.assertEqual(len(self.sec_map), 350)


class TestFullUniverseSectorRelativeMetricsPipeline(unittest.TestCase):
    """
    Adversarial stress-testing of calc_sector_relative_metrics
    across the entire 350-constituent universe under diverse valuation regimes.
    """

    def setUp(self):
        self.univ_path = ROOT / "data" / "universe.csv"
        self.univ_df = pd.read_csv(self.univ_path)
        self.sec_map = load_sector_mapping()

    def test_01_full_universe_dataframe_all_receive_valid_sectors(self):
        """Every single stock in 350-stock DataFrame must receive its official sector."""
        df = self.univ_df[["ticker", "name"]].copy()
        df["current_fwd_pe"] = 15.0  # Constant test multiple

        enriched_df = calc_sector_relative_metrics(df, self.sec_map)

        self.assertEqual(len(enriched_df), 350)
        self.assertIn("sector", enriched_df.columns)
        self.assertIn("sector_median_pe", enriched_df.columns)
        self.assertIn("sector_relative_pe", enriched_df.columns)
        self.assertIn("sector_pe_percentile", enriched_df.columns)
        self.assertIn("sector_stock_count", enriched_df.columns)

        # 100% of stocks must have valid sector
        self.assertTrue(enriched_df["sector"].notna().all())
        self.assertFalse((enriched_df["sector"] == "기타/미분류").any())
        self.assertTrue(all(s in VALID_WICS_12_SECTORS for s in enriched_df["sector"]))

    def test_02_full_universe_ratio_validity_positive_pes(self):
        """
        With heterogeneous positive P/Es (5.0 ~ 80.0x), all stocks must have:
        - sector_relative_pe > 0.0
        - sector_pe_percentile in (0.0, 100.0)
        - sector_stock_count == count of sector constituents in universe
        """
        df = self.univ_df[["ticker", "name"]].copy()
        rng = np.random.default_rng(42)
        # Assign realistic positive P/Es
        df["current_fwd_pe"] = rng.uniform(5.0, 50.0, size=350)

        enriched = calc_sector_relative_metrics(df, self.sec_map)

        # Relative P/E must be strictly positive and finite
        self.assertTrue((enriched["sector_relative_pe"] > 0.0).all())
        self.assertFalse(np.isinf(enriched["sector_relative_pe"]).any())
        self.assertFalse(enriched["sector_relative_pe"].isna().any())

        # Sector percentile must be strictly within (0, 100)
        self.assertTrue((enriched["sector_pe_percentile"] > 0.0).all())
        self.assertTrue((enriched_df_pct := enriched["sector_pe_percentile"] < 100.0).all())

        # Stock count per sector matches total universe count per sector
        sec_counts = enriched["sector"].value_counts().to_dict()
        for idx, row in enriched.iterrows():
            expected_n = sec_counts[row["sector"]]
            self.assertEqual(row["sector_stock_count"], expected_n)

    def test_03_full_universe_null_safety_deficits_and_outliers(self):
        """
        Adversarial scenario:
        Mixture of profitable stocks, deficits (PE < 0), zero P/Es, extreme bubble outliers (> 150),
        and NaNs across all 350 stocks.
        """
        df = self.univ_df[["ticker", "name"]].copy()
        rng = np.random.default_rng(2026)

        pe_values = []
        for i in range(350):
            case = i % 5
            if case == 0:
                pe_values.append(rng.uniform(8.0, 45.0))   # Normal profitable
            elif case == 1:
                pe_values.append(rng.uniform(-50.0, -1.0)) # Deficit / Loss
            elif case == 2:
                pe_values.append(rng.uniform(150.1, 500.0))# Extreme bubble outlier
            elif case == 3:
                pe_values.append(0.0)                      # Zero PE
            else:
                pe_values.append(np.nan)                   # Missing consensus

        df["current_fwd_pe"] = pe_values
        enriched = calc_sector_relative_metrics(df, self.sec_map)

        for idx, row in enriched.iterrows():
            pe = row["current_fwd_pe"]
            if pd.notna(pe) and 0.0 < pe <= 150.0:
                # Valid stock: if sector has at least 1 valid stock, metrics must be valid
                if row["sector_stock_count"] > 0:
                    self.assertGreater(row["sector_relative_pe"], 0.0)
                    self.assertGreaterEqual(row["sector_pe_percentile"], 0.0)
                    self.assertLessEqual(row["sector_pe_percentile"], 100.0)
            else:
                # Invalid stock: relative P/E and percentile MUST be NaN
                self.assertTrue(
                    pd.isna(row["sector_relative_pe"]),
                    f"Deficit/outlier stock {row['ticker']} (PE={pe}) should have NaN relative PE, got {row['sector_relative_pe']}"
                )
                self.assertTrue(
                    pd.isna(row["sector_pe_percentile"]),
                    f"Deficit/outlier stock {row['ticker']} (PE={pe}) should have NaN sector percentile, got {row['sector_pe_percentile']}"
                )

    def test_04_monotonicity_within_sectors(self):
        """
        Within every sector, for any two profitable stocks A and B:
        PE_A < PE_B implies relative_PE_A < relative_PE_B and pct_A <= pct_B.
        """
        df = self.univ_df[["ticker", "name"]].copy()
        rng = np.random.default_rng(999)
        df["current_fwd_pe"] = rng.uniform(4.0, 80.0, size=350)

        enriched = calc_sector_relative_metrics(df, self.sec_map)

        for sector, grp in enriched.groupby("sector"):
            valid_grp = grp.dropna(subset=["sector_relative_pe", "sector_pe_percentile"]).sort_values("current_fwd_pe")
            pes = valid_grp["current_fwd_pe"].tolist()
            rel_pes = valid_grp["sector_relative_pe"].tolist()
            pcts = valid_grp["sector_pe_percentile"].tolist()

            for i in range(len(pes) - 1):
                if pes[i] < pes[i + 1] - 1e-4:
                    self.assertLess(rel_pes[i], rel_pes[i + 1])
                    self.assertLessEqual(pcts[i], pcts[i + 1])

    def test_05_polymorphism_list_vs_dataframe_exact_match(self):
        """
        Parity check: Running calc_sector_relative_metrics on List[PEBandResult]
        must produce identical values to pd.DataFrame for all 350 universe stocks.
        """
        rng = np.random.default_rng(1234)
        pe_vals = [float(rng.uniform(6.0, 60.0)) if i % 10 != 0 else -10.0 for i in range(350)]

        # 1. Create List[PEBandResult]
        results_list = []
        for i, row in self.univ_df.iterrows():
            r = PEBandResult(
                ticker=row["ticker"],
                name=row["name"],
                current_price=50000.0,
                current_fwd_eps=5000.0,
                current_fwd_pe=pe_vals[i],
                pe_percentile=50.0,
                pe_min=5.0, pe_p25=8.0, pe_median=10.0, pe_p75=12.0, pe_max=15.0, pe_mean=10.0,
                target_bear=0, target_base=0, target_bull=0, upside_bear=0, upside_base=0, upside_bull=0,
                hist_pe_series=pd.Series(), hist_price_series=pd.Series(), hist_eps_series=pd.Series(),
                sector=self.sec_map.get(row["ticker"], "기타/미분류"),
            )
            results_list.append(r)

        enriched_list = calc_sector_relative_metrics(results_list, self.sec_map)

        # 2. Create pd.DataFrame
        df = self.univ_df[["ticker", "name"]].copy()
        df["current_fwd_pe"] = pe_vals
        df["sector"] = [self.sec_map.get(t, "기타/미분류") for t in df["ticker"]]
        enriched_df = calc_sector_relative_metrics(df, self.sec_map)

        # 3. Verify exact parity across all 350 stocks
        for i in range(350):
            r_item = enriched_list[i]
            df_row = enriched_df.iloc[i]

            self.assertEqual(r_item.sector, df_row["sector"])
            self.assertEqual(r_item.sector_stock_count, df_row["sector_stock_count"])

            if r_item.sector_median_pe is None:
                self.assertTrue(pd.isna(df_row["sector_median_pe"]))
            else:
                self.assertAlmostEqual(r_item.sector_median_pe, df_row["sector_median_pe"], places=4)

            if r_item.sector_relative_pe is None:
                self.assertTrue(pd.isna(df_row["sector_relative_pe"]))
            else:
                self.assertAlmostEqual(r_item.sector_relative_pe, df_row["sector_relative_pe"], places=4)

            if r_item.sector_pe_percentile is None:
                self.assertTrue(pd.isna(df_row["sector_pe_percentile"]))
            else:
                self.assertAlmostEqual(r_item.sector_pe_percentile, df_row["sector_pe_percentile"], places=4)

    def test_06_all_deficit_sector_in_full_universe(self):
        """
        Adversarial edge case:
        If every constituent of an entire sector (e.g. 건설/부동산 with 8 stocks) is loss-making,
        pipeline must not crash and must set median=None, rel=None, pct=None, count=0.
        """
        df = self.univ_df[["ticker", "name"]].copy()
        df["current_fwd_pe"] = 15.0
        # Make all Construction & Real Estate stocks loss-making
        sec_name = "건설/부동산"
        const_tickers = [t for t, s in self.sec_map.items() if s == sec_name]
        df.loc[df["ticker"].isin(const_tickers), "current_fwd_pe"] = -10.0

        enriched = calc_sector_relative_metrics(df, self.sec_map)
        const_rows = enriched[enriched["ticker"].isin(const_tickers)]

        self.assertEqual(len(const_rows), len(const_tickers))
        self.assertTrue((const_rows["sector_stock_count"] == 0).all())
        self.assertTrue(const_rows["sector_median_pe"].isna().all())
        self.assertTrue(const_rows["sector_relative_pe"].isna().all())
        self.assertTrue(const_rows["sector_pe_percentile"].isna().all())

    def test_07_single_profitable_survivor_in_68_stock_semiconductor_sector(self):
        """
        Adversarial edge case:
        In the largest sector (반도체, 68 stocks), only 1 stock is profitable, 67 are in deficit.
        The survivor must receive N=1 continuity-corrected percentile: strictly 50.0% and relative_pe: 1.0x.
        """
        df = self.univ_df[["ticker", "name"]].copy()
        df["current_fwd_pe"] = -5.0 # default all deficit
        # Samsung Electronics (005930) is the sole survivor at 10.0x
        df.loc[df["ticker"] == "005930", "current_fwd_pe"] = 10.0

        enriched = calc_sector_relative_metrics(df, self.sec_map)
        samsung = enriched[enriched["ticker"] == "005930"].iloc[0]

        self.assertEqual(samsung["sector"], "반도체")
        self.assertEqual(samsung["sector_stock_count"], 1)
        self.assertAlmostEqual(samsung["sector_median_pe"], 10.0, places=4)
        self.assertAlmostEqual(samsung["sector_relative_pe"], 1.0, places=4)
        self.assertAlmostEqual(samsung["sector_pe_percentile"], 50.0, places=4)

        # Other semiconductor constituents must be NaN
        other_semis = enriched[(enriched["sector"] == "반도체") & (enriched["ticker"] != "005930")]
        self.assertEqual(len(other_semis), 67)
        self.assertTrue(other_semis["sector_relative_pe"].isna().all())
        self.assertTrue(other_semis["sector_pe_percentile"].isna().all())


if __name__ == "__main__":
    unittest.main()
