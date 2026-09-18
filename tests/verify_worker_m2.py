"""
tests/verify_worker_m2.py
-------------------------
Verification script for Milestone 2 (Sector Mapping & Relative Valuation Engine).
Tests 350-stock WICS mapping, load_sector_mapping, calc_sector_median_pe,
calc_sector_pe_percentile, calc_sector_relative_metrics, PEBandResult, and run_screener.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calculator import (
    load_sector_mapping,
    calc_sector_median_pe,
    calc_sector_pe_percentile,
    calc_sector_relative_metrics,
    calc_pe_band,
    run_screener,
    PEBandResult,
    SectorMappingDict,
)
from tests.oracle import (
    oracle_sector_median_pe,
    oracle_sector_pe_percentile,
    clean_ticker,
)


def verify_all():
    print("=" * 70)
    print(">>> Milestone 2 Verification: Sector Mapping & Relative Valuation Engine")
    print("=" * 70)

    # 1. 350-Stock WICS Sector Mapping File Check
    print("\n[Test 1] 350-Stock WICS Sector Mapping in data/sector_mapping.json")
    json_path = ROOT / "data" / "sector_mapping.json"
    assert json_path.exists(), "data/sector_mapping.json does not exist!"
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert isinstance(mapping, dict), "sector_mapping.json must be a JSON object (dict)"
    print(f"  Count of mapped stocks in JSON: {len(mapping)}")
    assert len(mapping) == 350, f"Expected exactly 350 stocks, got {len(mapping)}"

    # Check against universe.csv
    univ_path = ROOT / "data" / "universe.csv"
    assert univ_path.exists(), "data/universe.csv does not exist!"
    univ_df = pd.read_csv(univ_path)
    univ_tickers = [clean_ticker(t) for t in univ_df["ticker"].dropna().tolist()]
    assert len(univ_tickers) == 350, f"Universe should have 350 stocks, found {len(univ_tickers)}"

    missing_in_mapping = [t for t in univ_tickers if t not in mapping]
    assert len(missing_in_mapping) == 0, f"Unmapped stocks found: {missing_in_mapping}"

    valid_sectors = {
        "반도체", "2차전지", "자동차", "바이오/헬스케어", "금융/지주",
        "화학/에너지", "조선/방산/중공업", "IT플랫폼/소프트웨어", "소비재/유통",
        "유틸리티/통신", "철강/소재", "건설/부동산", "기타/미분류"
    }
    for t, sec in mapping.items():
        assert sec in valid_sectors, f"Stock {t} has unknown sector {sec}"

    # Bluechip verification
    assert mapping["005930"] == "반도체"
    assert mapping["000660"] == "반도체"
    assert mapping["005380"] == "자동차"
    assert mapping["105560"] == "금융/지주"
    print("  PASS: 100% of 350 stocks mapped with zero unmapped stocks; bluechips correctly classified.")

    # 2. load_sector_mapping function
    print("\n[Test 2] load_sector_mapping & SectorMappingDict")
    sec_map = load_sector_mapping()
    assert isinstance(sec_map, dict)
    assert len(sec_map) == 350
    # Clean ticker lookup
    assert sec_map["005930"] == "반도체"
    assert sec_map.get("005930") == "반도체"
    # A-prefix ticker lookup
    assert sec_map["A005930"] == "반도체"
    assert sec_map.get("A005930") == "반도체"
    # Unknown fallback
    assert sec_map.get("UNKNOWN_XYZ", "기타") == "기타"
    print("  PASS: load_sector_mapping returns SectorMappingDict supporting clean & A-prefixed lookups.")

    # 3. calc_sector_median_pe & Oracle Parity
    print("\n[Test 3] calc_sector_median_pe & Oracle Parity")
    test_cases_med = [
        [10.0, 12.0, 15.0, 18.0, 20.0],
        [10.0, 12.0, 15.0, 18.0, 140.0], # Outlier resistant
        [10.0, 12.0, 16.0, 20.0],        # Even elements
        [-5.0, 0.0, 10.0, 20.0],         # Non-positive filtered
        [10.0, 20.0, 180.0],             # Cap > 150 filtered
        [12.0],                          # N=1
        [-5.0, -10.0, 0.0],              # All deficits -> None
        [],                              # Empty -> None
    ]
    for pe_list in test_cases_med:
        actual = calc_sector_median_pe(pe_list)
        expected = oracle_sector_median_pe(pe_list)
        if expected is None:
            assert actual is None, f"For {pe_list}, expected None, got {actual}"
        else:
            assert actual is not None and abs(actual - expected) < 1e-4, f"For {pe_list}, expected {expected}, got {actual}"
    print("  PASS: calc_sector_median_pe matches oracle 100% across all cases and edge conditions.")

    # 4. calc_sector_pe_percentile & Continuity Correction
    print("\n[Test 4] calc_sector_pe_percentile & Continuity Correction Formula")
    # Small sample N=1
    assert abs(calc_sector_pe_percentile(15.0, [15.0]) - 50.0) < 1e-4
    # Small sample N=2: (0.5/2)*100 = 25.0%, (1.5/2)*100 = 75.0%
    assert abs(calc_sector_pe_percentile(10.0, [10.0, 20.0]) - 25.0) < 1e-4
    assert abs(calc_sector_pe_percentile(20.0, [10.0, 20.0]) - 75.0) < 1e-4
    # Small sample N=3: 16.67%, 50.0%, 83.33%
    assert abs(calc_sector_pe_percentile(20.0, [10.0, 20.0, 30.0]) - 50.0) < 1e-4
    # Ties handling
    assert abs(calc_sector_pe_percentile(20.0, [10.0, 20.0, 20.0, 30.0]) - 50.0) < 1e-4
    # Excluded conditions
    assert calc_sector_pe_percentile(-5.0, [10.0, 20.0]) is None
    assert calc_sector_pe_percentile(180.0, [10.0, 20.0]) is None
    assert calc_sector_pe_percentile(15.0, [-5.0, 0.0]) is None

    # Parity with oracle
    test_cases_pct = [
        (10.0, [10.0, 20.0, 30.0, 40.0]),
        (40.0, [10.0, 20.0, 30.0, 40.0]),
        (25.0, [10.0, 20.0, 30.0, 40.0]),
        (8.0, [8.0, 12.0, 16.0, 24.0]),
        (16.0, [8.0, 12.0, 16.0, 24.0]),
    ]
    for tgt, plist in test_cases_pct:
        act = calc_sector_pe_percentile(tgt, plist)
        exp = oracle_sector_pe_percentile(tgt, plist)
        assert abs(act - exp) < 1e-4, f"Mismatch for {tgt} in {plist}: {act} vs {exp}"
    print("  PASS: Continuity-corrected percentile formula verified against all oracle edge cases.")

    # 5. calc_sector_relative_metrics on List[PEBandResult]
    print("\n[Test 5] calc_sector_relative_metrics on List[PEBandResult]")
    r1 = PEBandResult(
        ticker="005930", name="삼성전자", current_price=70000.0, current_fwd_eps=7000.0,
        current_fwd_pe=10.0, pe_percentile=30.0, pe_min=8.0, pe_p25=11.0, pe_median=13.0,
        pe_p75=16.0, pe_max=20.0, pe_mean=13.5, target_bear=77000.0, target_base=91000.0,
        target_bull=112000.0, upside_bear=10.0, upside_base=30.0, upside_bull=60.0,
        hist_pe_series=pd.Series(), hist_price_series=pd.Series(), hist_eps_series=pd.Series(),
        sector="반도체",
    )
    r2 = PEBandResult(
        ticker="000660", name="SK하이닉스", current_price=150000.0, current_fwd_eps=10000.0,
        current_fwd_pe=15.0, pe_percentile=50.0, pe_min=8.0, pe_p25=11.0, pe_median=13.0,
        pe_p75=16.0, pe_max=20.0, pe_mean=13.5, target_bear=110000.0, target_base=130000.0,
        target_bull=160000.0, upside_bear=-26.7, upside_base=-13.3, upside_bull=6.7,
        hist_pe_series=pd.Series(), hist_price_series=pd.Series(), hist_eps_series=pd.Series(),
        sector="반도체",
    )
    r3 = PEBandResult(
        ticker="005380", name="현대차", current_price=200000.0, current_fwd_eps=30000.0,
        current_fwd_pe=6.67, pe_percentile=20.0, pe_min=5.0, pe_p25=6.0, pe_median=7.5,
        pe_p75=9.0, pe_max=12.0, pe_mean=7.7, target_bear=180000.0, target_base=225000.0,
        target_bull=270000.0, upside_bear=-10.0, upside_base=12.5, upside_bull=35.0,
        hist_pe_series=pd.Series(), hist_price_series=pd.Series(), hist_eps_series=pd.Series(),
        sector="자동차",
    )

    enriched = calc_sector_relative_metrics([r1, r2, r3])
    # Semiconductor sector has N=2 stocks [10.0, 15.0]
    # Median = 12.5
    assert abs(r1.sector_median_pe - 12.5) < 1e-4
    assert abs(r2.sector_median_pe - 12.5) < 1e-4
    assert r1.sector_stock_count == 2
    assert r2.sector_stock_count == 2
    # Relative P/E: 10 / 12.5 = 0.8x, 15 / 12.5 = 1.2x
    assert abs(r1.sector_relative_pe - 0.8) < 1e-4
    assert abs(r2.sector_relative_pe - 1.2) < 1e-4
    # Continuity-corrected percentiles for N=2: 25.0% and 75.0%
    assert abs(r1.sector_pe_percentile - 25.0) < 1e-4
    assert abs(r2.sector_pe_percentile - 75.0) < 1e-4

    # Automotive sector has N=1 stock [6.67] -> percentile = 50.0%
    assert abs(r3.sector_median_pe - 6.67) < 1e-2
    assert r3.sector_stock_count == 1
    assert abs(r3.sector_pe_percentile - 50.0) < 1e-4
    print("  PASS: calc_sector_relative_metrics enriches List[PEBandResult] with median, relative PE, and continuity-corrected percentile.")

    # 6. calc_sector_relative_metrics on pd.DataFrame
    print("\n[Test 6] calc_sector_relative_metrics on pd.DataFrame")
    df_test = pd.DataFrame({
        "ticker": ["005930", "000660", "005380"],
        "current_fwd_pe": [10.0, 15.0, 6.67],
        "sector": ["반도체", "반도체", "자동차"],
    })
    df_out = calc_sector_relative_metrics(df_test)
    assert "sector_median_pe" in df_out.columns
    assert "sector_relative_pe" in df_out.columns
    assert "sector_pe_percentile" in df_out.columns
    assert "sector_stock_count" in df_out.columns
    assert abs(df_out.loc[0, "sector_median_pe"] - 12.5) < 1e-4
    assert abs(df_out.loc[0, "sector_pe_percentile"] - 25.0) < 1e-4
    assert abs(df_out.loc[1, "sector_pe_percentile"] - 75.0) < 1e-4
    assert abs(df_out.loc[2, "sector_pe_percentile"] - 50.0) < 1e-4
    print("  PASS: calc_sector_relative_metrics enriches pd.DataFrame seamlessly.")

    # 7. run_screener with sector mapping integration
    print("\n[Test 7] run_screener Integration")
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    price_df = pd.DataFrame({
        "005930": [50000.0 + i * 500 for i in range(36)],
        "000660": [100000.0 + i * 1000 for i in range(36)],
    }, index=dates)
    eps_df = pd.DataFrame({
        "005930": [4000.0 + i * 50 for i in range(36)],
        "000660": [8000.0 + i * 100 for i in range(36)],
    }, index=dates)
    names = {"005930": "삼성전자", "000660": "SK하이닉스"}

    screen_res = run_screener(price_df, eps_df, names)
    assert len(screen_res) == 2
    for r in screen_res:
        assert r.sector == "반도체"
        assert r.sector_median_pe is not None
        assert r.sector_relative_pe is not None
        assert r.sector_pe_percentile is not None
        assert r.sector_stock_count == 2
    print("  PASS: run_screener integrates sector mapping and attaches sector metrics to all returned PEBandResults.")

    print("\n" + "=" * 70)
    print(">>> ALL MILESTONE 2 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()
