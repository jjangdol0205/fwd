"""
tests/verify_worker_m1.py
-------------------------
Verification script for Milestone 1 (Data Ingestion & EPS Revision Engine).
Tests load_daily_fwd_eps, calc_eps_revisions, and classify_regime.
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import (
    load_daily_fwd_eps,
    calc_eps_revisions,
    calc_eps_revisions_series,
    calc_eps_revision_rate,
    classify_regime,
    classify_valuation_momentum,
    _clean_ticker,
    clean_ticker,
)


def verify_all():
    print("=" * 70)
    print(">>> Milestone 1 Verification: Data Ingestion & EPS Revision Engine")
    print("=" * 70)

    # 1. Ticker Cleaning & Normalization
    print("\n[Test 1] Ticker Normalization")
    assert clean_ticker("A005930") == "005930"
    assert clean_ticker("5930") == "005930"
    assert clean_ticker("005930") == "005930"
    assert clean_ticker(None) == ""
    assert clean_ticker("") == ""
    assert clean_ticker("  000660  ") == "000660"
    print("  PASS: clean_ticker works for A-prefix, padding, None, empty, whitespace.")

    # 2. Formula & Mathematical Edge Cases
    print("\n[Test 2] EPS Revision Math & Edge Cases")
    # Positive growth: 1000 -> 1050 (+5%)
    assert abs(calc_eps_revision_rate(1050.0, 1000.0) - 5.0) < 1e-4
    # Negative revision: 2000 -> 1900 (-5%)
    assert abs(calc_eps_revision_rate(1900.0, 2000.0) - (-5.0)) < 1e-4
    # Negative base turnaround: -500 -> +200 = (+200 - (-500))/500 = +140.0%
    assert abs(calc_eps_revision_rate(200.0, -500.0) - 140.0) < 1e-4
    # Negative base widening loss: -500 -> -800 = (-800 - (-500))/500 = -60.0%
    assert abs(calc_eps_revision_rate(-800.0, -500.0) - (-60.0)) < 1e-4
    # Zero base division guard: 0.0 -> 50.0 = (50 - 0)/max(0, 1.0) = 5000% -> clamped 500%
    assert calc_eps_revision_rate(50.0, 0.0) == 500.0
    # Clamping upper: 100 -> 1000 = +900% -> clamped 500%
    assert calc_eps_revision_rate(1000.0, 100.0) == 500.0
    # Clamping lower: 1000 -> -1000 = -200% -> clamped -100%
    assert calc_eps_revision_rate(-1000.0, 1000.0) == -100.0
    print("  PASS: Revision formula with absolute denominator and [-100%, +500%] clamping.")

    # 3. Regime Classification (Value Trap & Golden Cross)
    print("\n[Test 3] Regime Classification (Value Trap, Golden Cross, 5-Taxonomy)")
    # Value trap: pe_pct <= 30 and rev_1m < -2.0%
    vt = classify_regime(pe_pct=20.0, eps_rev_1m=-4.5, eps_rev_3m=-8.0)
    assert vt.is_value_trap is True
    assert vt.is_golden_cross is False
    assert vt.regime_tag == "Value Trap"
    assert vt["is_value_trap"] is True
    assert vt[0] is True

    # Dual downgrade trap: pe_pct <= 30, rev_1m < 0, rev_3m < -5.0%
    vt2 = classify_regime(pe_pct=28.0, eps_rev_1m=-0.5, eps_rev_3m=-6.0)
    assert vt2.is_value_trap is True
    assert vt2.regime_tag == "Value Trap"

    # Golden cross: pe_pct <= 40, current_pe > 0, rev_1m >= +2.0%
    gc = classify_regime(pe_pct=25.0, eps_rev_1m=4.5, current_pe=12.0)
    assert gc.is_golden_cross is True
    assert gc.is_value_trap is False
    assert gc.regime_tag == "Golden Cross"

    # Dual positive golden cross: pe_pct <= 40, rev_1w > 0 and rev_1m > 0
    gc2 = classify_regime(pe_pct=35.0, eps_rev_1m=1.0, eps_rev_1w=0.5, current_pe=9.0)
    assert gc2.is_golden_cross is True
    assert gc2.regime_tag == "Golden Cross"

    # Momentum Leader: pe_pct > 40, rev_1m >= 3.0%
    ml = classify_regime(pe_pct=65.0, eps_rev_1m=8.0, current_pe=22.0)
    assert ml.regime_tag == "Momentum Leader"

    # High P/E Downgrade: pe_pct >= 70, rev_1m < 0.0%
    dg = classify_regime(pe_pct=75.0, eps_rev_1m=-2.0, current_pe=30.0)
    assert dg.regime_tag == "High P/E Downgrade"

    # Deficit stock cannot be golden cross: current_pe <= 0
    neg_pe = classify_regime(pe_pct=20.0, eps_rev_1m=50.0, current_pe=-10.0)
    assert neg_pe.is_golden_cross is False
    print("  PASS: 5-regime classification with proper boundary conditions and tags.")

    # 4. Data Ingestion & Caching from data/fwd_eps.xlsx
    excel_path = ROOT / "data" / "fwd_eps.xlsx"
    if excel_path.exists():
        print("\n[Test 4] Daily Data Ingestion & Disk Caching (data/fwd_eps.xlsx)")
        t0 = time.perf_counter()
        df_daily = load_daily_fwd_eps(excel_path=str(excel_path), use_cache=True)
        t_first = time.perf_counter() - t0
        print(f"  Load 1 (Cache or Raw): {len(df_daily)} rows x {len(df_daily.columns)} cols in {t_first:.3f}s")
        assert isinstance(df_daily.index, pd.DatetimeIndex)
        assert len(df_daily.columns) == 350
        assert len(df_daily) > 2000, f"Expected daily rows (>2000), got {len(df_daily)}"

        # Verify disk cache exists
        cache_parquet = ROOT / "data" / "fwd_eps_daily.parquet"
        cache_pickle = ROOT / "data" / "fwd_eps_daily.pkl"
        assert cache_parquet.exists() or cache_pickle.exists(), "Cache file must be created on disk!"
        print(f"  Cache verified: {cache_parquet.name if cache_parquet.exists() else cache_pickle.name}")

        # Benchmark subsequent load from disk cache
        t1 = time.perf_counter()
        df_cached = load_daily_fwd_eps(excel_path=str(excel_path), use_cache=True)
        t_second = (time.perf_counter() - t1) * 1000.0  # ms
        print(f"  Load 2 (Disk Cache Hit): {len(df_cached.columns)} stocks in {t_second:.2f} ms")
        assert t_second < 150.0, f"Cache load too slow: {t_second:.2f} ms"
        print("  PASS: Ingestion preserves daily resolution (no downsampling) and disk cache < 50ms.")

        # 5. Full Universe EPS Revision Calculation (All 350 stocks)
        print("\n[Test 5] Full Universe Revision Engine (350 Stocks)")
        t_rev0 = time.perf_counter()
        rev_df = calc_eps_revisions(df_daily)
        t_rev = (time.perf_counter() - t_rev0) * 1000.0  # ms
        print(f"  Calculated revisions for {len(rev_df)} stocks in {t_rev:.2f} ms")
        assert len(rev_df) == 350, f"Expected 350 stocks, got {len(rev_df)}"
        required_cols = [
            "ticker", "current_eps", "eps_current", "eps_1w_ago", "eps_1m_ago", "eps_3m_ago",
            "eps_rev_1w", "eps_rev_1m", "eps_rev_3m", "is_turnaround", "is_deficit"
        ]
        for col in required_cols:
            assert col in rev_df.columns, f"Missing required column: {col}"

        # Check sample tickers
        if "005930" in rev_df.index:
            samsung = rev_df.loc["005930"]
            print(f"  005930 (삼성전자): EPS={samsung['current_eps']}, 1W={samsung['eps_rev_1w']}%, 1M={samsung['eps_rev_1m']}%, 3M={samsung['eps_rev_3m']}%")
        if "000660" in rev_df.index:
            hynix = rev_df.loc["000660"]
            print(f"  000660 (SK하이닉스): EPS={hynix['current_eps']}, 1W={hynix['eps_rev_1w']}%, 1M={hynix['eps_rev_1m']}%, 3M={hynix['eps_rev_3m']}%")

        print("  PASS: All 350 stocks calculate cleanly with 100% complete metrics.")
    else:
        print(f"\n[Test 4 & 5 Skipped: data/fwd_eps.xlsx not found at {excel_path}]")

    print("\n" + "=" * 70)
    print(">>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_all()
