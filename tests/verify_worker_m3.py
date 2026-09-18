"""
tests/verify_worker_m3.py
-------------------------
Comprehensive Verification Suite for Milestone 3 (Research Standard Bands & Outlier Handling).
Tests:
  1. winsorize_pe (95% two-sided Winsorization, pre-filtering [1, 150], <10 sample guard, oracle parity)
  2. calc_mean_sd_bands (Mean ± 1/2SD, floor rule max(floor_pe, min_obs*0.8), <4 sample guard, oracle parity)
  3. calc_fixed_multiple_bands (Fixed multiples 8x/10x/12x/15x, targets & upsides, edge guards, oracle parity)
  4. Upgraded PEBandResult dataclass & calc_pe_band (band_model switching, full time-series band generation)
  5. M2 Review Polish (SectorMappingDict.__contains__ with normalization, _calc_sector_relative_df empty guard)
  6. run_screener integration with M3 band parameters
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calculator import (
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    calc_pe_band,
    run_screener,
    PEBandResult,
    SectorMappingDict,
    load_sector_mapping,
    _calc_sector_relative_df,
    _clean_ticker,
)
from tests.oracle import (
    oracle_winsorize,
    oracle_mean_sd_bands,
    oracle_fixed_multiple_bands,
    clean_ticker,
)


def verify_all():
    print("=" * 75)
    print(">>> Milestone 3 Verification: Research Standard Bands & Outlier Handling")
    print("=" * 75)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. winsorize_pe
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 1] winsorize_pe — 95% Two-sided Winsorization & Pre-filtering")
    
    # Normal series with extreme outliers
    vals = [10.0] * 20 + [12.0] * 20 + [14.0] * 20 + [145.0, 148.0]
    raw_s = pd.Series(vals)
    win_s = winsorize_pe(raw_s, lower_q=0.025, upper_q=0.975)
    oracle_s = oracle_winsorize(raw_s, lower_q=0.025, upper_q=0.975)
    assert len(win_s) == len(raw_s), "Length must be preserved"
    assert win_s.max() < 145.0, f"Upper spike must be clipped, got {win_s.max()}"
    assert win_s.var() < raw_s.var(), "Variance must be reduced"
    assert abs(win_s.median() - raw_s.median()) < 1e-4, "Median must be preserved"
    assert np.allclose(win_s.values, oracle_s.values), "Must match oracle 100%"
    print("  PASS: Extreme upper percentile capped, variance reduced, oracle parity verified.")

    # Lower percentile flooring
    low_vals = [1.2, 1.5] + [10.0] * 20 + [12.0] * 20
    win_low = winsorize_pe(pd.Series(low_vals), lower_q=0.05, upper_q=0.95)
    assert win_low.min() > 1.2, "Lower extreme must be floored"
    print("  PASS: Extreme lower percentile floored.")

    # Pre-filtering [min_pe=1.0, max_pe=150.0]
    dirty_s = pd.Series([-5.0, 0.0, 0.5, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 140.0, 160.0, 250.0])
    win_dirty = winsorize_pe(dirty_s, min_pe=1.0, max_pe=150.0)
    assert (win_dirty >= 1.0).all(), "All values must be >= 1.0"
    assert (win_dirty <= 150.0).all(), "All values must be <= 150.0"
    assert len(win_dirty) < len(dirty_s), "Out-of-bounds values must be filtered"
    print("  PASS: Pre-filtering [1.0, 150.0] excludes non-positive, near-zero, and extreme >150 P/Es.")

    # Small sample guard (< 10 valid elements)
    short_s = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    win_short = winsorize_pe(short_s)
    assert len(win_short) == 5, "Short series must pass through unmodified"
    assert list(win_short) == [10.0, 12.0, 14.0, 16.0, 18.0]
    print("  PASS: Small sample guard (<10 samples) returns copy without distortion.")

    # NaNs and Infs handling
    inf_s = pd.Series([10.0, np.nan, 12.0, np.inf, -np.inf, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])
    win_inf = winsorize_pe(inf_s)
    assert not win_inf.isna().any(), "NaNs must be dropped"
    assert not np.isinf(win_inf).any(), "Infs must be filtered"
    print("  PASS: NaNs and positive/negative infinities safely filtered.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. calc_mean_sd_bands
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 2] calc_mean_sd_bands — Research Standard Mean ± 1/2SD Bands")

    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    test_hist_pe = pd.Series([10.0 + (i % 6) * 1.5 for i in range(36)], index=dates)
    bands = calc_mean_sd_bands(test_hist_pe, winsorize=False)
    oracle_b = oracle_mean_sd_bands(test_hist_pe, winsorize=False)
    for k in ["mean", "std", "p1sd", "p2sd", "m1sd", "m2sd"]:
        assert k in bands, f"Missing key {k} in bands"
        assert abs(bands[k] - oracle_b[k]) < 1e-4, f"Mismatch on {k}: {bands[k]} vs {oracle_b[k]}"

    # Monotonic ordering
    assert bands["m2sd"] <= bands["m1sd"] <= bands["mean"] <= bands["p1sd"] <= bands["p2sd"], "Monotonic order failed"
    # ddof=1 sample std
    assert abs(bands["std"] - float(test_hist_pe.std(ddof=1))) < 1e-4, "Must use ddof=1"
    print("  PASS: Mean ± 1/2SD structure, monotonic ordering, ddof=1, and oracle parity verified.")

    # Floor protection rule on high-volatility series
    volatile_s = pd.Series([1.5, 2.0, 3.0, 14.0, 15.0])
    bands_vol = calc_mean_sd_bands(volatile_s, floor_pe=1.0, winsorize=False)
    oracle_vol = oracle_mean_sd_bands(volatile_s, floor_pe=1.0, winsorize=False)
    min_obs = float(volatile_s.min())
    expected_floor = max(1.0, min_obs * 0.8)
    assert bands_vol["m2sd"] >= expected_floor, f"m2sd {bands_vol['m2sd']} must be >= {expected_floor}"
    assert bands_vol["m1sd"] >= expected_floor, f"m1sd {bands_vol['m1sd']} must be >= {expected_floor}"
    assert abs(bands_vol["m2sd"] - oracle_vol["m2sd"]) < 1e-4
    print(f"  PASS: Floor protection rule enforced: floor_val={expected_floor:.2f}, m2sd={bands_vol['m2sd']:.2f} >= 1.0.")

    # Winsorize toggle impact on bands
    spike_s = pd.Series([10.0] * 20 + [12.0] * 20 + [145.0])
    b_raw = calc_mean_sd_bands(spike_s, winsorize=False)
    b_win = calc_mean_sd_bands(spike_s, winsorize=True)
    assert b_win["std"] < b_raw["std"], "Winsorized std must be strictly lower than raw std"
    assert b_win["mean"] < b_raw["mean"], "Winsorized mean must be lower than raw mean"
    print("  PASS: Winsorization toggle effectively insulates Mean ± SD bands from extreme spikes.")

    # Insufficient data guard (< 4 observations)
    assert calc_mean_sd_bands(pd.Series([10.0, 12.0, 14.0])) == {}, "Must return {} for <4 samples"
    assert calc_mean_sd_bands(pd.Series([])) == {}, "Must return {} for empty series"
    print("  PASS: Guard correctly returns empty dict for <4 observations.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. calc_fixed_multiple_bands
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 3] calc_fixed_multiple_bands — Fixed Multiple Valuation Multiples")

    multiples = [8.0, 10.0, 12.0, 15.0]
    eps_val = 5000.0
    price_val = 50000.0
    fm_res = calc_fixed_multiple_bands(multiples, current_eps=eps_val, current_price=price_val)
    oracle_fm = oracle_fixed_multiple_bands(multiples, current_eps=eps_val, current_price=price_val)

    assert fm_res["targets"][8.0] == 40000.0
    assert fm_res["targets"][10.0] == 50000.0
    assert fm_res["targets"][12.0] == 60000.0
    assert fm_res["targets"][15.0] == 75000.0
    assert abs(fm_res["upsides"][8.0] - (-20.0)) < 1e-4
    assert abs(fm_res["upsides"][10.0] - 0.0) < 1e-4
    assert abs(fm_res["upsides"][12.0] - 20.0) < 1e-4
    assert abs(fm_res["upsides"][15.0] - 50.0) < 1e-4
    assert fm_res == oracle_fm, "Must match oracle_fixed_multiple_bands 100%"
    print("  PASS: Fixed multiple targets (8x/10x/12x/15x) and upside percentages verified against oracle.")

    # Edge cases
    # Negative EPS -> target=0.0
    neg_eps_res = calc_fixed_multiple_bands([10.0], current_eps=-500.0, current_price=50000.0)
    assert neg_eps_res["targets"][10.0] == 0.0
    # Zero Price -> upside=0.0
    zero_p_res = calc_fixed_multiple_bands([10.0], current_eps=5000.0, current_price=0.0)
    assert zero_p_res["upsides"][10.0] == 0.0
    # Empty list
    empty_m_res = calc_fixed_multiple_bands([], current_eps=5000.0, current_price=50000.0)
    assert empty_m_res["targets"] == {} and empty_m_res["upsides"] == {}
    print("  PASS: Edge cases (negative EPS, zero price, empty multiples list) guarded.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PEBandResult & calc_pe_band Upgraded Engine
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 4] PEBandResult & calc_pe_band Upgraded Models & Time-Series Arrays")

    # Generate synthetic price and EPS data (36 months)
    dates_36 = pd.date_range("2021-01-01", periods=36, freq="MS")
    p_series = pd.Series([50000.0 + i * 500 for i in range(36)], index=dates_36)
    e_series = pd.Series([4000.0 + i * 50 for i in range(36)], index=dates_36)

    # (A) Default Percentile Mode
    res_pct = calc_pe_band(
        ticker="005930",
        name="삼성전자",
        price_series=p_series,
        eps_series=e_series,
        band_years=5,
        band_model="percentile",
        winsorize=True,
    )
    assert res_pct is not None
    assert res_pct.band_model == "percentile"
    assert res_pct.band_series_pct is not None
    assert list(res_pct.band_series_pct.columns) == ["p10", "p25", "p50", "p75", "p90"]
    assert res_pct.band_series_sd is not None
    assert list(res_pct.band_series_sd.columns) == ["-2SD", "-1SD", "Mean", "+1SD", "+2SD"]
    assert res_pct.band_series_fixed is not None
    assert list(res_pct.band_series_fixed.columns) == ["8x", "10x", "12x", "15x"]
    assert res_pct.mean_sd_metrics is not None
    assert res_pct.fixed_targets is not None
    assert res_pct.fixed_upsides is not None
    # Percentile target verification
    assert abs(res_pct.target_base - (res_pct.current_fwd_eps * res_pct.pe_median)) < 1e-4
    print("  PASS: Percentile mode generates full time-series band arrays for instant UI switching.")

    # (B) Research Standard Mean ± SD Mode
    res_sd = calc_pe_band(
        ticker="005930",
        name="삼성전자",
        price_series=p_series,
        eps_series=e_series,
        band_years=5,
        band_model="mean_sd",
        winsorize=True,
    )
    assert res_sd is not None
    assert res_sd.band_model == "mean_sd"
    # Target prices must align with Mean ± SD
    expected_bear_sd = res_sd.current_fwd_eps * res_sd.mean_sd_metrics["m1sd"]
    expected_base_sd = res_sd.current_fwd_eps * res_sd.mean_sd_metrics["mean"]
    expected_bull_sd = res_sd.current_fwd_eps * res_sd.mean_sd_metrics["p1sd"]
    assert abs(res_sd.target_bear - expected_bear_sd) < 1e-4
    assert abs(res_sd.target_base - expected_base_sd) < 1e-4
    assert abs(res_sd.target_bull - expected_bull_sd) < 1e-4
    # Upsides must update dynamically
    expected_up_base = ((expected_base_sd / res_sd.current_price) - 1.0) * 100.0
    assert abs(res_sd.upside_base - expected_up_base) < 1e-4
    print("  PASS: Mean ± SD mode sets target_bear=m1sd, target_base=mean, target_bull=p1sd with dynamic upsides.")

    # (C) Fixed Multiples Mode
    res_fixed = calc_pe_band(
        ticker="005930",
        name="삼성전자",
        price_series=p_series,
        eps_series=e_series,
        band_years=5,
        band_model="fixed",
        fixed_multiples=[8.0, 10.0, 12.0, 15.0],
    )
    assert res_fixed is not None
    assert res_fixed.band_model == "fixed"
    assert res_fixed.fixed_targets[10.0] == res_fixed.current_fwd_eps * 10.0
    assert res_fixed.fixed_targets[15.0] == res_fixed.current_fwd_eps * 15.0
    assert res_fixed.target_bear <= res_fixed.target_base <= res_fixed.target_bull
    print("  PASS: Fixed multiple mode attaches fixed_targets/upsides and monotonic scenario targets.")

    # (D) Backward Compatibility: Old Positional Call
    res_old_pos = calc_pe_band(
        "005930",
        "삼성전자",
        p_series,
        e_series,
        5,
        "반도체",  # 6th positional arg was sector in M2
    )
    assert res_old_pos is not None
    assert res_old_pos.sector == "반도체", f"Expected sector '반도체', got {res_old_pos.sector}"
    assert res_old_pos.band_model == "percentile"
    print("  PASS: Backward compatibility preserved for legacy positional signatures.")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. M2 Review Feedback Polish
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 5] M2 Review Feedback Polish")

    sec_map = load_sector_mapping()
    assert isinstance(sec_map, SectorMappingDict)
    # 1. SectorMappingDict.__contains__ normalization
    assert "005930" in sec_map, "Clean ticker must be in mapping"
    assert "A005930" in sec_map, "A-prefixed ticker must evaluate to True via __contains__"
    assert "000660" in sec_map
    assert "A000660" in sec_map
    assert "UNKNOWN_999" not in sec_map
    print("  PASS: SectorMappingDict.__contains__ supports normalized and raw tickers flawlessly.")

    # 2. _calc_sector_relative_df empty DataFrame guard
    empty_df = pd.DataFrame()
    res_empty = _calc_sector_relative_df(empty_df, sec_map)
    assert res_empty.empty

    col_empty_df = pd.DataFrame(index=[0, 1, 2])
    res_col_empty = _calc_sector_relative_df(col_empty_df, sec_map)
    assert len(res_col_empty.columns) == 0
    print("  PASS: _calc_sector_relative_df empty DataFrame guards prevent IndexError/crash.")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. run_screener Integration with M3 Band Models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 6] run_screener Integration with M3 Options")

    price_df = pd.DataFrame({"005930": p_series, "000660": p_series * 1.5})
    eps_df = pd.DataFrame({"005930": e_series, "000660": e_series * 1.8})
    names = {"005930": "삼성전자", "000660": "SK하이닉스"}

    # Run screener in mean_sd mode
    screen_sd = run_screener(price_df, eps_df, names, band_model="mean_sd", winsorize=True)
    assert len(screen_sd) == 2
    for r in screen_sd:
        assert r.band_model == "mean_sd"
        assert r.band_series_pct is not None
        assert r.band_series_sd is not None
        assert r.band_series_fixed is not None
        assert r.sector == "반도체"
    print("  PASS: run_screener propagates M3 band options and enriches all results.")

    print("\n" + "=" * 75)
    print(">>> ALL 6 MILESTONE 3 VERIFICATION SUITES PASSED WITH 100% PARITY!")
    print("=" * 75)


if __name__ == "__main__":
    verify_all()
