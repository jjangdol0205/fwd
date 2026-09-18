"""
tests/oracle.py
Authoritative Reference Specification (Oracle) for Dataguide Valuation System.
Derived strictly from ORIGINAL_REQUEST.md, PROJECT.md, and TEST_INFRA.md.
Provides independent mathematical benchmarks for all 14 features across Tiers 1-4.
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import re


def clean_ticker(val: Any) -> str:
    """Normalize ticker code: remove 'A'/'a' prefix, strip, 6-digit zfill."""
    if val is None:
        return ""
    s = str(val).strip()
    s = re.sub(r"^[Aa](?=\d)", "", s)
    return s.zfill(6) if s.isdigit() else s


def oracle_eps_revision(
    current_eps: float,
    base_eps: float,
    clamp_min: float = -100.0,
    clamp_max: float = 500.0,
) -> Optional[float]:
    """
    Authoritative EPS revision formula:
      Revision (%) = (EPS_t - EPS_base) / max(|EPS_base|, 1.0) * 100
      Clamped to [clamp_min, clamp_max].
    """
    if current_eps is None or base_eps is None:
        return None
    if np.isnan(current_eps) or np.isnan(base_eps):
        return None

    denom = max(abs(float(base_eps)), 1.0)
    pct = ((float(current_eps) - float(base_eps)) / denom) * 100.0
    return float(np.clip(pct, clamp_min, clamp_max))


def oracle_calc_eps_revisions_series(
    eps_series: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    Authoritative calculation of 1W (5 trading days), 1M (20 trading days),
    and 3M (60 trading days) EPS revision rates from a sorted daily EPS series.
    """
    s = eps_series.dropna().sort_index()
    n = len(s)
    res: Dict[str, Any] = {
        "current_eps": None,
        "eps_1w_ago": None,
        "eps_1m_ago": None,
        "eps_3m_ago": None,
        "eps_rev_1w": None,
        "eps_rev_1m": None,
        "eps_rev_3m": None,
        "is_turnaround": False,
        "is_deficit": False,
    }

    if n == 0:
        return res

    latest = float(s.iloc[-1])
    res["current_eps"] = latest
    res["is_deficit"] = latest <= 0.0

    if n > 5:
        base_1w = float(s.iloc[-6])
        res["eps_1w_ago"] = base_1w
        res["eps_rev_1w"] = oracle_eps_revision(latest, base_1w)
    if n > 20:
        base_1m = float(s.iloc[-21])
        res["eps_1m_ago"] = base_1m
        res["eps_rev_1m"] = oracle_eps_revision(latest, base_1m)
        if base_1m <= 0.0 and latest > 0.0:
            res["is_turnaround"] = True
    if n > 60:
        base_3m = float(s.iloc[-61])
        res["eps_3m_ago"] = base_3m
        res["eps_rev_3m"] = oracle_eps_revision(latest, base_3m)

    return res


def oracle_classify_regime(
    pe_pct: float,
    rev_1m: Optional[float],
    rev_3m: Optional[float] = None,
    rev_1w: Optional[float] = None,
    current_pe: float = 10.0,
) -> Dict[str, Any]:
    """
    Authoritative 5-regime classification:
      - Value Trap: pe_pct <= 30.0 and (rev_1m < -2.0 or (rev_1m < 0.0 and rev_3m < -5.0))
      - Golden Cross: pe_pct <= 40.0 and current_pe > 0 and (rev_1m >= +2.0 or (rev_1w > 0 and rev_1m > 0))
      - Momentum Leader: pe_pct > 40.0 and rev_1m >= +3.0
      - High P/E Downgrade: pe_pct >= 70.0 and rev_1m < 0.0
      - Neutral: all others
    """
    if rev_1m is None or current_pe <= 0.0 or pe_pct is None:
        return {
            "is_value_trap": False,
            "is_golden_cross": False,
            "regime_tag": "Neutral",
        }

    is_trap = (pe_pct <= 30.0) and (
        rev_1m < -2.0 or (rev_1m < 0.0 and rev_3m is not None and rev_3m < -5.0)
    )

    is_golden = (
        (pe_pct <= 40.0)
        and (current_pe > 0.0)
        and (rev_1m >= 2.0 or (rev_1w is not None and rev_1w > 0.0 and rev_1m > 0.0))
    )

    # Disjointness: if both match corner boundary, golden cross has precedence if rev_1m > 0
    if is_golden:
        tag = "Golden Cross"
    elif is_trap:
        tag = "Value Trap"
    elif pe_pct > 40.0 and rev_1m >= 3.0:
        tag = "Momentum Leader"
    elif pe_pct >= 70.0 and rev_1m < 0.0:
        tag = "High P/E Downgrade"
    else:
        tag = "Neutral"

    return {
        "is_value_trap": bool(is_trap),
        "is_golden_cross": bool(is_golden),
        "regime_tag": tag,
    }


def oracle_sector_median_pe(
    pe_list: List[float], max_pe_cap: float = 150.0
) -> Optional[float]:
    """
    Authoritative Sector Median P/E calculation:
      Excludes non-positive values and extreme values > max_pe_cap.
    """
    valid = [p for p in pe_list if p is not None and not np.isnan(p) and 0.0 < p <= max_pe_cap]
    if not valid:
        return None
    return float(np.median(valid))


def oracle_sector_pe_percentile(
    target_pe: float,
    all_sector_pe_list: List[float],
    max_pe_cap: float = 150.0,
) -> Optional[float]:
    """
    Continuity-corrected hazard-free sector percentile ranking:
      Sector_Pct = (rank_0 + 0.5) / N * 100
      Where rank_0 is 0-indexed position in ascending sorted positive P/E list.
    """
    if target_pe is None or np.isnan(target_pe) or target_pe <= 0.0 or target_pe > max_pe_cap:
        return None

    valid = sorted(
        [p for p in all_sector_pe_list if p is not None and not np.isnan(p) and 0.0 < p <= max_pe_cap]
    )
    n = len(valid)
    if n == 0:
        return None

    # Find position (average rank if duplicate)
    indices = [i for i, v in enumerate(valid) if np.isclose(v, target_pe, atol=1e-5)]
    if indices:
        avg_rank = float(np.mean(indices))
    else:
        # Interpolated rank
        avg_rank = float(np.searchsorted(valid, target_pe))
        avg_rank = min(avg_rank, n - 0.5)

    pct = ((avg_rank + 0.5) / n) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


def oracle_winsorize(
    series: pd.Series,
    lower_q: float = 0.025,
    upper_q: float = 0.975,
    min_pe: float = 1.0,
    max_pe: float = 150.0,
) -> pd.Series:
    """
    Authoritative 95% Two-sided Winsorization:
      Pre-filters P/E into [min_pe, max_pe], clips extremes to [q_low, q_high].
    """
    s = series.dropna()
    valid = s[(s >= min_pe) & (s <= max_pe)]
    if len(valid) < 10:
        return valid.copy()

    q_low = float(valid.quantile(lower_q))
    q_high = float(valid.quantile(upper_q))
    return valid.clip(lower=q_low, upper=q_high)


def oracle_mean_sd_bands(
    hist_pe: pd.Series,
    floor_pe: float = 1.0,
    winsorize: bool = True,
) -> Dict[str, float]:
    """
    Authoritative Research Standard Mean ± 1/2SD Bands with floor protection.
    """
    s = oracle_winsorize(hist_pe) if winsorize else hist_pe.dropna()
    if len(s) < 4:
        return {}

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    min_observed = float(s.min())
    floor_val = max(floor_pe, min_observed * 0.8)

    return {
        "mean": mean,
        "std": std,
        "p2sd": mean + 2.0 * std,
        "p1sd": mean + 1.0 * std,
        "m1sd": max(mean - 1.0 * std, floor_val),
        "m2sd": max(mean - 2.0 * std, floor_val),
    }


def oracle_fixed_multiple_bands(
    multiples: List[float],
    current_eps: float,
    current_price: float,
) -> Dict[str, Dict[float, float]]:
    """
    Authoritative Fixed Multiple Targets and Upsides.
    """
    targets = {}
    upsides = {}
    for m in multiples:
        tgt = current_eps * m if current_eps > 0 else 0.0
        targets[m] = float(tgt)
        if current_price > 0:
            upsides[m] = float(((tgt / current_price) - 1.0) * 100.0)
        else:
            upsides[m] = 0.0

    return {"targets": targets, "upsides": upsides}


def oracle_apply_quick_preset(
    df: pd.DataFrame, preset_key: str
) -> pd.DataFrame:
    """
    Authoritative quick preset filters:
      - 'ALL': returns df
      - 'TURNAROUND': (pe_pct <= 40 | sector_pe_pct <= 40) & (eps_rev_1m > 0) & (upside_base >= 10.0)
      - 'EPS_TOP': eps_rev_1m >= 3.0 & current_eps > 0 (sorted desc by eps_rev_1m, top 25)
      - 'VALUE': pe_pct <= 25 & eps_rev_1m >= -2.0 & (0.0 < current_fwd_pe <= 15.0) & upside_base >= 15.0
      - 'TRAP': pe_pct <= 30 & eps_rev_1m < -3.0
    """
    if df is None or df.empty or len(df.columns) == 0:
        return pd.DataFrame() if df is None else df.copy()

    key = str(preset_key).strip().upper() if preset_key is not None else "ALL"
    if key == "ALL" or not key:
        return df.copy()

    if key == "TURNAROUND":
        s_pe_pct = df["pe_percentile"] if "pe_percentile" in df.columns else pd.Series(100.0, index=df.index)
        s_sec_pct = df["sector_pe_percentile"] if "sector_pe_percentile" in df.columns else pd.Series(100.0, index=df.index)
        s_rev_1m = df["eps_rev_1m"] if "eps_rev_1m" in df.columns else pd.Series(-999.0, index=df.index)
        s_upside = df["upside_base"] if "upside_base" in df.columns else pd.Series(-999.0, index=df.index)

        cond_pe = (s_pe_pct <= 40.0) | (s_sec_pct <= 40.0)
        cond_rev = s_rev_1m > 0.0
        cond_up = s_upside >= 10.0
        return df[cond_pe & cond_rev & cond_up]

    elif key == "EPS_TOP":
        s_rev_1m = df["eps_rev_1m"] if "eps_rev_1m" in df.columns else pd.Series(-999.0, index=df.index)
        s_fwd_eps = df["current_fwd_eps"] if "current_fwd_eps" in df.columns else pd.Series(-999.0, index=df.index)

        cond_rev = s_rev_1m >= 3.0
        cond_eps = s_fwd_eps > 0.0
        sub = df[cond_rev & cond_eps]
        if "eps_rev_1m" in sub.columns:
            return sub.sort_values(by="eps_rev_1m", ascending=False).head(25)
        return sub.head(25)

    elif key == "VALUE":
        s_pe_pct = df["pe_percentile"] if "pe_percentile" in df.columns else pd.Series(100.0, index=df.index)
        s_rev_1m = df["eps_rev_1m"] if "eps_rev_1m" in df.columns else pd.Series(-999.0, index=df.index)
        s_fwd_pe = df["current_fwd_pe"] if "current_fwd_pe" in df.columns else pd.Series(999.0, index=df.index)
        s_upside = df["upside_base"] if "upside_base" in df.columns else pd.Series(-999.0, index=df.index)

        cond_pct = s_pe_pct <= 25.0
        cond_rev = s_rev_1m >= -2.0
        cond_pe = (s_fwd_pe > 0.0) & (s_fwd_pe <= 15.0)
        cond_up = s_upside >= 15.0
        return df[cond_pct & cond_rev & cond_pe & cond_up]

    elif key == "TRAP":
        s_pe_pct = df["pe_percentile"] if "pe_percentile" in df.columns else pd.Series(100.0, index=df.index)
        s_rev_1m = df["eps_rev_1m"] if "eps_rev_1m" in df.columns else pd.Series(999.0, index=df.index)

        cond_pct = s_pe_pct <= 30.0
        cond_rev = s_rev_1m < -3.0
        return df[cond_pct & cond_rev]

    return df.copy()


def generate_synthetic_stock_series(
    n_days: int = 100,
    base_price: float = 50000.0,
    base_eps: float = 4000.0,
    eps_trend: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Generates synthetic daily trading dates, prices, and EPS."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-07-10", periods=n_days)
    
    price_rets = rng.normal(0.0005, 0.015, n_days)
    price = base_price * np.cumprod(1.0 + price_rets)
    
    eps_rets = rng.normal(eps_trend, 0.005, n_days)
    eps = base_eps * np.cumprod(1.0 + eps_rets)
    
    return pd.Series(price, index=dates), pd.Series(eps, index=dates)
