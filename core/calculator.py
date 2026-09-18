"""
calculator.py
-------------
12개월 Fwd EPS 기반 P/E 밴드 및 섹터 상대 밸류에이션 계산 모듈 (M1, M2 통합)
"""

import os
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# M1 모듈 임포트 (Data Ingestion & Revision Engine)
try:
    from core.data_loader import (
        _clean_ticker as loader_clean_ticker,
        calc_eps_revisions_series,
        calc_eps_revisions,
        classify_regime,
    )
except ImportError:
    loader_clean_ticker = None
    calc_eps_revisions_series = None
    calc_eps_revisions = None
    classify_regime = None


def _clean_ticker(val: Any) -> str:
    """종목코드 정규화 (A prefix 제거, 공백 제거, 6자리 zfill)"""
    if loader_clean_ticker is not None:
        return loader_clean_ticker(val)
    if val is None:
        return ""
    val = str(val).strip()
    if not val or val.lower() in ("nan", "none"):
        return ""
    val = re.sub(r"^[Aa](?=\d)", "", val)
    return val.zfill(6) if val.isdigit() else val


class SectorMappingDict(dict):
    """
    원시 종목코드('A005930')와 정규화 종목코드('005930') 양방향 조회를 지원하는 딕셔너리.
    len()은 고유 종목수(350개)를 정확히 반영합니다.
    """
    def __getitem__(self, key: Any) -> str:
        if super().__contains__(key):
            return super().__getitem__(key)
        ck = _clean_ticker(key)
        if super().__contains__(ck):
            return super().__getitem__(ck)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        ck = _clean_ticker(key)
        return super().__contains__(ck)

    def get(self, key: Any, default: Any = "기타/미분류") -> Any:
        if super().__contains__(key):
            return super().get(key, default)
        ck = _clean_ticker(key)
        if super().__contains__(ck):
            return super().get(ck, default)
        return default


def load_sector_mapping(path: str = "data/sector_mapping.json") -> SectorMappingDict:
    """
    350개 KOSPI200/KOSDAQ150 종목 WICS 섹터 매핑 로드.
    반환: {ticker: sector_name}
    """
    p = Path(path)
    mapping = SectorMappingDict()
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                ck = _clean_ticker(k)
                mapping[ck] = str(v)
            return mapping
        except Exception as e:
            logger.warning(f"Failed to load sector mapping from {path}: {e}")

    # Fallback to data/universe.csv if sector column exists
    univ_path = Path("data/universe.csv")
    if univ_path.exists():
        try:
            u_df = pd.read_csv(univ_path)
            if "sector" in u_df.columns:
                for _, row in u_df.iterrows():
                    ck = _clean_ticker(row["ticker"])
                    mapping[ck] = str(row["sector"])
        except Exception:
            pass

    return mapping


@dataclass
class PEBandResult:
    ticker: str
    name: str
    current_price: float
    current_fwd_eps: float
    current_fwd_pe: float
    pe_percentile: float        # 현재 P/E의 역사적 백분위 (0~100)
    pe_min: float
    pe_p25: float
    pe_median: float
    pe_p75: float
    pe_max: float
    pe_mean: float
    target_bear: float          # 25th percentile P/E × 현재 Fwd EPS
    target_base: float          # Median P/E × 현재 Fwd EPS
    target_bull: float          # 75th percentile P/E × 현재 Fwd EPS
    upside_bear: float          # %
    upside_base: float          # %
    upside_bull: float          # %
    hist_pe_series: pd.Series   # 역사적 Fwd P/E 시계열 (차트용)
    hist_price_series: pd.Series
    hist_eps_series: pd.Series

    # ── R2: Sector Relative Metrics (M2) ──
    sector: str = "기타/미분류"
    sector_median_pe: Optional[float] = None
    sector_relative_pe: Optional[float] = None
    sector_pe_percentile: Optional[float] = None
    sector_stock_count: int = 0

    # ── R1: Revision & Momentum Metrics (M1) ──
    eps_rev_1w: Optional[float] = None
    eps_rev_1m: Optional[float] = None
    eps_rev_3m: Optional[float] = None
    is_value_trap: bool = False
    is_golden_cross: bool = False
    regime_tag: str = "Neutral"

    # ── R3: Research Standard Bands & Fixed Multiple Metrics (M3) ──
    band_model: str = "percentile"
    band_series_pct: Optional[pd.DataFrame] = None
    band_series_sd: Optional[pd.DataFrame] = None
    band_series_fixed: Optional[pd.DataFrame] = None
    mean_sd_metrics: Optional[Dict[str, float]] = None
    fixed_targets: Optional[Dict[float, float]] = None
    fixed_upsides: Optional[Dict[float, float]] = None


def calc_fwd_pe_series(price: pd.Series, fwd_eps: pd.Series) -> pd.Series:
    """
    시계열 Fwd P/E 계산
    - 음수 EPS는 NaN 처리 (P/E 의미 없음)
    """
    pe = price / fwd_eps
    pe[fwd_eps <= 0] = np.nan
    pe[pe > 200] = np.nan   # 극단값 제거 (P/E 200 초과)
    pe[pe < 0] = np.nan
    return pe


def winsorize_pe(
    series: pd.Series,
    lower_q: float = 0.025,
    upper_q: float = 0.975,
    min_pe: float = 1.0,
    max_pe: float = 150.0,
) -> pd.Series:
    """
    95% Two-sided Winsorization (R3 / AC 33)
    일시적 적자나 일회성 손익으로 인한 비정상 P/E 왜곡을 보정하기 위해
    [min_pe, max_pe] 유효 구간으로 사전 필터링하고,
    하위 lower_q (2.5%) 및 상위 upper_q (97.5%) 분위수로 클리핑합니다.
    유효 표본이 10개 미만인 경우 원본 복사본을 반환합니다.
    """
    s = series.dropna()
    valid = s[(s >= min_pe) & (s <= max_pe)]
    if len(valid) < 10:
        return valid.copy()

    q_low = float(valid.quantile(lower_q))
    q_high = float(valid.quantile(upper_q))
    return valid.clip(lower=q_low, upper=q_high)


def calc_mean_sd_bands(
    hist_pe: pd.Series,
    floor_pe: float = 1.0,
    winsorize: bool = True,
) -> Dict[str, float]:
    """
    증권사 리서치 센터 표준 Mean ± 1SD / ± 2SD 밴드 산출 (R3 / AC 33)
    - winsorize=True인 경우 95% Winsorization 사전 적용
    - 유효 관측치 4개 미만 시 빈 딕셔너리 반환
    - 표본표준편차(ddof=1) 사용
    - Floor Rule: 경기순환 하강 국면에서 하단 밴드가 음수 또는 비정상 수준으로 왜곡되는 것을 방지하기 위해
      floor_val = max(floor_pe, min_observed * 0.8)를 적용하여 m1sd, m2sd에 하한 설정.
    """
    s = winsorize_pe(hist_pe) if winsorize else hist_pe.dropna()
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


def calc_fixed_multiple_bands(
    multiples: List[float],
    current_eps: float,
    current_price: float,
) -> Dict[str, Dict[float, float]]:
    """
    고정 배수 밴드 목표가 및 업사이드 산출 (R3 / AC 33)
    - targets: {multiple: current_eps * multiple (if current_eps > 0 else 0.0)}
    - upsides: {multiple: ((target / current_price) - 1.0) * 100.0 (if current_price > 0 else 0.0)}
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


def calc_sector_median_pe(
    pe_list: List[float],
    max_pe_cap: float = 150.0,
) -> Optional[float]:
    """
    섹터 중위수 P/E 산출:
    0 < P/E <= max_pe_cap 조건의 유효값에 대해서만 중위수 계산.
    유효값이 없으면 None 반환.
    """
    valid = [p for p in pe_list if p is not None and not np.isnan(p) and 0.0 < p <= max_pe_cap]
    if not valid:
        return None
    return float(np.median(valid))


def calc_sector_pe_percentile(
    target_pe: float,
    all_sector_pe_list: List[float],
    max_pe_cap: float = 150.0,
) -> Optional[float]:
    """
    연속성 보정(Continuity-Corrected) 소표본 왜곡 없는 섹터 백분위 순위 산출:
      Sector_Pct = (rank_0 + 0.5) / N * 100
      (N=1 -> 50.0%, N=2 -> 25.0% 및 75.0%)
    """
    if target_pe is None or np.isnan(target_pe) or target_pe <= 0.0 or target_pe > max_pe_cap:
        return None

    valid = sorted(
        [p for p in all_sector_pe_list if p is not None and not np.isnan(p) and 0.0 < p <= max_pe_cap]
    )
    n = len(valid)
    if n == 0:
        return None

    # 중복값(동일 P/E) 순위 평균 처리
    indices = [i for i, v in enumerate(valid) if np.isclose(v, target_pe, atol=1e-5)]
    if indices:
        avg_rank = float(np.mean(indices))
    else:
        avg_rank = float(np.searchsorted(valid, target_pe))
        avg_rank = min(avg_rank, n - 0.5)

    pct = ((avg_rank + 0.5) / n) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


def _calc_sector_relative_list(
    results: List[PEBandResult],
    sector_mapping: Dict[str, str],
) -> List[PEBandResult]:
    from collections import defaultdict

    # 1. 섹터 미분류 종목 매핑
    for r in results:
        if not r.sector or r.sector in ("기타", "기타/미분류"):
            clean_code = _clean_ticker(r.ticker)
            sec = sector_mapping.get(clean_code, sector_mapping.get(r.ticker, "기타/미분류"))
            if sec:
                r.sector = sec

    # 2. 섹터별 그룹핑
    groups: Dict[str, List[PEBandResult]] = defaultdict(list)
    for r in results:
        groups[r.sector].append(r)

    # 3. 섹터별 메트릭스 계산
    for sector, items in groups.items():
        valid_items = [
            x for x in items
            if x.current_fwd_pe is not None
            and not np.isnan(x.current_fwd_pe)
            and 0.0 < x.current_fwd_pe <= 150.0
        ]
        valid_pes = [x.current_fwd_pe for x in valid_items]
        n_sec = len(valid_pes)
        sec_median = calc_sector_median_pe(valid_pes)

        for item in items:
            item.sector_median_pe = sec_median
            item.sector_stock_count = n_sec

            if (
                item.current_fwd_pe is not None
                and not np.isnan(item.current_fwd_pe)
                and 0.0 < item.current_fwd_pe <= 150.0
            ):
                item.sector_relative_pe = (
                    item.current_fwd_pe / sec_median
                    if (sec_median is not None and sec_median > 0)
                    else 1.0
                )
                item.sector_pe_percentile = calc_sector_pe_percentile(item.current_fwd_pe, valid_pes)
            else:
                item.sector_relative_pe = None
                item.sector_pe_percentile = None

    return results


def _calc_sector_relative_df(
    df: pd.DataFrame,
    sector_mapping: Dict[str, str],
) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return df.copy()

    df = df.copy()

    # 종목코드 식별
    if "ticker" in df.columns:
        ticker_series = df["ticker"]
    elif df.index.name == "ticker":
        ticker_series = df.index.to_series()
    else:
        ticker_series = df.iloc[:, 0]

    # 섹터 컬럼 보강
    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = ticker_series.apply(
            lambda t: sector_mapping.get(_clean_ticker(t), "기타/미분류")
        )
    else:
        df["sector"] = df["sector"].fillna(
            ticker_series.apply(lambda t: sector_mapping.get(_clean_ticker(t), "기타/미분류"))
        )

    # P/E 컬럼 식별
    pe_col = None
    for cand in ("current_fwd_pe", "fwd_pe", "pe", "current_pe"):
        if cand in df.columns:
            pe_col = cand
            break

    df["sector_median_pe"] = np.nan
    df["sector_relative_pe"] = np.nan
    df["sector_pe_percentile"] = np.nan
    df["sector_stock_count"] = 0

    if pe_col is None:
        return df

    for sector in df["sector"].unique():
        sec_mask = df["sector"] == sector
        sec_sub = df.loc[sec_mask]

        valid_mask = (sec_sub[pe_col] > 0.0) & (sec_sub[pe_col] <= 150.0) & sec_sub[pe_col].notna()
        valid_pes = sec_sub.loc[valid_mask, pe_col].dropna().tolist()
        n_sec = len(valid_pes)
        sec_median = calc_sector_median_pe(valid_pes)

        df.loc[sec_mask, "sector_median_pe"] = sec_median
        df.loc[sec_mask, "sector_stock_count"] = n_sec

        for idx, row in sec_sub.iterrows():
            pe_val = row[pe_col]
            if pd.notna(pe_val) and 0.0 < pe_val <= 150.0:
                rel_pe = (
                    pe_val / sec_median
                    if (sec_median is not None and sec_median > 0)
                    else 1.0
                )
                pct = calc_sector_pe_percentile(pe_val, valid_pes)
                df.loc[idx, "sector_relative_pe"] = rel_pe
                df.loc[idx, "sector_pe_percentile"] = pct
            else:
                df.loc[idx, "sector_relative_pe"] = np.nan
                df.loc[idx, "sector_pe_percentile"] = np.nan

    return df


def calc_sector_relative_metrics(
    data: Union[List[PEBandResult], pd.DataFrame],
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Union[List[PEBandResult], pd.DataFrame]:
    """
    섹터별 상대 밸류에이션 지표 산출 엔진 (M2 / R2 / AC 32)
    - Sector Median P/E (0 < PE <= 150)
    - Sector Relative P/E (PE / Sector_Median_PE)
    - Continuity-Corrected Sector Percentile ((rank - 0.5) / N_S * 100)
    - Sector Stock Count (N_S)
    
    Parameters:
      data: List[PEBandResult] 또는 pd.DataFrame
      sector_mapping: Optional {ticker: sector} 딕셔너리
      
    Returns:
      동일한 타입(List[PEBandResult] 또는 pd.DataFrame)으로 섹터 지표가 보강된 객체
    """
    if sector_mapping is None:
        sector_mapping = load_sector_mapping()

    if isinstance(data, pd.DataFrame):
        return _calc_sector_relative_df(data, sector_mapping)
    elif isinstance(data, list):
        return _calc_sector_relative_list(data, sector_mapping)
    else:
        return data


def calc_pe_band(
    ticker: str,
    name: str,
    price_series: pd.Series,
    eps_series: pd.Series,
    band_years: int = 5,
    band_model: str = "percentile",
    winsorize: bool = True,
    fixed_multiples: Optional[List[float]] = None,
    sector: Optional[str] = None,
    eps_rev_1w: Optional[float] = None,
    eps_rev_1m: Optional[float] = None,
    eps_rev_3m: Optional[float] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
) -> Optional[PEBandResult]:
    """
    단일 종목 P/E 밴드 및 밸류에이션 모멘텀 지표 계산 (M1, M2, M3 통합)
    - 백분위(Percentile), Mean ± 1/2SD, 고정배수(Fixed) 밴드 모델 지원
    - 95% Winsorization 이상치 보정 지원
    - 전체 시계열 밴드 배열(band_series_pct, band_series_sd, band_series_fixed) 생성
    """
    # 하위 호환성 가드: 6번째 인자로 sector 문자열이 전달된 경우 처리
    if band_model not in ("percentile", "mean_sd", "fixed") and sector is None:
        sector = band_model
        band_model = "percentile"

    if fixed_multiples is None:
        fixed_multiples = [8.0, 10.0, 12.0, 15.0]
    else:
        fixed_multiples = [float(m) for m in fixed_multiples]

    # 데이터 정렬 및 공통 인덱스 추출
    price_series = price_series.sort_index().dropna()
    eps_series = eps_series.sort_index().dropna()

    common_idx = price_series.index.intersection(eps_series.index)
    if len(common_idx) < 6:
        # 월별 주가와 일간 EPS 간 날짜 불일치 시 asof/ffill 정렬
        aligned_eps = (
            eps_series.reindex(eps_series.index.union(price_series.index))
            .ffill()
            .loc[price_series.index]
            .dropna()
        )
        common_idx = price_series.index.intersection(aligned_eps.index)
        if len(common_idx) >= 6:
            eps_series = aligned_eps
        else:
            return None  # 데이터 부족 (6개월 미만)

    price = price_series.loc[common_idx]
    eps = eps_series.loc[common_idx]
    pe_series = calc_fwd_pe_series(price, eps)

    # 역사적 밴드 구간 (최근 N년)
    cutoff = pe_series.index.max() - pd.DateOffset(years=band_years)
    hist_pe = pe_series[pe_series.index >= cutoff].dropna()

    if len(hist_pe) < 4:
        return None  # 밴드 계산에 충분한 데이터 없음

    # 현재 값
    current_price = float(price.iloc[-1])
    current_eps = float(eps.iloc[-1])

    if current_eps <= 0:
        return None  # 최신 EPS가 음수이면 P/E 의미 없음

    current_pe = current_price / current_eps

    # 이상치 보정 (Winsorization) 적용 여부
    pe_for_bands = winsorize_pe(hist_pe) if winsorize else hist_pe.dropna()
    if len(pe_for_bands) < 4:
        pe_for_bands = hist_pe.dropna()
        if len(pe_for_bands) < 4:
            return None

    # 밴드 통계 (p10, p25, median, p75, p90, min, max, mean)
    pe_min    = float(pe_for_bands.min())
    pe_p10    = float(pe_for_bands.quantile(0.10))
    pe_p25    = float(pe_for_bands.quantile(0.25))
    pe_median = float(pe_for_bands.median())
    pe_p75    = float(pe_for_bands.quantile(0.75))
    pe_p90    = float(pe_for_bands.quantile(0.90))
    pe_max    = float(pe_for_bands.max())
    pe_mean   = float(pe_for_bands.mean())

    # 현재 P/E의 역사적 백분위
    pe_percentile = float((pe_for_bands < current_pe).mean() * 100)

    # 1. Mean ± SD 메트릭 계산
    mean_sd_metrics = calc_mean_sd_bands(hist_pe, floor_pe=1.0, winsorize=winsorize)

    # 2. Fixed Multiples 메트릭 계산
    fixed_res = calc_fixed_multiple_bands(fixed_multiples, current_eps=current_eps, current_price=current_price)
    fixed_targets = fixed_res["targets"]
    fixed_upsides = fixed_res["upsides"]

    # 3. 전체 시계열 밴드 배열 생성 (UI 즉시 전환 지원)
    eps_clean = eps.where(eps > 0, np.nan)

    # (1) Percentile Band Series
    band_series_pct = pd.DataFrame({
        "p10": eps_clean * pe_p10,
        "p25": eps_clean * pe_p25,
        "p50": eps_clean * pe_median,
        "p75": eps_clean * pe_p75,
        "p90": eps_clean * pe_p90,
    }, index=eps.index)

    # (2) Mean ± SD Band Series
    if mean_sd_metrics:
        band_series_sd = pd.DataFrame({
            "-2SD": eps_clean * mean_sd_metrics["m2sd"],
            "-1SD": eps_clean * mean_sd_metrics["m1sd"],
            "Mean": eps_clean * mean_sd_metrics["mean"],
            "+1SD": eps_clean * mean_sd_metrics["p1sd"],
            "+2SD": eps_clean * mean_sd_metrics["p2sd"],
        }, index=eps.index)
    else:
        band_series_sd = None

    # (3) Fixed Multiples Band Series
    fixed_cols = {}
    for m in fixed_multiples:
        col_name = f"{int(m)}x" if float(m).is_integer() else f"{m}x"
        fixed_cols[col_name] = eps_clean * float(m)
    band_series_fixed = pd.DataFrame(fixed_cols, index=eps.index)

    # 4. 모델별 목표가 및 업사이드 설정
    if band_model == "mean_sd" and mean_sd_metrics:
        target_bear = current_eps * mean_sd_metrics["m1sd"]
        target_base = current_eps * mean_sd_metrics["mean"]
        target_bull = current_eps * mean_sd_metrics["p1sd"]
    elif band_model == "fixed":
        sorted_m = sorted(fixed_multiples)
        if len(sorted_m) >= 3:
            bear_m = sorted_m[0]
            base_m = sorted_m[1] if len(sorted_m) <= 4 else sorted_m[len(sorted_m) // 2]
            bull_m = sorted_m[-1]
        elif len(sorted_m) == 2:
            bear_m = sorted_m[0]
            base_m = (sorted_m[0] + sorted_m[1]) / 2.0
            bull_m = sorted_m[1]
        elif len(sorted_m) == 1:
            bear_m = base_m = bull_m = sorted_m[0]
        else:
            bear_m = base_m = bull_m = 0.0

        target_bear = current_eps * bear_m if current_eps > 0 else 0.0
        target_base = current_eps * base_m if current_eps > 0 else 0.0
        target_bull = current_eps * bull_m if current_eps > 0 else 0.0
    else:  # "percentile" (기본)
        target_bear = current_eps * pe_p25
        target_base = current_eps * pe_median
        target_bull = current_eps * pe_p75

    upside_bear = ((target_bear / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0
    upside_base = ((target_base / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0
    upside_bull = ((target_bull / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0

    # 섹터 정보 결정
    clean_code = _clean_ticker(ticker)
    resolved_sector = sector
    if not resolved_sector:
        if sector_mapping is not None:
            resolved_sector = sector_mapping.get(clean_code, "기타/미분류")
        else:
            sec_map = load_sector_mapping()
            resolved_sector = sec_map.get(clean_code, "기타/미분류")

    # M1 EPS Revision 산출 (인자로 전달되지 않은 경우 시계열에서 산출 시도)
    if eps_rev_1m is None and calc_eps_revisions_series is not None:
        try:
            rev_dict = calc_eps_revisions_series(eps_series)
            eps_rev_1w = rev_dict.get("eps_rev_1w")
            eps_rev_1m = rev_dict.get("eps_rev_1m")
            eps_rev_3m = rev_dict.get("eps_rev_3m")
        except Exception:
            pass

    # 레짐 분류
    is_trap, is_golden, regime_tag = False, False, "Neutral"
    if classify_regime is not None and eps_rev_1m is not None:
        reg_res = classify_regime(
            pe_pct=pe_percentile,
            eps_rev_1m=eps_rev_1m,
            eps_rev_3m=eps_rev_3m,
            eps_rev_1w=eps_rev_1w,
            current_pe=current_pe,
        )
        is_trap = reg_res.is_value_trap
        is_golden = reg_res.is_golden_cross
        regime_tag = reg_res.regime_tag

    return PEBandResult(
        ticker=clean_code,
        name=name,
        current_price=current_price,
        current_fwd_eps=current_eps,
        current_fwd_pe=current_pe,
        pe_percentile=pe_percentile,
        pe_min=pe_min,
        pe_p25=pe_p25,
        pe_median=pe_median,
        pe_p75=pe_p75,
        pe_max=pe_max,
        pe_mean=pe_mean,
        target_bear=target_bear,
        target_base=target_base,
        target_bull=target_bull,
        upside_bear=upside_bear,
        upside_base=upside_base,
        upside_bull=upside_bull,
        hist_pe_series=pe_series,
        hist_price_series=price,
        hist_eps_series=eps,
        sector=resolved_sector,
        eps_rev_1w=eps_rev_1w,
        eps_rev_1m=eps_rev_1m,
        eps_rev_3m=eps_rev_3m,
        is_value_trap=is_trap,
        is_golden_cross=is_golden,
        regime_tag=regime_tag,
        band_model=band_model,
        band_series_pct=band_series_pct,
        band_series_sd=band_series_sd,
        band_series_fixed=band_series_fixed,
        mean_sd_metrics=mean_sd_metrics,
        fixed_targets=fixed_targets,
        fixed_upsides=fixed_upsides,
    )


def run_screener(
    price_df: pd.DataFrame,
    eps_df: pd.DataFrame,
    ticker_names: dict,
    band_years: int = 5,
    band_model: str = "percentile",
    winsorize: bool = True,
    fixed_multiples: Optional[List[float]] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
) -> list[PEBandResult]:
    """
    전 종목 일괄 밸류에이션 및 섹터 상대 스크리닝 (M1, M2 통합)
    """
    if sector_mapping is None:
        sector_mapping = load_sector_mapping()

    # Pre-calculate revisions for all tickers if calc_eps_revisions is available
    rev_table = {}
    if calc_eps_revisions is not None:
        try:
            rev_df = calc_eps_revisions(eps_df)
            if rev_df is not None and not rev_df.empty:
                for t, row in rev_df.iterrows():
                    rev_table[_clean_ticker(str(t))] = row
        except Exception as e:
            logger.warning(f"Error pre-computing EPS revisions: {e}")

    results = []
    col_map = {_clean_ticker(c): c for c in eps_df.columns}
    tickers = [t for t in price_df.columns if _clean_ticker(t) in col_map]

    for ticker in tickers:
        clean_code = _clean_ticker(ticker)
        eps_col = col_map[clean_code]
        name = ticker_names.get(ticker, ticker_names.get(clean_code, ticker))
        sec = sector_mapping.get(clean_code, "기타/미분류")

        # 사전 계산된 리비전 매핑
        rev_row = rev_table.get(clean_code)
        rev_1w = rev_row["eps_rev_1w"] if rev_row is not None and pd.notna(rev_row.get("eps_rev_1w")) else None
        rev_1m = rev_row["eps_rev_1m"] if rev_row is not None and pd.notna(rev_row.get("eps_rev_1m")) else None
        rev_3m = rev_row["eps_rev_3m"] if rev_row is not None and pd.notna(rev_row.get("eps_rev_3m")) else None

        result = calc_pe_band(
            ticker=clean_code,
            name=name,
            price_series=price_df[ticker],
            eps_series=eps_df[eps_col],
            band_years=band_years,
            band_model=band_model,
            winsorize=winsorize,
            fixed_multiples=fixed_multiples,
            sector=sec,
            eps_rev_1w=rev_1w,
            eps_rev_1m=rev_1m,
            eps_rev_3m=rev_3m,
            sector_mapping=sector_mapping,
        )
        if result is not None:
            results.append(result)

    # 섹터 상대 지표 일괄 보강
    results = calc_sector_relative_metrics(results, sector_mapping)

    # Base 업사이드 기준 내림차순 정렬
    results.sort(key=lambda r: r.upside_base, reverse=True)
    return results


def apply_quick_preset(df: pd.DataFrame, preset_key: str) -> pd.DataFrame:
    """
    애널리스트 퀵 전략 스크리닝 프리셋 필터 적용 (R4 / Feature 13 / AC 39)
      - 'ALL': 전체 종목 반환
      - 'TURNAROUND': (pe_pct <= 40 | sector_pe_pct <= 40) & (eps_rev_1m > 0) & (upside_base >= 10.0)
      - 'EPS_TOP': eps_rev_1m >= 3.0 & current_eps > 0 (eps_rev_1m 기준 내림차순 정렬 상위 25종목)
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
