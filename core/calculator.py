"""
calculator.py
-------------
12개월 Fwd EPS 기반 P/E 밴드 및 목표가 계산 모듈
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


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


def calc_pe_band(
    ticker: str,
    name: str,
    price_series: pd.Series,
    eps_series: pd.Series,
    band_years: int = 5
) -> Optional[PEBandResult]:
    """
    단일 종목 P/E 밴드 계산

    Parameters
    ----------
    ticker : str
        종목코드
    name : str
        종목명
    price_series : pd.Series
        월별 주가 시계열 (index = datetime)
    eps_series : pd.Series
        월별 12M Fwd EPS 시계열 (index = datetime)
    band_years : int
        역사적 밴드 산출 기간 (년)
    """
    # 데이터 정렬 및 공통 인덱스 추출
    price_series = price_series.sort_index().dropna()
    eps_series = eps_series.sort_index().dropna()
    common_idx = price_series.index.intersection(eps_series.index)

    if len(common_idx) < 6:
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

    # 밴드 통계
    pe_min    = float(hist_pe.min())
    pe_p25    = float(hist_pe.quantile(0.25))
    pe_median = float(hist_pe.median())
    pe_p75    = float(hist_pe.quantile(0.75))
    pe_max    = float(hist_pe.max())
    pe_mean   = float(hist_pe.mean())

    # 현재 P/E의 역사적 백분위
    pe_percentile = float((hist_pe < current_pe).mean() * 100)

    # 목표가
    target_bear = current_eps * pe_p25
    target_base = current_eps * pe_median
    target_bull = current_eps * pe_p75

    # 업사이드
    upside_bear = (target_bear / current_price - 1) * 100
    upside_base = (target_base / current_price - 1) * 100
    upside_bull = (target_bull / current_price - 1) * 100

    return PEBandResult(
        ticker=ticker,
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
    )


def run_screener(
    price_df: pd.DataFrame,
    eps_df: pd.DataFrame,
    ticker_names: dict,
    band_years: int = 5
) -> list[PEBandResult]:
    """
    여러 종목 일괄 스크리닝

    Parameters
    ----------
    price_df : pd.DataFrame
        열 = 종목코드, 행 = 날짜, 값 = 주가
    eps_df : pd.DataFrame
        열 = 종목코드, 행 = 날짜, 값 = 12M Fwd EPS
    ticker_names : dict
        {종목코드: 종목명}
    band_years : int
        역사적 밴드 기간
    """
    results = []
    tickers = [t for t in price_df.columns if t in eps_df.columns]

    for ticker in tickers:
        name = ticker_names.get(ticker, ticker)
        result = calc_pe_band(
            ticker=ticker,
            name=name,
            price_series=price_df[ticker],
            eps_series=eps_df[ticker],
            band_years=band_years,
        )
        if result is not None:
            results.append(result)

    # Base 업사이드 기준 내림차순 정렬
    results.sort(key=lambda r: r.upside_base, reverse=True)
    return results
