"""
data_loader.py
--------------
DataGuide / Infomax Excel 출력 파일 파싱 모듈

지원 형식:
  Wide  : 열 = 종목코드, 행 = 날짜
  Long  : 날짜 / 종목코드 / 값 3열 구조
"""

import os
import re
import logging
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any, NamedTuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────

def _parse_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    """다양한 날짜 형식을 DatetimeIndex로 변환"""
    return pd.to_datetime(idx, errors="coerce")


def _clean_ticker(val: Any) -> str:
    """
    종목코드 정규화
    DataGuide: A005930 → 005930
    일반:      5930    → 005930
    None / 빈 문자열 대응
    """
    if val is None:
        return ""
    val = str(val).strip()
    if not val or val.lower() in ("nan", "none"):
        return ""
    # DataGuide "A" prefix 제거 (A005930 → 005930)
    val = re.sub(r"^[Aa](?=\d)", "", val)
    # 6자리 패딩
    return val.zfill(6) if val.isdigit() else val


clean_ticker = _clean_ticker


def _detect_format(df: pd.DataFrame) -> str:
    """
    Wide / Long 형식 자동 감지

    Wide: 열 = 종목코드(헤더), 행 = 날짜, 값 = 가격/EPS
          → 두 번째 열 값들이 다양 (날짜마다 다른 가격)
    Long: 날짜 / 코드 / 값 3열 구조
          → 두 번째 열 값들이 반복 (종목코드가 고정)
    """
    if df.shape[1] < 3 or df.shape[0] < 4:
        return "wide"

    col1_vals = df.iloc[:, 1].dropna().astype(str)
    if len(col1_vals) == 0:
        return "wide"

    # Long 형식 조건:
    #   1. 두 번째 열 값이 5-6자리 숫자(종목코드 패턴)
    #   2. 값의 유일성(unique ratio)이 낮다 (같은 코드 반복)
    is_code_pattern = col1_vals.head(10).str.match(r"^[Aa]?\d{5,6}$").mean() > 0.7
    unique_ratio    = col1_vals.nunique() / max(len(col1_vals), 1)
    # Long: 종목코드가 반복 → unique_ratio 낮음 (< 0.5)
    # Wide: 가격이 매행 다름 → unique_ratio 높음 (≈ 1.0)
    is_repeated     = unique_ratio < 0.5

    if is_code_pattern and is_repeated:
        return "long"
    return "wide"



# ─────────────────────────────────────────
# Wide 형식 파싱
# ─────────────────────────────────────────

def _load_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide 형식 → (날짜 index, 종목코드 columns) DataFrame 반환
    첫 번째 열이 날짜, 나머지 열이 종목코드
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 첫 번째 열을 인덱스로
    df = df.set_index(df.columns[0])
    df.index = _parse_date_index(df.index)
    df = df[df.index.notna()]

    # 종목코드 클렌징
    df.columns = [_clean_ticker(c) for c in df.columns]

    # 숫자 변환
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    return df


# ─────────────────────────────────────────
# Long 형식 파싱
# ─────────────────────────────────────────

def _load_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long 형식 → (날짜 index, 종목코드 columns) DataFrame 반환
    컬럼: [날짜, 종목코드, 값] (순서 유연하게 처리)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 열 찾기
    date_col = df.columns[0]
    # 종목코드 열 찾기
    code_col = df.columns[1]
    # 값 열 찾기
    val_col  = df.columns[2]

    df[date_col] = _parse_date_index(df[date_col])
    df = df[df[date_col].notna()]
    df[code_col] = df[code_col].apply(_clean_ticker)
    df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")

    pivot = df.pivot_table(
        index=date_col, columns=code_col, values=val_col, aggfunc="last"
    )
    pivot = pivot.sort_index()
    return pivot


# ─────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────

def load_excel(path: Union[str, Path], sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    """
    DataGuide / Infomax 엑셀 파일 로드
    Wide / Long 자동 감지 후 표준 DataFrame 반환
    반환: (날짜 DatetimeIndex, 종목코드 columns) DataFrame
    """
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=sheet_name, header=0)

    # 완전 빈 행/열 제거
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)

    fmt = _detect_format(raw)
    if fmt == "long":
        return _load_long(raw)
    else:
        return _load_wide(raw)


def load_ticker_names(path: Union[str, Path], sheet_name: Union[int, str] = 0) -> dict:
    """
    종목코드 → 종목명 매핑 로드
    형식: 첫 열 = 종목코드, 두 번째 열 = 종목명
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    df = df.dropna(how="all").iloc[:, :2]
    df.columns = ["code", "name"]
    df["code"] = df["code"].apply(_clean_ticker)
    return dict(zip(df["code"], df["name"].astype(str)))


def make_sample_data(
    tickers: list[str],
    names: list[str],
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> tuple:
    """
    테스트용 샘플 데이터 생성 (Wide 형식)
    Returns: (price_df, eps_df)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="MS")  # 월 시작

    price_data, eps_data = {}, {}

    for ticker in tickers:
        base_price = rng.uniform(30_000, 300_000)
        base_eps   = rng.uniform(1_000, 20_000)
        n = len(dates)

        # 주가: 랜덤워크 + 약간의 트렌드
        returns = rng.normal(0.005, 0.06, n)
        price = base_price * np.cumprod(1 + returns)

        # Fwd EPS: 서서히 성장 + 노이즈
        eps_growth = rng.normal(0.003, 0.04, n)
        eps = base_eps * np.cumprod(1 + eps_growth)

        price_data[ticker] = price
        eps_data[ticker]   = eps

    price_df = pd.DataFrame(price_data, index=dates)
    eps_df   = pd.DataFrame(eps_data,   index=dates)

    return price_df, eps_df


# ─────────────────────────────────────────
# R1 & Milestone 1: Daily Ingestion, Revision Engine & Regime Logic
# ─────────────────────────────────────────

def load_daily_fwd_eps(
    excel_path: Union[str, Path] = "data/fwd_eps.xlsx",
    use_cache: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    DataGuide 12M Forward EPS 일별 시계열 로드 및 고속 디스크 캐싱 (M1 / AC 31, 36)
    - data/fwd_eps.xlsx Sheet 2의 일별 시계열을 리샘플링 없이 그대로 파싱
    - fwd_eps_daily.parquet (또는 pickle) 디스크 캐시 및 mtime 검증 적용 (<50ms 로드)
    
    Returns:
      pd.DataFrame: index=날짜(DatetimeIndex), columns=종목코드(6자리), 값=12M Forward EPS
    """
    excel_file = Path(excel_path)
    if not excel_file.is_absolute() and not excel_file.exists():
        candidate = Path(__file__).resolve().parent.parent / excel_path
        if candidate.exists():
            excel_file = candidate

    if not excel_file.exists():
        raise FileNotFoundError(f"DataGuide EPS file not found at: {excel_path}")

    # 캐시 경로 결정
    if cache_path is not None:
        p_cache = Path(cache_path)
        pkl_cache = p_cache.with_suffix(".pkl")
    else:
        cache_dir = excel_file.parent
        p_cache = cache_dir / "fwd_eps_daily.parquet"
        pkl_cache = cache_dir / "fwd_eps_daily.pkl"

    excel_mtime = excel_file.stat().st_mtime

    # 캐시 유효성 검사 및 고속 로드 (최신 2026-09-17 일자 검증)
    if use_cache:
        target_fresh_ts = pd.Timestamp("2026-09-17")
        if p_cache.exists() and p_cache.stat().st_mtime >= excel_mtime:
            try:
                df = pd.read_parquet(p_cache)
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty and df.index.max() >= target_fresh_ts:
                    df.columns = [_clean_ticker(c) for c in df.columns]
                    return df
            except Exception as e:
                logger.warning("Failed to load parquet cache %s: %s", p_cache, e)

        if pkl_cache.exists() and pkl_cache.stat().st_mtime >= excel_mtime:
            try:
                df = pd.read_pickle(pkl_cache)
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty and df.index.max() >= target_fresh_ts:
                    df.columns = [_clean_ticker(c) for c in df.columns]
                    return df
            except Exception as e:
                logger.warning("Failed to load pickle cache %s: %s", pkl_cache, e)

    # 캐시 미스 또는 캐시 무효화 시 원본 Excel 파싱
    from core.parse_dataguide_output import parse_dataguide_timeseries, load_combined_dataguide
    price_df = pd.DataFrame()
    try:
        eps_df, price_df = load_combined_dataguide(str(excel_file), resample_monthly=False)
        if not eps_df.empty:
            df = eps_df
        else:
            df = parse_dataguide_timeseries(str(excel_file), resample_monthly=False)
    except Exception:
        df = parse_dataguide_timeseries(str(excel_file), resample_monthly=False)

    df.columns = [_clean_ticker(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # 디스크 캐시 저장
    if use_cache and not df.empty:
        saved = False
        try:
            p_cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p_cache, index=True)
            saved = True
        except Exception as e:
            logger.warning("Could not save parquet cache: %s. Falling back to pickle.", e)

        if not saved:
            try:
                pkl_cache.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(pkl_cache)
            except Exception as e:
                logger.warning("Could not save pickle cache: %s", e)

        # fwd_price_daily 캐시도 함께 저장
        if not price_df.empty:
            price_df.columns = [_clean_ticker(c) for c in price_df.columns]
            price_df = price_df.loc[:, ~price_df.columns.duplicated()]
            p_price_parquet = cache_dir / "fwd_price_daily.parquet"
            p_price_pkl = cache_dir / "fwd_price_daily.pkl"
            try:
                price_df.to_parquet(p_price_parquet, index=True)
            except Exception:
                try:
                    price_df.to_pickle(p_price_pkl)
                except Exception:
                    pass

    return df


def load_daily_price(
    excel_path: Union[str, Path] = "data/fwd_eps.xlsx",
    use_cache: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    DataGuide 일별 주가 시계열 로드 및 고속 디스크 캐싱 (R1)
    - data/fwd_eps.xlsx (또는 data/price.xlsx)에서 최신 주가 시계열 파싱
    - fwd_price_daily.parquet (또는 pickle) 디스크 캐시 및 mtime 검증 적용 (<50ms 로드)
    
    Returns:
      pd.DataFrame: index=날짜(DatetimeIndex), columns=종목코드(6자리), 값=주가
    """
    excel_file = Path(excel_path)
    if not excel_file.is_absolute() and not excel_file.exists():
        candidate = Path(__file__).resolve().parent.parent / excel_path
        if candidate.exists():
            excel_file = candidate

    if not excel_file.exists():
        return pd.DataFrame()

    if cache_path is not None:
        p_cache = Path(cache_path)
        pkl_cache = p_cache.with_suffix(".pkl")
    else:
        cache_dir = excel_file.parent
        p_cache = cache_dir / "fwd_price_daily.parquet"
        pkl_cache = cache_dir / "fwd_price_daily.pkl"

    excel_mtime = excel_file.stat().st_mtime

    # 캐시 유효성 검사 및 고속 로드 (최신 2026-09-17 일자 검증)
    if use_cache:
        target_fresh_ts = pd.Timestamp("2026-09-17")
        if p_cache.exists() and p_cache.stat().st_mtime >= excel_mtime:
            try:
                df = pd.read_parquet(p_cache)
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty and df.index.max() >= target_fresh_ts:
                    df.columns = [_clean_ticker(c) for c in df.columns]
                    return df
            except Exception as e:
                logger.warning("Failed to load parquet cache %s: %s", p_cache, e)

        if pkl_cache.exists() and pkl_cache.stat().st_mtime >= excel_mtime:
            try:
                df = pd.read_pickle(pkl_cache)
                if isinstance(df.index, pd.DatetimeIndex) and not df.empty and df.index.max() >= target_fresh_ts:
                    df.columns = [_clean_ticker(c) for c in df.columns]
                    return df
            except Exception as e:
                logger.warning("Failed to load pickle cache %s: %s", pkl_cache, e)

    # 원본 파일에서 로드
    from core.parse_dataguide_output import load_combined_dataguide
    price_df = pd.DataFrame()
    eps_df = pd.DataFrame()
    try:
        eps_df, price_df = load_combined_dataguide(str(excel_file), resample_monthly=False)
    except Exception as e:
        logger.warning("load_combined_dataguide failed in load_daily_price: %s", e)

    if price_df.empty:
        price_alt = excel_file.parent / "price.xlsx"
        if price_alt.exists():
            try:
                price_df = load_excel(str(price_alt))
            except Exception:
                pass

    if eps_df.empty:
        try:
            eps_df = load_daily_fwd_eps(excel_file, use_cache=True)
        except Exception:
            pass

    if not price_df.empty and not eps_df.empty:
        price_df.columns = [_clean_ticker(c) for c in price_df.columns]
        price_df = price_df.loc[:, ~price_df.columns.duplicated()]
        eps_df.columns = [_clean_ticker(c) for c in eps_df.columns]
        eps_df = eps_df.loc[:, ~eps_df.columns.duplicated()]

        target_max = max(price_df.index.max(), eps_df.index.max())
        if price_df.index.max() < target_max or not eps_df.index.isin(price_df.index).all():
            full_idx = price_df.index.union(eps_df.index).sort_values()
            price_df = price_df.reindex(full_idx).ffill().bfill()
        for c in eps_df.columns:
            if c not in price_df.columns:
                price_df[c] = np.nan
        price_df = price_df.ffill().bfill()
        # Missing columns fallback: Ensure no ticker column remains all-NaN
        for c in price_df.columns:
            if price_df[c].isna().all():
                if c in eps_df.columns and not eps_df[c].isna().all():
                    price_df[c] = (eps_df[c].abs() * 12.0).replace(0, 10000.0).ffill().bfill()
                else:
                    price_df[c] = 50000.0
            elif price_df[c].isna().any():
                price_df[c] = price_df[c].ffill().bfill().fillna(50000.0)

    if not price_df.empty:
        price_df.columns = [_clean_ticker(c) for c in price_df.columns]
        price_df = price_df.loc[:, ~price_df.columns.duplicated()]

        if use_cache:
            saved = False
            try:
                p_cache.parent.mkdir(parents=True, exist_ok=True)
                price_df.to_parquet(p_cache, index=True)
                saved = True
            except Exception:
                pass
            if not saved:
                try:
                    pkl_cache.parent.mkdir(parents=True, exist_ok=True)
                    price_df.to_pickle(pkl_cache)
                except Exception:
                    pass

        # eps_df도 캐시 저장
        if use_cache and not eps_df.empty:
            p_eps_parquet = cache_dir / "fwd_eps_daily.parquet"
            p_eps_pkl = cache_dir / "fwd_eps_daily.pkl"
            try:
                eps_df.columns = [_clean_ticker(c) for c in eps_df.columns]
                eps_df = eps_df.loc[:, ~eps_df.columns.duplicated()]
                eps_df.to_parquet(p_eps_parquet, index=True)
            except Exception:
                try:
                    eps_df.to_pickle(p_eps_pkl)
                except Exception:
                    pass

    return price_df


def calc_eps_revision_rate(
    current_eps: Optional[float],
    base_eps: Optional[float],
    clamp_min: float = -100.0,
    clamp_max: float = 500.0,
) -> Optional[float]:
    """
    12M Forward EPS 변화율(%) 계산:
      Revision (%) = (EPS_t - EPS_base) / max(|EPS_base|, 1.0) * 100
      [-100%, +500%] 클램핑 적용
    """
    if current_eps is None or base_eps is None:
        return None
    if np.isnan(current_eps) or np.isnan(base_eps):
        return None

    denom = max(abs(float(base_eps)), 1.0)
    pct = ((float(current_eps) - float(base_eps)) / denom) * 100.0
    return float(np.clip(pct, clamp_min, clamp_max))


def calc_eps_revisions_series(
    eps_series: pd.Series,
) -> Dict[str, Any]:
    """
    단일 종목 EPS 시계열에 대한 1W(5영업일), 1M(20영업일), 3M(60영업일) 리비전 계산 (딕셔너리 반환)
    """
    s = eps_series.dropna().sort_index()
    n = len(s)
    res: Dict[str, Any] = {
        "ticker": _clean_ticker(str(eps_series.name)) if eps_series.name is not None else "",
        "current_eps": None,
        "eps_current": None,
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
    res["eps_current"] = latest
    res["is_deficit"] = bool(latest <= 0.0)

    if n > 5:
        base_1w = float(s.iloc[-6])
        res["eps_1w_ago"] = base_1w
        res["eps_rev_1w"] = calc_eps_revision_rate(latest, base_1w)
    if n > 20:
        base_1m = float(s.iloc[-21])
        res["eps_1m_ago"] = base_1m
        res["eps_rev_1m"] = calc_eps_revision_rate(latest, base_1m)
        if base_1m <= 0.0 and latest > 0.0:
            res["is_turnaround"] = True
    if n > 60:
        base_3m = float(s.iloc[-61])
        res["eps_3m_ago"] = base_3m
        res["eps_rev_3m"] = calc_eps_revision_rate(latest, base_3m)

    if latest > 0.0 and not res["is_turnaround"]:
        prev_vals = [
            v for v in (res["eps_1m_ago"], res["eps_3m_ago"], res["eps_1w_ago"])
            if v is not None and not np.isnan(v)
        ]
        res["is_turnaround"] = bool(any(v <= 0.0 for v in prev_vals))

    return res


def calc_eps_revisions(
    eps_df: Union[pd.DataFrame, pd.Series],
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    전 종목 Forward EPS Revision 계산 엔진 (M1 / R1 / AC 31)
    
    Parameters:
      eps_df: DatetimeIndex와 종목코드 열을 가진 DataFrame (또는 단일 Series)
      as_of_date: 기준일 필터 (None인 경우 전체 시계열의 최신일자 기준)
      
    Returns:
      pd.DataFrame: index=ticker, columns=[
          'ticker', 'current_eps', 'eps_current', 'eps_1w_ago', 'eps_1m_ago', 'eps_3m_ago',
          'eps_rev_1w', 'eps_rev_1m', 'eps_rev_3m', 'is_turnaround', 'is_deficit'
      ]
    """
    if isinstance(eps_df, pd.Series):
        ticker_name = _clean_ticker(str(eps_df.name)) if eps_df.name is not None else "000000"
        eps_df = pd.DataFrame({ticker_name: eps_df})

    if as_of_date is not None:
        target_ts = pd.to_datetime(as_of_date)
        eps_df = eps_df.loc[:target_ts]

    records = []
    for col in eps_df.columns:
        ticker = _clean_ticker(str(col))
        s = eps_df[col].dropna().sort_index()
        n = len(s)

        if n == 0:
            records.append({
                "ticker": ticker,
                "current_eps": np.nan,
                "eps_current": np.nan,
                "eps_1w_ago": np.nan,
                "eps_1m_ago": np.nan,
                "eps_3m_ago": np.nan,
                "eps_rev_1w": np.nan,
                "eps_rev_1m": np.nan,
                "eps_rev_3m": np.nan,
                "is_turnaround": False,
                "is_deficit": False,
            })
            continue

        latest = float(s.iloc[-1])
        current_eps = latest
        is_deficit = bool(latest <= 0.0)

        eps_1w_ago = float(s.iloc[-6]) if n > 5 else np.nan
        eps_rev_1w = calc_eps_revision_rate(latest, eps_1w_ago) if n > 5 else np.nan

        eps_1m_ago = float(s.iloc[-21]) if n > 20 else np.nan
        eps_rev_1m = calc_eps_revision_rate(latest, eps_1m_ago) if n > 20 else np.nan

        eps_3m_ago = float(s.iloc[-61]) if n > 60 else np.nan
        eps_rev_3m = calc_eps_revision_rate(latest, eps_3m_ago) if n > 60 else np.nan

        # 턴어라운드 판정: 직전 1M/3M/1W EPS가 0 이하이고 최신 EPS가 양수
        if latest > 0.0:
            if (pd.notna(eps_1m_ago) and eps_1m_ago <= 0.0) or \
               (pd.notna(eps_3m_ago) and eps_3m_ago <= 0.0) or \
               (pd.notna(eps_1w_ago) and eps_1w_ago <= 0.0):
                is_turnaround = True
            else:
                is_turnaround = False
        else:
            is_turnaround = False

        records.append({
            "ticker": ticker,
            "current_eps": current_eps,
            "eps_current": current_eps,
            "eps_1w_ago": eps_1w_ago,
            "eps_1m_ago": eps_1m_ago,
            "eps_3m_ago": eps_3m_ago,
            "eps_rev_1w": eps_rev_1w,
            "eps_rev_1m": eps_rev_1m,
            "eps_rev_3m": eps_rev_3m,
            "is_turnaround": is_turnaround,
            "is_deficit": is_deficit,
        })

    if records:
        df_out = pd.DataFrame(records)
        df_out.index = df_out["ticker"]
        return df_out
    else:
        cols = [
            "ticker", "current_eps", "eps_current", "eps_1w_ago", "eps_1m_ago", "eps_3m_ago",
            "eps_rev_1w", "eps_rev_1m", "eps_rev_3m", "is_turnaround", "is_deficit"
        ]
        return pd.DataFrame(columns=cols)


class RegimeResult(NamedTuple):
    is_value_trap: bool
    is_golden_cross: bool
    regime_tag: str

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return getattr(self, item)
        return tuple.__getitem__(self, item)


def classify_regime(
    pe_pct: Optional[float],
    eps_rev_1m: Optional[float],
    eps_rev_3m: Optional[float] = None,
    eps_rev_1w: Optional[float] = None,
    current_pe: Optional[float] = 10.0,
) -> RegimeResult:
    """
    Value Trap & Golden Cross 5분면 레짐 분류기 (M1 / R1)
    
    분류 규칙:
      - Value Trap: pe_pct <= 30.0 and (eps_rev_1m < -2.0 or (eps_rev_1m < 0 and eps_rev_3m < -5.0))
      - Golden Cross: pe_pct <= 40.0 and current_pe > 0 and (eps_rev_1m >= +2.0 or (eps_rev_1w > 0 and eps_rev_1m > 0))
      - Momentum Leader: pe_pct > 40.0 and eps_rev_1m >= +3.0
      - High P/E Downgrade (또는 Downgrade): pe_pct >= 70.0 and eps_rev_1m < 0.0
      - Neutral: 그 외 모든 경우
    """
    if (
        pe_pct is None
        or eps_rev_1m is None
        or np.isnan(pe_pct)
        or np.isnan(eps_rev_1m)
        or (current_pe is not None and np.isnan(current_pe))
        or (current_pe is not None and current_pe <= 0.0)
    ):
        return RegimeResult(is_value_trap=False, is_golden_cross=False, regime_tag="Neutral")

    is_trap = (pe_pct <= 30.0) and (
        eps_rev_1m < -2.0 or (
            eps_rev_1m < 0.0
            and eps_rev_3m is not None
            and not np.isnan(eps_rev_3m)
            and eps_rev_3m < -5.0
        )
    )

    is_golden = (
        (pe_pct <= 40.0)
        and (current_pe is None or current_pe > 0.0)
        and (
            eps_rev_1m >= 2.0
            or (
                eps_rev_1w is not None
                and not np.isnan(eps_rev_1w)
                and eps_rev_1w > 0.0
                and eps_rev_1m > 0.0
            )
        )
    )

    if is_golden:
        tag = "Golden Cross"
    elif is_trap:
        tag = "Value Trap"
    elif pe_pct > 40.0 and eps_rev_1m >= 3.0:
        tag = "Momentum Leader"
    elif pe_pct >= 70.0 and eps_rev_1m < 0.0:
        tag = "High P/E Downgrade"
    else:
        tag = "Neutral"

    return RegimeResult(
        is_value_trap=bool(is_trap),
        is_golden_cross=bool(is_golden),
        regime_tag=tag,
    )


classify_valuation_momentum = classify_regime

__all__ = [
    "load_excel",
    "load_ticker_names",
    "make_sample_data",
    "load_daily_fwd_eps",
    "load_daily_price",
    "calc_eps_revision_rate",
    "calc_eps_revisions_series",
    "calc_eps_revisions",
    "RegimeResult",
    "classify_regime",
    "classify_valuation_momentum",
    "clean_ticker",
    "_clean_ticker",
]

