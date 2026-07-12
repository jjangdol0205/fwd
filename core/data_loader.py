"""
data_loader.py
--------------
DataGuide / Infomax Excel 출력 파일 파싱 모듈

지원 형식:
  Wide  : 열 = 종목코드, 행 = 날짜
  Long  : 날짜 / 종목코드 / 값 3열 구조
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Union


# ─────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────

def _parse_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    """다양한 날짜 형식을 DatetimeIndex로 변환"""
    return pd.to_datetime(idx, errors="coerce")


def _clean_ticker(val: str) -> str:
    """
    종목코드 정규화
    DataGuide: A005930 → 005930
    일반:      5930    → 005930
    """
    val = str(val).strip()
    # DataGuide "A" prefix 제거 (A005930 → 005930)
    val = re.sub(r"^[Aa](?=\d)", "", val)
    # 6자리 패딩
    return val.zfill(6) if val.isdigit() else val


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
