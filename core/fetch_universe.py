"""
fetch_universe.py
-----------------
KOSPI200 + KOSDAQ150 구성종목 자동 수집

방법 우선순위:
  1) KRX 데이터 포털 API (무인증, 가장 정확)
  2) pykrx get_index_portfolio_deposit_file (KRX 계정 필요)
  3) FinanceDataReader (전체 시장 리스트에서 대형주 필터)
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 오늘 날짜
TODAY = datetime.today().strftime("%Y%m%d")


# ─────────────────────────────────────────
# 방법 1: KRX 데이터 포털 직접 호출
# ─────────────────────────────────────────

KRX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.krx.co.kr/",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

INDEX_CODES = {
    "KOSPI200":  "1028",
    "KOSDAQ150": "2203",
}


def _get_krx_otp(bld: str) -> str:
    """KRX OTP 토큰 발급"""
    resp = requests.get(
        "https://data.krx.co.kr/contents/COM/GenerateOTP.jspx",
        params={"bld": bld, "name": "form"},
        headers=KRX_HEADERS,
        timeout=10,
    )
    return resp.text.strip()


def get_index_components_krx(index_name: str, date: str = None) -> pd.DataFrame:
    """
    KRX 데이터 포털에서 지수 구성종목 조회
    date: YYYYMMDD, None이면 오늘

    반환: DataFrame [ticker, name, market]
    """
    date = date or TODAY
    idx_code = INDEX_CODES.get(index_name)
    if not idx_code:
        raise ValueError(f"지원하지 않는 지수: {index_name}")

    # OTP 발급
    try:
        otp = _get_krx_otp("dbms/MDC/STAT/standard/MDCSTAT00601")
    except Exception as e:
        logger.warning(f"KRX OTP 발급 실패: {e}")
        return pd.DataFrame()

    # 구성종목 조회
    try:
        resp = requests.post(
            "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            data={
                "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
                "indIdx": idx_code,
                "indIdx2": "00",
                "strtDd": date,
                "endDd":  date,
                "share":  "1",
                "money":  "1",
                "csvxls_isNo": "false",
                "code": otp,
            },
            headers=KRX_HEADERS,
            timeout=15,
        )
        raw = resp.json()
    except Exception as e:
        logger.warning(f"KRX 구성종목 조회 실패: {e}")
        return pd.DataFrame()

    # 데이터 파싱
    items = raw.get("OutBlock_1", [])
    if not items:
        logger.warning(f"KRX 응답 데이터 없음 (지수={index_name})")
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # 컬럼명 매핑 (KRX API 응답 필드명)
    col_map = {
        "ISU_SRT_CD": "ticker",     # 종목코드
        "ISU_ABBRV":  "name",       # 종목명
        "MKT_NM":     "market_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 종목코드 6자리 정규화
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    if "name" not in df.columns and df.shape[1] >= 2:
        df.columns.values[1] = "name"

    df["market"] = index_name
    df = df[["ticker", "name", "market"]].dropna(subset=["ticker"])
    logger.info(f"{index_name}: {len(df)}개 종목 (KRX 포털)")
    return df


# ─────────────────────────────────────────
# 방법 2: pykrx 폴백
# ─────────────────────────────────────────

def get_index_components_pykrx(index_name: str, date: str = None) -> pd.DataFrame:
    """pykrx를 이용한 지수 구성종목 조회 (KRX 계정 필요할 수 있음)"""
    try:
        from pykrx import stock as krx
        date = date or TODAY
        idx_code = INDEX_CODES.get(index_name)
        tickers  = krx.get_index_portfolio_deposit_file(idx_code, date)
        names    = {t: krx.get_market_ticker_name(t) for t in tickers}
        df = pd.DataFrame({
            "ticker": tickers,
            "name":   [names.get(t, t) for t in tickers],
            "market": index_name,
        })
        logger.info(f"{index_name}: {len(df)}개 종목 (pykrx)")
        return df
    except Exception as e:
        logger.warning(f"pykrx {index_name} 조회 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# 방법 3: FinanceDataReader 폴백
# ─────────────────────────────────────────

def get_index_components_fdr(index_name: str) -> pd.DataFrame:
    """FinanceDataReader로 대형주 필터링 (근사치)"""
    try:
        import FinanceDataReader as fdr
        market = "KOSPI" if "KOSPI" in index_name else "KOSDAQ"
        df_all = fdr.StockListing(market)

        # 시가총액 기준 상위 N개 선택
        top_n = 200 if "200" in index_name else 150
        if "Marcap" in df_all.columns:
            df_all = df_all.nlargest(top_n, "Marcap")
        else:
            df_all = df_all.head(top_n)

        df = pd.DataFrame({
            "ticker": df_all["Code"].astype(str).str.zfill(6),
            "name":   df_all.get("Name", df_all.get("ISU_ABBRV", "")),
            "market": index_name,
        })
        logger.info(f"{index_name}: {len(df)}개 종목 (FinanceDataReader 근사)")
        return df
    except Exception as e:
        logger.warning(f"FinanceDataReader {index_name} 조회 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# 통합 조회 (우선순위 폴백)
# ─────────────────────────────────────────

def get_index_components(index_name: str, date: str = None) -> pd.DataFrame:
    """
    지수 구성종목 조회 (KRX → pykrx → FDR 순으로 시도)
    """
    # 1) KRX 데이터 포털
    df = get_index_components_krx(index_name, date)
    if not df.empty:
        return df

    # 2) pykrx
    df = get_index_components_pykrx(index_name, date)
    if not df.empty:
        return df

    # 3) FinanceDataReader
    df = get_index_components_fdr(index_name)
    return df


def get_universe(date: str = None, save_path: str = None) -> pd.DataFrame:
    """
    KOSPI200 + KOSDAQ150 통합 유니버스 반환
    중복 종목은 KOSPI200 우선
    """
    date = date or TODAY
    print(f"\n📋 유니버스 수집 중... (기준일: {date})")

    df200  = get_index_components("KOSPI200",  date)
    time.sleep(0.5)
    df150  = get_index_components("KOSDAQ150", date)

    universe = pd.concat([df200, df150], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker", keep="first")
    universe = universe.reset_index(drop=True)

    print(f"  ✅ KOSPI200: {len(df200)}개  |  KOSDAQ150: {len(df150)}개")
    print(f"  ✅ 최종 유니버스: {len(universe)}개 종목 (중복 제거)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  💾 저장: {save_path}")

    return universe


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_universe(save_path=r"D:\Dataguide\data\universe.csv")
    print("\n샘플:")
    print(df.head(10).to_string(index=False))
