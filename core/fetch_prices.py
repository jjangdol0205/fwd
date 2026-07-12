"""
fetch_prices.py
---------------
pykrx를 이용한 월별 수정주가 자동 수집
KOSPI200 + KOSDAQ150 전 종목 배치 처리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock as krx
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_monthly_price(
    ticker: str,
    start: str,
    end: str,
    retries: int = 3,
) -> pd.Series:
    """
    단일 종목 월별 수정주가 조회
    반환: Series (index=DatetimeIndex, values=수정주가)
    """
    for attempt in range(retries):
        try:
            df = krx.get_market_ohlcv(start, end, ticker, adjusted=True)
            if df.empty:
                return pd.Series(dtype=float, name=ticker)

            df.index = pd.to_datetime(df.index)
            # 월말 기준 리샘플링 (월 마지막 거래일 종가)
            monthly = df["종가"].resample("ME").last().dropna()
            monthly.name = ticker
            return monthly
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logger.warning(f"{ticker} 주가 조회 실패: {e}")
                return pd.Series(dtype=float, name=ticker)


def fetch_all_prices(
    tickers: list[str],
    start: str = "20200101",
    end: str   = None,
    save_path: str = None,
    batch_size: int = 50,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    """
    전 종목 월별 수정주가 배치 수집

    Parameters
    ----------
    tickers : list[str]
        종목코드 리스트
    start : str
        시작일 (YYYYMMDD)
    end : str
        종료일 (YYYYMMDD), None이면 오늘
    save_path : str
        저장 경로 (.xlsx 또는 .csv)
    batch_size : int
        배치 크기 (KRX 서버 부하 방지)
    sleep_sec : float
        종목 간 대기 시간 (초)
    """
    end = end or datetime.today().strftime("%Y%m%d")
    all_series = []
    total = len(tickers)

    print(f"\n[주가 수집 시작] {total}개 종목 ({start} ~ {end})")
    print("=" * 50)

    for i, ticker in enumerate(tickers):
        series = get_monthly_price(ticker, start, end)
        all_series.append(series)
        time.sleep(sleep_sec)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {(i+1)/total*100:.1f}% 완료")

    price_df = pd.concat(all_series, axis=1)
    price_df.index.name = "date"
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    print(f"\n[완료] 주가 수집: {price_df.shape[1]}종목 x {price_df.shape[0]}개월")
    print(f"   유효 데이터: {price_df.notna().sum().sum():,}개 셀")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if save_path.endswith(".xlsx"):
            price_df.to_excel(save_path)
        else:
            price_df.to_csv(save_path, encoding="utf-8-sig")
        print(f"   [저장] {save_path}")

    return price_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # 테스트: 소수 종목만
    test_tickers = ["005930", "000660", "035420", "005380", "051910"]
    df = fetch_all_prices(
        test_tickers,
        start="20200101",
        save_path=r"D:\Dataguide\data\price.xlsx",
    )
    print(df.tail())
