"""
fetch_realtime_price.py
-----------------------
pykrx 실시간 현재가 조회 (매일 자동 갱신)
"""

import pandas as pd
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _latest_trading_date(offset: int = 0) -> str:
    """최근 N번째 영업일 (YYYYMMDD)"""
    today = datetime.today() - timedelta(days=offset)
    for i in range(15):
        d = today - timedelta(days=i)
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
    return today.strftime("%Y%m%d")


def get_current_prices(
    tickers: list[str],
    sleep_sec: float = 0.1,
) -> pd.Series:
    """
    pykrx로 당일 종가(현재가) 일괄 조회

    장 마감 전이면 전일 종가, 마감 후면 당일 종가 반환
    Returns: Series  index=종목코드, value=주가(원)
    """
    try:
        from pykrx import stock as krx
    except ImportError:
        logger.error("pykrx 미설치")
        return pd.Series(dtype=float)

    # 오늘 → 데이터 없으면 전일 시도
    for offset in range(5):
        date = _latest_trading_date(offset)
        try:
            result = {}
            for ticker in tickers:
                try:
                    df = krx.get_market_ohlcv(date, date, ticker, adjusted=True)
                    if not df.empty and "종가" in df.columns:
                        price = float(df["종가"].iloc[-1])
                        if price > 0:
                            result[ticker] = price
                    time.sleep(sleep_sec)
                except Exception:
                    continue

            if result:
                logger.info(f"현재가 조회: {len(result)}개 (기준일: {date})")
                return pd.Series(result)
        except Exception as e:
            logger.warning(f"현재가 조회 실패 ({date}): {e}")
            continue

    return pd.Series(dtype=float)


def get_current_prices_batch(
    tickers: list[str],
) -> pd.Series:
    """
    KRX 일괄 API 방식 (더 빠름)
    get_market_ohlcv_by_date 대신 당일 전체 시장 데이터 한 번에 조회
    """
    try:
        from pykrx import stock as krx
    except ImportError:
        return pd.Series(dtype=float)

    for offset in range(5):
        date = _latest_trading_date(offset)
        try:
            # KOSPI 전체
            df_kospi  = krx.get_market_ohlcv(date, date, adjusted=True, market="KOSPI")
            df_kosdaq = krx.get_market_ohlcv(date, date, adjusted=True, market="KOSDAQ")
            df_all = pd.concat([df_kospi, df_kosdaq])

            if df_all.empty:
                continue

            prices = df_all["종가"].reindex(tickers).dropna()
            if not prices.empty:
                logger.info(f"배치 현재가: {len(prices)}개 ({date})")
                return prices
        except Exception as e:
            logger.debug(f"배치 조회 실패: {e}")
            continue

    # 폴백: 개별 조회
    return get_current_prices(tickers)
