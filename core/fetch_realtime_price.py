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


def _fetch_market_ohlcv(krx, date: str, market: str) -> pd.DataFrame:
    """pykrx 버전에 따라 다른 API 시그니처 처리"""
    # 최신 pykrx: get_market_ohlcv(date, market=...)
    # 구버전: get_market_ohlcv(date, date, market=...)
    try:
        return krx.get_market_ohlcv(date, date, market=market)
    except TypeError:
        pass
    try:
        return krx.get_market_ohlcv(date, market=market)
    except TypeError:
        pass
    try:
        return krx.get_market_ohlcv(date, date)
    except Exception:
        return pd.DataFrame()


def get_current_prices_batch(tickers: list) -> pd.Series:
    """
    KRX 일괄 API 방식 — 당일 전체 시장 데이터를 한 번에 조회 (빠름)
    실패 시 개별 조회로 폴백
    """
    try:
        from pykrx import stock as krx
    except ImportError:
        logger.warning("pykrx 미설치 — 현재가 조회 불가")
        return pd.Series(dtype=float)

    for offset in range(5):
        date = _latest_trading_date(offset)
        try:
            df_kospi  = _fetch_market_ohlcv(krx, date, "KOSPI")
            df_kosdaq = _fetch_market_ohlcv(krx, date, "KOSDAQ")

            frames = [df for df in [df_kospi, df_kosdaq] if not df.empty]
            if not frames:
                continue

            df_all = pd.concat(frames)

            # 종가 컬럼 찾기 (버전마다 다를 수 있음)
            close_col = None
            for col in ["종가", "Close", "close"]:
                if col in df_all.columns:
                    close_col = col
                    break

            if close_col is None:
                logger.warning("종가 컬럼을 찾을 수 없음: %s", list(df_all.columns))
                continue

            prices = df_all[close_col].reindex(tickers).dropna()
            prices = prices[prices > 0]

            if not prices.empty:
                logger.info("배치 현재가: %d개 (%s)", len(prices), date)
                return prices

        except Exception as e:
            logger.debug("배치 조회 실패 (%s): %s", date, e)
            continue

    # 폴백: 개별 조회
    return _get_prices_individual(tickers)


def _get_prices_individual(tickers: list, sleep_sec: float = 0.05) -> pd.Series:
    """종목별 개별 현재가 조회 (폴백)"""
    try:
        from pykrx import stock as krx
    except ImportError:
        return pd.Series(dtype=float)

    result = {}
    for offset in range(3):
        date = _latest_trading_date(offset)
        for ticker in tickers:
            if ticker in result:
                continue
            try:
                df = krx.get_market_ohlcv(date, date, ticker)
                if df.empty:
                    continue
                for col in ["종가", "Close", "close"]:
                    if col in df.columns:
                        price = float(df[col].iloc[-1])
                        if price > 0:
                            result[ticker] = price
                        break
                time.sleep(sleep_sec)
            except Exception:
                continue
        if result:
            break

    return pd.Series(result)


# 하위 호환용 alias
def get_current_prices(tickers: list, sleep_sec: float = 0.1) -> pd.Series:
    return _get_prices_individual(tickers, sleep_sec)
