"""
app.py — 12M Fwd P/E 밸류에이션 및 실적 모멘텀 종합 대시보드 v3.0
실행: streamlit run app.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import (
    load_excel,
    load_daily_fwd_eps,
    calc_eps_revisions,
    calc_eps_revisions_series,
    classify_regime,
    _clean_ticker,
)
from core.parse_dataguide_output import load_combined_dataguide
from core.calculator import (
    run_screener,
    calc_pe_band,
    PEBandResult,
    load_sector_mapping,
    calc_sector_relative_metrics,
    winsorize_pe,
    calc_mean_sd_bands,
    calc_fixed_multiple_bands,
    apply_quick_preset,
)
from core.fetch_realtime_price import get_current_prices_batch


# ──────────────────────────────────────────────
# 공통 헬퍼 함수
# ──────────────────────────────────────────────
def signal_badge(pct: Optional[float]) -> str:
    if pct is None or np.isnan(pct):
        return '<span class="sig sig-neu">N/A</span>'
    if pct < 20:   return '<span class="sig sig-sb">Strong Buy</span>'
    if pct < 40:   return '<span class="sig sig-b">Buy</span>'
    if pct < 60:   return '<span class="sig sig-h">Hold</span>'
    if pct < 80:   return '<span class="sig sig-s">Sell</span>'
    return             '<span class="sig sig-ss">Strong Sell</span>'


def signal_label(pct: Optional[float]) -> str:
    if pct is None or np.isnan(pct):
        return "⚪ N/A"
    if pct < 20:   return "🟢🟢 Strong Buy"
    if pct < 40:   return "🟢 Buy"
    if pct < 60:   return "🟡 Hold"
    if pct < 80:   return "🔴 Sell"
    return             "🔴🔴 Strong Sell"


def pe_bar_color(pct: Optional[float]) -> str:
    if pct is None or np.isnan(pct):
        return "#9ca3af"
    if pct < 30: return "#34d399"
    if pct < 60: return "#fbbf24"
    return "#f87171"


def get_strategy_signal(r: PEBandResult) -> str:
    """전략 신호 분류 (R1 / R4): ✨골든크로스, ⚠️밸류트랩, 🚀어닝모멘텀, 💎가치주, Neutral"""
    if r.is_golden_cross or r.regime_tag == "Golden Cross":
        return "✨골든크로스"
    elif r.is_value_trap or r.regime_tag == "Value Trap":
        return "⚠️밸류트랩"
    elif (r.eps_rev_1m is not None and r.eps_rev_1m >= 5.0) or r.regime_tag == "Momentum Leader":
        return "🚀어닝모멘텀"
    elif (r.pe_percentile is not None and r.pe_percentile <= 25.0 and
          r.current_fwd_pe is not None and 0.0 < r.current_fwd_pe <= 15.0 and
          r.eps_rev_1m is not None and r.eps_rev_1m >= -2.0):
        return "💎가치주"
    elif r.regime_tag == "High P/E Downgrade":
        return "🔻고P/E하향"
    return "Neutral"


def strategy_badge_html(sig: str) -> str:
    if "골든크로스" in sig:
        return '<span class="sig sig-gc">✨ 골든크로스</span>'
    elif "밸류트랩" in sig:
        return '<span class="sig sig-vt">⚠️ 밸류트랩</span>'
    elif "어닝모멘텀" in sig:
        return '<span class="sig sig-em">🚀 어닝모멘텀</span>'
    elif "가치주" in sig:
        return '<span class="sig sig-val">💎 가치주</span>'
    elif "고P/E하향" in sig:
        return '<span class="sig sig-hd">🔻 고P/E하향</span>'
    return '<span class="sig sig-neu">Neutral</span>'


# apply_quick_preset is imported from core.calculator


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#9ca3af", size=11),
)
AX = dict(
    gridcolor="rgba(255,255,255,0.05)",
    zerolinecolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=10),
    title_font=dict(size=10)
)


# ──────────────────────────────────────────────
# Tier 1 Caching: 원시 데이터 로딩 (Parquet & Excel 분리)
# ──────────────────────────────────────────────
def _get_file_mtimes() -> Tuple[int, ...]:
    """데이터 파일들의 수정 시간 및 크기 튜플 (캐시 무효화 키)"""
    paths = [
        ROOT / "data" / "fwd_eps_daily.parquet",
        ROOT / "data" / "fwd_eps.xlsx",
        ROOT / "data" / "price.xlsx",
        ROOT / "data" / "universe.csv",
        ROOT / "data" / "sector_mapping.json",
    ]
    vals = []
    for p in paths:
        if p.exists():
            vals.extend([int(p.stat().st_mtime), int(p.stat().st_size)])
        else:
            vals.extend([0, 0])
    return tuple(vals)


@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_market_data(mtimes: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
    """
    Tier 1 Caching: 원시 마켓 데이터 로딩 (<50ms via Parquet)
    band_years나 preset 클릭과 완전히 분리되어 한 번만 로드됨.
    """
    eps_path = ROOT / "data" / "fwd_eps.xlsx"
    price_path = ROOT / "data" / "price.xlsx"
    uni_path = ROOT / "data" / "universe.csv"
    sec_path = ROOT / "data" / "sector_mapping.json"

    sector_mapping = load_sector_mapping(str(sec_path)) if sec_path.exists() else load_sector_mapping()

    if eps_path.exists():
        price_hist = pd.DataFrame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                eps_df = load_daily_fwd_eps(str(eps_path), use_cache=True)
            except Exception:
                eps_df, price_hist = load_combined_dataguide(str(eps_path))

            if price_path.exists():
                price_hist = load_excel(str(price_path))
            elif price_hist.empty:
                _, price_hist = load_combined_dataguide(str(eps_path))

        names, markets = {}, {}
        if uni_path.exists():
            uni = pd.read_csv(str(uni_path), dtype=str)
            names = dict(zip(uni["ticker"], uni["name"]))
            markets = dict(zip(uni["ticker"], uni["market"]))

        eps_df = eps_df.copy()
        eps_df.columns = [_clean_ticker(c) for c in eps_df.columns]
        eps_df = eps_df.loc[:, ~eps_df.columns.duplicated()]

        price_hist = price_hist.copy()
        price_hist.columns = [_clean_ticker(c) for c in price_hist.columns]
        price_hist = price_hist.loc[:, ~price_hist.columns.duplicated()]

        common_clean = [c for c in eps_df.columns if c in price_hist.columns]
        price_sub = price_hist[common_clean]
        eps_sub = eps_df[common_clean]

        return price_sub, eps_sub, names, markets, sector_mapping

    return _make_sample_raw_data()


def _make_sample_raw_data():
    """Streamlit Cloud 또는 데이터 부재 시 테스트용 샘플 데이터셋"""
    sample_stocks = [
        ("005930", "삼성전자", "KOSPI200", "반도체"),
        ("000660", "SK하이닉스", "KOSPI200", "반도체"),
        ("035420", "NAVER", "KOSPI200", "IT서비스"),
        ("005380", "현대차", "KOSPI200", "자동차"),
        ("051910", "LG화학", "KOSPI200", "화학"),
        ("006400", "삼성SDI", "KOSPI200", "2차전지"),
        ("000270", "기아", "KOSPI200", "자동차"),
        ("012330", "현대모비스", "KOSPI200", "자동차"),
        ("003550", "LG", "KOSPI200", "지주사"),
        ("034730", "SK", "KOSPI200", "지주사"),
        ("035900", "JYP Ent.", "KOSDAQ150", "엔터"),
        ("041510", "에스엠", "KOSDAQ150", "엔터"),
        ("263750", "펄어비스", "KOSDAQ150", "게임"),
        ("293490", "카카오게임즈", "KOSDAQ150", "게임"),
        ("145020", "휴젤", "KOSDAQ150", "바이오"),
    ]
    np.random.seed(42)
    dates_daily = pd.bdate_range("2015-01-01", periods=2800)
    p_data, e_data, names, markets, sec_map = {}, {}, {}, {}, {}

    for ticker, name, mkt, sec in sample_stocks:
        base_p = np.random.uniform(30000, 180000)
        base_e = base_p / np.random.uniform(8.0, 22.0)
        p_noise = np.random.normal(0.0003, 0.015, len(dates_daily))
        e_noise = np.random.normal(0.0002, 0.005, len(dates_daily))
        p_data[ticker] = base_p * np.cumprod(1.0 + p_noise)
        e_data[ticker] = base_e * np.cumprod(1.0 + e_noise)
        names[ticker] = name
        markets[ticker] = mkt
        sec_map[ticker] = sec

    return pd.DataFrame(p_data, index=dates_daily), pd.DataFrame(e_data, index=dates_daily), names, markets, sec_map


# ──────────────────────────────────────────────
# Tier 2 Caching: 인메모리 스크리너 연산 (<100ms)
# ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_screener_results(
    band_years: int,
    band_model: str = "percentile",
    winsorize: bool = True,
    mtimes: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
) -> List[PEBandResult]:
    price_df, eps_df, names, markets, sector_mapping = load_raw_market_data(mtimes=mtimes)
    results = run_screener(
        price_df=price_df,
        eps_df=eps_df,
        ticker_names=names,
        band_years=band_years,
        band_model=band_model,
        winsorize=winsorize,
        sector_mapping=sector_mapping,
    )
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime(tickers: Tuple[str, ...]) -> Dict[str, float]:
    prices = get_current_prices_batch(list(tickers))
    return prices.to_dict()


def apply_realtime_prices(
    results: List[PEBandResult],
    rt_prices: Dict[str, float],
    sector_mapping: Dict[str, str],
) -> List[PEBandResult]:
    """실시간 시세를 반영하여 P/E, 백분위, 업사이드, 전략 레짐 및 섹터 상대 지표를 고속 업데이트"""
    if not rt_prices:
        return results

    import dataclasses
    updated = []
    need_recalc_sector = False

    for r in results:
        rt_p = rt_prices.get(r.ticker)
        # 유효한 실시간 가격이고 기존 주가와 다른 경우에만 업데이트
        if rt_p is not None and rt_p > 0 and not np.isnan(rt_p) and rt_p != r.current_price:
            # 1. 흑자 기업 (current_fwd_eps > 0): P/E, 백분위, 업사이드 및 전략 레짐 동적 재평가
            if r.current_fwd_eps is not None and not np.isnan(r.current_fwd_eps) and r.current_fwd_eps > 0:
                need_recalc_sector = True
                rt_pe = rt_p / r.current_fwd_eps
                hist_pe = (
                    r.hist_pe_series.dropna()
                    if (r.hist_pe_series is not None and hasattr(r.hist_pe_series, "dropna"))
                    else pd.Series(dtype=float)
                )
                rt_pct = (
                    float((hist_pe < rt_pe).mean() * 100.0)
                    if len(hist_pe) > 0
                    else (r.pe_percentile if r.pe_percentile is not None else np.nan)
                )
                up_bear = (
                    ((r.target_bear / rt_p) - 1.0) * 100.0
                    if (r.target_bear is not None and not np.isnan(r.target_bear))
                    else r.upside_bear
                )
                up_base = (
                    ((r.target_base / rt_p) - 1.0) * 100.0
                    if (r.target_base is not None and not np.isnan(r.target_base))
                    else r.upside_base
                )
                up_bull = (
                    ((r.target_bull / rt_p) - 1.0) * 100.0
                    if (r.target_bull is not None and not np.isnan(r.target_bull))
                    else r.upside_bull
                )

                # 실시간 주가 변동에 따른 5-레짐 동적 재분류 (M1 / R1)
                reg_res = classify_regime(
                    pe_pct=rt_pct,
                    eps_rev_1m=r.eps_rev_1m,
                    eps_rev_3m=r.eps_rev_3m,
                    eps_rev_1w=r.eps_rev_1w,
                    current_pe=rt_pe,
                )

                r2 = dataclasses.replace(
                    r,
                    current_price=rt_p,
                    current_fwd_pe=rt_pe,
                    pe_percentile=rt_pct,
                    upside_bear=up_bear,
                    upside_base=up_base,
                    upside_bull=up_bull,
                    is_golden_cross=reg_res.is_golden_cross,
                    is_value_trap=reg_res.is_value_trap,
                    regime_tag=reg_res.regime_tag,
                )
                updated.append(r2)
            else:
                # 2. 적자/턴어라운드 기업 (current_fwd_eps <= 0 또는 결측):
                # 실시간 현재가는 즉시 갱신하되, 0/음수 분모로 인한 P/E 왜곡 및 인위적 -100% 목표가 산출 차단
                r2 = dataclasses.replace(
                    r,
                    current_price=rt_p,
                )
                updated.append(r2)
        else:
            updated.append(r)

    if need_recalc_sector:
        updated = calc_sector_relative_metrics(updated, sector_mapping)

    return updated


CUSTOM_CSS = """
<style>
/* Global Dark Theme Overrides */
.stApp {
    background-color: #0a0a14 !important;
    color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #0d0d1a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
button[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 8px 16px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    overflow: hidden;
}

/* Header Styling */
.dg-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 12px;
}
.dg-logo {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
}
.dg-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 4px;
}
.dg-date {
    font-size: 0.78rem;
    color: #64748b;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background: rgba(255, 255, 255, 0.04);
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* KPI Row & Cards */
.kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
    width: 100%;
}
.kpi-card {
    flex: 1 1 180px;
    min-width: 160px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    box-sizing: border-box;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.15);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 2px 0;
}
.kpi-blue { color: #818cf8; }
.kpi-yellow { color: #fbbf24; }
.kpi-green { color: #34d399; }
.kpi-red { color: #f87171; }
.kpi-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
}

/* P/E Percentile Progress Bar */
.pe-bar {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
    box-sizing: border-box;
}
.pe-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* Target Price Scenario Grid */
.target-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 6px;
}
.target-box {
    border-radius: 8px;
    padding: 8px 4px;
    text-align: center;
    box-sizing: border-box;
}
.target-bear {
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
}
.target-base {
    background: rgba(129, 140, 248, 0.08);
    border: 1px solid rgba(129, 140, 248, 0.25);
}
.target-bull {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
}
.t-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #94a3b8;
}
.t-price {
    font-size: 0.82rem;
    font-weight: 700;
    margin: 2px 0;
}
.t-upside {
    font-size: 0.72rem;
    font-weight: 700;
}
.t-bear-c { color: #f87171; }
.t-base-c { color: #818cf8; }
.t-bull-c { color: #34d399; }

/* Valuation Badges */
.sig {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    line-height: 1.4;
    text-align: center;
    white-space: nowrap;
}
.sig-sb {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.35);
}
.sig-b {
    background: rgba(96, 165, 250, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(96, 165, 250, 0.35);
}
.sig-h {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.35);
}
.sig-s {
    background: rgba(249, 115, 22, 0.15);
    color: #fb923c;
    border: 1px solid rgba(249, 115, 22, 0.35);
}
.sig-ss {
    background: rgba(248, 113, 113, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.35);
}
.sig-neu {
    background: rgba(156, 163, 175, 0.15);
    color: #9ca3af;
    border: 1px solid rgba(156, 163, 175, 0.3);
}

/* Strategy Signal Badges */
.sig-gc {
    background: rgba(251, 191, 36, 0.15);
    color: #fde047;
    border: 1px solid rgba(251, 191, 36, 0.45);
    font-weight: 700;
}
.sig-vt {
    background: rgba(248, 113, 113, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(248, 113, 113, 0.45);
    font-weight: 700;
}
.sig-em {
    background: rgba(56, 189, 248, 0.15);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.45);
    font-weight: 700;
}
.sig-val {
    background: rgba(192, 132, 252, 0.15);
    color: #d8b4fe;
    border: 1px solid rgba(192, 132, 252, 0.45);
    font-weight: 700;
}
.sig-hd {
    background: rgba(244, 63, 94, 0.15);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.45);
    font-weight: 700;
}
</style>
"""


# ──────────────────────────────────────────────
# Streamlit App Execution Entrypoint
# ──────────────────────────────────────────────
def main():
    # ──────────────────────────────────────────
    # 1. 페이지 설정 및 PWA/테마 CSS 인젝션
    # ──────────────────────────────────────────
    try:
        st.set_page_config(
            page_title="Fwd P/E & 실적 모멘텀 분석 시스템",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html("""
    <script>
    const parentDoc = window.parent.document;
    if (!parentDoc.getElementById("pwa-manifest")) {
        const manifest = {
            "name": "Fwd P/E Screener",
            "short_name": "Screener",
            "start_url": ".",
            "display": "standalone",
            "background_color": "#0a0a14",
            "theme_color": "#6366f1"
        };
        const blob = new Blob([JSON.stringify(manifest)], {type: 'application/json'});
        const manifestURL = URL.createObjectURL(blob);
        const link = parentDoc.createElement('link');
        link.rel = 'manifest';
        link.id = 'pwa-manifest';
        link.href = manifestURL;
        parentDoc.head.appendChild(link);
    }
    </script>
    """, height=0, width=0)

    # ──────────────────────────────────────────
    # 2. 사이드바 컨트롤 & 전역 필터
    # ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 분석 파라미터")
        st.markdown("---")

        band_years = st.select_slider(
            "📅 역사적 밴드 기간",
            options=[1, 2, 3, 5, 7, 10, 15],
            value=5,
            help="과거 N년의 시계열을 바탕으로 P/E 밴드를 산출합니다. (인메모리 연산으로 50ms 내 업데이트)",
        )

        st.markdown("---")
        st.markdown("**🔍 스크리닝 필터**")

        market_filter = st.multiselect(
            "지수 유니버스",
            ["KOSPI200", "KOSDAQ150"],
            default=["KOSPI200", "KOSDAQ150"],
        )

        strategy_options = ["✨골든크로스", "⚠️밸류트랩", "🚀어닝모멘텀", "💎가치주", "🔻고P/E하향", "Neutral"]
        strategy_filter = st.multiselect(
            "전략 신호 필터",
            strategy_options,
            default=strategy_options,
        )

        min_upside = st.slider("Base 업사이드 최소 (%)", -50, 100, -30)
        pe_pct_range = st.slider("전체 P/E 위치 범위 (%)", 0, 100, (0, 90))

        st.markdown("---")
        if st.button("🔄 데이터 캐시 즉시 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.session_state.active_preset = "ALL"
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#4b5563'>"
            "DataGuide × pykrx<br>Fwd P/E & Momentum System v3.0</div>",
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────
    # 3. 마켓 데이터 로딩 및 스크리닝 연산 (Two-Tier Caching)
    # ──────────────────────────────────────────
    file_mtimes = _get_file_mtimes()

    with st.spinner("마켓 데이터 로드 중..."):
        price_df, eps_df, names, markets, sector_mapping = load_raw_market_data(mtimes=file_mtimes)

    if eps_df.empty or price_df.empty:
        st.error("data/fwd_eps.xlsx 또는 data/price.xlsx 파일을 찾을 수 없습니다.")
        st.stop()

    # Screener dataset 계산 (Tier 2 인메모리 캐시)
    with st.spinner("스크리너 계산 중..."):
        base_results = get_cached_screener_results(
            band_years=band_years,
            band_model="percentile",
            winsorize=True,
            mtimes=file_mtimes,
        )

    # 실시간 현재가 반영
    tickers_tuple = tuple(r.ticker for r in base_results)
    rt_dict = fetch_realtime(tickers_tuple)
    all_results = apply_realtime_prices(base_results, rt_dict, sector_mapping)

    # 필터링 적용
    filtered_results = []
    for r in all_results:
        mkt = markets.get(r.ticker, "")
        if market_filter and mkt not in market_filter:
            continue
        sig = get_strategy_signal(r)
        if strategy_filter and sig not in strategy_filter:
            continue
        if r.upside_base < min_upside:
            continue
        if not (pe_pct_range[0] <= r.pe_percentile <= pe_pct_range[1]):
            continue
        filtered_results.append(r)

    filtered_results.sort(key=lambda r: r.upside_base, reverse=True)

    # ──────────────────────────────────────────
    # 4. 헤더 및 글로벌 KPI 요약
    # ──────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="dg-header">
      <div>
        <div class="dg-logo">Fwd P/E & 실적 모멘텀 분석 대시보드</div>
        <div class="dg-sub">12M Forward EPS 밴드 모델 · WICS 섹터 상대 밸류에이션 · 애널리스트 실무형 프리셋</div>
      </div>
      <div class="dg-date">기준시각: {now_str}</div>
    </div>
    """, unsafe_allow_html=True)

    total_cnt = len(filtered_results)
    gc_cnt = sum(1 for r in filtered_results if r.is_golden_cross or get_strategy_signal(r) == "✨골든크로스")
    vt_cnt = sum(1 for r in filtered_results if r.is_value_trap or get_strategy_signal(r) == "⚠️밸류트랩")
    top_mom_cnt = sum(1 for r in filtered_results if (r.eps_rev_1m or 0) >= 3.0)
    avg_upside = np.mean([r.upside_base for r in filtered_results]) if filtered_results else 0.0
    top_pick = filtered_results[0].name if filtered_results else "-"
    top_pick_up = filtered_results[0].upside_base if filtered_results else 0.0

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">분석 대상 종목</div>
        <div class="kpi-value kpi-blue">{total_cnt}<span style="font-size:1rem">개</span></div>
        <div class="kpi-sub">KOSPI200 + KOSDAQ150</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">✨ 골든크로스 (저평가+반등)</div>
        <div class="kpi-value kpi-yellow">{gc_cnt}<span style="font-size:1rem">개</span></div>
        <div class="kpi-sub">P/E ≤ 40% & 1M EPS 반등</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🚀 어닝 상향 종목</div>
        <div class="kpi-value kpi-green">{top_mom_cnt}<span style="font-size:1rem">개</span></div>
        <div class="kpi-sub">1M EPS Revision ≥ +3%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">⚠️ 밸류트랩 경고</div>
        <div class="kpi-value kpi-red">{vt_cnt}<span style="font-size:1rem">개</span></div>
        <div class="kpi-sub">P/E ≤ 30% & 1M EPS 급락</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">평균 Base 업사이드</div>
        <div class="kpi-value {'kpi-green' if avg_upside>=0 else 'kpi-red'}">{avg_upside:+.1f}<span style="font-size:1rem">%</span></div>
        <div class="kpi-sub">Top Pick: {top_pick} ({top_pick_up:+.1f}%)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # 5. 전역 검색 바 (Global Search)
    # ──────────────────────────────────────────
    if "sel_ticker" not in st.session_state:
        st.session_state.sel_ticker = None
    if "_synced_ticker" not in st.session_state:
        st.session_state._synced_ticker = None

    sc1, sc2 = st.columns([5, 1])
    with sc1:
        search_q = st.text_input(
            "종목 검색",
            placeholder="🔍  종목명 또는 6자리 코드 입력 (예: 삼성전자, 005930, 현대차, 반도체)",
            label_visibility="collapsed",
            key="global_search_input",
        )
    with sc2:
        if st.button("❌ 선택 초기화", use_container_width=True):
            st.session_state.sel_ticker = None
            st.session_state._synced_ticker = None
            st.rerun()

    if search_q and search_q.strip():
        sq = search_q.strip().lower()
        matches = [
            r for r in all_results
            if sq in r.name.lower() or sq in r.ticker.lower() or sq in r.sector.lower()
        ]
        matches.sort(key=lambda r: r.upside_base, reverse=True)

        if not matches:
            st.info(f"🔍 '{search_q}' 검색 결과가 없습니다.")
        else:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:8px'>"
                f"🔍 '{search_q}' 검색 결과: {len(matches)}개 종목 (카드를 클릭하면 즉시 상세 대시보드로 이동합니다)</div>",
                unsafe_allow_html=True,
            )
            srch_cols = st.columns(min(len(matches), 4))
            for idx, m in enumerate(matches[:4]):
                with srch_cols[idx]:
                    is_sel = (st.session_state.sel_ticker == m.ticker)
                    bcol = "#818cf8" if is_sel else "rgba(255,255,255,0.1)"
                    up_c = "#34d399" if m.upside_base >= 0 else "#f87171"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1.5px solid {bcol};
                                border-radius:12px;padding:12px 14px;margin-bottom:8px;">
                      <div style="font-size:0.7rem;color:#6b7280;">{m.ticker} · {m.sector}</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:2px 0;">{m.name}</div>
                      <div style="font-size:0.8rem;color:#9ca3af;">현재가: <b>₩{m.current_price:,.0f}</b></div>
                      <div style="font-size:0.78rem;color:#9ca3af;margin-top:2px;">Fwd P/E: <b>{m.current_fwd_pe:.1f}x</b> ({m.pe_percentile:.0f}%)</div>
                      <div style="font-size:0.82rem;font-weight:700;color:{up_c};margin-top:4px;">Base {m.upside_base:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("📊 상세 분석 보기", key=f"btn_search_{m.ticker}", use_container_width=True,
                                 type="primary" if is_sel else "secondary"):
                        st.session_state.sel_ticker = m.ticker
                        st.rerun()

    st.markdown(
        '<div style="height:2px;background:linear-gradient(90deg,#4f46e5,#7c3aed,#ec4899);'
        'border-radius:2px;margin:12px 0 16px;"></div>',
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────
    # 전역 스크리닝 데이터셋 구축 (탭 간 독립 스코프)
    # ──────────────────────────────────────────
    display_cols = [
        "전략 신호", "종목명", "코드", "섹터", "지수", "현재가", "Fwd EPS",
        "1W %", "1M %", "3M %", "Fwd P/E", "P/E 위치(%)", "섹터 P/E 위치(%)",
        "Bear목표", "Base목표", "Bull목표", "Base%"
    ]
    table_rows = []
    for r in filtered_results:
        mkt = markets.get(r.ticker, "")
        sig_str = get_strategy_signal(r)
        table_rows.append({
            "전략 신호": sig_str,
            "종목명": r.name,
            "코드": r.ticker,
            "섹터": r.sector,
            "지수": mkt,
            "현재가": float(r.current_price),
            "Fwd EPS": float(r.current_fwd_eps),
            "1W %": float(r.eps_rev_1w) if r.eps_rev_1w is not None else np.nan,
            "1M %": float(r.eps_rev_1m) if r.eps_rev_1m is not None else np.nan,
            "3M %": float(r.eps_rev_3m) if r.eps_rev_3m is not None else np.nan,
            "Fwd P/E": float(r.current_fwd_pe) if r.current_fwd_pe is not None else np.nan,
            "P/E 위치(%)": float(r.pe_percentile) if r.pe_percentile is not None else np.nan,
            "섹터 P/E 위치(%)": float(r.sector_pe_percentile) if r.sector_pe_percentile is not None else np.nan,
            "Bear목표": float(r.target_bear),
            "Base목표": float(r.target_base),
            "Bull목표": float(r.target_bull),
            "Base%": float(r.upside_base),
            "pe_percentile": float(r.pe_percentile) if r.pe_percentile is not None else 100.0,
            "sector_pe_percentile": float(r.sector_pe_percentile) if r.sector_pe_percentile is not None else 100.0,
            "eps_rev_1m": float(r.eps_rev_1m) if r.eps_rev_1m is not None else -999.0,
            "current_fwd_eps": float(r.current_fwd_eps),
            "current_fwd_pe": float(r.current_fwd_pe) if r.current_fwd_pe is not None else 999.0,
            "upside_base": float(r.upside_base),
        })
    df_table_all = pd.DataFrame(table_rows)
    df_screener_all = df_table_all[display_cols].copy() if not df_table_all.empty else pd.DataFrame(columns=display_cols)

    # ──────────────────────────────────────────
    # 6. 메인 탭 네비게이션
    # ──────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📋  스크리닝 테이블",
        "🔍  원페이지 통합 대시보드",
        "🪷  밸류에이션 버블 차트",
    ])

    # ══════════════════════════════════════════
    # TAB 1: 스크리닝 테이블 & 1-Click 전략 프리셋
    # ══════════════════════════════════════════
    with tab1:
        st.markdown("#### ⚡ 애널리스트 실무형 퀵 스크리닝 프리셋 (1-Click Presets)")

        if "active_preset" not in st.session_state:
            st.session_state.active_preset = "ALL"

        p_cols = st.columns(5)
        presets = [
            ("ALL", "🌟 전체 (ALL)", "초기화 및 전체 유니버스 탐색"),
            ("TURNAROUND", "✨ 저평가+실적 반등", "P/E ≤ 40% & 1M EPS 반등 (>0%) & Base 업사이드 ≥ 10%"),
            ("EPS_TOP", "🚀 어닝 상향 Top", "1M EPS Revision ≥ +3% 최상위 모멘텀 Top 25"),
            ("VALUE", "💎 가치주", "P/E ≤ 25% & Fwd P/E ≤ 15x & 1M EPS ≥ -2% & Base 업사이드 ≥ 15%"),
            ("TRAP", "⚠️ 밸류트랩 주의", "P/E ≤ 30% & 1M EPS 급락 (<-3%) 경고 종목군"),
        ]

        for col, (pkey, plabel, pdesc) in zip(p_cols, presets):
            is_active = (st.session_state.active_preset == pkey)
            if col.button(
                plabel,
                key=f"preset_pill_{pkey}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                help=pdesc,
            ):
                st.session_state.active_preset = pkey
                st.rerun()

        if not df_table_all.empty:
            df_preset = apply_quick_preset(df_table_all, st.session_state.active_preset)
        else:
            df_preset = df_table_all

        active_p_name = next((p[1] for p in presets if p[0] == st.session_state.active_preset), "전체")
        st.caption(f"현재 적용된 전략 프리셋: **{active_p_name}** ({len(df_preset)}개 종목 매칭)")

        if df_preset.empty:
            df_display = pd.DataFrame(columns=display_cols)
            st.warning("선택된 전략 프리셋 및 필터 조건에 부합하는 종목이 없습니다.")
        else:
            df_display = df_preset[display_cols].copy()

            # 다운로드 버튼
            import io
            dl1, dl2, _ = st.columns([1.2, 1.2, 3])
            with dl1:
                buf_xl = io.BytesIO()
                df_display.to_excel(buf_xl, index=False)
                st.download_button(
                    "📥 스크리닝 Excel 다운로드",
                    data=buf_xl.getvalue(),
                    file_name=f"pe_screener_{st.session_state.active_preset}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with dl2:
                csv_bytes = df_display.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 스크리닝 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"pe_screener_{st.session_state.active_preset}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # 업그레이드된 인터랙티브 테이블 렌더링
            event = st.dataframe(
                df_display,
                column_config={
                    "현재가": st.column_config.NumberColumn("현재가", format="₩%,.0f"),
                    "Fwd EPS": st.column_config.NumberColumn("Fwd EPS", format="₩%,.0f"),
                    "1W %": st.column_config.NumberColumn("1W %", format="%+.1f%%"),
                    "1M %": st.column_config.NumberColumn("1M %", format="%+.1f%%"),
                    "3M %": st.column_config.NumberColumn("3M %", format="%+.1f%%"),
                    "Fwd P/E": st.column_config.NumberColumn("Fwd P/E", format="%.1fx"),
                    "P/E 위치(%)": st.column_config.NumberColumn("P/E 위치(%)", format="%.0f%%"),
                    "섹터 P/E 위치(%)": st.column_config.NumberColumn("섹터 P/E 위치(%)", format="%.0f%%"),
                    "Bear목표": st.column_config.NumberColumn("Bear목표", format="₩%,.0f"),
                    "Base목표": st.column_config.NumberColumn("Base목표", format="₩%,.0f"),
                    "Bull목표": st.column_config.NumberColumn("Bull목표", format="₩%,.0f"),
                    "Base%": st.column_config.NumberColumn("Base%", format="%+.1f%%"),
                },
                use_container_width=True,
                height=min(80 + len(df_display) * 36, 560),
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
            )

            if hasattr(event, "selection") and event.selection.rows:
                sel_idx = event.selection.rows[0]
                if sel_idx < len(df_display):
                    clicked_ticker = df_display.iloc[sel_idx]["코드"]
                    if st.session_state.get("sel_ticker") != clicked_ticker:
                        st.session_state.sel_ticker = clicked_ticker
                        st.rerun()

            if st.session_state.sel_ticker:
                matched_name = names.get(st.session_state.sel_ticker, st.session_state.sel_ticker)
                st.info(
                    f"선택된 종목: **{matched_name} ({st.session_state.sel_ticker})** — 상단의 **'🔍 원페이지 통합 대시보드'** 탭에서 2단 연동 차트와 세부 시나리오를 확인하실 수 있습니다."
                )

    # ══════════════════════════════════════════
    # TAB 2: 원페이지 통합 대시보드
    # ══════════════════════════════════════════
    with tab2:
        if not all_results:
            st.warning("분석 가능한 종목 데이터가 없습니다.")
        else:
            all_ticker_map = {
                f"{r.name} ({r.ticker}) · {r.sector} · {get_strategy_signal(r)}": r.ticker
                for r in all_results
            }
            all_options = list(all_ticker_map.keys())

            # 외부(테이블 클릭/검색)에서 sel_ticker가 변경되었을 경우 selectbox 위젯 키를 선제 동기화
            if st.session_state.sel_ticker and st.session_state.sel_ticker != st.session_state._synced_ticker:
                matched_opt = next((opt for opt, tk in all_ticker_map.items() if tk == st.session_state.sel_ticker), None)
                if matched_opt:
                    st.session_state["dashboard_stock_selector"] = matched_opt
                st.session_state._synced_ticker = st.session_state.sel_ticker
            elif not st.session_state.sel_ticker and all_options:
                st.session_state.sel_ticker = all_ticker_map[all_options[0]]
                st.session_state._synced_ticker = st.session_state.sel_ticker
                st.session_state["dashboard_stock_selector"] = all_options[0]

            sel_box_val = st.selectbox(
                "분석 대상 종목 선택",
                all_options,
                label_visibility="collapsed",
                key="dashboard_stock_selector",
            )
            active_ticker = all_ticker_map.get(sel_box_val, st.session_state.sel_ticker)

            if st.session_state.sel_ticker != active_ticker:
                st.session_state.sel_ticker = active_ticker
                st.session_state._synced_ticker = active_ticker

            target_r = next((x for x in all_results if x.ticker == active_ticker), None)

            if not target_r:
                st.error("선택된 종목의 정보를 불러올 수 없습니다.")
            else:
                mkt_str = markets.get(target_r.ticker, "")
                strat_sig = get_strategy_signal(target_r)

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                            border-radius:14px;padding:16px 20px;margin-bottom:14px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                    <div>
                      <span style="font-size:0.85rem;color:#818cf8;font-weight:600;">{target_r.ticker}</span>
                      <span style="color:#6b7280;margin:0 6px;">·</span>
                      <span style="font-size:0.85rem;color:#9ca3af;">{target_r.sector}</span>
                      <span style="color:#6b7280;margin:0 6px;">·</span>
                      <span style="font-size:0.85rem;color:#9ca3af;">{mkt_str}</span>
                      <h2 style="margin:4px 0 0 0;font-size:1.8rem;font-weight:800;color:#f1f5f9;">{target_r.name}</h2>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:0.75rem;color:#9ca3af;">실시간 현재가</div>
                      <div style="font-size:1.9rem;font-weight:800;color:#f1f5f9;">₩{target_r.current_price:,.0f}</div>
                      <div style="margin-top:4px;">
                        {strategy_badge_html(strat_sig)}
                        <span style="margin-left:6px;">{signal_badge(target_r.pe_percentile)}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # 인터랙티브 밴드 모델 스위처
                ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([3, 2, 2])
                with ctrl_c1:
                    band_model_choice = st.radio(
                        "🎯 밸류에이션 밴드 산출 모델",
                        ["백분위수 밴드 (10~90%)", "리서치 표준 밴드 (Mean ± 1/2SD)", "고정 배수 밴드 (8x~15x)"],
                        index=0,
                        horizontal=True,
                        key=f"band_model_toggle_{target_r.ticker}",
                    )
                with ctrl_c2:
                    use_winsorize = st.checkbox(
                        "이상치 제거 (95% Winsorization)",
                        value=True,
                        key=f"winsor_toggle_{target_r.ticker}",
                        help="일시적 적자나 일회성 손익으로 인한 극단적 P/E 왜곡을 보정합니다.",
                    )
                with ctrl_c3:
                    model_period = st.selectbox(
                        "📅 시계열 분석 기간",
                        options=[1, 2, 3, 5, 7, 10, 15],
                        index=3 if band_years == 5 else ([1, 2, 3, 5, 7, 10, 15].index(band_years) if band_years in [1, 2, 3, 5, 7, 10, 15] else 3),
                        key=f"period_toggle_{target_r.ticker}",
                    )

                model_key = "percentile"
                if "Mean" in band_model_choice:
                    model_key = "mean_sd"
                elif "고정" in band_model_choice:
                    model_key = "fixed"

                p_series = target_r.hist_price_series.dropna().sort_index()
                e_series = eps_df[target_r.ticker].dropna().sort_index() if target_r.ticker in eps_df.columns else target_r.hist_eps_series.dropna().sort_index()

                r_active = calc_pe_band(
                    ticker=target_r.ticker,
                    name=target_r.name,
                    price_series=p_series,
                    eps_series=e_series,
                    band_years=model_period,
                    band_model=model_key,
                    winsorize=use_winsorize,
                    sector=target_r.sector,
                    eps_rev_1w=target_r.eps_rev_1w,
                    eps_rev_1m=target_r.eps_rev_1m,
                    eps_rev_3m=target_r.eps_rev_3m,
                )

                if r_active is None:
                    r_active = target_r
                else:
                    r_active.current_price = target_r.current_price
                    r_active.current_fwd_pe = target_r.current_fwd_pe
                    if target_r.current_price > 0:
                        r_active.upside_bear = ((r_active.target_bear / target_r.current_price) - 1.0) * 100.0
                        r_active.upside_base = ((r_active.target_base / target_r.current_price) - 1.0) * 100.0
                        r_active.upside_bull = ((r_active.target_bull / target_r.current_price) - 1.0) * 100.0

                    r_active.sector = target_r.sector
                    r_active.sector_median_pe = target_r.sector_median_pe
                    r_active.sector_relative_pe = target_r.sector_relative_pe
                    r_active.sector_pe_percentile = target_r.sector_pe_percentile
                    r_active.sector_stock_count = target_r.sector_stock_count
                    r_active.regime_tag = target_r.regime_tag
                    r_active.is_value_trap = target_r.is_value_trap
                    r_active.is_golden_cross = target_r.is_golden_cross

                # 4-Card 애널리스트 KPI 덱
                kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

                with kpi_c1:
                    up_col = "#34d399" if r_active.upside_base >= 0 else "#f87171"
                    st.markdown(f"""
                    <div class="kpi-card" style="height:100%">
                      <div class="kpi-label">1. 주가 & 전략 신호</div>
                      <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin:4px 0 2px;">
                        ₩{r_active.current_price:,.0f}
                      </div>
                      <div style="margin:6px 0;">
                        {strategy_badge_html(get_strategy_signal(r_active))}
                      </div>
                      <div style="font-size:0.78rem;font-weight:600;color:{up_col};margin-top:4px;">
                        Base 업사이드 {r_active.upside_base:+.1f}%
                      </div>
                      <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                        밸류에이션 평가: {signal_label(r_active.pe_percentile)}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with kpi_c2:
                    pct_col = pe_bar_color(r_active.pe_percentile)
                    sec_med_str = f"{r_active.sector_median_pe:.1f}x" if r_active.sector_median_pe else "N/A"
                    sec_pct_str = f"{r_active.sector_pe_percentile:.0f}%" if r_active.sector_pe_percentile is not None else "N/A"
                    sec_pct_col = pe_bar_color(r_active.sector_pe_percentile if r_active.sector_pe_percentile is not None else 50)
                    st.markdown(f"""
                    <div class="kpi-card" style="height:100%">
                      <div class="kpi-label">2. 밸류에이션 & 섹터 순위</div>
                      <div style="font-size:1.6rem;font-weight:700;color:#818cf8;margin:4px 0 2px;">
                        {r_active.current_fwd_pe:.1f}<span style="font-size:1rem">x</span>
                      </div>
                      <div style="font-size:0.78rem;color:#9ca3af;margin-top:4px;">
                        역사적 백분위: <b style="color:{pct_col}">{r_active.pe_percentile:.0f}%</b>
                      </div>
                      <div class="pe-bar"><div class="pe-fill" style="width:{min(100, max(0, r_active.pe_percentile)):.0f}%;background:{pct_col}"></div></div>
                      <div style="font-size:0.75rem;color:#9ca3af;margin-top:6px;">
                        동일 섹터({r_active.sector}): 중앙값 <b style="color:#f1f5f9">{sec_med_str}</b> · 위치 <b style="color:{sec_pct_col}">{sec_pct_str}</b>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with kpi_c3:
                    rev_1w_str = f"{r_active.eps_rev_1w:+.1f}%" if r_active.eps_rev_1w is not None else "N/A"
                    rev_1m_str = f"{r_active.eps_rev_1m:+.1f}%" if r_active.eps_rev_1m is not None else "N/A"
                    rev_3m_str = f"{r_active.eps_rev_3m:+.1f}%" if r_active.eps_rev_3m is not None else "N/A"
                    col_1w = "#34d399" if (r_active.eps_rev_1w or 0) > 0 else ("#f87171" if (r_active.eps_rev_1w or 0) < 0 else "#9ca3af")
                    col_1m = "#34d399" if (r_active.eps_rev_1m or 0) > 0 else ("#f87171" if (r_active.eps_rev_1m or 0) < 0 else "#9ca3af")
                    col_3m = "#34d399" if (r_active.eps_rev_3m or 0) > 0 else ("#f87171" if (r_active.eps_rev_3m or 0) < 0 else "#9ca3af")
                    st.markdown(f"""
                    <div class="kpi-card" style="height:100%">
                      <div class="kpi-label">3. 12M Fwd EPS & 리비전</div>
                      <div style="font-size:1.6rem;font-weight:700;color:#60a5fa;margin:4px 0 2px;">
                        ₩{r_active.current_fwd_eps:,.0f}
                      </div>
                      <div style="font-size:0.75rem;color:#9ca3af;margin-top:6px;display:flex;justify-content:space-between;">
                        <span>1W: <b style="color:{col_1w}">{rev_1w_str}</b></span>
                        <span>1M: <b style="color:{col_1m}">{rev_1m_str}</b></span>
                        <span>3M: <b style="color:{col_3m}">{rev_3m_str}</b></span>
                      </div>
                      <div style="font-size:0.72rem;color:#6b7280;margin-top:6px;">
                        과거 {model_period}년 평균 P/E: {r_active.pe_mean:.1f}x · 중앙값: {r_active.pe_median:.1f}x
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with kpi_c4:
                    model_lbl = "백분위 (25/50/75)" if model_key == "percentile" else ("리서치 표준 (±1SD)" if model_key == "mean_sd" else "고정 배수")
                    st.markdown(f"""
                    <div class="kpi-card" style="height:100%">
                      <div class="kpi-label">4. 목표가 시나리오 ({model_lbl})</div>
                      <div class="target-grid" style="margin-top:6px;">
                        <div class="target-box target-bear">
                          <div class="t-label">Bear</div>
                          <div class="t-price t-bear-c">₩{r_active.target_bear:,.0f}</div>
                          <div class="t-upside t-bear-c">{r_active.upside_bear:+.1f}%</div>
                        </div>
                        <div class="target-box target-base">
                          <div class="t-label">Base</div>
                          <div class="t-price t-base-c">₩{r_active.target_base:,.0f}</div>
                          <div class="t-upside t-base-c">{r_active.upside_base:+.1f}%</div>
                        </div>
                        <div class="target-box target-bull">
                          <div class="t-label">Bull</div>
                          <div class="t-price t-bull-c">₩{r_active.target_bull:,.0f}</div>
                          <div class="t-upside t-bull-c">{r_active.upside_bull:+.1f}%</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2단 수직 연동 종합 차트 (Synchronized Subplots)
                common_idx = p_series.index.intersection(e_series.index)
                if len(common_idx) < 4:
                    aligned_e = e_series.reindex(e_series.index.union(p_series.index)).ffill().loc[p_series.index].dropna()
                    common_idx = p_series.index.intersection(aligned_e.index)
                    if len(common_idx) >= 4:
                        e_series = aligned_e

                if len(common_idx) < 2:
                    st.warning("시계열 데이터가 부족하여 차트를 표시할 수 없습니다.")
                else:
                    cutoff_date = common_idx.max() - pd.DateOffset(years=model_period)
                    plot_idx = common_idx[common_idx >= cutoff_date]
                    if len(plot_idx) < 2:
                        plot_idx = common_idx

                    p_plot = p_series.loc[plot_idx]
                    e_plot = e_series.loc[plot_idx]
                    pe_plot = (p_plot / e_plot).where(e_plot > 0, np.nan)
                    pe_plot = pe_plot.where((pe_plot > 0) & (pe_plot <= 200), np.nan)

                    fig_integrated = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_heights=[0.65, 0.35],
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                    )

                    # Subplot 1 (Upper): 주가 및 동적 밸류에이션 밴드
                    fig_integrated.add_trace(
                        go.Scatter(
                            x=p_plot.index, y=p_plot.values,
                            mode="lines",
                            name="실제 주가",
                            line=dict(color="#38bdf8", width=2.5),
                        ),
                        row=1, col=1
                    )

                    if model_key == "percentile" and r_active.band_series_pct is not None:
                        b_df = r_active.band_series_pct.loc[r_active.band_series_pct.index.isin(plot_idx)]
                        if "p90" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["p90"], mode="lines", name="p90 밴드",
                                                                line=dict(color="rgba(192,132,252,0.6)", width=1.2, dash="dash")), row=1, col=1)
                        if "p75" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["p75"], mode="lines", name="Bull 밴드 (75th)",
                                                                line=dict(color="rgba(52,211,153,0.85)", width=1.6, dash="dash")), row=1, col=1)
                        if "p50" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["p50"], mode="lines", name="Base 밴드 (Median)",
                                                                line=dict(color="rgba(251,191,36,0.9)", width=2.0, dash="dash")), row=1, col=1)
                        if "p25" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["p25"], mode="lines", name="Bear 밴드 (25th)",
                                                                line=dict(color="rgba(248,113,113,0.85)", width=1.6, dash="dash")), row=1, col=1)
                        if "p10" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["p10"], mode="lines", name="p10 밴드",
                                                                line=dict(color="rgba(244,63,94,0.6)", width=1.2, dash="dash")), row=1, col=1)

                    elif model_key == "mean_sd" and r_active.band_series_sd is not None:
                        b_df = r_active.band_series_sd.loc[r_active.band_series_sd.index.isin(plot_idx)]
                        if "+2SD" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["+2SD"], mode="lines", name="+2SD 밴드",
                                                                line=dict(color="rgba(192,132,252,0.6)", width=1.2, dash="dash")), row=1, col=1)
                        if "+1SD" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["+1SD"], mode="lines", name="Bull 밴드 (+1SD)",
                                                                line=dict(color="rgba(52,211,153,0.85)", width=1.6, dash="dash")), row=1, col=1)
                        if "Mean" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["Mean"], mode="lines", name="Base 밴드 (Mean)",
                                                                line=dict(color="rgba(251,191,36,0.9)", width=2.0, dash="dash")), row=1, col=1)
                        if "-1SD" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["-1SD"], mode="lines", name="Bear 밴드 (-1SD)",
                                                                line=dict(color="rgba(248,113,113,0.85)", width=1.6, dash="dash")), row=1, col=1)
                        if "-2SD" in b_df:
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df["-2SD"], mode="lines", name="-2SD 밴드",
                                                                line=dict(color="rgba(244,63,94,0.6)", width=1.2, dash="dash")), row=1, col=1)

                    elif model_key == "fixed" and r_active.band_series_fixed is not None:
                        b_df = r_active.band_series_fixed.loc[r_active.band_series_fixed.index.isin(plot_idx)]
                        f_colors = ["#f87171", "#fbbf24", "#34d399", "#c084fc", "#38bdf8"]
                        for i, c_name in enumerate(b_df.columns):
                            c_col = f_colors[i % len(f_colors)]
                            fig_integrated.add_trace(go.Scatter(x=b_df.index, y=b_df[c_name], mode="lines", name=f"{c_name} 밴드",
                                                                line=dict(color=c_col, width=1.5, dash="dash")), row=1, col=1)

                    latest_date = p_plot.index[-1]
                    fig_integrated.add_trace(
                        go.Scatter(
                            x=[latest_date], y=[r_active.current_price],
                            mode="markers+text",
                            marker=dict(size=9, color="#ffffff", line=dict(color="#38bdf8", width=2)),
                            text=[f" 현재가 ₩{r_active.current_price:,.0f}"],
                            textposition="middle right",
                            name="현재가",
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                    fig_integrated.add_hline(y=r_active.target_bull, line_dash="dot", line_color="#34d399", line_width=1,
                                            annotation_text=f"Bull ₩{r_active.target_bull:,.0f} ({r_active.upside_bull:+.1f}%)",
                                            annotation_position="top left", annotation_font=dict(size=9, color="#34d399"), row=1, col=1)
                    fig_integrated.add_hline(y=r_active.target_base, line_dash="dot", line_color="#fbbf24", line_width=1,
                                            annotation_text=f"Base ₩{r_active.target_base:,.0f} ({r_active.upside_base:+.1f}%)",
                                            annotation_position="top left", annotation_font=dict(size=9, color="#fbbf24"), row=1, col=1)
                    fig_integrated.add_hline(y=r_active.target_bear, line_dash="dot", line_color="#f87171", line_width=1,
                                            annotation_text=f"Bear ₩{r_active.target_bear:,.0f} ({r_active.upside_bear:+.1f}%)",
                                            annotation_position="top left", annotation_font=dict(size=9, color="#f87171"), row=1, col=1)

                    # Subplot 2 (Lower): 12M Fwd EPS 추이 & P/E 멀티플
                    fig_integrated.add_trace(
                        go.Scatter(
                            x=e_plot.index, y=e_plot.values,
                            mode="lines",
                            name="12M Fwd EPS",
                            line=dict(color="#60a5fa", width=2.0),
                            fill="tozeroy",
                            fillcolor="rgba(96,165,250,0.08)"
                        ),
                        row=2, col=1, secondary_y=False
                    )
                    ma12_e = e_plot.rolling(12, min_periods=3).mean()
                    fig_integrated.add_trace(
                        go.Scatter(
                            x=ma12_e.index, y=ma12_e.values,
                            mode="lines",
                            name="EPS MA12",
                            line=dict(color="#93c5fd", width=1.5, dash="dot")
                        ),
                        row=2, col=1, secondary_y=False
                    )

                    fig_integrated.add_trace(
                        go.Scatter(
                            x=pe_plot.index, y=pe_plot.values,
                            mode="lines",
                            name="12M Fwd P/E",
                            line=dict(color="#c084fc", width=2.0)
                        ),
                        row=2, col=1, secondary_y=True
                    )

                    if r_active.pe_median and r_active.pe_median > 0:
                        fig_integrated.add_hline(
                            y=r_active.pe_median, line_dash="dash", line_color="rgba(251,191,36,0.6)", line_width=1,
                            annotation_text=f"Med {r_active.pe_median:.1f}x", annotation_position="bottom right",
                            annotation_font=dict(size=9, color="#fbbf24"), row=2, col=1, secondary_y=True
                        )

                    fig_integrated.update_layout(
                        **CHART_LAYOUT,
                        height=620,
                        margin=dict(l=10, r=20, t=30, b=30),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
                    )
                    fig_integrated.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True, linecolor="rgba(255,255,255,0.1)")
                    fig_integrated.update_yaxes(title_text="주가 (원)", tickformat=",.0f", showgrid=True, gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
                    fig_integrated.update_yaxes(title_text="EPS (원)", tickformat=",.0f", showgrid=True, gridcolor="rgba(255,255,255,0.04)", row=2, col=1, secondary_y=False)
                    fig_integrated.update_yaxes(title_text="P/E 배수", tickformat=".1f", ticksuffix="x", showgrid=False, row=2, col=1, secondary_y=True)

                    st.plotly_chart(fig_integrated, use_container_width=True)

                # 종목별 리서치 데이터 내보내기
                import io as _io
                d1, d2, d3 = st.columns(3)
                with d1:
                    stock_summary_df = pd.DataFrame({
                        "항목": [
                            "종목명", "종목코드", "섹터", "지수", "전략 신호", "현재가",
                            "12M Fwd EPS", "1W EPS 변동률", "1M EPS 변동률", "3M EPS 변동률",
                            "Fwd P/E", "P/E 역사적 백분위", "섹터 P/E 중앙값", "섹터 P/E 백분위",
                            f"Bear 목표가 ({model_lbl})", f"Base 목표가 ({model_lbl})", f"Bull 목표가 ({model_lbl})",
                            "Bear 업사이드%", "Base 업사이드%", "Bull 업사이드%",
                        ],
                        "값": [
                            r_active.name, r_active.ticker, r_active.sector, mkt_str, strat_sig, f"{r_active.current_price:,.0f}",
                            f"{r_active.current_fwd_eps:,.0f}", f"{r_active.eps_rev_1w:+.1f}%" if r_active.eps_rev_1w is not None else "N/A",
                            f"{r_active.eps_rev_1m:+.1f}%" if r_active.eps_rev_1m is not None else "N/A",
                            f"{r_active.eps_rev_3m:+.1f}%" if r_active.eps_rev_3m is not None else "N/A",
                            f"{r_active.current_fwd_pe:.2f}x", f"{r_active.pe_percentile:.1f}%",
                            f"{r_active.sector_median_pe:.2f}x" if r_active.sector_median_pe else "N/A",
                            f"{r_active.sector_pe_percentile:.1f}%" if r_active.sector_pe_percentile is not None else "N/A",
                            f"{r_active.target_bear:,.0f}", f"{r_active.target_base:,.0f}", f"{r_active.target_bull:,.0f}",
                            f"{r_active.upside_bear:+.1f}%", f"{r_active.upside_base:+.1f}%", f"{r_active.upside_bull:+.1f}%",
                        ]
                    })
                    buf_stock = _io.BytesIO()
                    stock_summary_df.to_excel(buf_stock, index=False)
                    st.download_button(
                        f"📥 {r_active.name} 분석 요약 Excel",
                        data=buf_stock.getvalue(),
                        file_name=f"{r_active.ticker}_{r_active.name}_valuation_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                with d2:
                    if 'p_plot' in locals() and 'e_plot' in locals() and 'pe_plot' in locals():
                        ts_export = pd.concat([
                            p_plot.rename("Price"),
                            e_plot.rename("Fwd_EPS"),
                            pe_plot.rename("Fwd_PE"),
                        ], axis=1)
                        ts_export.index.name = "Date"
                        csv_ts = ts_export.reset_index().to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 시계열 데이터 CSV",
                            data=csv_ts,
                            file_name=f"{r_active.ticker}_{r_active.name}_timeseries.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                with d3:
                    buf_all = _io.BytesIO()
                    df_screener_all.to_excel(buf_all, index=False)
                    st.download_button(
                        "📥 스크리닝 전체 결과 Excel",
                        data=buf_all.getvalue(),
                        file_name=f"screener_universe_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

    # ══════════════════════════════════════════
    # TAB 3: 밸류에이션 버블 차트
    # ══════════════════════════════════════════
    with tab3:
        if not filtered_results:
            st.warning("표시할 데이터가 없습니다.")
        else:
            b_c1, b_c2 = st.columns([3, 1])
            with b_c2:
                show_lbl = st.checkbox("종목명 상시 표시", value=False, key="bubble_show_label")
                min_eps_filter = st.number_input("최소 Fwd EPS 필터", value=0, step=500, key="bubble_min_eps")

            bubble_data = [r for r in filtered_results if r.current_fwd_eps >= min_eps_filter]

            SIG_COLOR_MAP = {
                "✨골든크로스": "#fbbf24",
                "🚀어닝모멘텀": "#38bdf8",
                "💎가치주": "#c084fc",
                "⚠️밸류트랩": "#f87171",
                "Neutral": "#9ca3af",
                "🔻고P/E하향": "#f43f5e",
            }

            fig_bubble = go.Figure()
            for sig_key, sig_color in SIG_COLOR_MAP.items():
                grp = [r for r in bubble_data if get_strategy_signal(r) == sig_key]
                if not grp:
                    continue
                fig_bubble.add_trace(go.Scatter(
                    x=[r.pe_percentile for r in grp],
                    y=[r.upside_base for r in grp],
                    mode="markers+text" if show_lbl else "markers",
                    name=sig_key,
                    text=[r.name for r in grp],
                    textposition="top center",
                    textfont=dict(size=9, color=sig_color),
                    marker=dict(
                        size=[max(8, min(28, r.current_fwd_eps / 800)) for r in grp],
                        color=sig_color,
                        opacity=0.75,
                        line=dict(width=1, color="rgba(255,255,255,0.2)"),
                    ),
                    customdata=[[
                        r.name, r.ticker, r.sector,
                        f"{r.current_price:,.0f}", f"{r.current_fwd_pe:.1f}",
                        f"{r.current_fwd_eps:,.0f}", f"{r.target_base:,.0f}",
                        f"{r.upside_base:+.1f}%", f"{r.eps_rev_1m:+.1f}%" if r.eps_rev_1m is not None else "N/A"
                    ] for r in grp],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]}) · %{customdata[2]}<br>"
                        "현재가: ₩%{customdata[3]}<br>"
                        "Fwd P/E: %{customdata[4]}x (위치: %{x:.0f}%)<br>"
                        "Fwd EPS: ₩%{customdata[5]} (1M 리비전: %{customdata[8]})<br>"
                        "─────────────<br>"
                        "Base 목표가: ₩%{customdata[6]}<br>"
                        "<b>Base 업사이드: %{customdata[7]}</b><extra></extra>"
                    ),
                ))

            fig_bubble.add_vline(x=50, line_color="rgba(255,255,255,0.15)", line_dash="dash",
                                 annotation_text="P/E 중앙값", annotation_font_color="#6b7280")
            fig_bubble.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dash",
                                 annotation_text="현재가 = Base 목표가", annotation_font_color="#6b7280")

            fig_bubble.update_layout(
                **CHART_LAYOUT,
                height=540,
                margin=dict(l=10, r=20, t=40, b=10),
                title=dict(
                    text="P/E 역사적 위치 vs Base 업사이드 (버블 크기 = 12M Fwd EPS)",
                    x=0, font=dict(size=13, color="#c4c4e0")
                ),
                xaxis=dict(**AX, title="P/E 역사적 위치 (%)", range=[-2, 102]),
                yaxis=dict(**AX, title="Base 업사이드 (%)"),
                legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            )

            with b_c1:
                st.plotly_chart(fig_bubble, use_container_width=True)


if __name__ == "__main__":
    main()
