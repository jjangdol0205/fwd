"""
app.py — 12M Fwd P/E 밴드 스크리닝 대시보드 v2
실행: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.data_loader import load_excel
from core.parse_dataguide_output import load_dataguide_eps
from core.calculator import run_screener, PEBandResult
from core.fetch_realtime_price import get_current_prices_batch

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fwd P/E 스크리너",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ──────────────────────────────────────────────
# PWA (앱 설치) 지원 인젝터
# ──────────────────────────────────────────────
import streamlit.components.v1 as components

def inject_pwa():
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
            "theme_color": "#6366f1",
            "icons": [{
                "src": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
                "sizes": "512x512",
                "type": "image/png"
            }, {
                "src": "https://cdn-icons-png.flaticon.com/512/2933/2933116.png",
                "sizes": "192x192",
                "type": "image/png"
            }]
        };
        const blob = new Blob([JSON.stringify(manifest)], {type: 'application/json'});
        const manifestURL = URL.createObjectURL(blob);
        
        const link = parentDoc.createElement('link');
        link.rel = 'manifest';
        link.id = 'pwa-manifest';
        link.href = manifestURL;
        parentDoc.head.appendChild(link);
        
        const meta1 = parentDoc.createElement('meta');
        meta1.name = 'apple-mobile-web-app-capable';
        meta1.content = 'yes';
        parentDoc.head.appendChild(meta1);
        
        const meta2 = parentDoc.createElement('meta');
        meta2.name = 'apple-mobile-web-app-status-bar-style';
        meta2.content = 'black-translucent';
        parentDoc.head.appendChild(meta2);
        
        const linkIcon = parentDoc.createElement('link');
        linkIcon.rel = 'apple-touch-icon';
        linkIcon.href = 'https://cdn-icons-png.flaticon.com/512/2933/2933116.png';
        parentDoc.head.appendChild(linkIcon);
    }
    </script>
    """, height=0, width=0)

inject_pwa()


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a14; color: #e2e2f0; }
section[data-testid="stSidebar"] {
    background: #0d0d1f !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * { color: #c4c4e0 !important; }

/* ── 헤더 ── */
.dg-header { display: flex; align-items: flex-start; gap: 16px; padding: 8px 0 20px 0; flex-wrap: wrap; }
.dg-logo {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.dg-sub { font-size: 0.85rem; color: #6366f1; font-weight: 500; margin-top: 2px; }
.dg-date { font-size: 0.78rem; color: #4b5563; margin-left: auto; }

/* ── KPI 카드 ── */
.kpi-row {
    display: flex; gap: 10px; margin: 16px 0;
    flex-wrap: wrap;
}
.kpi-card {
    flex: 1 1 160px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 14px 16px; transition: border-color .2s;
    min-width: 0;
}
.kpi-card:hover { border-color: rgba(129,140,248,0.4); }
.kpi-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 1.7rem; font-weight: 700; margin: 4px 0 2px; }
.kpi-sub   { font-size: 0.72rem; color: #6b7280; }
.kpi-green { color: #34d399; } .kpi-red { color: #f87171; }
.kpi-blue  { color: #60a5fa; } .kpi-purple { color: #a78bfa; } .kpi-yellow { color: #fbbf24; }

/* ── 신호 배지 ── */
.sig { display: inline-block; border-radius: 6px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }
.sig-sb { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.sig-b  { background: rgba(96,165,250,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.sig-h  { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.sig-s  { background: rgba(251,146,60,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
.sig-ss { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04) !important; border-radius: 8px !important;
    color: #9ca3af !important; border: 1px solid rgba(255,255,255,0.07) !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #818cf8 !important; border-color: rgba(99,102,241,0.4) !important;
}

/* ── 버튼 ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #047857, #059669) !important;
    box-shadow: 0 4px 12px rgba(16,185,129,0.3) !important;
}

/* ── 상세 카드 ── */
.detail-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 16px;
    height: 100%;
}
.detail-ticker { font-size: 0.75rem; color: #6366f1; font-weight: 600; }
.detail-name   { font-size: 1.2rem; font-weight: 700; margin: 2px 0 10px; line-height: 1.3; }
.price-tag { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }

/* ── 목표가 그리드 ── */
.target-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin: 10px 0; }
.target-box  { border-radius: 10px; padding: 10px; text-align: center; }
.target-bear { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.2); }
.target-base { background: rgba(251,191,36,0.08);  border: 1px solid rgba(251,191,36,0.2); }
.target-bull { background: rgba(52,211,153,0.08);  border: 1px solid rgba(52,211,153,0.2); }
.t-label { font-size: 0.65rem; color: #9ca3af; text-transform: uppercase; letter-spacing: .06em; }
.t-price { font-size: 0.95rem; font-weight: 700; margin: 3px 0 2px; }
.t-upside{ font-size: 0.75rem; font-weight: 600; }
.t-bear-c { color: #f87171; } .t-base-c { color: #fbbf24; } .t-bull-c { color: #34d399; }
.pe-bar { height: 6px; border-radius: 3px; background: #1f2937; margin-top: 4px; overflow: hidden; }
.pe-fill { height: 100%; border-radius: 3px; }

/* ════════════════════════════════════
   📱 모바일 반응형
   ════════════════════════════════════ */
@media (max-width: 768px) {
    .dg-logo { font-size: 1.6rem; }
    .dg-sub  { font-size: 0.75rem; }
    .dg-date { margin-left: 0; margin-top: 4px; }
    .kpi-card  { flex: 1 1 calc(50% - 5px); }
    .kpi-value { font-size: 1.35rem; }
    .detail-name { font-size: 1rem; }
    .price-tag   { font-size: 1.2rem; }
    .target-box  { padding: 8px 4px; }
    .t-price     { font-size: 0.8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.72rem !important; padding: 5px 8px !important; }
}
@media (max-width: 480px) {
    .kpi-card  { flex: 1 1 100%; }
    .kpi-value { font-size: 1.25rem; }
    .target-grid { grid-template-columns: 1fr; gap: 6px; }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def signal_badge(pct):
    if pct < 20:   return '<span class="sig sig-sb">Strong Buy</span>'
    if pct < 40:   return '<span class="sig sig-b">Buy</span>'
    if pct < 60:   return '<span class="sig sig-h">Hold</span>'
    if pct < 80:   return '<span class="sig sig-s">Sell</span>'
    return             '<span class="sig sig-ss">Strong Sell</span>'

def signal_label(pct):
    if pct < 20:   return "🟢🟢 Strong Buy"
    if pct < 40:   return "🟢 Buy"
    if pct < 60:   return "🟡 Hold"
    if pct < 80:   return "🔴 Sell"
    return             "🔴🔴 Strong Sell"

def pe_bar_color(pct):
    if pct < 30: return "#34d399"
    if pct < 60: return "#fbbf24"
    return "#f87171"

# 공통 차트 레이아웃 (xaxis/yaxis 없음 - 개별 지정)
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#9ca3af", size=11),
    margin=dict(l=10, r=20, t=40, b=10),
)
# 공통 축 스타일
AX = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)")


# ──────────────────────────────────────────────
# 데이터 로딩 (캐시)
# ──────────────────────────────────────────────
def _get_file_mtimes():
    """파일 수정 시간 반환 — 캐시 무효화 키로 사용"""
    import os
    price_path = ROOT / "data" / "price.xlsx"
    eps_path   = ROOT / "data" / "fwd_eps.xlsx"
    mt_price = int(os.path.getmtime(price_path)) if price_path.exists() else 0
    mt_eps   = int(os.path.getmtime(eps_path))   if eps_path.exists()   else 0
    return mt_price, mt_eps

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(band_years, _mtimes=(0, 0)):
    """_mtimes 는 파일 수정 시간 — 파일 변경 시 캐시 자동 무효화"""
    price_path = ROOT / "data" / "price.xlsx"
    eps_path   = ROOT / "data" / "fwd_eps.xlsx"
    uni_path   = ROOT / "data" / "universe.csv"

    # ── 실제 데이터 파일이 있으면 로드 ──────────────
    if price_path.exists() and eps_path.exists():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            price_hist = load_excel(str(price_path))
            eps_df     = load_dataguide_eps(str(eps_path))

        names, markets = {}, {}
        if uni_path.exists():
            uni     = pd.read_csv(str(uni_path), dtype=str)
            names   = dict(zip(uni["ticker"], uni["name"]))
            markets = dict(zip(uni["ticker"], uni["market"]))

        common   = [t for t in eps_df.columns if t in price_hist.columns]
        price_df = price_hist[common]
        eps_df   = eps_df[common]
        results  = run_screener(price_df, eps_df, names, band_years=band_years)
        return price_df, eps_df, names, markets, results

    # ── 데이터 없음 → 샘플 데이터 자동 생성 (Cloud 배포용) ──
    return _make_sample_data(band_years)


def _make_sample_data(band_years: int):
    """Streamlit Cloud / 데이터 없는 환경용 샘플 데이터 생성"""
    SAMPLE = [
        ("005930","삼성전자","KOSPI200"), ("000660","SK하이닉스","KOSPI200"),
        ("035420","NAVER","KOSPI200"),    ("005380","현대차","KOSPI200"),
        ("051910","LG화학","KOSPI200"),   ("006400","삼성SDI","KOSPI200"),
        ("000270","기아","KOSPI200"),     ("012330","현대모비스","KOSPI200"),
        ("003550","LG","KOSPI200"),       ("034730","SK","KOSPI200"),
        ("035900","JYP Ent.","KOSDAQ150"),("041510","에스엠","KOSDAQ150"),
        ("263750","펄어비스","KOSDAQ150"),("293490","카카오게임즈","KOSDAQ150"),
        ("145020","휴젤","KOSDAQ150"),
    ]
    np.random.seed(42)
    dates_p = pd.date_range("2015-01-31", periods=12*band_years+12, freq="ME")
    dates_e = pd.date_range("2010-01-31", periods=12*15+12, freq="ME")

    price_data, eps_data = {}, {}
    names, markets = {}, {}
    for ticker, name, mkt in SAMPLE:
        base_p = np.random.uniform(30000, 200000)
        trend  = np.random.uniform(0.995, 1.012)
        noise  = np.random.normal(1, 0.06, len(dates_p))
        prices = base_p * np.cumprod(trend * noise)
        price_data[ticker] = pd.Series(prices, index=dates_p)

        base_e   = base_p / np.random.uniform(10, 25)
        eps_vals = base_e * (1 + np.random.normal(0.005, 0.03, len(dates_e)))
        eps_data[ticker] = pd.Series(np.maximum(eps_vals, 100), index=dates_e)
        names[ticker]    = name
        markets[ticker]  = mkt

    price_df = pd.DataFrame(price_data)
    eps_df   = pd.DataFrame(eps_data)
    results  = run_screener(price_df, eps_df, names, band_years=band_years)
    return price_df, eps_df, names, markets, results



@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime(tickers):
    prices = get_current_prices_batch(list(tickers))
    return prices.to_dict()


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    st.markdown("---")

    band_years = st.select_slider(
        "📅 역사적 밴드 기간",
        options=[1, 2, 3, 5, 7, 10, 15],
        value=10,
    )

    st.markdown("---")
    st.markdown("**🔍 필터**")

    market_filter = st.multiselect(
        "지수",
        ["KOSPI200", "KOSDAQ150"],
        default=["KOSPI200", "KOSDAQ150"],
    )
    signal_filter = st.multiselect(
        "투자 신호",
        ["🟢🟢 Strong Buy", "🟢 Buy", "🟡 Hold", "🔴 Sell", "🔴🔴 Strong Sell"],
        default=["🟢🟢 Strong Buy", "🟢 Buy", "🟡 Hold"],
    )
    min_upside   = st.slider("Base 업사이드 최소 (%)", -50, 100, -30)
    pe_pct_range = st.slider("P/E 위치 범위 (%)", 0, 100, (0, 80))

    st.markdown("---")
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#374151'>"
        "DataGuide × pykrx<br>Fwd P/E Screener v2.0</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div class="dg-header">
  <div>
    <div class="dg-logo">Fwd P/E Screener</div>
    <div class="dg-sub">12M Forward EPS 기반 역사적 P/E 밴드 분석 · KOSPI200 + KOSDAQ150</div>
  </div>
  <div class="dg-date">기준: {now_str}</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
with st.spinner("데이터 로드 중..."):
    price_df, eps_df, names, markets, all_results = load_all_data(band_years, _mtimes=_get_file_mtimes())

if not all_results:
    st.error("data/price.xlsx 또는 data/fwd_eps.xlsx 파일이 없습니다.")
    st.stop()

IS_SAMPLE = not (ROOT / "data" / "fwd_eps.xlsx").exists()
if IS_SAMPLE:
    st.info("⚠️ 실제 데이터 파일이 없어 **샘플 데이터 (데모 모드)**로 실행 중입니다. DataGuide에서 fwd_eps.xlsx와 price.xlsx를 data/ 폴더에 넣으면 실제 데이터가 로드됩니다.")

eps_months = eps_df.shape[0]
eps_start  = eps_df.index.min().strftime("%Y-%m")
eps_end    = eps_df.index.max().strftime("%Y-%m")

# 실시간 현재가 조회
result_tickers = [r.ticker for r in all_results]
with st.spinner("현재가 업데이트 중..."):
    rt_prices = fetch_realtime(tuple(result_tickers))

# 현재가로 P/E 재계산
updated_results = []
for r in all_results:
    rt_p = rt_prices.get(r.ticker)
    if rt_p and rt_p > 0 and r.current_fwd_eps > 0:
        rt_pe  = rt_p / r.current_fwd_eps
        hist_pe = r.hist_pe_series.dropna()
        cutoff  = hist_pe.index.max() - pd.DateOffset(years=band_years)
        hist_pe = hist_pe[hist_pe.index >= cutoff]
        rt_pct  = float((hist_pe < rt_pe).mean() * 100) if len(hist_pe) > 0 else r.pe_percentile

        r2 = PEBandResult(
            ticker=r.ticker, name=r.name,
            current_price=rt_p, current_fwd_eps=r.current_fwd_eps,
            current_fwd_pe=rt_pe, pe_percentile=rt_pct,
            pe_min=r.pe_min, pe_p25=r.pe_p25, pe_median=r.pe_median,
            pe_p75=r.pe_p75, pe_max=r.pe_max, pe_mean=r.pe_mean,
            target_bear=r.current_fwd_eps * r.pe_p25,
            target_base=r.current_fwd_eps * r.pe_median,
            target_bull=r.current_fwd_eps * r.pe_p75,
            upside_bear=(r.current_fwd_eps * r.pe_p25 / rt_p - 1) * 100,
            upside_base=(r.current_fwd_eps * r.pe_median / rt_p - 1) * 100,
            upside_bull=(r.current_fwd_eps * r.pe_p75 / rt_p - 1) * 100,
            hist_pe_series=r.hist_pe_series,
            hist_price_series=r.hist_price_series,
            hist_eps_series=r.hist_eps_series,
        )
        updated_results.append(r2)
    else:
        updated_results.append(r)

# 필터 적용
filtered = []
for r in updated_results:
    mkt = markets.get(r.ticker, "")
    if market_filter and mkt not in market_filter:
        continue
    sig = signal_label(r.pe_percentile)
    if signal_filter and sig not in signal_filter:
        continue
    if r.upside_base < min_upside:
        continue
    if not (pe_pct_range[0] <= r.pe_percentile <= pe_pct_range[1]):
        continue
    filtered.append(r)

filtered.sort(key=lambda r: r.upside_base, reverse=True)


# ──────────────────────────────────────────────
# KPI 요약 카드
# ──────────────────────────────────────────────
total      = len(filtered)
strong_buy = sum(1 for r in filtered if r.pe_percentile < 20)
buy_cnt    = sum(1 for r in filtered if 20 <= r.pe_percentile < 40)
hold_cnt   = sum(1 for r in filtered if 40 <= r.pe_percentile < 60)
sell_cnt   = sum(1 for r in filtered if r.pe_percentile >= 60)
avg_up     = np.mean([r.upside_base for r in filtered]) if filtered else 0
top_name   = filtered[0].name if filtered else "-"
top_up     = filtered[0].upside_base if filtered else 0
eps_note   = f"EPS {eps_start}~{eps_end} ({eps_months}M)"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">분석 종목</div>
    <div class="kpi-value kpi-blue">{total}<span style="font-size:1rem">개</span></div>
    <div class="kpi-sub">{eps_note}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">🟢🟢 Strong Buy</div>
    <div class="kpi-value kpi-green">{strong_buy}<span style="font-size:1rem">개</span></div>
    <div class="kpi-sub">P/E &lt; 20th percentile</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">🟢 Buy</div>
    <div class="kpi-value kpi-blue">{buy_cnt}<span style="font-size:1rem">개</span></div>
    <div class="kpi-sub">P/E 20~40th percentile</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">평균 Base 업사이드</div>
    <div class="kpi-value {'kpi-green' if avg_up>=0 else 'kpi-red'}">{avg_up:+.1f}<span style="font-size:1rem">%</span></div>
    <div class="kpi-sub">Hold/Sell {hold_cnt+sell_cnt}개</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Top Pick (Base)</div>
    <div class="kpi-value kpi-purple" style="font-size:1.2rem">{top_name}</div>
    <div class="kpi-sub {'kpi-green' if top_up>=0 else 'kpi-red'}">{top_up:+.1f}% 업사이드</div>
  </div>
</div>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────
# 키워드 검색
# ──────────────────────────────────────────────
if "sel_ticker" not in st.session_state:
    st.session_state.sel_ticker = None

st.markdown(
    '<div style="height:4px;background:linear-gradient(90deg,#4f46e5,#7c3aed,#ec4899);'
    'border-radius:4px;margin:16px 0 14px;"></div>',
    unsafe_allow_html=True,
)

sc1, sc2 = st.columns([4, 1])
with sc1:
    search_q = st.text_input(
        "클 업 검색",
        placeholder="⛌  종목명 또는 코드 입력  (예: 삼성전자, 005930)",
        label_visibility="collapsed",
        key="global_search",
    )
with sc2:
    if st.button("❌ 검색 초기화", use_container_width=True):
        st.session_state.sel_ticker = None
        st.rerun()

# 검색 진행
if search_q and search_q.strip():
    q       = search_q.strip().lower()
    # 전체 updated_results 에서 검색 (필터 무시)
    matches = [
        r for r in updated_results
        if q in r.name.lower() or q in r.ticker.lower()
    ]
    matches.sort(key=lambda r: r.upside_base, reverse=True)

    if not matches:
        st.info("🔍 검색 결과가 없습니다.")
    else:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#6366f1;font-weight:600;margin-bottom:8px">'
            f'🔍 &nbsp;\'{search_q}\' 검색 결과  {len(matches)}개 종목</div>',
            unsafe_allow_html=True,
        )

        # 융합 소형 카드 (4열 최대 8개)
        display_matches = matches[:8]
        cols_per_row    = 4
        for row_start in range(0, len(display_matches), cols_per_row):
            row_items = display_matches[row_start : row_start + cols_per_row]
            cols      = st.columns(cols_per_row)
            for col, m in zip(cols, row_items):
                mkt_m    = markets.get(m.ticker, "")
                col_now  = pe_bar_color(m.pe_percentile)
                up_col   = "#34d399" if m.upside_base >= 0 else "#f87171"
                rt_pm    = rt_prices.get(m.ticker, m.current_price)
                with col:
                    is_selected = st.session_state.sel_ticker == m.ticker
                    border_col  = "#6366f1" if is_selected else "rgba(255,255,255,0.08)"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.04);border:1.5px solid {border_col};
                                border-radius:12px;padding:14px 16px;margin-bottom:6px;">
                      <div style="font-size:0.68rem;color:#6b7280;">{m.ticker} &middot; {mkt_m}</div>
                      <div style="font-size:1rem;font-weight:700;margin:3px 0 6px;">{m.name}</div>
                      <div style="font-size:0.78rem;color:#9ca3af;">
                        현재가&nbsp; <b style='color:#f1f5f9'>&#8361;{rt_pm:,.0f}</b>
                      </div>
                      <div style="font-size:0.78rem;margin-top:3px;color:#9ca3af;">
                        Fwd P/E&nbsp;<b style='color:#818cf8'>{m.current_fwd_pe:.1f}x</b>
                        &nbsp;&middot;&nbsp;위치 <b style='color:{col_now}'>{m.pe_percentile:.0f}%</b>
                      </div>
                      <div style="font-size:0.82rem;font-weight:700;color:{up_col};margin-top:6px;">
                        Base {m.upside_base:+.1f}%
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("📈 상세 분석", key=f"srch_{m.ticker}",
                                 use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.sel_ticker = m.ticker
                        st.rerun()

        # 선택된 종목 상세 분석 인라인 표시
        if st.session_state.sel_ticker:
            sel_r = next((x for x in updated_results
                          if x.ticker == st.session_state.sel_ticker), None)
            if sel_r:
                _rt  = rt_prices.get(sel_r.ticker, sel_r.current_price)
                _mkt = markets.get(sel_r.ticker, "")
                _col = pe_bar_color(sel_r.pe_percentile)

                st.markdown("<hr style='border-color:rgba(99,102,241,0.3);margin:16px 0;'>",
                            unsafe_allow_html=True)
                st.markdown(f"#### 📈 {sel_r.name} ({sel_r.ticker}) · {_mkt} — 상세 분석")

                # 요약 카드
                _h1, _h2, _h3, _h4 = st.columns(4)
                with _h1:
                    st.markdown(f"""
                    <div class="detail-card">
                      <div class="detail-ticker">{sel_r.ticker} &middot; {_mkt}</div>
                      <div class="detail-name">{sel_r.name}</div>
                      <div class="price-tag">₩{_rt:,.0f}</div>
                      <div style="font-size:0.75rem;color:#6b7280;margin-top:4px">실시간 현재가</div>
                      {signal_badge(sel_r.pe_percentile)}
                    </div>""", unsafe_allow_html=True)
                with _h2:
                    st.markdown(f"""
                    <div class="detail-card">
                      <div class="kpi-label">현재 Fwd P/E</div>
                      <div style="font-size:2rem;font-weight:700;color:#818cf8">{sel_r.current_fwd_pe:.1f}x</div>
                      <div class="kpi-label" style="margin-top:8px">역사적 위치</div>
                      <div style="font-size:1.4rem;font-weight:700;color:{_col}">{sel_r.pe_percentile:.0f}%</div>
                      <div class="pe-bar"><div class="pe-fill" style="width:{sel_r.pe_percentile:.0f}%;background:{_col}"></div></div>
                    </div>""", unsafe_allow_html=True)
                with _h3:
                    st.markdown(f"""
                    <div class="detail-card">
                      <div class="kpi-label">12M Fwd EPS</div>
                      <div style="font-size:1.6rem;font-weight:700;color:#60a5fa">₩{sel_r.current_fwd_eps:,.0f}</div>
                      <div style="margin-top:10px">
                        <div class="kpi-label">P/E 밴드 ({band_years}년)</div>
                        <div style="font-size:0.82rem;color:#9ca3af;margin-top:4px">
                          25th {sel_r.pe_p25:.1f}x &middot; Med {sel_r.pe_median:.1f}x &middot; 75th {sel_r.pe_p75:.1f}x
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with _h4:
                    st.markdown(f"""
                    <div class="detail-card">
                      <div class="kpi-label">목표가 요약</div>
                      <div class="target-grid" style="margin-top:8px">
                        <div class="target-box target-bear">
                          <div class="t-label">Bear</div>
                          <div class="t-price t-bear-c">₩{sel_r.target_bear:,.0f}</div>
                          <div class="t-upside t-bear-c">{sel_r.upside_bear:+.1f}%</div>
                        </div>
                        <div class="target-box target-base">
                          <div class="t-label">Base</div>
                          <div class="t-price t-base-c">₩{sel_r.target_base:,.0f}</div>
                          <div class="t-upside t-base-c">{sel_r.upside_base:+.1f}%</div>
                        </div>
                        <div class="target-box target-bull">
                          <div class="t-label">Bull</div>
                          <div class="t-price t-bull-c">₩{sel_r.target_bull:,.0f}</div>
                          <div class="t-upside t-bull-c">{sel_r.upside_bull:+.1f}%</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # 차트
                _c1, _c2 = st.columns([3, 2])
                with _c1:
                    _ph = sel_r.hist_price_series.dropna()
                    _p2 = _ph[_ph.index >= _ph.index.max() - pd.DateOffset(years=2)]
                    _fp = go.Figure()
                    _fp.add_trace(go.Scatter(x=_p2.index, y=_p2.values, mode="lines",
                                            line=dict(color="#60a5fa", width=2.5),
                                            fill="tozeroy", fillcolor="rgba(96,165,250,0.07)", name="주가"))
                    _fp.add_trace(go.Scatter(x=[_p2.index[-1]], y=[_rt], mode="markers+text",
                                            marker=dict(size=12, color="#f1f5f9", symbol="circle",
                                                        line=dict(width=2, color="#818cf8")),
                                            text=[f" 현재가 ₩{_rt:,.0f}"],
                                            textposition="middle right",
                                            textfont=dict(size=9, color="#f1f5f9"),
                                            showlegend=False))
                    for _v, _l, _c, _d in [
                        (sel_r.target_bull, f"🐂 Bull ₩{sel_r.target_bull:,.0f} ({sel_r.upside_bull:+.1f}%)", "#34d399", "dot"),
                        (sel_r.target_base, f"📍 Base ₩{sel_r.target_base:,.0f} ({sel_r.upside_base:+.1f}%)", "#fbbf24", "dash"),
                        (sel_r.target_bear, f"🐻 Bear ₩{sel_r.target_bear:,.0f} ({sel_r.upside_bear:+.1f}%)", "#f87171", "dot"),
                    ]:
                        _fp.add_hline(y=_v, line_dash=_d, line_color=_c, line_width=1.5,
                                      annotation_text=_l, annotation_position="top left",
                                      annotation_font_color=_c, annotation_font_size=10,
                                      annotation_bgcolor="rgba(0,0,0,0.5)")
                    _fp.add_hrect(y0=sel_r.target_bear, y1=sel_r.target_bull,
                                  fillcolor="rgba(251,191,36,0.06)", line_width=0)
                    _fp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)", font=dict(color="#9ca3af", size=11), height=280,
                                      title=dict(text=f"{sel_r.name} — 주가 추이(2년) & 목표가",
                                                 x=0, font=dict(size=12, color="#c4c4e0")),
                                      xaxis=dict(**AX), yaxis=dict(**AX, title="주가 (원)"),
                                      hovermode="x unified")
                    st.plotly_chart(_fp, use_container_width=True)

                with _c2:
                    _pct = sel_r.pe_percentile
                    _cn  = pe_bar_color(_pct)
                    _pe_full = sel_r.hist_pe_series.dropna()
                    _fig_g = go.Figure()
                    _fig_g.add_trace(go.Scatter(
                        x=list(range(101)), y=[0.5]*101, mode="markers",
                        marker=dict(size=20, symbol="square", color=list(range(101)),
                                    colorscale=[[0.0,"rgba(52,211,153,0.7)"],[0.35,"rgba(251,191,36,0.6)"],
                                                [0.65,"rgba(251,191,36,0.6)"],[1.0,"rgba(248,113,113,0.7)"]],
                                    opacity=0.85),
                        hoverinfo="skip", showlegend=False,
                    ))
                    _fig_g.add_trace(go.Scatter(x=[_pct], y=[0.5], mode="markers",
                                               marker=dict(size=18, color="white", symbol="triangle-down",
                                                           line=dict(width=2.5, color=_cn)),
                                               hoverinfo="skip", showlegend=False))
                    for _xv, _txt, _cl in [
                        (0, f"Min\n{sel_r.pe_min:.0f}x", "#9ca3af"),
                        (25, f"25th\n{sel_r.pe_p25:.0f}x", "#f87171"),
                        (50, f"Med\n{sel_r.pe_median:.0f}x", "#fbbf24"),
                        (75, f"75th\n{sel_r.pe_p75:.0f}x", "#34d399"),
                        (100, f"Max\n{sel_r.pe_max:.0f}x", "#9ca3af"),
                    ]:
                        _fig_g.add_shape(type="line", x0=_xv, x1=_xv, y0=0.25, y1=0.39,
                                         line=dict(color=_cl, width=1.2))
                        _fig_g.add_annotation(x=_xv, y=0.16, text=_txt,
                                              font=dict(size=9, color=_cl),
                                              showarrow=False, align="center")
                    _fig_g.add_annotation(
                        x=_pct, y=0.78,
                        text=f"현재 Fwd P/E<br><b>{sel_r.current_fwd_pe:.1f}x</b><br>({_pct:.0f}% 위치)",
                        font=dict(size=11, color="white"),
                        showarrow=True, arrowhead=2, arrowcolor=_cn, ax=0, ay=22,
                        bgcolor="rgba(0,0,0,0.75)", bordercolor=_cn, borderwidth=1.5,
                        align="center")
                    _fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.02)",
                        font=dict(color="#9ca3af", size=11),
                        height=280,
                        title=dict(text=f"P/E 역사적 위치 — <b style='color:{_cn}'>{_pct:.0f}%</b>",
                                   x=0, font=dict(size=12, color="#c4c4e0")),
                        xaxis=dict(range=[-3,103], showticklabels=False, showgrid=False, zeroline=False,
                                   title="← 저평가 (싸다)                     고평가 (비싸다) →"),
                        yaxis=dict(range=[0,1], showticklabels=False, showgrid=False, zeroline=False),
                        margin=dict(l=10, r=20, t=50, b=55),
                    )
                    st.plotly_chart(_fig_g, use_container_width=True)

                st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:16px 0;'>",
                            unsafe_allow_html=True)

st.markdown(
    '<div style="height:4px;background:linear-gradient(90deg,#4f46e5,#7c3aed,#ec4899);'
    'border-radius:4px;margin:14px 0 16px;"></div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 메인 탭
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋  스크리닝 테이블", "🪷  버블 차트", "🔍  종목 상세"])



# ══════════════════════════════════════════════
# TAB 1: 스크리닝 테이블
# ══════════════════════════════════════════════
with tab1:
    if not filtered:
        st.warning("필터 조건에 맞는 종목이 없습니다.")
    else:
        rows = []
        for r in filtered:
            mkt = markets.get(r.ticker, "")
            rows.append({
                "신호":      signal_label(r.pe_percentile),
                "종목명":    r.name,
                "코드":      r.ticker,
                "지수":      mkt,
                "현재가":    f"{r.current_price:>10,.0f}",
                "Fwd EPS":   f"{r.current_fwd_eps:>9,.0f}",
                "Fwd P/E":   f"{r.current_fwd_pe:.1f}x",
                "P/E 위치":  f"{r.pe_percentile:.0f}%",
                "P/E 중앙값":f"{r.pe_median:.1f}x",
                "🐻Bear목표":f"{r.target_bear:>10,.0f}",
                "📍Base목표":f"{r.target_base:>10,.0f}",
                "🐂Bull목표":f"{r.target_bull:>10,.0f}",
                "Bear%":     f"{r.upside_bear:+.1f}%",
                "Base%":     f"{r.upside_base:+.1f}%",
                "Bull%":     f"{r.upside_bull:+.1f}%",
            })

        df_show = pd.DataFrame(rows)

        # ── 다운로드 버튼 (테이블 위에 배치) ──
        import io
        _dl1, _dl2, _ = st.columns([1, 1, 2])
        with _dl1:
            buf_xl = io.BytesIO()
            df_show.to_excel(buf_xl, index=False)
            st.download_button(
                "📥 Excel 다운로드",
                data=buf_xl.getvalue(),
                file_name=f"pe_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with _dl2:
            csv_data = df_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 CSV 다운로드",
                data=csv_data,
                file_name=f"pe_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            df_show, use_container_width=True,
            height=min(80 + len(rows) * 36, 560),
            hide_index=True,
        )


# ══════════════════════════════════════════════
# TAB 2: 버블 차트
# ══════════════════════════════════════════════
with tab2:
    if not filtered:
        st.warning("데이터 없음")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_b:
            show_label = st.checkbox("종목명 표시", value=False)
            min_eps    = st.number_input("최소 EPS 필터", value=0, step=500)

        display = [r for r in filtered if r.current_fwd_eps >= min_eps]

        SIG_COLORS = {
            "🟢🟢 Strong Buy": "#34d399",
            "🟢 Buy":          "#60a5fa",
            "🟡 Hold":         "#fbbf24",
            "🔴 Sell":         "#fb923c",
            "🔴🔴 Strong Sell":"#f87171",
        }

        fig = go.Figure()
        for sig_name, color in SIG_COLORS.items():
            grp = [r for r in display if signal_label(r.pe_percentile) == sig_name]
            if not grp:
                continue
            fig.add_trace(go.Scatter(
                x     = [r.pe_percentile for r in grp],
                y     = [r.upside_base for r in grp],
                mode  = "markers+text" if show_label else "markers",
                name  = sig_name,
                text  = [r.name for r in grp],
                textposition="top center",
                textfont=dict(size=9, color=color),
                marker=dict(
                    size   = [max(8, min(30, r.current_fwd_eps / 800)) for r in grp],
                    color  = color, opacity=0.75,
                    line   = dict(width=1, color="rgba(255,255,255,0.2)"),
                ),
                customdata=[[
                    r.name, r.ticker,
                    f"{r.current_price:,.0f}", f"{r.current_fwd_pe:.1f}",
                    f"{r.current_fwd_eps:,.0f}",
                    f"{r.target_bear:,.0f}", f"{r.target_base:,.0f}", f"{r.target_bull:,.0f}",
                    f"{r.upside_base:+.1f}%",
                ] for r in grp],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "현재가: ₩%{customdata[2]}<br>"
                    "Fwd P/E: %{customdata[3]}x<br>"
                    "Fwd EPS: %{customdata[4]}<br>"
                    "─────────────<br>"
                    "Bear: ₩%{customdata[5]}<br>"
                    "Base: ₩%{customdata[6]}<br>"
                    "Bull: ₩%{customdata[7]}<br>"
                    "<b>Base 업사이드: %{customdata[8]}</b><extra></extra>"
                ),
            ))

        fig.add_vline(x=50, line_color="rgba(255,255,255,0.15)", line_dash="dash",
                      annotation_text="P/E 중앙값", annotation_font_color="#6b7280")
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dash",
                      annotation_text="현재가=목표가", annotation_font_color="#6b7280")

        fig.update_layout(
            **CHART_LAYOUT, height=520,
            title=dict(text="P/E 위치 vs Base 업사이드  (버블 크기 = Fwd EPS)", x=0, font=dict(size=13, color="#c4c4e0")),
            xaxis=dict(**AX, title="P/E 역사적 위치 (%)", range=[-2, 102]),
            yaxis=dict(**AX, title="Base 업사이드 (%)"),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
        )
        with col_a:
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: 종목 상세
# ══════════════════════════════════════════════
with tab3:
    if not filtered:
        st.warning("데이터 없음")
    else:
        opts = {f"{r.name}  |  {r.ticker}  |  {signal_label(r.pe_percentile)}": r for r in filtered}
        sel  = st.selectbox("종목 선택", list(opts.keys()), label_visibility="collapsed")
        r: PEBandResult = opts[sel]
        mkt  = markets.get(r.ticker, "")
        rt_p = rt_prices.get(r.ticker, r.current_price)

        # ── 상단 요약 카드 4개 ──────────────────
        h1, h2, h3, h4 = st.columns([2, 2, 2, 2])

        with h1:
            st.markdown(f"""
            <div class="detail-card">
              <div class="detail-ticker">{r.ticker} · {mkt}</div>
              <div class="detail-name">{r.name}</div>
              <div class="price-tag">₩{rt_p:,.0f}</div>
              <div style="font-size:0.75rem;color:#6b7280;margin-top:4px">실시간 현재가</div>
              {signal_badge(r.pe_percentile)}
            </div>
            """, unsafe_allow_html=True)

        with h2:
            st.markdown(f"""
            <div class="detail-card">
              <div class="kpi-label">현재 Fwd P/E</div>
              <div style="font-size:2rem;font-weight:700;color:#818cf8">{r.current_fwd_pe:.1f}x</div>
              <div class="kpi-label" style="margin-top:8px">역사적 위치</div>
              <div style="font-size:1.4rem;font-weight:700;color:{pe_bar_color(r.pe_percentile)}">{r.pe_percentile:.0f}%</div>
              <div class="pe-bar"><div class="pe-fill" style="width:{r.pe_percentile:.0f}%;background:{pe_bar_color(r.pe_percentile)}"></div></div>
            </div>
            """, unsafe_allow_html=True)

        with h3:
            st.markdown(f"""
            <div class="detail-card">
              <div class="kpi-label">12M Fwd EPS</div>
              <div style="font-size:1.6rem;font-weight:700;color:#60a5fa">₩{r.current_fwd_eps:,.0f}</div>
              <div style="margin-top:10px">
                <div class="kpi-label">P/E 밴드 ({band_years}년)</div>
                <div style="font-size:0.82rem;color:#9ca3af;margin-top:4px">
                  Min {r.pe_min:.1f}x · 25th {r.pe_p25:.1f}x<br>
                  Med {r.pe_median:.1f}x · 75th {r.pe_p75:.1f}x<br>
                  Max {r.pe_max:.1f}x
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with h4:
            st.markdown(f"""
            <div class="detail-card">
              <div class="kpi-label">목표가 요약</div>
              <div class="target-grid" style="margin-top:8px">
                <div class="target-box target-bear">
                  <div class="t-label">Bear</div>
                  <div class="t-price t-bear-c">₩{r.target_bear:,.0f}</div>
                  <div class="t-upside t-bear-c">{r.upside_bear:+.1f}%</div>
                </div>
                <div class="target-box target-base">
                  <div class="t-label">Base</div>
                  <div class="t-price t-base-c">₩{r.target_base:,.0f}</div>
                  <div class="t-upside t-base-c">{r.upside_base:+.1f}%</div>
                </div>
                <div class="target-box target-bull">
                  <div class="t-label">Bull</div>
                  <div class="t-price t-bull-c">₩{r.target_bull:,.0f}</div>
                  <div class="t-upside t-bull-c">{r.upside_bull:+.1f}%</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── 다운로드 ───────────────────────────────
        import io as _io
        _d1, _d2, _d3 = st.columns([1, 1, 1])
        with _d1:
            # 현재 종목 데이터
            _stock_rows = {
                "항목": ["종목명","코드","지수","현재가","Fwd EPS","Fwd P/E","P/E 위치",
                          "P/E Min","P/E 25th","P/E Median","P/E 75th","P/E Max",
                          "Bear목표가","Base목표가","Bull목표가",
                          "Bear업사이드%","Base업사이드%","Bull업사이드%"],
                "값": [r.name, r.ticker, mkt,
                        f"{rt_p:,.0f}", f"{r.current_fwd_eps:,.0f}",
                        f"{r.current_fwd_pe:.2f}x", f"{r.pe_percentile:.1f}%",
                        f"{r.pe_min:.2f}x", f"{r.pe_p25:.2f}x",
                        f"{r.pe_median:.2f}x", f"{r.pe_p75:.2f}x", f"{r.pe_max:.2f}x",
                        f"{r.target_bear:,.0f}", f"{r.target_base:,.0f}", f"{r.target_bull:,.0f}",
                        f"{r.upside_bear:+.1f}%", f"{r.upside_base:+.1f}%", f"{r.upside_bull:+.1f}%"],
            }
            _df_stock = pd.DataFrame(_stock_rows)
            _buf_s = _io.BytesIO()
            _df_stock.to_excel(_buf_s, index=False)
            st.download_button(
                f"📥 {r.name} Excel",
                data=_buf_s.getvalue(),
                file_name=f"{r.ticker}_{r.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with _d2:
            # EPS + 주가 시계열 CSV
            _eps_s = r.hist_eps_series.dropna().rename("Fwd_EPS")
            _prc_s = r.hist_price_series.dropna().rename("Price")
            _pe_s  = r.hist_pe_series.dropna().rename("Fwd_PE")
            _ts_df = pd.concat([_prc_s, _eps_s, _pe_s], axis=1)
            _ts_df.index.name = "Date"
            _csv_ts = _ts_df.reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 시계열 CSV",
                data=_csv_ts,
                file_name=f"{r.ticker}_{r.name}_timeseries.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with _d3:
            # 전체 스크리닝 결과
            _all_rows = []
            for _r2 in filtered:
                _all_rows.append({
                    "신호": signal_label(_r2.pe_percentile),
                    "종목명": _r2.name, "코드": _r2.ticker,
                    "지수": markets.get(_r2.ticker,""),
                    "현재가": f"{_r2.current_price:,.0f}",
                    "FwdEPS": f"{_r2.current_fwd_eps:,.0f}",
                    "FwdPE": f"{_r2.current_fwd_pe:.1f}x",
                    "PE위치%": f"{_r2.pe_percentile:.0f}%",
                    "Bear목표": f"{_r2.target_bear:,.0f}",
                    "Base목표": f"{_r2.target_base:,.0f}",
                    "Bull목표": f"{_r2.target_bull:,.0f}",
                    "Base%": f"{_r2.upside_base:+.1f}%",
                })
            _buf_all = _io.BytesIO()
            pd.DataFrame(_all_rows).to_excel(_buf_all, index=False)
            st.download_button(
                "📥 전체 결과 Excel",
                data=_buf_all.getvalue(),
                file_name=f"screener_all_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 행 1: 주가 추이(최근 2년) + 목표가 비교 ──

        c1, c2 = st.columns([3, 2])

        with c1:
            price_hist = r.hist_price_series.dropna()
            cutoff_2y  = price_hist.index.max() - pd.DateOffset(years=2)
            price_2y   = price_hist[price_hist.index >= cutoff_2y]

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=price_2y.index, y=price_2y.values,
                mode="lines", name="월별 주가",
                line=dict(color="#60a5fa", width=2.5),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
            ))
            fig_p.add_trace(go.Scatter(
                x=[price_2y.index[-1]], y=[rt_p],
                mode="markers+text",
                marker=dict(size=12, color="#f1f5f9", symbol="circle",
                            line=dict(width=2, color="#818cf8")),
                text=[f" 현재가 ₩{rt_p:,.0f}"],
                textposition="middle right",
                textfont=dict(size=9, color="#f1f5f9"),
                name="현재가", showlegend=False,
            ))
            for val, lbl, col, dash in [
                (r.target_bull, f"🐂 Bull ₩{r.target_bull:,.0f} ({r.upside_bull:+.1f}%)", "#34d399", "dot"),
                (r.target_base, f"📍 Base ₩{r.target_base:,.0f} ({r.upside_base:+.1f}%)", "#fbbf24", "dash"),
                (r.target_bear, f"🐻 Bear ₩{r.target_bear:,.0f} ({r.upside_bear:+.1f}%)", "#f87171", "dot"),
            ]:
                fig_p.add_hline(
                    y=val, line_dash=dash, line_color=col, line_width=1.5,
                    annotation_text=lbl, annotation_position="top left",
                    annotation_font_color=col, annotation_font_size=10,
                    annotation_bgcolor="rgba(0,0,0,0.5)",
                )
            fig_p.add_hrect(
                y0=r.target_bear, y1=r.target_bull,
                fillcolor="rgba(251,191,36,0.06)", line_width=0,
            )
            fig_p.update_layout(
                **CHART_LAYOUT, height=300,
                title=dict(text=f"{r.name} — 주가 추이(최근 2년) & Bear/Base/Bull 목표가",
                           x=0, font=dict(size=12, color="#c4c4e0")),
                xaxis=dict(**AX),
                yaxis=dict(**AX, title="주가 (원)"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            # 목표가 비교 — 현재가 vs Bear/Base/Bull 수평 바
            fig_tgt = go.Figure()
            targets = [r.target_bear, r.target_base, r.target_bull]
            upsides = [r.upside_bear, r.upside_base, r.upside_bull]
            colors  = ["#f87171", "#fbbf24", "#34d399"]
            labels  = ["🐻 Bear", "📍 Base", "🐂 Bull"]

            fig_tgt.add_vline(
                x=rt_p, line_color="#f1f5f9", line_width=2,
                annotation_text=f"현재가\n₩{rt_p:,.0f}",
                annotation_position="top",
                annotation_font_color="#f1f5f9", annotation_font_size=10,
            )
            for i, (tgt, ups, col, lbl) in enumerate(zip(targets, upsides, colors, labels)):
                x0, x1 = min(rt_p, tgt), max(rt_p, tgt)
                fig_tgt.add_shape(
                    type="rect", x0=x0, x1=x1, y0=i - 0.35, y1=i + 0.35,
                    fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.25)",
                    line=dict(color=col, width=1.5),
                )
                fig_tgt.add_annotation(
                    x=tgt, y=i,
                    text=f"₩{tgt:,.0f}<br><b>{ups:+.1f}%</b>",
                    font=dict(size=10, color=col),
                    showarrow=True, arrowhead=2, arrowcolor=col,
                    ax=35 if tgt >= rt_p else -35, ay=0,
                    bgcolor="rgba(0,0,0,0.7)", bordercolor=col, borderwidth=1,
                )
            fig_tgt.update_layout(
                **CHART_LAYOUT, height=300,
                title=dict(text="현재가 기준 Bear/Base/Bull 업사이드",
                           x=0, font=dict(size=12, color="#c4c4e0")),
                xaxis=dict(**AX, title="주가 (원)"),
                yaxis=dict(tickvals=[0, 1, 2], ticktext=labels,
                           showgrid=False, zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_tgt, use_container_width=True)

        # ── 행 2: P/E 역사적 밴드 + P/E 백분위 위치 게이지 ──
        c3, c4 = st.columns([3, 2])

        with c3:
            pe_full = r.hist_pe_series.dropna()
            fig_pe  = go.Figure()
            fig_pe.add_trace(go.Scatter(
                x=pe_full.index, y=pe_full.values,
                mode="lines", name="Fwd P/E",
                line=dict(color="#a78bfa", width=1.8),
            ))
            fig_pe.add_hrect(
                y0=r.pe_p25, y1=r.pe_p75,
                fillcolor="rgba(99,102,241,0.1)", line_width=0,
                annotation_text="IQR 범위 (25~75th)",
                annotation_position="top left",
                annotation_font_color="#6b7280", annotation_font_size=9,
            )
            for val, lbl, col in [
                (r.pe_p75,    f"75th  {r.pe_p75:.1f}x",    "#34d399"),
                (r.pe_median, f"Median {r.pe_median:.1f}x", "#fbbf24"),
                (r.pe_p25,    f"25th  {r.pe_p25:.1f}x",    "#f87171"),
            ]:
                fig_pe.add_hline(y=val, line_dash="dash", line_color=col, line_width=1,
                                 annotation_text=lbl, annotation_position="right",
                                 annotation_font_color=col, annotation_font_size=10)
            fig_pe.add_hline(
                y=r.current_fwd_pe, line_color="#f1f5f9", line_width=2, line_dash="dot",
                annotation_text=f"현재 {r.current_fwd_pe:.1f}x",
                annotation_position="right", annotation_font_color="#f1f5f9", annotation_font_size=11,
            )
            fig_pe.update_layout(
                **CHART_LAYOUT, height=280,
                title=dict(text=f"Fwd P/E 역사적 밴드 ({band_years}년 기준) — 현재 {r.pe_percentile:.0f}% 위치",
                           x=0, font=dict(size=12, color="#c4c4e0")),
                xaxis=dict(**AX),
                yaxis=dict(**AX, title="Fwd P/E"),
            )
            st.plotly_chart(fig_pe, use_container_width=True)

        with c4:
            # P/E 백분위 게이지 — 컬러 스펙트럼 바
            pct     = r.pe_percentile
            col_now = pe_bar_color(pct)

            fig_g = go.Figure()

            # 컬러 그라디언트 바 (정사각 마커 연속 배치)
            fig_g.add_trace(go.Scatter(
                x=list(range(101)), y=[0.5] * 101,
                mode="markers",
                marker=dict(
                    size=22, symbol="square",
                    color=list(range(101)),
                    colorscale=[
                        [0.00, "rgba(52,211,153,0.7)"],
                        [0.30, "rgba(251,191,36,0.6)"],
                        [0.70, "rgba(251,191,36,0.6)"],
                        [1.00, "rgba(248,113,113,0.7)"],
                    ],
                    opacity=0.85,
                ),
                hoverinfo="skip", showlegend=False,
            ))

            # 현재 P/E 포인터 (삼각형)
            fig_g.add_trace(go.Scatter(
                x=[pct], y=[0.5],
                mode="markers",
                marker=dict(size=20, color="white", symbol="triangle-down",
                            line=dict(width=2.5, color=col_now)),
                hoverinfo="skip", showlegend=False,
            ))

            # 키 레벨 눈금
            for xv, txt, col in [
                (0,   f"Min<br>{r.pe_min:.0f}x",     "#9ca3af"),
                (25,  f"25th<br>{r.pe_p25:.0f}x",    "#f87171"),
                (50,  f"Median<br>{r.pe_median:.0f}x","#fbbf24"),
                (75,  f"75th<br>{r.pe_p75:.0f}x",    "#34d399"),
                (100, f"Max<br>{r.pe_max:.0f}x",     "#9ca3af"),
            ]:
                fig_g.add_shape(type="line", x0=xv, x1=xv, y0=0.25, y1=0.39,
                                line=dict(color=col, width=1.2))
                fig_g.add_annotation(x=xv, y=0.16, text=txt,
                                     font=dict(size=9, color=col),
                                     showarrow=False, align="center")

            # 현재 P/E 말풍선
            fig_g.add_annotation(
                x=pct, y=0.78,
                text=f"현재 Fwd P/E<br><b>{r.current_fwd_pe:.1f}x</b><br>({pct:.0f}% 위치)",
                font=dict(size=11, color="white"),
                showarrow=True, arrowhead=2, arrowcolor=col_now,
                ax=0, ay=22,
                bgcolor="rgba(0,0,0,0.75)",
                bordercolor=col_now, borderwidth=1.5,
                align="center",
            )

            fig_g.update_layout(
                **CHART_LAYOUT, height=280,
                title=dict(
                    text=f"P/E 역사적 위치 — <b style='color:{col_now}'>{pct:.0f}%</b>  (저평가→고평가)",
                    x=0, font=dict(size=12, color="#c4c4e0"),
                ),
                xaxis=dict(range=[-3, 103], showticklabels=False, showgrid=False,
                           zeroline=False, title="← 저평가 (싸다)               고평가 (비싸다) →"),
                yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
                margin=dict(l=10, r=20, t=50, b=55),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # ── 행 3: Fwd EPS 연도별 추이 ──
        eps = r.hist_eps_series.dropna()
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Bar(
            x=eps.index, y=eps.values,
            marker_color=["#34d399" if v >= 0 else "#f87171" for v in eps.values],
            opacity=0.8, name="Fwd EPS",
        ))
        fig_eps.add_hline(
            y=r.current_fwd_eps, line_color="#f1f5f9", line_width=1.5, line_dash="dot",
            annotation_text=f"현재 EPS ₩{r.current_fwd_eps:,.0f}",
            annotation_position="right", annotation_font_color="#f1f5f9", annotation_font_size=10,
        )
        fig_eps.update_layout(
            **CHART_LAYOUT, height=240,
            title=dict(text="12M Fwd EPS 연도별 추이",
                       x=0, font=dict(size=12, color="#c4c4e0")),
            xaxis=dict(**AX),
            yaxis=dict(**AX, title="EPS (원)"),
            showlegend=False,
        )
        st.plotly_chart(fig_eps, use_container_width=True)
