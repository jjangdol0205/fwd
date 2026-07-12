"""
run_pipeline.py
---------------
KOSPI200 + KOSDAQ150 전 종목 데이터 수집 → 자동화 파이프라인

실행: python run_pipeline.py
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 경로 설정
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.fetch_universe import get_universe
from core.fetch_prices   import fetch_all_prices
from core.calculator     import run_screener
from core.data_loader    import load_excel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DATA_DIR      = ROOT / "data"
TEMPLATE_DIR  = ROOT / "templates"
DATA_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

UNIVERSE_PATH = DATA_DIR / "universe.csv"
PRICE_PATH    = DATA_DIR / "price.xlsx"
EPS_PATH      = DATA_DIR / "fwd_eps.xlsx"
RESULT_PATH   = DATA_DIR / "screening_result.xlsx"

START_DATE = "20200101"
BAND_YEARS = 5


# ─────────────────────────────────────────
# Step 1: 유니버스 수집
# ─────────────────────────────────────────
def step1_universe(force: bool = False):
    print("\n" + "═"*60)
    print(" STEP 1 / 3  │  KOSPI200 + KOSDAQ150 종목 리스트 수집")
    print("═"*60)

    if UNIVERSE_PATH.exists() and not force:
        import pandas as pd
        universe = pd.read_csv(UNIVERSE_PATH, dtype=str)
        print(f"  ✅ 기존 유니버스 로드: {len(universe)}개 종목")
        return universe

    universe = get_universe(save_path=str(UNIVERSE_PATH))
    return universe


# ─────────────────────────────────────────
# Step 2: 주가 수집 (pykrx)
# ─────────────────────────────────────────
def step2_prices(tickers: list[str], force: bool = False):
    print("\n" + "═"*60)
    print(" STEP 2 / 3  │  월별 수정주가 수집 (pykrx)")
    print("═"*60)

    if PRICE_PATH.exists() and not force:
        price_df = load_excel(str(PRICE_PATH))
        missing  = [t for t in tickers if t not in price_df.columns]
        if not missing:
            print(f"  ✅ 기존 주가 로드: {price_df.shape[1]}종목 × {price_df.shape[0]}개월")
            return price_df
        print(f"  ℹ️  {len(missing)}개 신규 종목 추가 수집")
        tickers = missing

    price_df = fetch_all_prices(
        tickers,
        start=START_DATE,
        save_path=str(PRICE_PATH),
    )
    return price_df


# ─────────────────────────────────────────
# Step 3: Fwd EPS 수집 (DataGuide)
# ─────────────────────────────────────────
def step3_fwd_eps(tickers: list[str], force: bool = False):
    print("\n" + "═"*60)
    print(" STEP 3 / 3  │  12M Fwd EPS 수집 (DataGuide)")
    print("═"*60)

    # 이미 수집된 파일이 있으면 스킵
    if EPS_PATH.exists() and not force:
        eps_df  = load_excel(str(EPS_PATH))
        missing = [t for t in tickers if t not in eps_df.columns]
        if not missing:
            print(f"  ✅ 기존 EPS 로드: {eps_df.shape[1]}종목 × {eps_df.shape[0]}개월")
            return eps_df
        print(f"  ℹ️  {len(missing)}개 종목 EPS 미수집")

    # DataGuide 자동화 시도
    try:
        from core.fetch_eps_dataguide import DataGuideEPSFetcher, create_dataguide_template

        print("  🔄 DataGuide Add-in 탐지 중...")
        fetcher = DataGuideEPSFetcher()
        addins  = fetcher.detect_dataguide_addin()

        if addins:
            print(f"  ✅ DataGuide 감지됨: {list(addins.values())}")
            eps_df = fetcher.fetch(
                tickers,
                start=START_DATE,
                save_path=str(EPS_PATH),
            )
            if not eps_df.empty:
                return eps_df

        # 자동화 실패 시 → 템플릿 생성 + 안내
        print("\n  ⚠️  DataGuide 자동화를 완료하지 못했습니다.")
        print("  📋 아래 안내에 따라 DataGuide에서 EPS를 수동으로 뽑아주세요:\n")
        create_dataguide_template(tickers, str(TEMPLATE_DIR / "dg_eps_template.xlsx"))

    except Exception as e:
        print(f"  ⚠️  DataGuide 자동화 오류: {e}")
        print(f"  📁 EPS 파일을 직접 저장해주세요: {EPS_PATH}")

    # EPS 파일이 나중에 업로드될 경우를 대비해 None 반환
    if EPS_PATH.exists():
        return load_excel(str(EPS_PATH))
    return None


# ─────────────────────────────────────────
# Step 4: 스크리닝 & 결과 저장
# ─────────────────────────────────────────
def step4_screen(price_df, eps_df, ticker_names: dict):
    print("\n" + "═"*60)
    print(" SCREENING  │  P/E 밴드 계산 & 목표가 산출")
    print("═"*60)

    results = run_screener(price_df, eps_df, ticker_names, band_years=BAND_YEARS)
    print(f"  ✅ 스크리닝 완료: {len(results)}개 종목 유효")

    if not results:
        return []

    # 결과 테이블
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "종목코드":      r.ticker,
            "종목명":        r.name,
            "현재가":        round(r.current_price),
            "Fwd EPS":      round(r.current_fwd_eps),
            "현재 Fwd P/E": round(r.current_fwd_pe, 1),
            "P/E 위치(%)":  round(r.pe_percentile, 1),
            "P/E Median":   round(r.pe_median, 1),
            "Bear 목표가":  round(r.target_bear),
            "Base 목표가":  round(r.target_base),
            "Bull 목표가":  round(r.target_bull),
            "Bear 업사이드(%)": round(r.upside_bear, 1),
            "Base 업사이드(%)": round(r.upside_base, 1),
            "Bull 업사이드(%)": round(r.upside_bull, 1),
        })

    df_result = pd.DataFrame(rows)
    df_result.to_excel(str(RESULT_PATH), index=False)
    print(f"  💾 결과 저장: {RESULT_PATH}")

    # Top 10 미리보기
    print("\n  📊 Base 업사이드 Top 10:")
    print(f"  {'종목명':<12} {'현재가':>8} {'Base목표가':>10} {'업사이드':>8} {'P/E위치':>8}")
    print("  " + "-"*52)
    for r in results[:10]:
        print(f"  {r.name:<12} {r.current_price:>8,.0f} {r.target_base:>10,.0f} "
              f"{r.upside_base:>7.1f}% {r.pe_percentile:>7.0f}%")

    return results


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KOSPI200+KOSDAQ150 Fwd P/E 스크리닝 파이프라인")
    parser.add_argument("--force",     action="store_true", help="기존 캐시 무시 후 재수집")
    parser.add_argument("--no-eps",    action="store_true", help="EPS 수집 건너뜀 (주가만 수집)")
    parser.add_argument("--no-launch", action="store_true", help="웹앱 자동 실행 비활성")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  📊  Fwd P/E 밴드 스크리닝 파이프라인")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*60)

    # Step 1: 유니버스
    universe = step1_universe(force=args.force)
    tickers  = universe["ticker"].tolist()
    names    = dict(zip(universe["ticker"], universe["name"]))

    # Step 2: 주가
    price_df = step2_prices(tickers, force=args.force)

    # Step 3: Fwd EPS
    eps_df = None
    if not args.no_eps:
        eps_df = step3_fwd_eps(tickers, force=args.force)

    # Step 4: 스크리닝 (EPS 있을 때만)
    if eps_df is not None and price_df is not None:
        step4_screen(price_df, eps_df, names)
    else:
        print("\n  ℹ️  EPS 데이터 없음 → 스크리닝 건너뜀")
        print(f"  EPS 파일 경로: {EPS_PATH}")
        print("  DataGuide에서 EPS 수집 후 웹앱에서 업로드하거나 위 경로에 저장하세요.")

    # 웹앱 실행
    if not args.no_launch:
        print("\n" + "═"*60)
        print("  🌐  웹앱 실행: http://localhost:8501")
        print("═"*60)
        import subprocess
        subprocess.Popen(
            ["streamlit", "run", str(ROOT / "app.py"), "--server.port", "8501"],
            cwd=str(ROOT),
        )

    print("\n✅ 파이프라인 완료!")


if __name__ == "__main__":
    main()
