"""
fetch_eps_auto.py
-----------------
DataGuide Excel Add-in UI 직접 자동화로 12M Fwd EPS 수집

방식: pywin32 + pyautogui로 DataGuide 리본 버튼 클릭 → 대화상자 제어
VBA 권한 없이도 동작
"""

import time
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import win32gui
    import win32con
    import win32api
    import win32com.client
    WIN32_OK = True
except ImportError:
    WIN32_OK = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True   # 마우스 왼쪽 상단 이동 시 중단
    pyautogui.PAUSE    = 0.3
    PGUI_OK = True
except ImportError:
    PGUI_OK = False

try:
    import xlwings as xw
    XW_OK = True
except ImportError:
    XW_OK = False


# ─────────────────────────────────────────
# DataGuide 워크시트 함수 방식 (가장 안정적)
# ─────────────────────────────────────────

# DataGuide 6 시계열 함수 후보 (버전별로 다를 수 있음)
DG_FUNC_CANDIDATES = [
    # 형식: (함수명, 인수 순서 설명)
    ("DG6",          "ticker, item_code, start, end, period"),
    ("DG_DATA",      "ticker, item_code, start, end, period"),
    ("DGDATA",       "ticker, item_code, start, end, period"),
    ("FN_DATA",      "ticker, item_code, start, end, period"),
    ("DG_TIME",      "ticker, item_code, start, end, period"),
    ("DGTIMESERIES", "ticker, item_code, start, end, period"),
]

# DataGuide 12M Fwd EPS 항목코드 후보
EPS_ITEM_CODES = [
    "S182800",   # 컨센서스 12개월 선행 EPS (DataGuide 내부 코드)
    "E030200",
    "FWD_EPS",
    "EPS12F",
    "CONSENEPS",
]


def try_worksheet_function(app, func_name: str, ticker: str, item_code: str,
                            start: str, end: str) -> float | None:
    """
    DataGuide 워크시트 함수 테스트
    성공 시 값 반환, 실패 시 None
    """
    try:
        result = app.api.WorksheetFunction.Run(
            func_name, ticker, item_code, start, end, "M"
        )
        if result and not isinstance(result, str):
            return float(result)
    except Exception:
        pass
    return None


def detect_dg_worksheet_function(test_ticker: str = "005930") -> tuple[str, str] | None:
    """
    DataGuide 워크시트 함수 자동 탐지
    작동하는 함수명 + 항목코드 조합을 반환
    """
    if not XW_OK:
        return None

    app = xw.App(visible=False, add_book=False)
    wb  = app.books.add()
    ws  = wb.sheets[0]

    start = "20240101"
    end   = "20241231"

    try:
        for func_name, _ in DG_FUNC_CANDIDATES:
            for item_code in EPS_ITEM_CODES:
                # 워크시트에 함수 입력
                try:
                    formula = f'={func_name}("{test_ticker}","{item_code}","{start}","{end}","M")'
                    ws.range("A1").formula = formula
                    app.api.Calculate()
                    time.sleep(1)
                    val = ws.range("A1").value
                    if val and isinstance(val, (int, float)) and val > 0:
                        logger.info(f"DataGuide 함수 탐지 성공: {func_name}, 코드={item_code}, 값={val}")
                        return func_name, item_code
                except Exception:
                    continue
    finally:
        wb.close()
        app.quit()

    return None


# ─────────────────────────────────────────
# 워크시트 함수 기반 배치 EPS 수집
# ─────────────────────────────────────────

def fetch_eps_via_worksheet_function(
    tickers: list[str],
    func_name: str,
    item_code: str,
    start: str = "20200101",
    end:   str  = None,
    batch_size: int = 30,
    save_path: str  = None,
) -> pd.DataFrame:
    """
    DataGuide 워크시트 함수로 EPS 배치 수집

    한 시트에 종목×날짜 형태로 함수를 채워 넣고
    Excel이 계산한 뒤 값을 읽어 DataFrame으로 반환
    """
    if not XW_OK:
        raise RuntimeError("xlwings 미설치")

    end = end or datetime.today().strftime("%Y%m%d")

    # 날짜 범위 생성 (월별)
    dates = pd.date_range(start, end, freq="MS")
    date_strs = [d.strftime("%Y%m%d") for d in dates]

    all_dfs = []
    chunks  = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    total_chunks = len(chunks)

    print(f"\n[EPS 수집] 워크시트 함수 방식 | {len(tickers)}종목 | {total_chunks}배치")

    app = xw.App(visible=True, add_book=False)
    app.display_alerts = False

    try:
        for ci, chunk in enumerate(chunks):
            print(f"  배치 {ci+1}/{total_chunks} ({len(chunk)}종목)...")
            wb = app.books.add()
            ws = wb.sheets[0]

            # 헤더 행: 날짜
            ws.range("B1").options(transpose=False).value = date_strs

            result_data = {}
            for ri, ticker in enumerate(chunk):
                row = ri + 2  # 2행부터 시작
                ws.range(f"A{row}").value = ticker
                # 각 날짜별 함수 입력
                for ci2, date_str in enumerate(date_strs):
                    col = ci2 + 2  # B열부터
                    formula = f'={func_name}("{ticker}","{item_code}","{date_str}","{date_str}","M")'
                    ws.range(row, col).formula = formula

            # 전체 계산
            app.api.Calculate()
            time.sleep(max(5, len(chunk) * 0.3))  # 로딩 대기

            # 결과 읽기
            for ri, ticker in enumerate(chunk):
                row = ri + 2
                vals = ws.range(f"B{row}").expand("right").value
                if isinstance(vals, (int, float)):
                    vals = [vals]
                result_data[ticker] = vals

            df_chunk = pd.DataFrame(result_data, index=dates)
            all_dfs.append(df_chunk)
            wb.close()
            time.sleep(1)

    finally:
        app.quit()

    if not all_dfs:
        return pd.DataFrame()

    eps_df = pd.concat(all_dfs, axis=1).sort_index()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        eps_df.to_excel(save_path)
        print(f"  [저장] {save_path}")

    return eps_df


# ─────────────────────────────────────────
# DataGuide 리본 UI 자동화 (pyautogui 방식)
# ─────────────────────────────────────────

def open_excel_with_dataguide(xlsx_path: str = None) -> object:
    """xlwings로 Excel 열기"""
    if not XW_OK:
        raise RuntimeError("xlwings 미설치")
    app = xw.App(visible=True, add_book=False)
    if xlsx_path and Path(xlsx_path).exists():
        wb = app.books.open(xlsx_path)
    else:
        wb = app.books.add()
    # Excel 창을 앞으로 가져오기
    time.sleep(1)
    if WIN32_OK:
        hwnd = win32gui.FindWindow("XLMAIN", None)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    return app, wb


def click_dataguide_ribbon_button(button_name: str = "시계열데이터"):
    """
    DataGuide 리본 탭 → 특정 버튼 클릭 (pyautogui)
    button_name: '시계열데이터', '새로고침' 등
    """
    if not PGUI_OK:
        raise RuntimeError("pyautogui 미설치")

    # DataGuide 탭 클릭 (화면 상단 탭바에서 찾기)
    try:
        # DataGuide 탭 이미지 위치 탐지 (템플릿 매칭)
        # 실제 환경에서는 스크린샷 기반으로 위치 찾아야 함
        # 현재는 좌표 기반 (Excel 최대화 기준 근사값)
        import pyautogui
        
        # DataGuide6 탭 클릭 (화면 위쪽 탭 영역)
        tab_region = (0, 95, 1920, 30)  # Excel 리본 탭 영역
        tab_pos = pyautogui.locateOnScreen(
            r"D:\Dataguide\assets\dg_tab.png",
            region=tab_region,
            confidence=0.8
        )
        if tab_pos:
            pyautogui.click(tab_pos)
            time.sleep(0.5)
        
        # 버튼 클릭
        btn_pos = pyautogui.locateOnScreen(
            rf"D:\Dataguide\assets\dg_{button_name}.png",
            confidence=0.8
        )
        if btn_pos:
            pyautogui.click(btn_pos)
            return True
    except Exception as e:
        logger.warning(f"이미지 탐지 실패: {e}")
    
    return False


# ─────────────────────────────────────────
# 최적 방법 자동 선택 및 실행
# ─────────────────────────────────────────

def auto_fetch_eps(
    tickers: list[str],
    start: str = "20200101",
    end: str   = None,
    save_path: str = None,
) -> pd.DataFrame | None:
    """
    사용 가능한 방법 중 최적을 선택해 Fwd EPS 수집

    우선순위:
      1. DataGuide 워크시트 함수 (자동 탐지 후 사용)
      2. 기존 파일이 있으면 로드
      3. 가이드 출력 (수동 개입 필요)
    """
    end = end or datetime.today().strftime("%Y%m%d")

    # 1) 워크시트 함수 탐지
    print("\n[EPS 자동화] DataGuide 워크시트 함수 탐지 중...")
    found = detect_dg_worksheet_function()

    if found:
        func_name, item_code = found
        print(f"  [성공] 함수: {func_name}, 항목코드: {item_code}")
        return fetch_eps_via_worksheet_function(
            tickers, func_name, item_code,
            start=start, end=end, save_path=save_path,
        )

    # 2) 기존 EPS 파일 확인
    if save_path and Path(save_path).exists():
        print(f"  [로드] 기존 EPS 파일 사용: {save_path}")
        from core.data_loader import load_excel
        return load_excel(save_path)

    # 3) 수동 가이드
    print("\n" + "="*60)
    print("[안내] DataGuide 워크시트 함수를 자동 탐지하지 못했습니다.")
    print("       아래 방법으로 EPS 파일을 준비해주세요:\n")
    print("  방법 A — DataGuide에서 직접 추출:")
    print("    1. Excel DataGuide 탭 → [시계열데이터]")
    print("    2. 항목: '12개월 선행 EPS' or 'FWD EPS'")
    print("    3. 종목코드: data/universe.csv 참고 (350개)")
    print("    4. 기간: 2020-01-01 ~ 오늘, 월별")
    print(f"    5. 저장: {save_path or 'data/fwd_eps.xlsx'}")
    print()
    print("  방법 B — Excel VBA 권한 활성화 후 재시도:")
    print("    Excel → 파일 → 옵션 → 보안 센터 → 보안 센터 설정")
    print("    → 매크로 설정 → 'VBA 프로젝트 개체 모델에 대한 액세스 신뢰' 체크")
    print("    → 확인 후 python fetch_eps_auto.py 재실행")
    print("="*60)

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pandas as pd

    universe = pd.read_csv(r"D:\Dataguide\data\universe.csv", dtype=str)
    tickers  = universe["ticker"].tolist()

    result = auto_fetch_eps(
        tickers,
        start="20200101",
        save_path=r"D:\Dataguide\data\fwd_eps.xlsx",
    )

    if result is not None:
        print(f"\n[완료] EPS shape: {result.shape}")
    else:
        print("\n[대기] EPS 파일을 data/fwd_eps.xlsx 에 저장 후 웹앱을 실행하세요.")
        print("       웹앱: http://localhost:8501")
