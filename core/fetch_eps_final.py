"""
fetch_eps_final.py
------------------
DataGuide 컨센서스 EPS(Fwd.12M) 자동 수집
항목코드: FM30041100  (EPS Fwd.12M, 지배)

DataGuide xlwings 자동화:
  1) Excel + DataGuide Add-in 로드
  2) 시계열데이터 요청 (xlwings로 DataGuide COM 직접 호출)
  3) 결과 추출 → DataFrame 반환
"""

import time, sys, logging, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

EPS_ITEM_CODE = "FM30041100"   # EPS(Fwd.12M, 지배) — DataGuide 확인 완료
EPS_ITEM_NAME = "EPS(Fwd.12M, 지배)"
DG_SERVER     = "dataguide.co.kr"

try:
    import xlwings as xw
    XW_OK = True
except ImportError:
    XW_OK = False

try:
    import win32com.client, win32gui, win32con
    WIN32_OK = True
except ImportError:
    WIN32_OK = False


# ─────────────────────────────────────────
# DataGuide COM 직접 호출 시도
# ─────────────────────────────────────────

def _try_com_fetch(app, ticker_list: list[str], start: str, end: str) -> pd.DataFrame | None:
    """
    DataGuide Add-in COM 객체를 직접 호출해 EPS 데이터 수집
    여러 ProgID / 메서드명 조합 시도
    """
    prog_ids = [
        "DataGuide6.AddIn",
        "DataGuide.AddIn",
        "DataGuidePro.AddIn",
        "DataGuide6",
    ]
    method_names = [
        "GetTimeSeriesData",
        "GetData",
        "GetTimeSeries",
        "RequestData",
        "GetConsensusData",
    ]

    codes_str = ",".join(ticker_list)

    for prog_id in prog_ids:
        try:
            dg = app.api.COMAddIns(prog_id).Object
            if dg is None:
                continue
            logger.info(f"DataGuide COM 연결: {prog_id}")

            for method in method_names:
                try:
                    func = getattr(dg, method)
                    result = func(codes_str, EPS_ITEM_CODE, start, end, "M")
                    if result is not None:
                        logger.info(f"COM 메서드 성공: {method}")
                        # result를 DataFrame으로 변환
                        df = _result_to_df(result, ticker_list)
                        if df is not None:
                            return df
                except Exception as me:
                    logger.debug(f"  {method} 실패: {me}")
        except Exception as pe:
            logger.debug(f"COM {prog_id} 실패: {pe}")

    return None


def _result_to_df(result, tickers: list[str]) -> pd.DataFrame | None:
    """COM 반환값을 DataFrame으로 변환"""
    try:
        if hasattr(result, '__iter__'):
            arr = list(result)
            if len(arr) > 0:
                return pd.DataFrame(arr)
        if isinstance(result, (list, tuple)):
            return pd.DataFrame(result)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────
# DataGuide 시계열 템플릿 방식
# ─────────────────────────────────────────

def _build_dg_template(tickers: list[str], start: str, end: str,
                        save_path: str) -> str:
    """
    DataGuide 시계열 쿼리 템플릿 Excel 파일 생성
    사용자가 이 파일을 DataGuide에서 열고 새로고침하면 EPS 데이터가 채워짐

    Returns: 저장된 파일 경로
    """
    if not XW_OK:
        raise RuntimeError("xlwings 미설치")

    app = xw.App(visible=False, add_book=False)
    wb  = app.books.add()
    try:
        # 시트 1: 종목 리스트
        ws_code = wb.sheets[0]
        ws_code.name = "종목리스트"
        ws_code["A1"].value = "종목코드"
        ws_code["B1"].value = "종목명"
        for i, t in enumerate(tickers):
            ws_code[f"A{i+2}"].value = t
        ws_code["D1"].value  = "DataGuide 설정값"
        ws_code["D2"].value  = f"항목코드: {EPS_ITEM_CODE}"
        ws_code["D3"].value  = f"항목명: {EPS_ITEM_NAME}"
        ws_code["D4"].value  = f"시작일: {start}"
        ws_code["D5"].value  = f"종료일: {end}"
        ws_code["D6"].value  = "주기: 월별(M)"

        # 시트 2: EPS 출력 (DataGuide가 여기에 데이터를 채움)
        ws_eps = wb.sheets.add("FWD_EPS")
        ws_eps["A1"].value = "※ DataGuide [시계열데이터] → FM30041100 → 이 시트에 출력"

        wb.save(save_path)
        logger.info(f"템플릿 저장: {save_path}")
        return save_path
    finally:
        wb.close()
        app.quit()


# ─────────────────────────────────────────
# DataGuide Refresh 자동화
# ─────────────────────────────────────────

def refresh_and_extract(
    template_path: str,
    sheet_name: str = "FWD_EPS",
    wait_sec: int   = 15,
) -> pd.DataFrame | None:
    """
    기존 DataGuide 템플릿 파일을 열고 Refresh → 데이터 추출

    Parameters
    ----------
    template_path : str
        DataGuide 쿼리가 등록된 Excel 파일 경로
    sheet_name : str
        EPS 데이터가 출력된 시트명
    wait_sec : int
        Refresh 후 데이터 로딩 대기 시간(초)
    """
    if not XW_OK:
        raise RuntimeError("xlwings 미설치")

    app = xw.App(visible=True, add_book=False)
    app.display_alerts = False
    wb = app.books.open(str(template_path))

    try:
        # DataGuide Refresh 매크로 실행 (여러 이름 시도)
        refresh_names = [
            "DataGuide6.xlam!RefreshAll",
            "DataGuide6.xlam!Refresh",
            "DataGuide.xlam!RefreshAll",
            "RefreshAll",
            "DGRefresh",
        ]
        refreshed = False
        for name in refresh_names:
            try:
                app.api.Run(name)
                refreshed = True
                logger.info(f"Refresh 성공: {name}")
                break
            except Exception:
                continue

        if not refreshed:
            # COM Add-in을 통한 Refresh 시도
            try:
                for addin in app.api.COMAddIns:
                    if addin.Connect and "DataGuide" in str(addin.Description):
                        obj = addin.Object
                        if hasattr(obj, "RefreshAll"):
                            obj.RefreshAll()
                            refreshed = True
                            break
                        if hasattr(obj, "Refresh"):
                            obj.Refresh()
                            refreshed = True
                            break
            except Exception as e:
                logger.warning(f"COM Refresh 실패: {e}")

        if refreshed:
            logger.info(f"Refresh 대기: {wait_sec}초")
            time.sleep(wait_sec)

        # 데이터 추출
        try:
            ws   = wb.sheets[sheet_name]
            used = ws.used_range
            df   = used.options(pd.DataFrame, index=False, header=True).value
            if df is not None and not df.empty:
                logger.info(f"데이터 추출 완료: {df.shape}")
                return df
        except Exception as e:
            logger.error(f"데이터 추출 실패: {e}")

    finally:
        wb.close()
        app.quit()

    return None


# ─────────────────────────────────────────
# 메인 수집 함수
# ─────────────────────────────────────────

def fetch_fwd_eps(
    tickers: list[str],
    ticker_names: dict  = None,
    start: str          = "20200101",
    end: str            = None,
    template_path: str  = r"D:\Dataguide\templates\dg_eps_template.xlsm",
    save_path: str      = r"D:\Dataguide\data\fwd_eps.xlsx",
    wait_refresh: int   = 20,
) -> pd.DataFrame | None:
    """
    12M Fwd EPS (FM30041100) 전 종목 자동 수집

    1차: 기존 템플릿 있으면 Refresh → 추출
    2차: 템플릿 없으면 생성 → 사용자 안내
    """
    end = end or datetime.today().strftime("%Y%m%d")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(template_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[EPS 수집] 항목: {EPS_ITEM_NAME} ({EPS_ITEM_CODE})")
    print(f"  종목: {len(tickers)}개  |  기간: {start} ~ {end}")

    # ── 기존 EPS 파일 있으면 즉시 반환 ──
    if Path(save_path).exists():
        from core.data_loader import load_excel
        df = load_excel(save_path)
        if not df.empty:
            print(f"  [로드] 기존 EPS 파일 사용: {df.shape}")
            return df

    # ── 기존 템플릿 있으면 Refresh 시도 ──
    if Path(template_path).exists():
        print("  [Refresh] DataGuide 템플릿 새로고침 시도...")
        df = refresh_and_extract(template_path, wait_sec=wait_refresh)
        if df is not None and not df.empty:
            from core.data_loader import load_excel
            import io
            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            result = load_excel(buf)
            result.to_excel(save_path)
            print(f"  [완료] EPS 수집: {result.shape}")
            return result

    # ── 템플릿 생성 ──
    print(f"  [안내] DataGuide 템플릿 파일 생성 중...")
    _build_dg_template(tickers, start, end, template_path)

    print("\n" + "="*65)
    print("  DataGuide에서 딱 한 번만 아래 순서대로 해주세요!")
    print("="*65)
    print(f"  1. 아래 파일을 Excel로 열기:")
    print(f"     {template_path}")
    print()
    print("  2. DataGuide6 탭 → [시계열데이터] 클릭")
    print()
    print("  3. 데이터 설정:")
    print(f"     - 종목코드 : '종목리스트' 시트 A열 전체 붙여넣기 ({len(tickers)}개)")
    print(f"     - 항목코드 : FM30041100  (EPS Fwd.12M, 지배)")
    print(f"     - 시작일   : {start[:4]}-{start[4:6]}-{start[6:]}")
    print(f"     - 종료일   : {end[:4]}-{end[4:6]}-{end[6:]}")
    print(f"     - 주기     : 월별")
    print(f"     - 출력시트 : FWD_EPS (A1)")
    print()
    print(f"  4. Submit → 저장 (Ctrl+S)")
    print()
    print(f"  5. 이후 자동화: python -c \"")
    print(f"     from core.fetch_eps_final import refresh_and_extract")
    print(f"     df = refresh_and_extract(r'{template_path}')\"")
    print("="*65)

    return None


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    universe = pd.read_csv(r"D:\Dataguide\data\universe.csv", dtype=str)
    tickers  = universe["ticker"].tolist()
    names    = dict(zip(universe["ticker"], universe["name"]))

    result = fetch_fwd_eps(
        tickers,
        ticker_names=names,
        start="20200101",
        save_path=r"D:\Dataguide\data\fwd_eps.xlsx",
        template_path=r"D:\Dataguide\templates\dg_eps_template.xlsm",
    )

    if result is not None:
        print(f"\n[완료] EPS DataFrame: {result.shape}")
        print(result.tail(3))
    else:
        print("\n[다음 단계] 위 안내에 따라 DataGuide에서 EPS를 뽑아주세요.")
        print("           그 후: python run_pipeline.py 실행")
