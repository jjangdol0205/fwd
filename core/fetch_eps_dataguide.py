"""
fetch_eps_dataguide.py
----------------------
DataGuide Excel Add-in을 이용한 12M Fwd EPS 자동 수집

전략:
  1) Excel을 xlwings로 열고 DataGuide Add-in 감지
  2) VBA 매크로로 DataGuide 시계열 함수 호출 (배치)
  3) 데이터 추출 후 DataFrame 반환
"""

import pandas as pd
import numpy as np
import time
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xlwings as xw
    import win32com.client
    XLWINGS_OK = True
except ImportError:
    XLWINGS_OK = False


# ─────────────────────────────────────────────────────────────────
# DataGuide VBA 매크로 (Excel 내부에서 실행될 코드)
# ─────────────────────────────────────────────────────────────────

DG_VBA_MODULE = '''
Option Explicit

' DataGuide 시계열 EPS 배치 추출 매크로
' DataGuide6 Add-in의 내부 함수명이 다를 경우 아래를 수정

Sub DG_FetchFwdEPS(tickers As String, startDate As String, endDate As String, outSheet As String)
    Dim ws As Worksheet
    Dim dg As Object
    
    ' 출력 시트 준비
    On Error Resume Next
    Set ws = ThisWorkbook.Sheets(outSheet)
    On Error GoTo 0
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = outSheet
    End If
    ws.Cells.Clear
    
    ' DataGuide COM 객체 접근 시도
    On Error GoTo FallbackMethod
    Set dg = Application.COMAddIns("DataGuide6.AddIn").Object
    
    ' DataGuide API 호출 (API 명칭은 버전에 따라 상이)
    ' 방법 A: GetTimeSeriesData 형식
    Dim result As Variant
    result = dg.GetTimeSeriesData( _
        tickers, _
        "S182800",  ' 12M FWD EPS 항목코드 (DataGuide 항목코드 확인 필요)
        startDate, _
        endDate, _
        "M" _       ' 월별
    )
    
    If Not IsEmpty(result) Then
        ws.Range("A1").Resize(UBound(result, 1), UBound(result, 2)).Value = result
    End If
    
    MsgBox "DataGuide EPS 수집 완료: " & outSheet, vbInformation
    Exit Sub

FallbackMethod:
    ' COM 직접 호출 실패 시 → DataGuide 리본 메뉴 방식
    Application.StatusBar = "DataGuide EPS 수집 중..."
    
    ' DataGuide6 Refresh 시도
    On Error Resume Next
    Application.Run "DataGuide6.xlam!RefreshAll"
    Application.Run "DataGuide.xlam!RefreshAll"
    Application.Run "RefreshAll"
    On Error GoTo 0
    
    Application.StatusBar = False
End Sub


Sub DG_CheckAddin()
    ' DataGuide Add-in 설치 여부 및 이름 확인
    Dim addin As COMAddIn
    Dim msg As String
    msg = "설치된 COM Add-in:" & vbCrLf
    For Each addin In Application.COMAddIns
        If addin.Connect Then
            msg = msg & "  [활성] " & addin.Description & " (" & addin.ProgID & ")" & vbCrLf
        End If
    Next
    MsgBox msg, vbInformation, "Add-in 목록"
End Sub


Sub DG_ListMacros()
    ' 사용 가능한 DataGuide 관련 매크로 목록 출력
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets.Add
    ws.Name = "_MacroList"
    
    Dim row As Integer
    row = 1
    ws.Cells(row, 1) = "Add-in Name"
    ws.Cells(row, 2) = "ProgID"
    ws.Cells(row, 3) = "Connected"
    row = 2
    
    Dim addin As COMAddIn
    For Each addin In Application.COMAddIns
        ws.Cells(row, 1) = addin.Description
        ws.Cells(row, 2) = addin.ProgID
        ws.Cells(row, 3) = addin.Connect
        row = row + 1
    Next
    MsgBox "매크로 목록이 _MacroList 시트에 저장됐습니다.", vbInformation
End Sub
'''


# ─────────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────────

class DataGuideEPSFetcher:
    """
    DataGuide Add-in을 통한 12M Fwd EPS 자동 수집기

    사용 예시:
        fetcher = DataGuideEPSFetcher()
        eps_df = fetcher.fetch(tickers, start="20200101", end="20251231")
        eps_df.to_excel(r"D:\\Dataguide\\data\\fwd_eps.xlsx")
    """

    def __init__(self, dg_login_id: str = "lofa00", wait_refresh: int = 15):
        self.dg_login_id  = dg_login_id
        self.wait_refresh = wait_refresh
        self._app = None
        self._wb  = None

    # ── Excel + DataGuide 초기화 ─────────────────────────────────

    def _open_excel(self, wb_path: str = None):
        if not XLWINGS_OK:
            raise RuntimeError("xlwings / pywin32 미설치")

        self._app = xw.App(visible=True, add_book=False)
        self._app.display_alerts = False

        if wb_path and Path(wb_path).exists():
            self._wb = self._app.books.open(wb_path)
        else:
            self._wb = self._app.books.add()

        logger.info("Excel 열기 완료")

    def _close_excel(self):
        try:
            if self._wb:
                self._wb.close()
            if self._app:
                self._app.quit()
        except Exception:
            pass

    # ── Add-in 정보 조회 ─────────────────────────────────────────

    def detect_dataguide_addin(self) -> dict:
        """DataGuide Add-in의 ProgID 및 매크로 이름 자동 탐지"""
        self._open_excel()
        try:
            # VBA 모듈 삽입 후 DG_CheckAddin 실행
            wb = self._wb
            vba_mod = wb.api.VBProject.VBComponents.Add(1)  # 1=Module
            vba_mod.CodeModule.AddFromString(DG_VBA_MODULE)

            # COM Add-in 목록 수집
            addins = {}
            for addin in self._app.api.COMAddIns:
                if addin.Connect:
                    addins[addin.ProgID] = addin.Description
            logger.info(f"활성 Add-in: {addins}")
            return addins
        except Exception as e:
            logger.warning(f"Add-in 탐지 실패: {e}")
            return {}
        finally:
            self._close_excel()

    # ── 배치 EPS 수집 ────────────────────────────────────────────

    def fetch(
        self,
        tickers: list[str],
        start: str = "20200101",
        end:   str = None,
        chunk_size: int = 30,
        save_path: str = None,
    ) -> pd.DataFrame:
        """
        전 종목 12M Fwd EPS 배치 수집

        DataGuide가 한 번에 처리 가능한 종목 수 제한으로
        chunk_size 단위로 분할 요청
        """
        from datetime import datetime
        end = end or datetime.today().strftime("%Y%m%d")

        chunks  = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
        results = []

        print(f"\n📊 Fwd EPS 수집 시작: {len(tickers)}종목, {len(chunks)}개 배치")

        for idx, chunk in enumerate(chunks):
            print(f"  배치 {idx+1}/{len(chunks)}: {len(chunk)}종목...")
            df_chunk = self._fetch_chunk(chunk, start, end)
            if df_chunk is not None and not df_chunk.empty:
                results.append(df_chunk)
            time.sleep(2)  # 서버 부하 방지

        if not results:
            logger.error("Fwd EPS 데이터 수집 실패")
            return pd.DataFrame()

        eps_df = pd.concat(results, axis=1)
        eps_df = eps_df.sort_index()
        eps_df.index = pd.to_datetime(eps_df.index)

        print(f"\n✅ Fwd EPS 수집 완료: {eps_df.shape[1]}종목 × {eps_df.shape[0]}개월")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            eps_df.to_excel(save_path)
            print(f"   저장 완료: {save_path}")

        return eps_df

    def _fetch_chunk(self, tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
        """
        청크 단위 EPS 수집
        DataGuide Excel Add-in의 시계열 데이터 함수 호출
        """
        tmp_path = Path(tempfile.mktemp(suffix=".xlsm", prefix="dg_eps_", dir=r"D:\Dataguide\data"))
        self._open_excel()

        try:
            wb  = self._wb
            ws  = wb.sheets[0]
            ws.name = "EPS_DATA"

            # VBA 매크로 삽입
            vba_mod = wb.api.VBProject.VBComponents.Add(1)
            vba_mod.CodeModule.AddFromString(DG_VBA_MODULE)

            # 종목코드 시트에 입력
            codes_ws = wb.sheets.add(name="CODES")
            codes_ws.range("A1").value = "ticker"
            codes_ws.range("A2").options(transpose=True).value = tickers

            # DataGuide 시계열 데이터 요청
            # 방법1: VBA 매크로 실행
            ticker_str = ",".join(tickers)
            try:
                wb.macro("DG_FetchFwdEPS")(ticker_str, start, end, "EPS_DATA")
                time.sleep(self.wait_refresh)
            except Exception as e:
                logger.warning(f"VBA 매크로 실행 실패: {e}")

            # 방법2: DataGuide 리프레시 버튼 트리거
            try:
                for macro_name in ["DataGuide6.xlam!RefreshAll", "DataGuide.RefreshAll", "RefreshAll"]:
                    try:
                        self._app.api.Run(macro_name)
                        time.sleep(self.wait_refresh)
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            # 데이터 추출
            df = ws.range("A1").options(pd.DataFrame, expand="table").value
            if df is not None and not df.empty:
                return df

        except Exception as e:
            logger.error(f"청크 처리 오류: {e}")
        finally:
            try:
                wb.save(str(tmp_path))
            except Exception:
                pass
            self._close_excel()

        return None


# ─────────────────────────────────────────────────────────────────
# DataGuide 템플릿 파일 생성
# (한 번 DataGuide에서 설정 후 자동 Refresh 가능한 파일 생성 가이드)
# ─────────────────────────────────────────────────────────────────

def create_dataguide_template(
    tickers: list[str],
    save_path: str = r"D:\Dataguide\templates\dg_eps_template.xlsx",
):
    """
    DataGuide 시계열 EPS 조회를 위한 Excel 템플릿 생성
    이 파일을 DataGuide에서 열고 [시계열데이터] → Submit 하면
    이후 Python이 자동 Refresh + 추출 가능
    """
    if not XLWINGS_OK:
        print("xlwings 미설치 — 수동으로 DataGuide에서 데이터를 뽑아주세요.")
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    app = xw.App(visible=False, add_book=False)
    wb  = app.books.add()

    try:
        # 종목 리스트 시트
        ws_code = wb.sheets[0]
        ws_code.name = "종목리스트"
        ws_code.range("A1").value = "종목코드"
        ws_code.range("A2").options(transpose=True).value = tickers
        ws_code.range("B1").value = "※ DataGuide [시계열데이터] 버튼 클릭 후 이 목록으로 조회하세요"

        # EPS 출력 시트 (DataGuide가 여기에 데이터를 채울 것)
        ws_eps = wb.sheets.add(name="FWD_EPS")
        ws_eps.range("A1").value = "DataGuide 시계열데이터 출력 위치"

        # 주가 출력 시트
        ws_price = wb.sheets.add(name="PRICE")
        ws_price.range("A1").value = "DataGuide 주가 출력 위치"

        wb.save(save_path)
        print(f"\n✅ DataGuide 템플릿 생성: {save_path}")
        print("\n📌 다음 단계:")
        print("  1. Excel에서 위 파일 열기")
        print("  2. DataGuide 탭 → [시계열데이터] 클릭")
        print("  3. 항목: '12개월 선행 EPS' (또는 FWD EPS 12M)")
        print("     종목코드: A 시트의 목록 복사·붙여넣기")
        print("     기간: 2020-01 ~ 현재, 월별")
        print("     출력 위치: FWD_EPS 시트 A1")
        print("  4. Submit → 저장")
        print("  5. 이후 Python이 자동으로 이 파일을 Refresh + 추출합니다")
    finally:
        app.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataGuideEPSFetcher()
    info = fetcher.detect_dataguide_addin()
    print("DataGuide Add-in 정보:", info)
