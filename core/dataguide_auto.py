"""
dataguide_auto.py
-----------------
xlwings를 이용한 DataGuide / Infomax Excel 자동화 모듈

사용 전제:
  - DataGuide 또는 Infomax Add-in이 설치된 Excel이 PC에 있어야 함
  - 한 번은 수동으로 쿼리 템플릿 파일을 만들어 저장해야 함
  - 이후에는 이 스크립트가 Refresh → 데이터 추출을 자동 수행
"""

import time
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
except ImportError:
    XLWINGS_AVAILABLE = False
    logger.warning("xlwings 미설치 — 자동화 기능 비활성화")


# ─────────────────────────────────────────
# DataGuide 자동화
# ─────────────────────────────────────────

class DataGuideAuto:
    """
    DataGuide Excel Add-in 자동화 클래스

    사용 예시:
        dg = DataGuideAuto(r"D:\\Dataguide\\templates\\fwd_eps.xlsx")
        df = dg.refresh_and_read(sheet_name="EPS")
    """

    # DataGuide 새로고침 관련 매크로 후보 이름들
    # (실제 이름은 Excel [개발도구 > 매크로]에서 확인)
    REFRESH_MACRO_CANDIDATES = [
        "DataGuide6.RefreshAll",
        "DataGuide.RefreshAll",
        "RefreshAll",
        "DG_Refresh",
        "Refresh",
    ]

    def __init__(self, template_path: str, wait_seconds: int = 8):
        """
        Parameters
        ----------
        template_path : str
            DataGuide 쿼리가 등록된 Excel 템플릿 파일 경로
        wait_seconds : int
            Refresh 후 데이터 로딩 대기 시간 (초)
        """
        self.template_path = Path(template_path)
        self.wait_seconds  = wait_seconds
        self._wb = None
        self._app = None

    def open(self):
        """Excel + 템플릿 파일 오픈"""
        if not XLWINGS_AVAILABLE:
            raise RuntimeError("xlwings가 설치되어 있지 않습니다.")
        self._app = xw.App(visible=True, add_book=False)
        self._wb  = self._app.books.open(str(self.template_path))
        logger.info(f"파일 열기 완료: {self.template_path}")

    def close(self):
        """Excel 닫기"""
        if self._wb:
            self._wb.close()
        if self._app:
            self._app.quit()

    def _try_refresh(self) -> bool:
        """DataGuide Refresh 매크로 실행 시도"""
        for macro_name in self.REFRESH_MACRO_CANDIDATES:
            try:
                self._app.macro(macro_name)()
                logger.info(f"Refresh 매크로 실행 성공: {macro_name}")
                return True
            except Exception:
                continue

        # 매크로 이름을 못 찾으면 Excel VBA로 직접 실행 시도
        try:
            self._wb.macro("RefreshAll")()
            return True
        except Exception:
            logger.warning("DataGuide Refresh 매크로를 찾지 못했습니다. 수동 Refresh 후 데이터를 저장하세요.")
            return False

    def refresh_and_read(
        self,
        sheet_name: str | int = 0,
        data_range: str | None = None,
    ) -> pd.DataFrame:
        """
        Refresh 실행 후 데이터 읽기

        Parameters
        ----------
        sheet_name : str | int
            읽을 시트 이름 또는 인덱스
        data_range : str | None
            읽을 셀 범위 (예: "A1:Z200"). None이면 자동 감지
        """
        self.open()
        try:
            self._try_refresh()
            time.sleep(self.wait_seconds)  # 데이터 로딩 대기

            sheet = self._wb.sheets[sheet_name]
            if data_range:
                rng = sheet.range(data_range)
            else:
                rng = sheet.used_range

            df = rng.options(pd.DataFrame, index=False, header=True).value
            logger.info(f"데이터 읽기 완료: {df.shape}")
            return df
        finally:
            self.close()


# ─────────────────────────────────────────
# Infomax 자동화
# ─────────────────────────────────────────

class InfomaxAuto:
    """
    Infomax Excel Add-in 자동화 클래스
    '수동 재조회' 버튼 트리거 후 데이터 추출
    """

    REFRESH_MACRO_CANDIDATES = [
        "Infomax.ManualRefresh",
        "IFX_Refresh",
        "ManualRefresh",
        "Refresh",
    ]

    def __init__(self, template_path: str, wait_seconds: int = 8):
        self.template_path = Path(template_path)
        self.wait_seconds  = wait_seconds
        self._wb = None
        self._app = None

    def open(self):
        if not XLWINGS_AVAILABLE:
            raise RuntimeError("xlwings가 설치되어 있지 않습니다.")
        self._app = xw.App(visible=True, add_book=False)
        self._wb  = self._app.books.open(str(self.template_path))

    def close(self):
        if self._wb:
            self._wb.close()
        if self._app:
            self._app.quit()

    def refresh_and_read(
        self,
        sheet_name: str | int = 0,
        data_range: str | None = None,
    ) -> pd.DataFrame:
        self.open()
        try:
            for macro_name in self.REFRESH_MACRO_CANDIDATES:
                try:
                    self._app.macro(macro_name)()
                    break
                except Exception:
                    continue

            time.sleep(self.wait_seconds)

            sheet = self._wb.sheets[sheet_name]
            rng   = sheet.range(data_range) if data_range else sheet.used_range
            df    = rng.options(pd.DataFrame, index=False, header=True).value
            return df
        finally:
            self.close()


# ─────────────────────────────────────────
# 매크로 이름 탐색 유틸
# ─────────────────────────────────────────

def list_macros(excel_path: str) -> list[str]:
    """
    Excel 파일에 등록된 매크로 이름 목록 반환
    DataGuide Refresh 매크로 이름 확인용
    """
    if not XLWINGS_AVAILABLE:
        return []

    app = xw.App(visible=False, add_book=False)
    wb  = app.books.open(excel_path)
    try:
        macros = []
        for component in wb.api.VBProject.VBComponents:
            for proc in range(1, component.CodeModule.CountOfProcedures + 1):
                name = component.CodeModule.ProcOfLine(proc, 0)
                if name and name not in macros:
                    macros.append(f"{component.Name}.{name}")
        return macros
    except Exception as e:
        logger.warning(f"매크로 목록 조회 실패: {e}")
        return []
    finally:
        wb.close()
        app.quit()
