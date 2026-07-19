"""
parse_dataguide_output.py
-------------------------
DataGuide 시계열 출력 Excel 파싱
DataGuide 고유 포맷: 상단 8행 메타데이터 + 코드/날짜 매트릭스
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 데이터가 아닌 메타 행의 첫 번째 셀 패턴
_META_KEYWORDS = {
    "코드", "코드명", "유형", "아이템코드", "아이템명", "최근주기",
    "refresh", "달력기준", "코드 포트폴리오", "아이템 포트폴리오",
    "출력주기", "비영업일", "주말포함", "기간",
}


def parse_dataguide_timeseries(path: str, sheet_idx: int = 1) -> pd.DataFrame:
    """
    DataGuide 시계열데이터 출력 Excel 파싱

    DataGuide 출력 포맷 (Sheet2 기준):
      Row 0:  Refresh / Last Updated
      Row 1:  달력기준
      Row 2:  코드 포트폴리오
      Row 3:  아이템 포트폴리오
      Row 4:  출력주기 / 일간 또는 월별 / 원화
      Row 5:  비영업일
      Row 6:  주말포함
      Row 7:  기간 / 시작일 / 종료일
      Row 8:  코드   / A005930 / A000660 / ...
      Row 9:  코드명 / 삼성전자 / SK하이닉스 / ...
      Row 10: 유형   / CON / CON / ...
      Row 11: 아이템코드 / FM30041100 / ...
      Row 12: 아이템명 / ...
      Row 13: 최근주기 / ...  (있을 수도 없을 수도 있음)
      Row 14+: 날짜 / 값 / ...

    반환:
      DataFrame: index=날짜(DatetimeIndex), columns=종목코드(6자리)
    """
    df_raw = pd.read_excel(path, sheet_name=sheet_idx, header=None)

    # ── 메타데이터 읽기 ─────────────────────
    meta = {}
    for i in range(min(10, len(df_raw))):
        row = df_raw.iloc[i].dropna().tolist()
        if row:
            meta[str(row[0])] = row[1:]

    logger.info("[DataGuide] 메타 키: %s", list(meta.keys())[:5])

    # ── 코드 헤더 행 찾기 ───────────────────
    # "코드" 텍스트가 있는 행
    header_row = None
    for i in range(min(20, len(df_raw))):
        first_val = str(df_raw.iloc[i, 0]).strip()
        if first_val == "코드":
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"DataGuide 코드 헤더 행을 찾지 못했습니다. "
            f"첫 번째 열 상위 값: {[str(df_raw.iloc[j,0]) for j in range(min(15,len(df_raw)))]}"
        )

    # 종목코드 추출 (A005930 → 005930)
    code_row = df_raw.iloc[header_row]
    tickers = []
    for v in code_row.iloc[1:]:
        if pd.notna(v):
            code = str(v).strip()
            if code.startswith("A") and code[1:].isdigit():
                code = code[1:]
            tickers.append(code.zfill(6))

    # ── 데이터 시작 행 탐색 ─────────────────
    # 코드 헤더 다음부터 날짜처럼 파싱되는 첫 행을 찾는다
    # (메타 키워드 행은 모두 건너뜀)
    data_start = header_row + 1
    for i in range(header_row + 1, min(header_row + 15, len(df_raw))):
        first_val = str(df_raw.iloc[i, 0]).strip()
        # 메타 키워드면 스킵
        if first_val.lower() in _META_KEYWORDS or first_val in _META_KEYWORDS:
            data_start = i + 1
            continue
        # 날짜로 파싱 가능하면 → 데이터 시작
        try:
            parsed = pd.to_datetime(first_val, errors="raise")
            data_start = i
            break
        except Exception:
            data_start = i + 1  # 파싱 안 되면 스킵

    logger.info("[파싱] 코드 헤더: row %d | 데이터 시작: row %d", header_row, data_start)
    logger.info("  종목: %d개 | 샘플: %s", len(tickers), tickers[:5])

    # ── 데이터 추출 ─────────────────────────
    data_df = df_raw.iloc[data_start:].copy()
    data_df = data_df.dropna(how="all")

    # 날짜 열 (첫 번째 열)
    dates = pd.to_datetime(data_df.iloc[:, 0], errors="coerce")
    valid = dates.notna()
    dates   = dates[valid]
    data_df = data_df[valid]

    # 값 열 (2번째부터)
    values = data_df.iloc[:, 1: len(tickers) + 1].copy()
    values = values.apply(pd.to_numeric, errors="coerce")
    values.columns = tickers[: len(values.columns)]
    values.index = dates

    # 월별 리샘플링 (일간 → 월말 기준)
    freq_val = ""
    for key in ("출력주기", "주기"):
        if key in meta and meta[key]:
            freq_val = str(meta[key][0])
            break

    if "일" in freq_val or "day" in freq_val.lower():
        logger.info("  [리샘플] 일간 → 월별 변환")
        values = values.resample("ME").last()

    values.index.name = "date"
    values = values.sort_index()

    logger.info("[완료] shape: %s | 날짜: %s ~ %s",
                values.shape,
                values.index.min().date() if len(values) > 0 else "?",
                values.index.max().date() if len(values) > 0 else "?")

    return values


def load_dataguide_eps(path: str) -> pd.DataFrame:
    """
    DataGuide fwd_eps.xlsx 자동 로드
    Sheet1 = 코드 목록, Sheet2 = 실제 시계열 데이터
    """
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names

    # DataGuide 출력 시트 찾기 (Sheet2 또는 Refresh가 있는 시트)
    data_sheet_idx = 1  # 기본값 Sheet2
    for i, sh in enumerate(sheets):
        df_check = pd.read_excel(path, sheet_name=i, header=None, nrows=2)
        if "Refresh" in str(df_check.iloc[0, 0] if len(df_check) > 0 else ""):
            data_sheet_idx = i
            break

    return parse_dataguide_timeseries(path, sheet_idx=data_sheet_idx)


def load_combined_dataguide(path: str):
    """
    DataGuide fwd_eps.xlsx (다중 아이템: EPS, 주가) 자동 로드
    반환: (eps_df, price_df)
    """
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names
    data_sheet_idx = 1
    for i, sh in enumerate(sheets):
        df_check = pd.read_excel(path, sheet_name=i, header=None, nrows=2)
        if "Refresh" in str(df_check.iloc[0, 0] if len(df_check) > 0 else ""):
            data_sheet_idx = i
            break
            
    df_raw = pd.read_excel(path, sheet_name=data_sheet_idx, header=None)
    
    header_row = -1
    for i in range(min(20, len(df_raw))):
        if str(df_raw.iloc[i, 0]).strip() == "코드":
            header_row = i
            break
            
    if header_row == -1:
        raise ValueError("코드 헤더를 찾을 수 없습니다.")
        
    code_row = df_raw.iloc[header_row]
    item_row = None
    for i in range(header_row, min(header_row + 10, len(df_raw))):
        if str(df_raw.iloc[i, 0]).strip() in ["항목명", "아이템명"]:
            item_row = df_raw.iloc[i]
            break
            
    data_start = header_row + 1
    for i in range(header_row + 1, min(header_row + 15, len(df_raw))):
        val = str(df_raw.iloc[i, 0]).strip()
        try:
            pd.to_datetime(val, errors="raise")
            data_start = i
            break
        except Exception:
            data_start = i + 1
            
    data_df = df_raw.iloc[data_start:].copy().dropna(how="all")
    dates = pd.to_datetime(data_df.iloc[:, 0], errors="coerce")
    valid = dates.notna()
    dates = dates[valid]
    data_df = data_df[valid]
    
    eps_cols = []
    price_cols = []
    
    for idx in range(1, len(code_row)):
        code = str(code_row.iloc[idx]).strip()
        if pd.isna(code_row.iloc[idx]) or not code or code == "nan":
            continue
        if code.startswith("A") and code[1:].isdigit():
            code = code[1:]
        code = code.zfill(6)
        
        item = str(item_row.iloc[idx]).strip() if item_row is not None else ""
        
        if "EPS" in item.upper() or "EARNING" in item.upper():
            eps_cols.append((idx, code))
        else:
            price_cols.append((idx, code))
            
    eps_values = data_df.iloc[:, [x[0] for x in eps_cols]].copy()
    eps_values.columns = [x[1] for x in eps_cols]
    eps_values.index = dates
    eps_values = eps_values.apply(pd.to_numeric, errors="coerce")
    
    price_values = data_df.iloc[:, [x[0] for x in price_cols]].copy()
    price_values.columns = [x[1] for x in price_cols]
    price_values.index = dates
    price_values = price_values.apply(pd.to_numeric, errors="coerce")
    
    eps_values = eps_values.resample("ME").last()
    price_values = price_values.resample("ME").last()
    
    eps_values.index.name = "date"
    price_values.index.name = "date"
    
    return eps_values.sort_index(), price_values.sort_index()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Dataguide\data\fwd_eps.xlsx"
    eps = load_dataguide_eps(path)
    print(eps.tail(5).to_string())

    # 저장 (정규화된 형태로)
    out = path.replace(".xlsx", "_parsed.xlsx")
    eps.to_excel(out)
    print(f"\n저장: {out}")
