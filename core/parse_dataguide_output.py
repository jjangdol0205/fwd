"""
parse_dataguide_output.py
-------------------------
DataGuide 시계열 출력 Excel 파싱
DataGuide 고유 포맷: 상단 8행 메타데이터 + 코드/날짜 매트릭스
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
      Row 12+: 날짜 / 값 / 값 / ...

    반환:
      DataFrame: index=날짜(DatetimeIndex), columns=종목코드(6자리)
    """
    df_raw = pd.read_excel(path, sheet_name=sheet_idx, header=None)

    # ── 메타데이터 읽기 ─────────────────────
    meta = {}
    for i in range(8):
        row = df_raw.iloc[i].dropna().tolist()
        if row:
            meta[row[0]] = row[1:]

    print("[DataGuide 메타데이터]")
    for k, v in meta.items():
        print(f"  {k}: {v[:3]}{'...' if len(v) > 3 else ''}")

    # ── 코드 헤더 행 찾기 ───────────────────
    # Row 8: "코드" + A005930, A000660, ...
    header_row = None
    for i in range(min(15, len(df_raw))):
        first_val = str(df_raw.iloc[i, 0]).strip()
        if first_val == "코드":
            header_row = i
            break

    if header_row is None:
        raise ValueError("DataGuide 코드 헤더 행을 찾지 못했습니다.")

    # 종목코드 추출 (A005930 → 005930)
    code_row  = df_raw.iloc[header_row]
    tickers   = []
    for v in code_row.iloc[1:]:
        if pd.notna(v):
            code = str(v).strip()
            # A prefix 제거
            if code.startswith("A") and code[1:].isdigit():
                code = code[1:]
            tickers.append(code.zfill(6))

    # 데이터 시작 행 (코드명, 유형, 아이템코드 행 스킵)
    data_start = header_row + 1
    for i in range(header_row + 1, header_row + 6):
        if i >= len(df_raw):
            break
        first_val = str(df_raw.iloc[i, 0]).strip()
        if first_val in ("코드명", "유형", "아이템코드", "아이템명"):
            data_start = i + 1
        else:
            break

    print(f"\n[파싱] 코드 헤더: row {header_row} | 데이터 시작: row {data_start}")
    print(f"  종목: {len(tickers)}개 | 샘플: {tickers[:5]}")

    # ── 데이터 추출 ─────────────────────────
    data_df = df_raw.iloc[data_start:].copy()
    data_df = data_df.dropna(how="all")

    # 날짜 열 (첫 번째 열)
    dates = pd.to_datetime(data_df.iloc[:, 0], errors="coerce")
    valid = dates.notna()
    dates    = dates[valid]
    data_df  = data_df[valid]

    # 값 열 (2번째부터)
    values = data_df.iloc[:, 1: len(tickers) + 1]
    values = values.apply(pd.to_numeric, errors="coerce")
    values.columns = tickers[:len(values.columns)]
    values.index   = dates

    # 월별 리샘플링 (일간 → 월말 기준)
    freq = meta.get("출력주기", [""])[0] if meta.get("출력주기") else "?"
    print(f"  출력주기: {freq}")

    if "일" in str(freq) or "day" in str(freq).lower():
        print("  [리샘플] 일간 → 월별 변환 중...")
        values = values.resample("ME").last()

    values.index.name = "date"
    values = values.sort_index()

    print(f"\n[완료] shape: {values.shape}")
    print(f"  날짜 범위: {values.index.min().date()} ~ {values.index.max().date()}")
    print(f"  유효 값: {values.notna().sum().sum():,}개")

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


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Dataguide\data\fwd_eps.xlsx"
    eps = load_dataguide_eps(path)
    print(eps.tail(5).to_string())

    # 저장 (정규화된 형태로)
    out = path.replace(".xlsx", "_parsed.xlsx")
    eps.to_excel(out)
    print(f"\n저장: {out}")
