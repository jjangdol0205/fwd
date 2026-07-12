"""
make_dg_codelist.py
-------------------
DataGuide 입력용 종목코드 리스트 생성
DataGuide 형식: A005930, A000660, ... (A + 6자리)
"""
import pandas as pd
from pathlib import Path

# 유니버스 로드
universe = pd.read_csv(r"D:\Dataguide\data\universe.csv", dtype=str)

# DataGuide 형식으로 변환: 005930 → A005930
universe["dg_code"] = "A" + universe["ticker"].str.zfill(6)

# KOSPI200 / KOSDAQ150 분리
kospi200  = universe[universe["market"] == "KOSPI200"]["dg_code"].tolist()
kosdaq150 = universe[universe["market"] == "KOSDAQ150"]["dg_code"].tolist()
all_codes = universe["dg_code"].tolist()

# 결과 저장
out_dir = Path(r"D:\Dataguide\data")

# 전체 코드 (DataGuide 붙여넣기용, 줄바꿈 구분)
with open(out_dir / "dg_codes_all.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_codes))

# 쉼표 구분 (일부 DataGuide 버전용)
with open(out_dir / "dg_codes_comma.txt", "w", encoding="utf-8") as f:
    f.write(",".join(all_codes))

# 코드+이름 Excel (DataGuide 확인용)
universe[["dg_code", "name", "market"]].rename(
    columns={"dg_code": "DataGuide코드", "name": "종목명", "market": "지수"}
).to_excel(out_dir / "dg_codes_named.xlsx", index=False)

print(f"[완료] 총 {len(all_codes)}개 코드 생성")
print(f"  KOSPI200:  {len(kospi200)}개")
print(f"  KOSDAQ150: {len(kosdaq150)}개")
print(f"\n샘플 (처음 5개): {all_codes[:5]}")
print(f"\n[저장 파일]")
print(f"  줄바꿈: {out_dir}/dg_codes_all.txt     ← DataGuide에 붙여넣기")
print(f"  쉼표:   {out_dir}/dg_codes_comma.txt")
print(f"  Excel:  {out_dir}/dg_codes_named.xlsx")
