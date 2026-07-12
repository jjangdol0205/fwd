"""
detect_dataguide.py
-------------------
DataGuide Excel Add-in의 COM 인터페이스 및 매크로 이름 자동 탐지
실행: python detect_dataguide.py
(Excel이 열려 있지 않아도 됩니다)
"""

import sys
import subprocess
import tempfile
import time
from pathlib import Path

DETECT_VBA = '''
Sub DetectAndReport()
    Dim fso As Object, f As Object
    Dim outPath As String
    outPath = "D:\\Dataguide\\data\\_dg_detect.txt"
    
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set f = fso.CreateTextFile(outPath, True, True)
    
    ' ===== 1. COM Add-in 목록 =====
    f.WriteLine "=== COM Add-ins ==="
    Dim addin As COMAddIn
    For Each addin In Application.COMAddIns
        f.WriteLine addin.ProgID & " | " & addin.Description & " | Connected=" & addin.Connect
    Next
    
    ' ===== 2. Excel Add-in 목록 =====
    f.WriteLine ""
    f.WriteLine "=== Excel Add-ins ==="
    Dim xaddin As AddIn
    For Each xaddin In Application.AddIns
        If xaddin.Installed Then
            f.WriteLine xaddin.Name & " | " & xaddin.FullName
        End If
    Next
    
    ' ===== 3. 열린 통합문서의 매크로 목록 =====
    f.WriteLine ""
    f.WriteLine "=== VBA Macros in Open Workbooks ==="
    Dim wb As Workbook
    Dim comp As Object
    For Each wb In Application.Workbooks
        f.WriteLine "WB: " & wb.Name
        On Error Resume Next
        For Each comp In wb.VBProject.VBComponents
            Dim i As Integer
            For i = 1 To comp.CodeModule.CountOfProcedures
                Dim procName As String
                procName = comp.CodeModule.ProcOfLine(i, 0)
                If procName <> "" Then
                    f.WriteLine "  " & comp.Name & "." & procName
                End If
            Next i
        Next comp
        On Error GoTo 0
    Next wb
    
    ' ===== 4. DataGuide 특화 탐지 =====
    f.WriteLine ""
    f.WriteLine "=== DataGuide Detection ==="
    Dim dgObj As Object
    
    Dim progids(5) As String
    progids(0) = "DataGuide6.AddIn"
    progids(1) = "DataGuide.AddIn"
    progids(2) = "DG6.Connect"
    progids(3) = "FnGuide.DataGuide"
    progids(4) = "DataGuidePro.AddIn"
    progids(5) = "DataGuide6"
    
    Dim pid As String
    For Each pid In progids
        On Error Resume Next
        Set dgObj = Application.COMAddIns(pid).Object
        If Not dgObj Is Nothing Then
            f.WriteLine "FOUND COM Object: " & pid
            ' 메서드 목록 시도
            On Error Resume Next
            f.WriteLine "  Type: " & TypeName(dgObj)
        End If
        On Error GoTo 0
        Set dgObj = Nothing
    Next
    
    f.Close
    MsgBox "탐지 완료! 결과: " & outPath, vbInformation
End Sub
'''

DETECT_XLSM_PATH = r"D:\Dataguide\data\_dg_detect.xlsm"
RESULT_PATH      = r"D:\Dataguide\data\_dg_detect.txt"


def run_detection():
    """
    Excel을 열어 DataGuide 탐지 매크로를 실행하고 결과를 읽음
    """
    try:
        import xlwings as xw
    except ImportError:
        print("[ERROR] xlwings 미설치: pip install xlwings")
        return

    print("[탐지 시작] DataGuide COM Add-in 분석 중...")

    # 임시 Excel 파일 생성 + VBA 삽입
    app = xw.App(visible=True, add_book=False)
    app.display_alerts = False
    wb  = app.books.add()

    try:
        # VBA 모듈 삽입
        vba_mod = wb.api.VBProject.VBComponents.Add(1)
        vba_mod.CodeModule.AddFromString(DETECT_VBA)

        # 매크로 파일로 저장
        wb.api.SaveAs(DETECT_XLSM_PATH, 52)  # 52 = xlOpenXMLWorkbookMacroEnabled

        # 매크로 실행
        print("[실행] DetectAndReport 매크로 실행 중...")
        app.macro("DetectAndReport")()
        time.sleep(3)

        # 결과 읽기
        if Path(RESULT_PATH).exists():
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            print("\n" + "="*60)
            print(content)
            print("="*60)
            return content
        else:
            print(f"[WARNING] 결과 파일 없음: {RESULT_PATH}")
            return None

    except Exception as e:
        print(f"[ERROR] 탐지 실패: {e}")
        return None
    finally:
        try:
            wb.close()
            app.quit()
        except:
            pass


def parse_result(content: str) -> dict:
    """탐지 결과 파싱"""
    result = {
        "com_addins": [],
        "xl_addins":  [],
        "macros":     [],
        "dg_found":   [],
    }
    if not content:
        return result

    section = None
    for line in content.splitlines():
        if "=== COM Add-ins ===" in line:
            section = "com"
        elif "=== Excel Add-ins ===" in line:
            section = "xl"
        elif "=== VBA Macros" in line:
            section = "macro"
        elif "=== DataGuide Detection ===" in line:
            section = "dg"
        elif line.strip():
            if section == "com":
                result["com_addins"].append(line.strip())
            elif section == "xl":
                result["xl_addins"].append(line.strip())
            elif section == "macro":
                result["macros"].append(line.strip())
            elif section == "dg":
                result["dg_found"].append(line.strip())

    return result


if __name__ == "__main__":
    content = run_detection()
    if content:
        info = parse_result(content)
        print("\n[요약]")
        print(f"  COM Add-ins: {len(info['com_addins'])}개")
        print(f"  DataGuide 감지: {info['dg_found']}")

        # 결과를 JSON으로 저장
        import json
        with open(r"D:\Dataguide\data\_dg_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] D:\\Dataguide\\data\\_dg_info.json")
