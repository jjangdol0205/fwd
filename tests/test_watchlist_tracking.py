"""
test_watchlist_tracking.py
---------------------------
주요 관심종목 (삼성전자, SK하이닉스, SK스퀘어, 알테오젠, HD현대일렉트릭)
Fwd EPS 트래킹 및 동적 관심종목 관리 기능 전용 단위/통합 테스트
"""

import os
import sys
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calculator import apply_quick_preset
from core.data_loader import _clean_ticker


class TestWatchlistTracking(unittest.TestCase):
    def setUp(self):
        self.watchlist_file = ROOT / "data" / "user_watchlist.json"
        self.original_content = None
        if self.watchlist_file.exists():
            try:
                self.original_content = self.watchlist_file.read_text(encoding="utf-8")
            except Exception:
                pass

    def tearDown(self):
        if self.original_content is not None:
            self.watchlist_file.write_text(self.original_content, encoding="utf-8")

    def test_01_default_five_stocks_watchlist(self):
        """기본 주요 관심종목 5개(삼성전자, SK하이닉스, SK스퀘어, 알테오젠, HD현대일렉트릭) 확인"""
        from app import DEFAULT_WATCHLIST, get_user_watchlist
        self.assertEqual(len(DEFAULT_WATCHLIST), 5)
        self.assertIn("005930", DEFAULT_WATCHLIST)  # 삼성전자
        self.assertIn("000660", DEFAULT_WATCHLIST)  # SK하이닉스
        self.assertIn("402340", DEFAULT_WATCHLIST)  # SK스퀘어
        self.assertIn("196170", DEFAULT_WATCHLIST)  # 알테오젠
        self.assertIn("267260", DEFAULT_WATCHLIST)  # HD현대일렉트릭

    def test_02_apply_quick_preset_default_watchlist(self):
        """apply_quick_preset('WATCHLIST') 실행 시 기본 5개 종목이 순서대로 필터링되는지 검증"""
        df = pd.DataFrame({
            "코드": ["005930", "000660", "035420", "402340", "196170", "267260", "005380"],
            "종목명": ["삼성전자", "SK하이닉스", "NAVER", "SK스퀘어", "알테오젠", "HD현대일렉트릭", "현대차"],
            "현재가": [262000, 1681000, 190000, 1050000, 241000, 606000, 250000],
        })
        res = apply_quick_preset(df, "WATCHLIST")
        self.assertEqual(len(res), 5)
        codes = res["코드"].tolist()
        self.assertEqual(codes, ["005930", "000660", "402340", "196170", "267260"])

    def test_03_apply_quick_preset_custom_tickers(self):
        """사용자가 커스텀 관심종목을 전달했을 때 해당 종목들만 정확히 반환되는지 검증"""
        df = pd.DataFrame({
            "ticker": ["A005930", "A035420", "A196170", "A000660"],
            "name": ["삼성전자", "NAVER", "알테오젠", "SK하이닉스"]
        })
        custom = ["196170", "005930"]
        res = apply_quick_preset(df, "WATCHLIST", custom_tickers=custom)
        self.assertEqual(len(res), 2)
        codes = [str(c).replace("A", "").zfill(6) for c in res["ticker"].tolist()]
        self.assertEqual(codes, ["196170", "005930"])

    def test_04_save_and_get_user_watchlist_persistence(self):
        """관심종목 저장 및 조회 시 디스크 영속성 및 중복제거 동작 검증"""
        from app import save_user_watchlist, get_user_watchlist
        import streamlit as st

        test_tickers = ["005930", "196170", "267260", "035420", "005930"]
        save_user_watchlist(test_tickers)

        loaded = get_user_watchlist()
        self.assertEqual(loaded, ["005930", "196170", "267260", "035420"])

    def test_05_curated_comments_coverage(self):
        """핵심 5개 종목에 대한 애널리스트 코멘트가 모두 준비되어 있는지 검증"""
        from app import render_watchlist_spotlight_deck
        import inspect

        source = inspect.getsource(render_watchlist_spotlight_deck)
        for t in ["005930", "000660", "402340", "196170", "267260"]:
            self.assertIn(t, source)


class TestWatchlistAppTest(unittest.TestCase):
    def test_headless_watchlist_interaction(self):
        """Streamlit AppTest를 통해 스포트라이트 덱 및 퀵픽 버튼 렌더링 검증"""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("AppTest not available")

        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"App execution failed: {at.exception}")

            # 퀵픽 버튼이 5개 이상 존재하는지 확인
            quick_picks = [b for b in at.button if "quick_pick_" in (getattr(b, "key", "") or "")]
            self.assertGreaterEqual(len(quick_picks), 5)

            # 삼성전자 및 알테오젠 퀵픽 버튼이 있는지 확인
            labels = [b.label for b in quick_picks]
            self.assertTrue(any("삼성전자" in lbl for lbl in labels))
            self.assertTrue(any("알테오젠" in lbl for lbl in labels))
            self.assertTrue(any("HD현대일렉트릭" in lbl for lbl in labels))
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)


if __name__ == "__main__":
    unittest.main()
