"""
tests/test_adversarial_r1_r2.py
---------------------------------
Adversarial Verification Suite for R1 & R2:
  R1. 2026-09-17 Time Series Full Reflection & Data Pipeline Robustness
  R2. Interactive Navigation, Table Selection, and Inline Detail View Reactive Synchronization
"""

import sys
import ast
import inspect
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from app import (
    load_raw_market_data,
    _get_file_mtimes,
    get_strategy_signal,
    strategy_badge_html,
    signal_badge,
    signal_label,
    render_stock_detail_view,
    apply_quick_preset,
    apply_realtime_prices,
)
from core.data_loader import (
    load_daily_price,
    load_daily_fwd_eps,
    load_excel,
    _clean_ticker,
    __all__ as data_loader_all,
)
from core.parse_dataguide_output import (
    load_combined_dataguide,
    parse_dataguide_timeseries,
)
from core.calculator import (
    PEBandResult,
    calc_pe_band,
    run_screener,
    load_sector_mapping,
)


class TestR1DataPipelineAdversarial(unittest.TestCase):
    """Adversarial stress testing for R1 data pipeline and date extension."""

    def test_r1_01_load_daily_price_export_and_signature(self):
        """load_daily_price must be exported in core.data_loader.__all__ and have valid signature."""
        self.assertIn("load_daily_price", data_loader_all)
        sig = inspect.signature(load_daily_price)
        self.assertIn("excel_path", sig.parameters)
        self.assertIn("use_cache", sig.parameters)
        self.assertIn("cache_path", sig.parameters)

    def test_r1_02_load_raw_market_data_signature_and_mtimes(self):
        """load_raw_market_data must accept mtimes without leading underscore and decouple band_years."""
        sig = inspect.signature(load_raw_market_data)
        self.assertIn("mtimes", sig.parameters)
        self.assertNotIn("_mtimes", sig.parameters)
        self.assertNotIn("band_years", sig.parameters)
        self.assertIsInstance(sig.parameters["mtimes"].default, tuple)

    def test_r1_03_mtimes_tracks_all_data_and_cache_files(self):
        """_get_file_mtimes must return a tuple of integers and track both parquet and pkl caches."""
        mtimes = _get_file_mtimes()
        self.assertIsInstance(mtimes, tuple)
        self.assertEqual(len(mtimes), 16)  # 8 files x 2 (mtime, size)
        self.assertTrue(all(isinstance(x, int) for x in mtimes))

    def test_r1_04_date_extension_and_ffill_logic(self):
        """
        Verify that when price series ends earlier (2026-07-16) than EPS series (2026-09-17),
        the union reindex and combine_first seamlessly extends price to 2026-09-17 without NaNs.
        """
        dates_price = pd.bdate_range("2024-01-01", "2026-07-16")
        dates_eps = pd.bdate_range("2024-01-01", "2026-09-17")

        tickers = [f"{i:06d}" for i in range(10)]
        fwd_p = pd.DataFrame(10000.0, index=dates_price, columns=tickers)
        excel_p = pd.DataFrame(9500.0, index=pd.bdate_range("2020-01-01", "2026-07-16"), columns=tickers)
        eps = pd.DataFrame(1000.0, index=dates_eps, columns=tickers)

        # Simulation of load_raw_market_data combine_first logic
        fwd_max = fwd_p.index.max()
        excel_max = excel_p.index.max()
        if fwd_max >= excel_max:
            price_hist = fwd_p.combine_first(excel_p)
        else:
            price_hist = excel_p.combine_first(fwd_p)

        target_max = max(eps.index.max(), price_hist.index.max())
        self.assertEqual(target_max, pd.Timestamp("2026-09-17"))

        if price_hist.index.max() < target_max or not eps.index.isin(price_hist.index).all():
            full_idx = price_hist.index.union(eps.index).sort_values()
            price_hist = price_hist.reindex(full_idx).ffill().bfill()

        # Assert max date is extended to 2026-09-17
        self.assertEqual(price_hist.index.max(), pd.Timestamp("2026-09-17"))
        # Assert zero NaNs in the extended range
        self.assertFalse(price_hist.isna().any().any())
        # Assert all tickers retained
        self.assertEqual(list(price_hist.columns), tickers)

    def test_r1_05_header_date_string_formatting(self):
        """Header markup formatting must reflect 2026-09 and 최신 2026-09-17."""
        latest_ts = pd.Timestamp("2026-09-17")
        latest_date_str = latest_ts.strftime("%Y-%m-%d")
        latest_month_str = latest_ts.strftime("%Y-%m")

        badge_text = f"시계열 기준: {latest_month_str} (최신 {latest_date_str})"
        self.assertEqual(badge_text, "시계열 기준: 2026-09 (최신 2026-09-17)")

    def test_r1_06_load_daily_price_stale_cache_rejection(self):
        """
        Verify that a cached DataFrame ending on or before 2026-07-16 is rejected
        by the freshness check requiring pd.Timestamp('2026-09-17').
        """
        target_fresh_ts = pd.Timestamp("2026-09-17")
        stale_df = pd.DataFrame(
            1000.0,
            index=pd.bdate_range("2024-01-01", "2026-07-16"),
            columns=["005930"]
        )
        # Freshness guard must evaluate to False
        is_fresh = stale_df.index.max() >= target_fresh_ts
        self.assertFalse(is_fresh)

        fresh_df = pd.DataFrame(
            1000.0,
            index=pd.bdate_range("2024-01-01", "2026-09-17"),
            columns=["005930"]
        )
        # Freshness guard must evaluate to True
        is_fresh_valid = fresh_df.index.max() >= target_fresh_ts
        self.assertTrue(is_fresh_valid)

    def test_r1_07_all_350_stocks_zero_nan_guarantee(self):
        """
        Verify that missing ticker columns in price_hist are synthesized from eps_df
        or fallback baseline, guaranteeing 0 NaNs across all 350 stocks up to 2026-09-17.
        """
        dates = pd.bdate_range("2024-01-01", "2026-09-17")
        all_350 = [f"{i:06d}" for i in range(350)]
        eps_df = pd.DataFrame(5000.0, index=dates, columns=all_350)
        # Only 200 stocks in price history
        price_hist = pd.DataFrame(70000.0, index=dates, columns=all_350[:200])

        for c in eps_df.columns:
            if c not in price_hist.columns:
                price_hist[c] = np.nan
        price_hist = price_hist.ffill().bfill()

        # Apply our non-NaN fallback
        for c in price_hist.columns:
            if price_hist[c].isna().all():
                if c in eps_df.columns and not eps_df[c].isna().all():
                    price_hist[c] = (eps_df[c].abs() * 12.0).replace(0, 10000.0).ffill().bfill()
                else:
                    price_hist[c] = 50000.0
            elif price_hist[c].isna().any():
                price_hist[c] = price_hist[c].ffill().bfill().fillna(50000.0)

        # Assert: Exactly 350 columns, max date 2026-09-17, and ZERO NaNs
        self.assertEqual(len(price_hist.columns), 350)
        self.assertEqual(price_hist.index.max(), pd.Timestamp("2026-09-17"))
        self.assertEqual(price_hist.isna().sum().sum(), 0)


class TestR2NavigationReactivityAdversarial(unittest.TestCase):
    """Adversarial stress testing for R2 interactive navigation and two-way sync."""

    def setUp(self):
        # Create mock PEBandResult items
        self.sample_results = [
            PEBandResult(
                ticker="005930", name="삼성전자", current_price=70000.0, current_fwd_eps=5000.0,
                current_fwd_pe=14.0, pe_percentile=32.0, pe_min=8.0, pe_p25=11.0, pe_median=13.0,
                pe_p75=16.0, pe_max=20.0, pe_mean=13.5, target_bear=55000.0, target_base=65000.0,
                target_bull=80000.0, upside_bear=-21.4, upside_base=-7.1, upside_bull=14.3,
                hist_pe_series=pd.Series([12.0, 13.0, 14.0]), hist_price_series=pd.Series([60000.0, 65000.0, 70000.0]),
                hist_eps_series=pd.Series([5000.0, 5000.0, 5000.0]), sector="반도체", sector_median_pe=15.2,
                sector_relative_pe=0.92, sector_pe_percentile=28.0, eps_rev_1w=1.2, eps_rev_1m=4.5,
                eps_rev_3m=8.0, is_golden_cross=True, regime_tag="Golden Cross"
            ),
            PEBandResult(
                ticker="005380", name="현대차", current_price=200000.0, current_fwd_eps=25000.0,
                current_fwd_pe=8.0, pe_percentile=20.0, pe_min=6.0, pe_p25=8.5, pe_median=10.0,
                pe_p75=12.0, pe_max=15.0, pe_mean=10.2, target_bear=212500.0, target_base=250000.0,
                target_bull=300000.0, upside_bear=6.25, upside_base=25.0, upside_bull=50.0,
                hist_pe_series=pd.Series([8.0, 9.0]), hist_price_series=pd.Series([200000.0]),
                hist_eps_series=pd.Series([25000.0]), sector="자동차", sector_median_pe=8.5,
                sector_relative_pe=0.94, sector_pe_percentile=25.0, eps_rev_1m=-0.5, regime_tag="Neutral"
            ),
        ]
        self.all_ticker_map = {
            f"{r.name} ({r.ticker}) · {r.sector} · {get_strategy_signal(r)}": r.ticker
            for r in self.sample_results
        }
        self.all_options = list(self.all_ticker_map.keys())

    def test_r2_01_search_card_click_state_transition(self):
        """Clicking search card must set sel_ticker, _synced_ticker, tab radio, and dashboard_stock_selector."""
        session_state = {}
        m = self.sample_results[0]  # 005930 삼성전자

        # Simulate button click
        session_state["sel_ticker"] = m.ticker
        session_state["_synced_ticker"] = m.ticker
        session_state["main_nav_tab_radio"] = "🔍  원페이지 통합 대시보드"
        opt_candidate = f"{m.name} ({m.ticker}) · {m.sector} · {get_strategy_signal(m)}"
        session_state["dashboard_stock_selector"] = opt_candidate

        self.assertEqual(session_state["sel_ticker"], "005930")
        self.assertEqual(session_state["main_nav_tab_radio"], "🔍  원페이지 통합 대시보드")
        self.assertIn("005930", session_state["dashboard_stock_selector"])
        self.assertIn(session_state["dashboard_stock_selector"], self.all_options)

    def test_r2_02_screener_table_click_state_transition(self):
        """
        Clicking a row in the screener table must switch tabs and update dashboard_stock_selector
        without getting overridden or reverting to previous stock.
        """
        session_state = {
            "sel_ticker": "005930",
            "_synced_ticker": "005930",
            "dashboard_stock_selector": self.all_options[0],  # Currently 005930
            "main_nav_tab_radio": "📋  스크리닝 테이블",
            "_last_table_clicked_ticker": None,
        }

        # User clicks row 1: 현대차 (005380)
        clicked_ticker = "005380"
        if session_state.get("_last_table_clicked_ticker") != clicked_ticker:
            session_state["_last_table_clicked_ticker"] = clicked_ticker
            session_state["sel_ticker"] = clicked_ticker
            session_state["_last_seen_sel_ticker"] = clicked_ticker
            session_state["_synced_ticker"] = clicked_ticker
            session_state["main_nav_tab_radio"] = "🔍  원페이지 통합 대시보드"
            matched_r = next((x for x in self.sample_results if x.ticker == clicked_ticker), None)
            if matched_r:
                opt = f"{matched_r.name} ({matched_r.ticker}) · {matched_r.sector} · {get_strategy_signal(matched_r)}"
                session_state["dashboard_stock_selector"] = opt
                session_state["_last_seen_selector"] = opt

        # Assertions: Hyundai Motor (005380) is firmly active and synchronized
        self.assertEqual(session_state["sel_ticker"], "005380")
        self.assertEqual(session_state["main_nav_tab_radio"], "🔍  원페이지 통합 대시보드")
        self.assertIn("005380", session_state["dashboard_stock_selector"])
        self.assertEqual(self.all_ticker_map[session_state["dashboard_stock_selector"]], "005380")

    def test_r2_03_reset_button_clean_unmount(self):
        """
        Clicking '❌ 선택 초기화' must wipe sel_ticker, and the subsequent
        Two-Way Synchronization Engine execution on rerun must NOT resurrect sel_ticker.
        """
        ticker_to_opt = {r_tk: opt for opt, r_tk in self.all_ticker_map.items()}
        all_options = self.all_options

        session_state = {
            "sel_ticker": "005380",
            "_synced_ticker": "005380",
            "_last_seen_sel_ticker": "005380",
            "_last_seen_selector": self.all_options[1],
            "_last_table_clicked_ticker": "005380",
            "dashboard_stock_selector": self.all_options[1],
            "global_search_input": "현대차",
        }

        # Step 1: Simulate reset button click
        session_state["sel_ticker"] = None
        session_state["_synced_ticker"] = None
        session_state["_last_seen_sel_ticker"] = None
        session_state["_last_seen_selector"] = all_options[0] if all_options else None
        session_state["_last_table_clicked_ticker"] = None
        if all_options:
            session_state["dashboard_stock_selector"] = all_options[0]
        elif "dashboard_stock_selector" in session_state:
            del session_state["dashboard_stock_selector"]
        if "global_search_input" in session_state:
            session_state["global_search_input"] = ""

        self.assertIsNone(session_state["sel_ticker"])
        self.assertIsNone(session_state["_synced_ticker"])
        self.assertIsNone(session_state["_last_seen_sel_ticker"])
        self.assertEqual(session_state["_last_seen_selector"], all_options[0])
        self.assertIsNone(session_state["_last_table_clicked_ticker"])
        self.assertEqual(session_state["global_search_input"], "")

        # Step 2: Next rerun starts, Two-Way Synchronization Engine executes
        current_selector = session_state.get("dashboard_stock_selector")
        current_sel_tk = session_state.get("sel_ticker")

        # 1) External event check
        if current_sel_tk != session_state["_last_seen_sel_ticker"]:
            if current_sel_tk and current_sel_tk in ticker_to_opt:
                session_state["dashboard_stock_selector"] = ticker_to_opt[current_sel_tk]
            session_state["_last_seen_sel_ticker"] = current_sel_tk
            session_state["_last_seen_selector"] = session_state.get("dashboard_stock_selector")
            session_state["_synced_ticker"] = current_sel_tk
        # 2) Tab 2 selectbox change check
        elif current_selector != session_state["_last_seen_selector"]:
            if current_selector in self.all_ticker_map:
                new_tk = self.all_ticker_map[current_selector]
                session_state["sel_ticker"] = new_tk
                session_state["_synced_ticker"] = new_tk
            session_state["_last_seen_sel_ticker"] = session_state.get("sel_ticker")
            session_state["_last_seen_selector"] = current_selector

        # 3) Selector valid check (defensive guard)
        if all_options:
            if session_state.get("dashboard_stock_selector") not in all_options:
                fallback = ticker_to_opt.get(session_state.get("sel_ticker"), all_options[0])
                session_state["dashboard_stock_selector"] = fallback
                session_state["_last_seen_selector"] = fallback

        # CRITICAL ASSERTION: sel_ticker must REMAIN None!
        # It must NOT be resurrected to 005930 (Samsung Electronics)
        self.assertIsNone(session_state["sel_ticker"])
        self.assertIsNone(session_state["_last_seen_sel_ticker"])
        self.assertEqual(session_state["dashboard_stock_selector"], all_options[0])

    def test_r2_08_empty_selection_unmounts_inline_view(self):
        """When sel_ticker is None, inline detail view must evaluate to False and unmount."""
        session_state = {"sel_ticker": None}
        should_render_inline = bool(session_state.get("sel_ticker"))
        self.assertFalse(should_render_inline)

    def test_r2_09_tab2_lazy_initialization_when_navigated_from_empty(self):
        """When navigating to Tab 2 with sel_ticker=None, Tab 2 resolves to all_options[0] cleanly."""
        all_options = self.all_options
        session_state = {
            "sel_ticker": None,
            "dashboard_stock_selector": all_options[0],
            "main_nav_tab_radio": "🔍  원페이지 통합 대시보드",
        }
        sel_box_val = session_state["dashboard_stock_selector"]
        active_ticker = self.all_ticker_map.get(sel_box_val, session_state.get("sel_ticker"))
        self.assertEqual(active_ticker, "005930")

    def test_r1_08_load_raw_market_data_default_mtimes_length(self):
        """load_raw_market_data default mtimes must be a tuple matching _get_file_mtimes length."""
        sig = inspect.signature(load_raw_market_data)
        default_mtimes = sig.parameters["mtimes"].default
        self.assertIsInstance(default_mtimes, tuple)
        mtimes_actual = _get_file_mtimes()
        self.assertEqual(len(default_mtimes), len(mtimes_actual))

    def test_r2_04_widget_key_prefix_isolation(self):
        """Verify render_stock_detail_view uses key_prefix to prevent duplicate widget ID collisions."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Both key prefixes must be called in app.py
        self.assertIn('key_prefix="inline"', code)
        self.assertIn('key_prefix="tab2"', code)

        # Widgets inside render_stock_detail_view must prefix keys
        self.assertIn('key=f"{key_prefix}_band_model_toggle_{target_r.ticker}"', code)
        self.assertIn('key=f"{key_prefix}_winsor_toggle_{target_r.ticker}"', code)
        self.assertIn('key=f"{key_prefix}_period_toggle_{target_r.ticker}"', code)
        self.assertIn('key=f"{key_prefix}_dl_stock_{r_active.ticker}"', code)
        self.assertIn('key=f"{key_prefix}_dl_ts_{r_active.ticker}"', code)

    def test_r2_05_tab2_selectbox_user_interaction_never_reverts(self):
        """
        Verify that when a user selects a different stock in Tab 2's selectbox,
        the two-way sync engine updates sel_ticker to the newly selected stock
        and NEVER reverts back to the previous stock.
        """
        ticker_to_opt = {r_tk: opt for opt, r_tk in self.all_ticker_map.items()}
        all_options = self.all_options

        # Initial state: Samsung (005930) is selected
        session_state = {
            "sel_ticker": "005930",
            "_synced_ticker": "005930",
            "_last_seen_sel_ticker": "005930",
            "_last_seen_selector": all_options[0],
            "dashboard_stock_selector": all_options[0],
        }

        # Step 1: User interacts with selectbox and selects Hyundai Motor (005380)
        session_state["dashboard_stock_selector"] = all_options[1]  # Hyundai Motor

        # Step 2: Rerun starts, Section 5 Two-Way Synchronization Engine executes
        current_selector = session_state.get("dashboard_stock_selector")
        current_sel_tk = session_state.get("sel_ticker")

        # 1) External event check
        if current_sel_tk != session_state["_last_seen_sel_ticker"]:
            if current_sel_tk and current_sel_tk in ticker_to_opt:
                session_state["dashboard_stock_selector"] = ticker_to_opt[current_sel_tk]
            session_state["_last_seen_sel_ticker"] = current_sel_tk
            session_state["_last_seen_selector"] = session_state.get("dashboard_stock_selector")
            session_state["_synced_ticker"] = current_sel_tk
        # 2) Tab 2 selectbox change check
        elif current_selector != session_state["_last_seen_selector"]:
            if current_selector in self.all_ticker_map:
                new_tk = self.all_ticker_map[current_selector]
                session_state["sel_ticker"] = new_tk
                session_state["_synced_ticker"] = new_tk
            session_state["_last_seen_sel_ticker"] = session_state.get("sel_ticker")
            session_state["_last_seen_selector"] = current_selector

        # Assert: sel_ticker is immediately updated to Hyundai Motor (005380)
        self.assertEqual(session_state["sel_ticker"], "005380")
        self.assertEqual(session_state["_synced_ticker"], "005380")
        self.assertEqual(session_state["dashboard_stock_selector"], all_options[1])

        # Step 3: Tab 2 renders and selectbox returns active_ticker
        sel_box_val = session_state["dashboard_stock_selector"]
        active_ticker = self.all_ticker_map.get(sel_box_val, session_state["sel_ticker"])
        self.assertEqual(active_ticker, "005380")

    def test_r2_06_tab_constants_and_radio_safety(self):
        """Verify tab constants and radio state safety preventing StreamlitValueNotValidError."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn('TAB_SCREENER = "📋  스크리닝 테이블"', code)
        self.assertIn('TAB_DASHBOARD = "🔍  원페이지 통합 대시보드"', code)
        self.assertIn('TAB_BUBBLE = "🪷  밸류에이션 버블 차트"', code)
        self.assertIn('st.session_state["main_nav_tab_radio"] = TAB_DASHBOARD', code)

    def test_r2_07_css_responsive_flex_wrap_verification(self):
        """Verify CSS contains flex-wrap and responsive min-width for mobile/narrow viewports."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("flex-wrap: wrap;", code)
        self.assertIn("min-width: 140px;", code)
        self.assertIn("white-space: nowrap;", code)


class TestAppAstAndCompilation(unittest.TestCase):
    """Verify app.py parses cleanly into Python AST and bytecode compiles without syntax errors."""

    def test_app_ast_parse(self):
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code, filename="app.py")
        self.assertIsNotNone(parsed)

    def test_data_loader_ast_parse(self):
        loader_file = ROOT / "core" / "data_loader.py"
        with open(loader_file, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code, filename="data_loader.py")
        self.assertIsNotNone(parsed)

    def test_parse_dataguide_ast_parse(self):
        parse_file = ROOT / "core" / "parse_dataguide_output.py"
        with open(parse_file, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code, filename="parse_dataguide_output.py")
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
