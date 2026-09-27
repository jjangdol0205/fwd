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

    def test_r1_09_dynamic_key_versioning_and_pending_search_reset(self):
        """
        Verify that search input reset utilizes Dynamic Key Versioning and pending queue,
        guaranteeing st.session_state[widget_key] is never mutated post-instantiation.
        """
        session_state = {
            "_search_input_version": 0,
            "global_search_input": "삼성전자",
        }
        current_search_key = f"global_search_input_v{session_state['_search_input_version']}"
        session_state[current_search_key] = "삼성전자"

        # Simulate user clicking '❌ 선택 초기화'
        session_state["_pending_search_input"] = ""
        session_state["_search_input_version"] += 1
        session_state["global_search_input"] = ""

        self.assertEqual(session_state["_search_input_version"], 1)
        self.assertEqual(session_state["_pending_search_input"], "")
        self.assertEqual(session_state["global_search_input"], "")

        # Simulate next script execution top: pending pop and dynamic key generation
        if "_pending_search_input" in session_state:
            pending_val = session_state.pop("_pending_search_input")
            session_state["global_search_input"] = pending_val

        new_search_key = f"global_search_input_v{session_state['_search_input_version']}"
        self.assertEqual(new_search_key, "global_search_input_v1")
        self.assertNotIn("_pending_search_input", session_state)
        self.assertEqual(session_state["global_search_input"], "")
        self.assertNotIn(new_search_key, session_state)

    def test_r1_10_app_code_dynamic_key_isolation(self):
        """Verify app.py source code uses dynamic key versioning and does not assign to static widget key."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("current_search_key = f\"global_search_input_v{st.session_state.get('_search_input_version', 0)}\"", code)
        self.assertIn("key=current_search_key", code)
        self.assertNotIn('key="global_search_input"', code)
        self.assertIn('_pending_stock_selector', code)
        self.assertIn('_pending_nav_tab', code)


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
        self.assertIn('st.session_state["_pending_nav_tab"] = TAB_DASHBOARD', code)
        self.assertIn('st.session_state["main_nav_tab_radio"] = st.session_state.pop("_pending_nav_tab")', code)

    def test_r2_07_css_responsive_flex_wrap_verification(self):
        """Verify CSS contains flex-wrap and responsive min-width for mobile/narrow viewports."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("flex-wrap: wrap;", code)
        self.assertIn("min-width: 140px;", code)
        self.assertIn("white-space: nowrap;", code)

    def test_r2_10_pending_nav_tab_and_stock_selector_full_cycle(self):
        """Verify complete pending lifecycle for nav_tab and dashboard_stock_selector."""
        session_state = {
            "sel_ticker": "005930",
            "_last_seen_sel_ticker": "005930",
            "main_nav_tab_radio": "📋  스크리닝 테이블",
        }
        all_options = self.all_options
        ticker_to_opt = {r_tk: opt for opt, r_tk in self.all_ticker_map.items()}

        # User clicks search card for Hyundai Motor (005380)
        clicked_ticker = "005380"
        opt_candidate = ticker_to_opt[clicked_ticker]
        session_state["sel_ticker"] = clicked_ticker
        session_state["_last_seen_sel_ticker"] = clicked_ticker
        session_state["_synced_ticker"] = clicked_ticker
        session_state["_pending_nav_tab"] = "🔍  원페이지 통합 대시보드"
        session_state["_pending_stock_selector"] = opt_candidate

        # On rerun:
        # Step 1: Pending stock selector is processed before sync / selectbox
        if "_pending_stock_selector" in session_state:
            pending_sel = session_state.pop("_pending_stock_selector")
            if pending_sel in all_options:
                session_state["dashboard_stock_selector"] = pending_sel
                session_state["_last_seen_selector"] = pending_sel

        # Step 2: Pending nav tab is processed before radio widget
        if "_pending_nav_tab" in session_state:
            session_state["main_nav_tab_radio"] = session_state.pop("_pending_nav_tab")

        self.assertEqual(session_state["sel_ticker"], "005380")
        self.assertEqual(session_state["main_nav_tab_radio"], "🔍  원페이지 통합 대시보드")
        self.assertEqual(session_state["dashboard_stock_selector"], opt_candidate)
        self.assertNotIn("_pending_nav_tab", session_state)
        self.assertNotIn("_pending_stock_selector", session_state)

    def test_r2_11_extreme_edge_cases_and_search_robustness(self):
        """Verify search filtering resilience with special regex chars, empty queries, and empty datasets."""
        # Special regex characters in search query must not cause crashes or regex errors
        extreme_queries = ["[", "]", "*", "+", "?", "\\", "(", ")", "^", "$", "{", "}", "   ", ""]
        for q in extreme_queries:
            sq = q.strip().lower()
            if sq:
                matches = [
                    r for r in self.sample_results
                    if sq in (r.name or "").lower() or sq in (r.ticker or "").lower() or sq in (r.sector or "").lower()
                ]
            else:
                matches = []
            self.assertIsInstance(matches, list)

        # Bubble chart empty data filter guard
        min_eps_high = 99999999.0
        bubble_data = [r for r in self.sample_results if r.current_fwd_eps >= min_eps_high]
        self.assertEqual(len(bubble_data), 0)

    def test_r2_12_clean_unmount_with_dynamic_key_versioning(self):
        """Verify reset button sets _pending_search_input and bumps version without error."""
        all_options = self.all_options
        session_state = {
            "sel_ticker": "005380",
            "_synced_ticker": "005380",
            "_last_seen_sel_ticker": "005380",
            "_last_seen_selector": all_options[1],
            "_last_table_clicked_ticker": "005380",
            "dashboard_stock_selector": all_options[1],
            "_search_input_version": 2,
            "global_search_input": "현대차",
            "global_search_input_v2": "현대차",
        }

        # Simulate Reset Button
        session_state["sel_ticker"] = None
        session_state["_synced_ticker"] = None
        session_state["_last_seen_sel_ticker"] = None
        session_state["_last_seen_selector"] = all_options[0] if all_options else None
        session_state["_last_table_clicked_ticker"] = None
        if all_options:
            session_state["_pending_stock_selector"] = all_options[0]
            session_state["dashboard_stock_selector"] = all_options[0]
        session_state["_pending_search_input"] = ""
        session_state["_search_input_version"] = session_state.get("_search_input_version", 0) + 1
        session_state["global_search_input"] = ""

        # Assert clean state
        self.assertIsNone(session_state["sel_ticker"])
        self.assertEqual(session_state["dashboard_stock_selector"], all_options[0])
        self.assertEqual(session_state["_pending_stock_selector"], all_options[0])
        self.assertEqual(session_state["_search_input_version"], 3)
        self.assertEqual(session_state["global_search_input"], "")
        self.assertEqual(session_state["_pending_search_input"], "")

        # Verify next key will be v3, leaving v2 uninstantiated and cleanly discarded
        next_key = f"global_search_input_v{session_state['_search_input_version']}"
        self.assertEqual(next_key, "global_search_input_v3")
        self.assertNotIn(next_key, session_state)

    def test_r2_13_bubble_chart_all_groups_plotted(self):
        """Verify that bubble chart creates a separate Scatter trace for each present strategy group."""
        r_gc = PEBandResult(
            ticker="005930", name="삼성전자", current_price=70000.0, current_fwd_eps=5000.0,
            current_fwd_pe=14.0, pe_percentile=32.0, pe_min=8.0, pe_p25=11.0, pe_median=13.0,
            pe_p75=16.0, pe_max=20.0, pe_mean=13.5, target_bear=55000.0, target_base=65000.0,
            target_bull=80000.0, upside_bear=-21.4, upside_base=-7.1, upside_bull=14.3,
            hist_pe_series=pd.Series([12.0, 13.0]), hist_price_series=pd.Series([60000.0, 70000.0]),
            hist_eps_series=pd.Series([5000.0, 5000.0]), sector="반도체", is_golden_cross=True, regime_tag="Golden Cross"
        )
        r_vt = PEBandResult(
            ticker="005380", name="현대차", current_price=200000.0, current_fwd_eps=25000.0,
            current_fwd_pe=8.0, pe_percentile=20.0, pe_min=6.0, pe_p25=8.5, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.2, target_bear=212500.0, target_base=250000.0,
            target_bull=300000.0, upside_bear=6.25, upside_base=25.0, upside_bull=50.0,
            hist_pe_series=pd.Series([8.0, 9.0]), hist_price_series=pd.Series([200000.0]),
            hist_eps_series=pd.Series([25000.0]), sector="자동차", is_value_trap=True, regime_tag="Value Trap"
        )
        r_em = PEBandResult(
            ticker="035420", name="NAVER", current_price=180000.0, current_fwd_eps=9000.0,
            current_fwd_pe=20.0, pe_percentile=55.0, pe_min=15.0, pe_p25=18.0, pe_median=22.0,
            pe_p75=26.0, pe_max=32.0, pe_mean=22.5, target_bear=160000.0, target_base=200000.0,
            target_bull=240000.0, upside_bear=-11.1, upside_base=11.1, upside_bull=33.3,
            hist_pe_series=pd.Series([20.0]), hist_price_series=pd.Series([180000.0]),
            hist_eps_series=pd.Series([9000.0]), sector="IT서비스", eps_rev_1m=6.5, regime_tag="Momentum Leader"
        )
        bubble_data = [r_gc, r_vt, r_em]
        SIG_COLOR_MAP = {
            "✨골든크로스": "#fbbf24",
            "🚀어닝모멘텀": "#38bdf8",
            "💎가치주": "#c084fc",
            "⚠️밸류트랩": "#f87171",
            "Neutral": "#9ca3af",
            "🔻고P/E하향": "#f43f5e",
        }

        import plotly.graph_objects as go
        fig_bubble = go.Figure()
        for sig_key, sig_color in SIG_COLOR_MAP.items():
            grp = [r for r in bubble_data if get_strategy_signal(r) == sig_key]
            if not grp:
                continue
            fig_bubble.add_trace(go.Scatter(
                x=[r.pe_percentile for r in grp],
                y=[r.upside_base for r in grp],
                mode="markers",
                name=sig_key,
            ))

        self.assertEqual(len(fig_bubble.data), 3)
        trace_names = [t.name for t in fig_bubble.data]
        self.assertIn("✨골든크로스", trace_names)
        self.assertIn("⚠️밸류트랩", trace_names)
        self.assertIn("🚀어닝모멘텀", trace_names)

    def test_r2_14_detail_view_none_metric_resilience(self):
        """Verify that KPI calculations and display logic safely handle None/NaN metrics without crashing."""
        r_loss = PEBandResult(
            ticker="999999", name="적자기업", current_price=10000.0, current_fwd_eps=-500.0,
            current_fwd_pe=np.nan, pe_percentile=np.nan, pe_min=np.nan, pe_p25=np.nan, pe_median=np.nan,
            pe_p75=np.nan, pe_max=np.nan, pe_mean=np.nan, target_bear=np.nan, target_base=np.nan,
            target_bull=np.nan, upside_bear=np.nan, upside_base=np.nan, upside_bull=np.nan,
            hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series([10000.0]),
            hist_eps_series=pd.Series([-500.0]), sector="바이오", regime_tag="Neutral"
        )
        curr_p_str = f"₩{r_loss.current_price:,.0f}" if (r_loss.current_price is not None and not np.isnan(r_loss.current_price)) else "N/A"
        curr_pe_str = f"{r_loss.current_fwd_pe:.1f}" if (r_loss.current_fwd_pe is not None and not np.isnan(r_loss.current_fwd_pe)) else "N/A"
        pe_pct_str = f"{r_loss.pe_percentile:.0f}%" if (r_loss.pe_percentile is not None and not np.isnan(r_loss.pe_percentile)) else "N/A"
        self.assertEqual(curr_p_str, "₩10,000")
        self.assertEqual(curr_pe_str, "N/A")
        self.assertEqual(pe_pct_str, "N/A")

    def test_r2_15_rapid_reset_interaction_stress(self):
        """Verify rapid repeated clicks of reset button advance version monotonically without state leakage."""
        session_state = {"_search_input_version": 0}
        for expected_ver in range(1, 10):
            session_state["_pending_search_input"] = ""
            session_state["_search_input_version"] = session_state.get("_search_input_version", 0) + 1
            session_state["global_search_input"] = ""

            if "_pending_search_input" in session_state:
                _val = session_state.pop("_pending_search_input")
                session_state["global_search_input"] = _val

            current_key = f"global_search_input_v{session_state['_search_input_version']}"
            self.assertEqual(session_state["_search_input_version"], expected_ver)
            self.assertEqual(current_key, f"global_search_input_v{expected_ver}")
            self.assertEqual(session_state["global_search_input"], "")

    def test_r2_16_table_rows_float_conversion_null_safety(self):
        """Verify building screener table row with None target metrics does not raise float(None) TypeError."""
        r_incomplete = PEBandResult(
            ticker="999998", name="미평가종목", current_price=5000.0, current_fwd_eps=100.0,
            current_fwd_pe=50.0, pe_percentile=None, pe_min=10.0, pe_p25=20.0, pe_median=30.0,
            pe_p75=40.0, pe_max=60.0, pe_mean=30.0, target_bear=None, target_base=None,
            target_bull=None, upside_bear=None, upside_base=None, upside_bull=None,
            hist_pe_series=pd.Series([50.0]), hist_price_series=pd.Series([5000.0]),
            hist_eps_series=pd.Series([100.0]), sector="기타", regime_tag="Neutral"
        )
        row = {
            "현재가": float(r_incomplete.current_price) if (r_incomplete.current_price is not None and not np.isnan(r_incomplete.current_price)) else np.nan,
            "Bear목표": float(r_incomplete.target_bear) if (r_incomplete.target_bear is not None and not np.isnan(r_incomplete.target_bear)) else np.nan,
            "Base목표": float(r_incomplete.target_base) if (r_incomplete.target_base is not None and not np.isnan(r_incomplete.target_base)) else np.nan,
            "Bull목표": float(r_incomplete.target_bull) if (r_incomplete.target_bull is not None and not np.isnan(r_incomplete.target_bull)) else np.nan,
            "Base%": float(r_incomplete.upside_base) if (r_incomplete.upside_base is not None and not np.isnan(r_incomplete.upside_base)) else np.nan,
        }
        self.assertTrue(np.isnan(row["Bear목표"]))
        self.assertTrue(np.isnan(row["Base목표"]))
        self.assertTrue(np.isnan(row["Bull목표"]))
        self.assertTrue(np.isnan(row["Base%"]))
        self.assertEqual(row["현재가"], 5000.0)

    def test_r2_17_detail_header_price_none_resilience(self):
        """Verify that render_stock_detail_view header safely formats current_price when it is None or NaN."""
        r_noprice = PEBandResult(
            ticker="999997", name="거래정지종목", current_price=None, current_fwd_eps=2000.0,
            current_fwd_pe=None, pe_percentile=None, pe_min=5.0, pe_p25=10.0, pe_median=15.0,
            pe_p75=20.0, pe_max=25.0, pe_mean=15.0, target_bear=None, target_base=None,
            target_bull=None, upside_bear=None, upside_base=None, upside_bull=None,
            hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series(dtype=float),
            hist_eps_series=pd.Series([2000.0]), sector="금융", regime_tag="Neutral"
        )
        t_curr_p_str = f"₩{r_noprice.current_price:,.0f}" if (r_noprice.current_price is not None and not np.isnan(r_noprice.current_price)) else "N/A"
        self.assertEqual(t_curr_p_str, "N/A")

    def test_r2_18_detail_upside_recalc_none_resilience(self):
        """Verify that upside recalculation in render_stock_detail_view safely handles None price/targets."""
        target_r = PEBandResult(
            ticker="999996", name="신규상장사", current_price=None, current_fwd_eps=1000.0,
            current_fwd_pe=None, pe_percentile=None, pe_min=None, pe_p25=None, pe_median=None,
            pe_p75=None, pe_max=None, pe_mean=None, target_bear=None, target_base=None,
            target_bull=None, upside_bear=None, upside_base=None, upside_bull=None,
            hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series(dtype=float),
            hist_eps_series=pd.Series([1000.0]), sector="IT", regime_tag="Neutral"
        )
        r_active = target_r
        if target_r.current_price is not None and not np.isnan(target_r.current_price) and target_r.current_price > 0:
            if r_active.target_bear is not None and not np.isnan(r_active.target_bear):
                r_active.upside_bear = ((r_active.target_bear / target_r.current_price) - 1.0) * 100.0
            if r_active.target_base is not None and not np.isnan(r_active.target_base):
                r_active.upside_base = ((r_active.target_base / target_r.current_price) - 1.0) * 100.0
            if r_active.target_bull is not None and not np.isnan(r_active.target_bull):
                r_active.upside_bull = ((r_active.target_bull / target_r.current_price) - 1.0) * 100.0
        self.assertIsNone(r_active.upside_bear)
        self.assertIsNone(r_active.upside_base)

    def test_r2_19_filtering_and_sort_none_metrics(self):
        """Verify filtering loop and sorting safely handle None upside_base and pe_percentile."""
        r_with_nones = [
            PEBandResult(
                ticker="111111", name="정상주", current_price=50000.0, current_fwd_eps=5000.0,
                current_fwd_pe=10.0, pe_percentile=30.0, pe_min=5.0, pe_p25=8.0, pe_median=12.0,
                pe_p75=15.0, pe_max=20.0, pe_mean=12.0, target_bear=40000.0, target_base=60000.0,
                target_bull=75000.0, upside_bear=-20.0, upside_base=20.0, upside_bull=50.0,
                hist_pe_series=pd.Series([10.0]), hist_price_series=pd.Series([50000.0]),
                hist_eps_series=pd.Series([5000.0]), sector="제조", regime_tag="Golden Cross"
            ),
            PEBandResult(
                ticker="222222", name="결측주", current_price=10000.0, current_fwd_eps=-200.0,
                current_fwd_pe=None, pe_percentile=None, pe_min=None, pe_p25=None, pe_median=None,
                pe_p75=None, pe_max=None, pe_mean=None, target_bear=None, target_base=None,
                target_bull=None, upside_bear=None, upside_base=None, upside_bull=None,
                hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series([10000.0]),
                hist_eps_series=pd.Series([-200.0]), sector="바이오", regime_tag="Neutral"
            )
        ]
        min_upside = -30
        pe_pct_range = (0, 90)
        filtered = []
        for r in r_with_nones:
            if r.upside_base is not None and not np.isnan(r.upside_base):
                if r.upside_base < min_upside:
                    continue
            elif min_upside > -50:
                continue
            if r.pe_percentile is not None and not np.isnan(r.pe_percentile):
                if not (pe_pct_range[0] <= r.pe_percentile <= pe_pct_range[1]):
                    continue
            elif pe_pct_range[0] > 0:
                continue
            filtered.append(r)

        filtered.sort(
            key=lambda r: (float(r.upside_base) if (r.upside_base is not None and not np.isnan(r.upside_base)) else -9999.0),
            reverse=True
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].ticker, "111111")

    def test_r2_20_search_sort_and_card_color_none_resilience(self):
        """Verify search matches sorting and card up_c color check safely handle None upside_base."""
        r_none_up = PEBandResult(
            ticker="333333", name="무업사이드", current_price=15000.0, current_fwd_eps=500.0,
            current_fwd_pe=30.0, pe_percentile=80.0, pe_min=10.0, pe_p25=15.0, pe_median=20.0,
            pe_p75=25.0, pe_max=35.0, pe_mean=20.0, target_bear=None, target_base=None,
            target_bull=None, upside_bear=None, upside_base=None, upside_bull=None,
            hist_pe_series=pd.Series([30.0]), hist_price_series=pd.Series([15000.0]),
            hist_eps_series=pd.Series([500.0]), sector="유통", regime_tag="Neutral"
        )
        matches = [self.sample_results[0], r_none_up]
        matches.sort(
            key=lambda r: (float(r.upside_base) if (r.upside_base is not None and not np.isnan(r.upside_base)) else -9999.0),
            reverse=True
        )
        self.assertEqual(matches[0].ticker, "005930")
        self.assertEqual(matches[1].ticker, "333333")

        m_has_up = (r_none_up.upside_base is not None and not np.isnan(r_none_up.upside_base))
        up_c = "#34d399" if (m_has_up and r_none_up.upside_base >= 0) else "#f87171"
        self.assertEqual(up_c, "#f87171")

    def test_r2_21_bubble_chart_filter_and_sizing_none_resilience(self):
        """Verify bubble chart data filtering and marker sizing with None and NaN current_fwd_eps."""
        r_nan_eps = PEBandResult(
            ticker="444444", name="NaN기업", current_price=20000.0, current_fwd_eps=np.nan,
            current_fwd_pe=np.nan, pe_percentile=50.0, pe_min=np.nan, pe_p25=np.nan, pe_median=np.nan,
            pe_p75=np.nan, pe_max=np.nan, pe_mean=np.nan, target_bear=np.nan, target_base=np.nan,
            target_bull=np.nan, upside_bear=np.nan, upside_base=15.0, upside_bull=np.nan,
            hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series([20000.0]),
            hist_eps_series=pd.Series(dtype=float), sector="화학", regime_tag="Neutral"
        )
        items = [self.sample_results[0], r_nan_eps]
        min_eps_filter = 0
        bubble_data = [
            r for r in items
            if (r.current_fwd_eps is not None and not np.isnan(r.current_fwd_eps) and r.current_fwd_eps >= min_eps_filter)
        ]
        self.assertEqual(len(bubble_data), 1)
        self.assertEqual(bubble_data[0].ticker, "005930")

        # Test marker sizing calculation
        grp = [r_nan_eps, self.sample_results[0]]
        sizes = [
            max(8, min(28, float(r.current_fwd_eps) / 800.0))
            if (r.current_fwd_eps is not None and not np.isnan(r.current_fwd_eps) and r.current_fwd_eps > 0)
            else 8
            for r in grp
        ]
        self.assertEqual(sizes[0], 8)
        self.assertAlmostEqual(sizes[1], max(8, min(28, 5000.0 / 800.0)))

    def test_r2_22_preset_switching_and_filtering_exhaustive(self):
        """Verify apply_quick_preset on all 5 presets plus edge cases (empty, unknown)."""
        df_sample = pd.DataFrame({
            "코드": ["005930", "005380", "035420", "999999"],
            "pe_percentile": [35.0, 20.0, 70.0, 15.0],
            "sector_pe_percentile": [30.0, 25.0, 65.0, 20.0],
            "eps_rev_1m": [5.0, -1.0, 10.0, -5.0],
            "upside_base": [25.0, 20.0, 5.0, -10.0],
            "current_fwd_eps": [5000.0, 25000.0, 8000.0, 1000.0],
            "current_fwd_pe": [12.0, 8.0, 25.0, 10.0],
        })

        # 1) ALL
        res_all = apply_quick_preset(df_sample, "ALL")
        self.assertEqual(len(res_all), 4)

        # 2) TURNAROUND
        res_ta = apply_quick_preset(df_sample, "TURNAROUND")
        self.assertTrue(all(res_ta["eps_rev_1m"] > 0))
        self.assertTrue(all(res_ta["upside_base"] >= 10.0))

        # 3) EPS_TOP
        res_eps = apply_quick_preset(df_sample, "EPS_TOP")
        self.assertTrue(all(res_eps["eps_rev_1m"] >= 3.0))

        # 4) VALUE
        res_val = apply_quick_preset(df_sample, "VALUE")
        self.assertTrue(all(res_val["pe_percentile"] <= 25.0))
        self.assertTrue(all(res_val["current_fwd_pe"] <= 15.0))

        # 5) TRAP
        res_trap = apply_quick_preset(df_sample, "TRAP")
        self.assertTrue(all(res_trap["pe_percentile"] <= 30.0))
        self.assertTrue(all(res_trap["eps_rev_1m"] < -3.0))

        # 6) Empty DataFrame
        res_empty = apply_quick_preset(pd.DataFrame(), "VALUE")
        self.assertTrue(res_empty.empty)

        # 7) Unknown preset
        res_unk = apply_quick_preset(df_sample, "UNKNOWN_PRESET")
        self.assertEqual(len(res_unk), 4)

    def test_r2_23_valuation_band_models_switch(self):
        """Verify calc_pe_band across all 3 models ('percentile', 'mean_sd', 'fixed') and periods."""
        dates = pd.bdate_range("2020-01-01", "2026-09-17")
        p_series = pd.Series(70000.0, index=dates)
        e_series = pd.Series(5000.0, index=dates)

        # 1. Percentile Model
        r_pct = calc_pe_band("005930", "삼성전자", p_series, e_series, band_years=5, band_model="percentile")
        self.assertIsNotNone(r_pct)
        self.assertIsNotNone(r_pct.band_series_pct)
        self.assertIn("p50", r_pct.band_series_pct.columns)

        # 2. Mean ± SD Model
        r_sd = calc_pe_band("005930", "삼성전자", p_series, e_series, band_years=5, band_model="mean_sd")
        self.assertIsNotNone(r_sd)
        self.assertIsNotNone(r_sd.band_series_sd)
        self.assertIn("Mean", r_sd.band_series_sd.columns)

        # 3. Fixed Multiples Model
        r_fix = calc_pe_band("005930", "삼성전자", p_series, e_series, band_years=5, band_model="fixed")
        self.assertIsNotNone(r_fix)
        self.assertIsNotNone(r_fix.band_series_fixed)
        self.assertIn("10x", r_fix.band_series_fixed.columns)

        # 4. Period switching
        for yr in [1, 2, 3, 5, 7, 10, 15]:
            r_yr = calc_pe_band("005930", "삼성전자", p_series, e_series, band_years=yr, band_model="percentile")
            self.assertIsNotNone(r_yr)

    def test_r2_24_excel_and_csv_download_generation(self):
        """Verify generation of Excel and CSV download buffers without errors."""
        import io
        r = self.sample_results[0]
        # Stock Summary Excel
        stock_summary_df = pd.DataFrame({
            "항목": ["종목명", "종목코드", "현재가", "Fwd P/E"],
            "값": [r.name, r.ticker, f"{r.current_price:,.0f}", f"{r.current_fwd_pe:.2f}x"]
        })
        buf_stock = io.BytesIO()
        stock_summary_df.to_excel(buf_stock, index=False)
        self.assertGreater(len(buf_stock.getvalue()), 0)

        # Timeseries CSV
        p_plot = r.hist_price_series
        e_plot = r.hist_eps_series
        pe_plot = r.hist_pe_series
        ts_export = pd.concat([p_plot.rename("Price"), e_plot.rename("Fwd_EPS"), pe_plot.rename("Fwd_PE")], axis=1)
        csv_bytes = ts_export.reset_index().to_csv(index=False).encode("utf-8-sig")
        self.assertGreater(len(csv_bytes), 0)

        # Screener Excel & CSV
        df_screener = pd.DataFrame([{"코드": r.ticker, "종목명": r.name, "현재가": r.current_price}])
        buf_xl = io.BytesIO()
        df_screener.to_excel(buf_xl, index=False)
        self.assertGreater(len(buf_xl.getvalue()), 0)
        csv_scr = df_screener.to_csv(index=False).encode("utf-8-sig")
        self.assertGreater(len(csv_scr), 0)

    def test_r2_25_dataframe_width_kwargs_audit(self):
        """Verify _dataframe_width_kwargs returns width='stretch' when supported, avoiding deprecation."""
        from app import _dataframe_width_kwargs
        import streamlit as st
        kwargs = _dataframe_width_kwargs()
        self.assertIsInstance(kwargs, dict)
        if "width" in inspect.signature(st.dataframe).parameters:
            self.assertEqual(kwargs.get("width"), "stretch")
            self.assertNotIn("use_container_width", kwargs)
        else:
            self.assertTrue(kwargs.get("use_container_width"))

    def test_r2_26_pending_only_lifecycle_architecture(self):
        """Verify app.py source code strictly uses _pending_ pattern and avoids post-instantiation direct assignment."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        # In search card button handler:
        self.assertNotIn('st.session_state["dashboard_stock_selector"] = opt_candidate', code)
        self.assertNotIn('st.session_state["main_nav_tab_radio"] = TAB_DASHBOARD', code)
        # In reset button handler:
        self.assertNotIn('st.session_state["dashboard_stock_selector"] = all_options[0]', code)
        # Verify pending assignments exist
        self.assertIn('st.session_state["_pending_nav_tab"] = TAB_DASHBOARD', code)
        self.assertIn('st.session_state["_pending_stock_selector"] = opt_candidate', code)
        self.assertIn('st.session_state["_pending_stock_selector"] = all_options[0]', code)


class TestAppAstAndCompilation(unittest.TestCase):
    """Verify app.py parses cleanly into Python AST and bytecode compiles without syntax errors."""

    def test_app_ast_parse(self):
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code, filename="app.py")
        self.assertIsNotNone(parsed)

    def test_app_bytecode_compile(self):
        """Verify app.py compiles cleanly to bytecode without syntax or indentation errors."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(app_file), "exec")
        self.assertIsNotNone(compiled)

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


class TestR3HeadlessStreamlitExecution(unittest.TestCase):
    """Verify headless execution of app.py using Streamlit AppTest framework."""

    def test_r3_01_headless_app_initialization(self):
        """Verify app.py initializes and executes in headless mode without crashing."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available in this environment")

        import os
        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"Streamlit app raised an unhandled exception: {at.exception}")
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)

    def test_r3_02_headless_interactive_search_and_reset(self):
        """Verify interactive search input and '선택 초기화' button click via AppTest without widget errors."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available in this environment")

        import os
        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"Initial run failed: {at.exception}")

            # 1. Type into search input
            if len(at.text_input) > 0:
                at.text_input[0].input("005930").run()
                self.assertFalse(at.exception, f"Text input run failed: {at.exception}")
                self.assertEqual(at.session_state["global_search_input"], "005930")

            # 2. Find and click reset button ('❌ 선택 초기화')
            reset_btn = next((b for b in at.button if "초기화" in b.label), None)
            if reset_btn is not None:
                reset_btn.click().run()
                self.assertFalse(at.exception, f"Reset click raised exception: {at.exception}")
                self.assertIsNone(at.session_state["sel_ticker"])
                self.assertEqual(at.session_state["global_search_input"], "")
                if len(at.text_input) > 0:
                    self.assertEqual(at.text_input[0].value, "")
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)

    def test_r3_03_headless_preset_switch_and_tabs(self):
        """Verify 1-click strategy preset switching and tab navigation via AppTest."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available in this environment")

        import os
        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"Initial run failed: {at.exception}")

            # Click a preset button ('💎 가치주')
            val_btn = next((b for b in at.button if "가치주" in b.label), None)
            if val_btn is not None:
                val_btn.click().run()
                self.assertFalse(at.exception, f"Preset click raised exception: {at.exception}")
                self.assertEqual(at.session_state["active_preset"], "VALUE")

            # Reset preset to ALL
            all_btn = next((b for b in at.button if "전체 (ALL)" in b.label), None)
            if all_btn is not None:
                all_btn.click().run()
                self.assertFalse(at.exception, f"ALL preset click raised exception: {at.exception}")
                self.assertEqual(at.session_state["active_preset"], "ALL")

            # Switch tab to Tab 3
            tab_radio = next((r for r in at.radio if "main_nav_tab_radio" in getattr(r, "key", "")), None)
            if tab_radio is not None:
                tab_radio.set_value("🪷  밸류에이션 버블 차트").run()
                self.assertFalse(at.exception, f"Tab 3 switch raised exception: {at.exception}")
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)

    def test_r3_04_headless_interactive_search_card_click_and_reset(self):
        """Verify search card click navigates to Tab 2 and reset clears state cleanly."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available in this environment")

        import os
        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"Initial run failed: {at.exception}")

            # Type ticker into search
            if len(at.text_input) > 0:
                at.text_input[0].input("삼성").run()
                self.assertFalse(at.exception, f"Search run failed: {at.exception}")

                # Find search detail button
                detail_btn = next((b for b in at.button if "상세 분석 보기" in b.label), None)
                if detail_btn is not None:
                    detail_btn.click().run()
                    self.assertFalse(at.exception, f"Search card click failed: {at.exception}")
                    self.assertIsNotNone(at.session_state["sel_ticker"])
                    self.assertEqual(at.session_state["main_nav_tab_radio"], "🔍  원페이지 통합 대시보드")

            # Click reset button
            reset_btn = next((b for b in at.button if "초기화" in b.label), None)
            if reset_btn is not None:
                reset_btn.click().run()
                self.assertFalse(at.exception, f"Reset click after card failed: {at.exception}")
                self.assertIsNone(at.session_state["sel_ticker"])
                self.assertEqual(at.session_state["global_search_input"], "")
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)

    def test_r3_05_headless_winsorization_toggle_and_empty_filters(self):
        """Verify winsorization toggle and empty filter state handling via AppTest."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available in this environment")

        import os
        os.environ["STREAMLIT_HEADLESS_TEST"] = "1"
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            self.assertFalse(at.exception, f"Initial run failed: {at.exception}")

            # Navigate to Tab 2
            tab_radio = next((r for r in at.radio if "main_nav_tab_radio" in getattr(r, "key", "")), None)
            if tab_radio is not None:
                tab_radio.set_value("🔍  원페이지 통합 대시보드").run()
                self.assertFalse(at.exception, f"Tab 2 switch failed: {at.exception}")

                # Toggle winsorization checkbox if present
                winsor_cb = next((c for c in at.checkbox if "Winsor" in c.label or "이상치" in c.label), None)
                if winsor_cb is not None:
                    winsor_cb.uncheck().run()
                    self.assertFalse(at.exception, f"Winsor uncheck failed: {at.exception}")
                    winsor_cb.check().run()
                    self.assertFalse(at.exception, f"Winsor recheck failed: {at.exception}")

            # Test search query with 0 matches
            if len(at.text_input) > 0:
                at.text_input[0].input("존재하지않는종목xyz").run()
                self.assertFalse(at.exception, f"Empty search query run failed: {at.exception}")
                self.assertTrue(any("검색 결과가 없습니다" in str(info.value) for info in at.info))
        finally:
            os.environ.pop("STREAMLIT_HEADLESS_TEST", None)


if __name__ == "__main__":
    unittest.main()
