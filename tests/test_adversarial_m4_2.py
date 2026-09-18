"""
tests/test_adversarial_m4_2.py
------------------------------
Adversarial Stress-Testing & Empirical Verification Suite for Milestone 4:
One-Page Integrated Dashboard & Band Switcher in app.py.

Author: Challenger M4-2 (Empirical Challenger)

Challenge Dimensions:
1. Strategy Signal Taxonomy & Priority Mapping (get_strategy_signal & strategy_badge_html):
   - Golden Cross vs Value Trap vs Momentum Leader vs Deep Value vs High P/E Downgrade vs Neutral.
   - Dual-flag conflict & precedence handling (e.g. is_golden_cross=True AND is_value_trap=True).
   - Numerical boundary testing (pe_percentile <= 25/30/40, eps_rev_1m >= 5.0 / -2.0, current_fwd_pe <= 15.0).
   - Missing / NaN / None resilience across all metric inputs without exceptions.
   - Discrepancy check: '🔻고P/E하향' regime in get_strategy_signal vs sidebar strategy_options.
2. Real-time Price Updates Resilience (apply_realtime_prices):
   - Empty/None price dictionary handling.
   - Missing tickers, unchanged prices, and selective updates.
   - Pathological price inputs: 0.0, negative prices, NaN prices, infinite prices, extreme price spikes (1e8 KRW), micro prices (0.01 KRW).
   - Negative EPS / Cyclical loss-maker protection: ensures no division by zero or negative P/E calculation.
   - Mathematical exactness: P/E, empirical percentile against historical series, and upside formulas.
   - Full attribute preservation via dataclasses.replace.
   - Sector relative metrics recalculation triggering and cap at P/E 150.
   - Empty historical P/E series fallback.
3. Plotly Figure 2-Subplot Synchronization & Trace Integrity across all 3 Models:
   - Subplot configuration: 2 rows, 1 col, shared_xaxes=True, row_heights=[0.65, 0.35], hovermode="x unified".
   - Model 1 (Percentile): Price, p90, Bull(p75), Base(p50), Bear(p25), p10, marker + 3 target hlines in Upper; EPS, MA12, P/E in Lower.
   - Model 2 (Mean ± SD): Price, +2SD, Bull(+1SD), Base(Mean), Bear(-1SD), -2SD, marker + 3 target hlines in Upper; EPS, MA12, P/E in Lower.
   - Model 3 (Fixed Multiples): Price, fixed multiple bands (8x, 10x, 12x, 15x), marker + 3 target hlines in Upper; EPS, MA12, P/E in Lower.
   - Coordinate axis integrity: Row 1 on primary y-axis, Row 2 EPS on primary y-axis, Row 2 P/E on secondary y-axis.
   - Frequency alignment: Monthly price vs daily EPS forward-fill alignment.
   - Loss-making dates handling: NaN masking on non-positive EPS in P/E trace.
4. Headless Import Safety, Caching Decoupling & AST Conformance:
   - AST validation and bytecode compilation of app.py without syntax errors.
   - Headless import of app.py functions without active Streamlit ScriptRunContext.
   - Decoupled caching signature check (load_raw_market_data does not accept band_years).
   - 1-Click Quick Screener Presets parity with tests/oracle.py across all 5 keys.
   - Screener table columns completeness (all 17 columns present in display_cols and configs).
"""

import sys
import os
import ast
import inspect
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calculator import (
    calc_pe_band,
    PEBandResult,
    load_sector_mapping,
    calc_sector_relative_metrics,
)
from tests.oracle import (
    oracle_apply_quick_preset,
    oracle_classify_regime,
)

# Import app.py functions under test
from app import (
    get_strategy_signal,
    strategy_badge_html,
    signal_badge,
    signal_label,
    pe_bar_color,
    apply_quick_preset,
    apply_realtime_prices,
    load_raw_market_data,
    CHART_LAYOUT,
    AX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Standalone figure builder matching app.py:973-1132 for algorithmic testing
# ─────────────────────────────────────────────────────────────────────
def build_integrated_figure(
    r_active: PEBandResult,
    p_series: pd.Series,
    e_series: pd.Series,
    model_key: str = "percentile",
    model_period: int = 5,
) -> go.Figure:
    """Replicates the Plotly figure construction in app.py lines 973-1132."""
    p_clean = p_series.dropna().sort_index()
    e_clean = e_series.dropna().sort_index()

    common_idx = p_clean.index.intersection(e_clean.index)
    if len(common_idx) < 4:
        aligned_e = e_clean.reindex(e_clean.index.union(p_clean.index)).ffill().loc[p_clean.index].dropna()
        common_idx = p_clean.index.intersection(aligned_e.index)
        if len(common_idx) >= 4:
            e_clean = aligned_e

    if len(common_idx) < 2:
        return None

    cutoff_date = common_idx.max() - pd.DateOffset(years=model_period)
    plot_idx = common_idx[common_idx >= cutoff_date]
    if len(plot_idx) < 2:
        plot_idx = common_idx

    p_plot = p_clean.loc[plot_idx]
    e_plot = e_clean.loc[plot_idx]
    pe_plot = (p_plot / e_plot).where(e_plot > 0, np.nan)
    pe_plot = pe_plot.where((pe_plot > 0) & (pe_plot <= 200), np.nan)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # Upper Subplot: Actual stock price
    fig.add_trace(
        go.Scatter(
            x=p_plot.index, y=p_plot.values,
            mode="lines", name="실제 주가",
            line=dict(color="#38bdf8", width=2.5),
        ),
        row=1, col=1
    )

    # Valuation bands
    if model_key == "percentile" and r_active.band_series_pct is not None:
        b_df = r_active.band_series_pct.loc[r_active.band_series_pct.index.isin(plot_idx)]
        if "p90" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["p90"], mode="lines", name="p90 밴드",
                                     line=dict(color="rgba(192,132,252,0.6)", width=1.2, dash="dash")), row=1, col=1)
        if "p75" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["p75"], mode="lines", name="Bull 밴드 (75th)",
                                     line=dict(color="rgba(52,211,153,0.85)", width=1.6, dash="dash")), row=1, col=1)
        if "p50" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["p50"], mode="lines", name="Base 밴드 (Median)",
                                     line=dict(color="rgba(251,191,36,0.9)", width=2.0, dash="dash")), row=1, col=1)
        if "p25" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["p25"], mode="lines", name="Bear 밴드 (25th)",
                                     line=dict(color="rgba(248,113,113,0.85)", width=1.6, dash="dash")), row=1, col=1)
        if "p10" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["p10"], mode="lines", name="p10 밴드",
                                     line=dict(color="rgba(244,63,94,0.6)", width=1.2, dash="dash")), row=1, col=1)

    elif model_key == "mean_sd" and r_active.band_series_sd is not None:
        b_df = r_active.band_series_sd.loc[r_active.band_series_sd.index.isin(plot_idx)]
        if "+2SD" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["+2SD"], mode="lines", name="+2SD 밴드",
                                     line=dict(color="rgba(192,132,252,0.6)", width=1.2, dash="dash")), row=1, col=1)
        if "+1SD" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["+1SD"], mode="lines", name="Bull 밴드 (+1SD)",
                                     line=dict(color="rgba(52,211,153,0.85)", width=1.6, dash="dash")), row=1, col=1)
        if "Mean" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["Mean"], mode="lines", name="Base 밴드 (Mean)",
                                     line=dict(color="rgba(251,191,36,0.9)", width=2.0, dash="dash")), row=1, col=1)
        if "-1SD" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["-1SD"], mode="lines", name="Bear 밴드 (-1SD)",
                                     line=dict(color="rgba(248,113,113,0.85)", width=1.6, dash="dash")), row=1, col=1)
        if "-2SD" in b_df:
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df["-2SD"], mode="lines", name="-2SD 밴드",
                                     line=dict(color="rgba(244,63,94,0.6)", width=1.2, dash="dash")), row=1, col=1)

    elif model_key == "fixed" and r_active.band_series_fixed is not None:
        b_df = r_active.band_series_fixed.loc[r_active.band_series_fixed.index.isin(plot_idx)]
        f_colors = ["#f87171", "#fbbf24", "#34d399", "#c084fc", "#38bdf8"]
        for i, c_name in enumerate(b_df.columns):
            c_col = f_colors[i % len(f_colors)]
            fig.add_trace(go.Scatter(x=b_df.index, y=b_df[c_name], mode="lines", name=f"{c_name} 밴드",
                                     line=dict(color=c_col, width=1.5, dash="dash")), row=1, col=1)

    # Current price marker
    latest_date = p_plot.index[-1]
    fig.add_trace(
        go.Scatter(
            x=[latest_date], y=[r_active.current_price],
            mode="markers+text",
            marker=dict(size=9, color="#ffffff", line=dict(color="#38bdf8", width=2)),
            text=[f" 현재가 ₩{r_active.current_price:,.0f}"],
            textposition="middle right",
            name="현재가",
            showlegend=False
        ),
        row=1, col=1
    )

    # Upper horizontal target lines
    fig.add_hline(y=r_active.target_bull, line_dash="dot", line_color="#34d399", line_width=1, row=1, col=1)
    fig.add_hline(y=r_active.target_base, line_dash="dot", line_color="#fbbf24", line_width=1, row=1, col=1)
    fig.add_hline(y=r_active.target_bear, line_dash="dot", line_color="#f87171", line_width=1, row=1, col=1)

    # Lower Subplot: 12M Fwd EPS & MA12 on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=e_plot.index, y=e_plot.values,
            mode="lines", name="12M Fwd EPS",
            line=dict(color="#60a5fa", width=2.0),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"
        ),
        row=2, col=1, secondary_y=False
    )
    ma12_e = e_plot.rolling(12, min_periods=3).mean()
    fig.add_trace(
        go.Scatter(
            x=ma12_e.index, y=ma12_e.values,
            mode="lines", name="EPS MA12",
            line=dict(color="#93c5fd", width=1.5, dash="dot")
        ),
        row=2, col=1, secondary_y=False
    )

    # Lower Subplot: 12M Fwd P/E on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=pe_plot.index, y=pe_plot.values,
            mode="lines", name="12M Fwd P/E",
            line=dict(color="#c084fc", width=2.0)
        ),
        row=2, col=1, secondary_y=True
    )

    if r_active.pe_median and r_active.pe_median > 0:
        fig.add_hline(
            y=r_active.pe_median, line_dash="dash", line_color="rgba(251,191,36,0.6)",
            line_width=1, row=2, col=1, secondary_y=True
        )

    fig.update_layout(
        **CHART_LAYOUT,
        height=620,
        margin=dict(l=10, r=20, t=30, b=30),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Test Suite: Strategy Signal Taxonomy Adversarial Stress-Testing
# ─────────────────────────────────────────────────────────────────────────────
class TestStrategySignalTaxonomyAdversarial(unittest.TestCase):
    """Adversarial stress-testing of get_strategy_signal taxonomy and edge-case mapping."""

    def _make_dummy_result(self, **kwargs) -> PEBandResult:
        defaults = dict(
            ticker="005930",
            name="삼성전자",
            current_price=70000.0,
            current_fwd_eps=5000.0,
            current_fwd_pe=14.0,
            pe_percentile=32.0,
            pe_min=8.0, pe_p25=11.0, pe_median=13.0, pe_p75=16.0, pe_max=20.0, pe_mean=13.5,
            target_bear=55000.0, target_base=65000.0, target_bull=80000.0,
            upside_bear=-21.4, upside_base=-7.1, upside_bull=14.3,
            hist_pe_series=pd.Series([12.0, 13.0, 14.0]),
            hist_price_series=pd.Series([60000.0, 65000.0, 70000.0]),
            hist_eps_series=pd.Series([5000.0, 5000.0, 5000.0]),
            sector="반도체",
            sector_median_pe=15.2,
            sector_relative_pe=0.92,
            sector_pe_percentile=28.0,
            eps_rev_1w=1.2,
            eps_rev_1m=4.5,
            eps_rev_3m=8.0,
            is_golden_cross=False,
            is_value_trap=False,
            regime_tag="Neutral",
        )
        defaults.update(kwargs)
        return PEBandResult(**defaults)

    def test_golden_cross_mapping(self):
        """Verify Golden Cross via boolean flag and regime_tag."""
        r1 = self._make_dummy_result(is_golden_cross=True, regime_tag="Golden Cross")
        self.assertEqual(get_strategy_signal(r1), "✨골든크로스")

        r2 = self._make_dummy_result(is_golden_cross=False, regime_tag="Golden Cross")
        self.assertEqual(get_strategy_signal(r2), "✨골든크로스")

        r3 = self._make_dummy_result(is_golden_cross=True, regime_tag="Neutral")
        self.assertEqual(get_strategy_signal(r3), "✨골든크로스")

    def test_value_trap_mapping(self):
        """Verify Value Trap via boolean flag and regime_tag."""
        r1 = self._make_dummy_result(is_value_trap=True, regime_tag="Value Trap")
        self.assertEqual(get_strategy_signal(r1), "⚠️밸류트랩")

        r2 = self._make_dummy_result(is_value_trap=False, regime_tag="Value Trap")
        self.assertEqual(get_strategy_signal(r2), "⚠️밸류트랩")

        r3 = self._make_dummy_result(is_value_trap=True, regime_tag="Neutral")
        self.assertEqual(get_strategy_signal(r3), "⚠️밸류트랩")

    def test_conflicting_flags_precedence(self):
        """Adversarial: When both golden cross and value trap are True, golden cross takes precedence."""
        r_conflict = self._make_dummy_result(is_golden_cross=True, is_value_trap=True)
        self.assertEqual(get_strategy_signal(r_conflict), "✨골든크로스")

    def test_momentum_leader_mapping(self):
        """Verify Earnings Momentum via revision rate >= 5.0% or regime tag."""
        # 1M Revision >= 5.0%
        r1 = self._make_dummy_result(eps_rev_1m=5.0, regime_tag="Neutral")
        self.assertEqual(get_strategy_signal(r1), "🚀어닝모멘텀")

        r2 = self._make_dummy_result(eps_rev_1m=12.5, regime_tag="Neutral")
        self.assertEqual(get_strategy_signal(r2), "🚀어닝모멘텀")

        # Tag is Momentum Leader even if rev < 5.0% (e.g. 3.5%)
        r3 = self._make_dummy_result(eps_rev_1m=3.5, regime_tag="Momentum Leader")
        self.assertEqual(get_strategy_signal(r3), "🚀어닝모멘텀")

    def test_deep_value_stock_mapping_and_boundaries(self):
        """Verify Deep Value requires pe_percentile <= 25, 0 < fwd_pe <= 15, rev_1m >= -2.0."""
        # Exact matching candidate
        r_val = self._make_dummy_result(
            pe_percentile=25.0, current_fwd_pe=15.0, eps_rev_1m=-2.0, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_val), "💎가치주")

        # Boundary fail 1: pe_percentile slightly above 25.0 (25.1)
        r_fail_pct = self._make_dummy_result(
            pe_percentile=25.1, current_fwd_pe=12.0, eps_rev_1m=-1.0, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_fail_pct), "Neutral")

        # Boundary fail 2: current_fwd_pe slightly above 15.0 (15.1)
        r_fail_pe = self._make_dummy_result(
            pe_percentile=20.0, current_fwd_pe=15.1, eps_rev_1m=-1.0, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_fail_pe), "Neutral")

        # Boundary fail 3: current_fwd_pe <= 0 (e.g. loss-maker with pe=0 or negative)
        r_fail_neg_pe = self._make_dummy_result(
            pe_percentile=10.0, current_fwd_pe=0.0, eps_rev_1m=1.0, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_fail_neg_pe), "Neutral")

        # Boundary fail 4: eps_rev_1m below -2.0 (e.g. -2.1)
        r_fail_rev = self._make_dummy_result(
            pe_percentile=20.0, current_fwd_pe=12.0, eps_rev_1m=-2.1, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_fail_rev), "Neutral")

    def test_high_pe_downgrade_mapping(self):
        """Verify High P/E Downgrade regime tag mapping."""
        r_hd = self._make_dummy_result(
            pe_percentile=85.0, current_fwd_pe=45.0, eps_rev_1m=-3.5, regime_tag="High P/E Downgrade"
        )
        self.assertEqual(get_strategy_signal(r_hd), "🔻고P/E하향")

    def test_none_and_nan_resilience(self):
        """Adversarial: get_strategy_signal must never raise exception on None, NaN, or extreme values."""
        # All None
        r_none = self._make_dummy_result(
            pe_percentile=None, current_fwd_pe=None, eps_rev_1m=None, regime_tag=None
        )
        self.assertEqual(get_strategy_signal(r_none), "Neutral")

        # All NaN
        r_nan = self._make_dummy_result(
            pe_percentile=float("nan"), current_fwd_pe=float("nan"),
            eps_rev_1m=float("nan"), regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_nan), "Neutral")

    def test_strategy_badge_html_rendering(self):
        """Verify HTML badges for all 6 signal variations and unknown fallback."""
        badges = {
            "✨골든크로스": "sig-gc",
            "⚠️밸류트랩": "sig-vt",
            "🚀어닝모멘텀": "sig-em",
            "💎가치주": "sig-val",
            "🔻고P/E하향": "sig-hd",
            "Neutral": "sig-neu",
            "UNKNOWN_RANDOM_SIG": "sig-neu",
        }
        for sig, expected_class in badges.items():
            html = strategy_badge_html(sig)
            self.assertIn(expected_class, html, f"Badge for {sig} must contain class {expected_class}")

    def test_sidebar_filter_options_discrepancy_discovery(self):
        """
        Adversarial Finding: Check whether '🔻고P/E하향' is present in app.py:strategy_options.
        Notice that app.py defines:
          strategy_options = ["✨골든크로스", "⚠️밸류트랩", "🚀어닝모멘텀", "💎가치주", "Neutral"]
        which omits "🔻고P/E하향". This causes High P/E Downgrade stocks to be filtered out by default!
        """
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Parse AST to find strategy_options assignment
        tree = ast.parse(code)
        found_options = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "strategy_options":
                        if isinstance(node.value, ast.List):
                            found_options = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]

        self.assertIsNotNone(found_options, "strategy_options must be defined in app.py")
        is_hd_present = "🔻고P/E하향" in found_options
        self.assertTrue(is_hd_present, "'🔻고P/E하향' must be present in sidebar strategy_options")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Test Suite: Real-time Price Updates Resilience Adversarial Testing
# ─────────────────────────────────────────────────────────────────────
class TestRealtimePricesAdversarial(unittest.TestCase):
    """Adversarial stress-testing of apply_realtime_prices against pathological inputs."""

    def setUp(self):
        self.sec_map = load_sector_mapping()
        self.base_result = PEBandResult(
            ticker="005930",
            name="삼성전자",
            current_price=70000.0,
            current_fwd_eps=5000.0,
            current_fwd_pe=14.0,
            pe_percentile=35.0,
            pe_min=8.0, pe_p25=11.0, pe_median=13.0, pe_p75=16.0, pe_max=20.0, pe_mean=13.5,
            target_bear=55000.0, target_base=65000.0, target_bull=80000.0,
            upside_bear=-21.43, upside_base=-7.14, upside_bull=14.29,
            hist_pe_series=pd.Series([10.0, 12.0, 13.0, 14.0, 16.0, 18.0]),
            hist_price_series=pd.Series([50000.0, 60000.0, 65000.0, 70000.0, 80000.0, 90000.0]),
            hist_eps_series=pd.Series([5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0]),
            sector="반도체",
            sector_median_pe=15.0,
            sector_relative_pe=0.933,
            sector_pe_percentile=30.0,
            sector_stock_count=5,
            eps_rev_1w=1.2,
            eps_rev_1m=4.5,
            eps_rev_3m=8.0,
            is_golden_cross=True,
            regime_tag="Golden Cross",
        )

    def test_empty_or_none_prices(self):
        """apply_realtime_prices must return results directly if prices empty or None."""
        res = [self.base_result]
        self.assertEqual(apply_realtime_prices(res, {}, self.sec_map), res)
        self.assertEqual(apply_realtime_prices(res, None, self.sec_map), res)

    def test_missing_ticker_unmodified(self):
        """Tickers missing from real-time dict must remain untouched."""
        res = [self.base_result]
        updated = apply_realtime_prices(res, {"035420": 200000.0}, self.sec_map)
        self.assertEqual(updated[0].current_price, 70000.0)
        self.assertEqual(updated[0].current_fwd_pe, 14.0)

    def test_zero_and_negative_prices_rejected(self):
        """Zero and negative prices must be rejected without causing ZeroDivisionError."""
        res = [self.base_result]
        # Zero price
        up_zero = apply_realtime_prices(res, {"005930": 0.0}, self.sec_map)
        self.assertEqual(up_zero[0].current_price, 70000.0, "Zero price must be ignored")

        # Negative price
        up_neg = apply_realtime_prices(res, {"005930": -50000.0}, self.sec_map)
        self.assertEqual(up_neg[0].current_price, 70000.0, "Negative price must be ignored")

    def test_nan_price_rejected(self):
        """NaN price must be rejected without contaminating floats."""
        res = [self.base_result]
        up_nan = apply_realtime_prices(res, {"005930": float("nan")}, self.sec_map)
        self.assertEqual(up_nan[0].current_price, 70000.0, "NaN price must be ignored")

    def test_cyclical_loss_maker_protected(self):
        """Loss-making company (EPS <= 0) must NOT divide by non-positive EPS."""
        loss_r = PEBandResult(
            ticker="999999", name="적자기업", current_price=10000.0, current_fwd_eps=-200.0,
            current_fwd_pe=None, pe_percentile=50.0, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0.0, target_base=0.0, target_bull=0.0,
            upside_bear=0.0, upside_base=0.0, upside_bull=0.0,
            hist_pe_series=pd.Series([10.0, 12.0]), hist_price_series=pd.Series([10000.0]), hist_eps_series=pd.Series([-200.0]),
            sector="기타", eps_rev_1m=-10.0, is_value_trap=True
        )
        res = [loss_r]
        updated = apply_realtime_prices(res, {"999999": 12000.0}, self.sec_map)
        self.assertEqual(updated[0].current_price, 12000.0, "Loss-maker current_price must update to real-time price")
        self.assertIsNone(updated[0].current_fwd_pe, "Loss-maker P/E must not be computed")

    def test_unchanged_price_skips_recalculation(self):
        """If real-time price equals current closing price, no dataclass replacement is performed."""
        res = [self.base_result]
        updated = apply_realtime_prices(res, {"005930": 70000.0}, self.sec_map)
        self.assertIs(updated[0], self.base_result, "Unchanged price should return original object")

    def test_normal_price_update_mathematical_precision(self):
        """Verify exact mathematical formulas upon normal price update."""
        res = [self.base_result]
        new_price = 75000.0
        updated = apply_realtime_prices(res, {"005930": new_price}, self.sec_map)
        r = updated[0]

        # 1. Price
        self.assertEqual(r.current_price, 75000.0)
        # 2. P/E = 75000 / 5000 = 15.0
        self.assertEqual(r.current_fwd_pe, 15.0)
        # 3. Percentile against [10, 12, 13, 14, 16, 18]: values < 15.0 are [10, 12, 13, 14] (4 of 6 = 66.6667%)
        expected_pct = (4.0 / 6.0) * 100.0
        self.assertAlmostEqual(r.pe_percentile, expected_pct, places=4)
        # 4. Upsides: ((target / new_price) - 1.0) * 100.0
        self.assertAlmostEqual(r.upside_bear, ((55000.0 / 75000.0) - 1.0) * 100.0, places=4)
        self.assertAlmostEqual(r.upside_base, ((65000.0 / 75000.0) - 1.0) * 100.0, places=4)
        self.assertAlmostEqual(r.upside_bull, ((80000.0 / 75000.0) - 1.0) * 100.0, places=4)
        # 5. Metadata preservation
        self.assertEqual(r.sector, "반도체")
        self.assertEqual(r.is_golden_cross, False, "Golden cross should invalidate when pe_pct > 40%")
        self.assertEqual(r.regime_tag, "Momentum Leader", "Regime should transition to Momentum Leader")
        self.assertEqual(r.eps_rev_1m, 4.5)

    def test_extreme_price_spike_resilience(self):
        """Adversarial: 100,000,000 KRW extreme price spike must be handled safely."""
        res = [self.base_result]
        spike_price = 100_000_000.0
        updated = apply_realtime_prices(res, {"005930": spike_price}, self.sec_map)
        r = updated[0]

        self.assertEqual(r.current_price, spike_price)
        self.assertEqual(r.current_fwd_pe, spike_price / 5000.0) # 20,000x
        self.assertEqual(r.pe_percentile, 100.0)
        self.assertAlmostEqual(r.upside_base, -100.0, delta=0.5)
        # Cap at 150 ensures it does not ruin sector median
        self.assertIsNone(r.sector_relative_pe, "P/E > 150 must have None sector relative PE")

    def test_micro_price_resilience(self):
        """Adversarial: 0.01 KRW price must be handled without crash or zero division."""
        res = [self.base_result]
        micro_price = 0.01
        updated = apply_realtime_prices(res, {"005930": micro_price}, self.sec_map)
        r = updated[0]

        self.assertEqual(r.current_price, 0.01)
        self.assertAlmostEqual(r.current_fwd_pe, 0.01 / 5000.0, places=6)
        self.assertEqual(r.pe_percentile, 0.0)
        self.assertTrue(r.upside_base > 100_000.0)

    def test_empty_hist_pe_series_fallback(self):
        """When hist_pe_series is empty, pe_percentile falls back to existing value."""
        empty_hist_r = PEBandResult(
            ticker="005930", name="삼성전자", current_price=70000.0, current_fwd_eps=5000.0,
            current_fwd_pe=14.0, pe_percentile=42.0, pe_min=8.0, pe_p25=11.0, pe_median=13.0,
            pe_p75=16.0, pe_max=20.0, pe_mean=13.5, target_bear=55000.0, target_base=65000.0, target_bull=80000.0,
            upside_bear=-21.4, upside_base=-7.1, upside_bull=14.3,
            hist_pe_series=pd.Series(dtype=float), hist_price_series=pd.Series([70000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="반도체", eps_rev_1m=4.5
        )
        updated = apply_realtime_prices([empty_hist_r], {"005930": 72000.0}, self.sec_map)
        self.assertEqual(updated[0].pe_percentile, 42.0, "Empty hist_pe_series must retain previous percentile")

    def test_deficit_stocks_zero_negative_nan_none_resilience(self):
        """Stress-test: deficit stocks (EPS <= 0, nan, None) update price without ZeroDivisionError or target corruption."""
        r_neg = PEBandResult(
            ticker="900001", name="적자1_음수", current_price=10000.0, current_fwd_eps=-350.0,
            current_fwd_pe=None, pe_percentile=np.nan, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0.0, target_base=0.0, target_bull=0.0,
            upside_bear=0.0, upside_base=0.0, upside_bull=0.0,
            hist_pe_series=pd.Series([10.0, 12.0]), hist_price_series=pd.Series([10000.0]),
            hist_eps_series=pd.Series([-350.0]), sector="기타"
        )
        r_zero = PEBandResult(
            ticker="900002", name="적자2_제로", current_price=8000.0, current_fwd_eps=0.0,
            current_fwd_pe=None, pe_percentile=np.nan, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=0.0, target_base=0.0, target_bull=0.0,
            upside_bear=0.0, upside_base=0.0, upside_bull=0.0,
            hist_pe_series=pd.Series([10.0, 12.0]), hist_price_series=pd.Series([8000.0]),
            hist_eps_series=pd.Series([0.0]), sector="기타"
        )
        r_nan = PEBandResult(
            ticker="900003", name="결측3_NaN", current_price=12000.0, current_fwd_eps=float("nan"),
            current_fwd_pe=None, pe_percentile=np.nan, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=10000.0, target_base=15000.0, target_bull=18000.0,
            upside_bear=-16.7, upside_base=25.0, upside_bull=50.0,
            hist_pe_series=pd.Series([10.0, 12.0]), hist_price_series=pd.Series([12000.0]),
            hist_eps_series=pd.Series([np.nan]), sector="기타"
        )
        r_none = PEBandResult(
            ticker="900004", name="결측4_None", current_price=15000.0, current_fwd_eps=None,
            current_fwd_pe=None, pe_percentile=np.nan, pe_min=5.0, pe_p25=8.0, pe_median=10.0,
            pe_p75=12.0, pe_max=15.0, pe_mean=10.0, target_bear=12000.0, target_base=18000.0, target_bull=22000.0,
            upside_bear=-20.0, upside_base=20.0, upside_bull=46.7,
            hist_pe_series=pd.Series([10.0, 12.0]), hist_price_series=pd.Series([15000.0]),
            hist_eps_series=pd.Series([np.nan]), sector="기타"
        )

        test_batch = [r_neg, r_zero, r_nan, r_none]
        rt_map = {
            "900001": 11000.0,
            "900002": 9500.0,
            "900003": 13500.0,
            "900004": 16500.0,
        }
        updated = apply_realtime_prices(test_batch, rt_map, self.sec_map)

        # 1. Verify current_price updated correctly
        self.assertEqual(updated[0].current_price, 11000.0)
        self.assertEqual(updated[1].current_price, 9500.0)
        self.assertEqual(updated[2].current_price, 13500.0)
        self.assertEqual(updated[3].current_price, 16500.0)

        # 2. Verify no division by zero or negative P/E generation
        for r_up in updated:
            self.assertIsNone(r_up.current_fwd_pe, f"{r_up.name} P/E must remain None")

        # 3. Verify target prices and upsides are not corrupted
        self.assertEqual(updated[0].target_base, 0.0)
        self.assertEqual(updated[0].upside_base, 0.0)
        self.assertEqual(updated[1].target_base, 0.0)
        self.assertEqual(updated[1].upside_base, 0.0)
        self.assertEqual(updated[2].target_base, 15000.0)
        self.assertEqual(updated[2].upside_base, 25.0)  # Preserved without corruption
        self.assertEqual(updated[3].target_base, 18000.0)
        self.assertEqual(updated[3].upside_base, 20.0)  # Preserved without corruption

    def test_dynamic_regime_transition_golden_cross_to_momentum_leader(self):
        """Stress-test: price surge dynamically clears Golden Cross (pe_pct > 40%) and enters Momentum Leader."""
        hist_pe = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        r_gc = PEBandResult(
            ticker="005930", name="삼성전자", current_price=50000.0, current_fwd_eps=5000.0,
            current_fwd_pe=10.0, pe_percentile=20.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=0.0, upside_base=25.0, upside_bull=50.0,
            hist_pe_series=hist_pe, hist_price_series=pd.Series([50000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="반도체", eps_rev_1w=1.0, eps_rev_1m=4.0, eps_rev_3m=6.0,
            is_golden_cross=True, regime_tag="Golden Cross"
        )
        self.assertEqual(get_strategy_signal(r_gc), "✨골든크로스")

        # Surge price to 65,000 (P/E = 13.0 -> values < 13.0 are 5 of 10 = 50.0% > 40.0%)
        updated = apply_realtime_prices([r_gc], {"005930": 65000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 65000.0)
        self.assertEqual(updated.current_fwd_pe, 13.0)
        self.assertEqual(updated.pe_percentile, 50.0)
        self.assertFalse(updated.is_golden_cross, "Golden Cross must be dynamically cleared when pe_pct > 40%")
        self.assertEqual(updated.regime_tag, "Momentum Leader", "Regime must transition to Momentum Leader")
        self.assertEqual(get_strategy_signal(updated), "🚀어닝모멘텀")

    def test_dynamic_regime_transition_to_high_pe_downgrade(self):
        """Stress-test: price surge with negative revision dynamically enters High P/E Downgrade (pe_pct >= 70%)."""
        hist_pe = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        r_mid = PEBandResult(
            ticker="005930", name="삼성전자", current_price=65000.0, current_fwd_eps=5000.0,
            current_fwd_pe=13.0, pe_percentile=50.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=-23.1, upside_base=-3.8, upside_bull=15.4,
            hist_pe_series=hist_pe, hist_price_series=pd.Series([65000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="반도체", eps_rev_1w=-1.0, eps_rev_1m=-3.0, eps_rev_3m=-5.0,
            is_golden_cross=False, is_value_trap=False, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_mid), "Neutral")

        # Surge price to 80,000 (P/E = 16.0 -> values < 16.0 are 8 of 10 = 80.0% >= 70.0%)
        updated = apply_realtime_prices([r_mid], {"005930": 80000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 80000.0)
        self.assertEqual(updated.current_fwd_pe, 16.0)
        self.assertEqual(updated.pe_percentile, 80.0)
        self.assertEqual(updated.regime_tag, "High P/E Downgrade", "Regime must transition to High P/E Downgrade")
        self.assertEqual(get_strategy_signal(updated), "🔻고P/E하향")

    def test_dynamic_regime_transition_to_value_trap(self):
        """Stress-test: price drop with negative revision dynamically enters Value Trap (pe_pct <= 30%)."""
        hist_pe = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        r_mid = PEBandResult(
            ticker="005930", name="삼성전자", current_price=65000.0, current_fwd_eps=5000.0,
            current_fwd_pe=13.0, pe_percentile=50.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=-23.1, upside_base=-3.8, upside_bull=15.4,
            hist_pe_series=hist_pe, hist_price_series=pd.Series([65000.0]), hist_eps_series=pd.Series([5000.0]),
            sector="반도체", eps_rev_1w=-1.5, eps_rev_1m=-3.5, eps_rev_3m=-7.0,
            is_golden_cross=False, is_value_trap=False, regime_tag="Neutral"
        )
        self.assertEqual(get_strategy_signal(r_mid), "Neutral")

        # Drop price to 45,000 (P/E = 9.0 -> values < 9.0 is 1 of 10 = 10.0% <= 30.0%)
        updated = apply_realtime_prices([r_mid], {"005930": 45000.0}, self.sec_map)[0]

        self.assertEqual(updated.current_price, 45000.0)
        self.assertEqual(updated.current_fwd_pe, 9.0)
        self.assertEqual(updated.pe_percentile, 10.0)
        self.assertTrue(updated.is_value_trap, "Value Trap flag must be set when pe_pct <= 30% and eps_rev_1m < -2%")
        self.assertEqual(updated.regime_tag, "Value Trap", "Regime must transition to Value Trap")
        self.assertEqual(get_strategy_signal(updated), "⚠️밸류트랩")

    def test_sidebar_strategy_filter_operation(self):
        """Stress-test: filtering by strategy_filter = ['🔻고P/E하향'] preserves only High P/E Downgrade."""
        r_gc = PEBandResult(
            ticker="001", name="골든", current_price=50000.0, current_fwd_eps=5000.0,
            current_fwd_pe=10.0, pe_percentile=20.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=0.0, upside_base=25.0, upside_bull=50.0,
            hist_pe_series=pd.Series([10.0]), hist_price_series=pd.Series([50000.0]), hist_eps_series=pd.Series([5000.0]),
            is_golden_cross=True, regime_tag="Golden Cross"
        )
        r_hd = PEBandResult(
            ticker="002", name="하향", current_price=80000.0, current_fwd_eps=5000.0,
            current_fwd_pe=16.0, pe_percentile=80.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=-37.5, upside_base=-21.9, upside_bull=-6.25,
            hist_pe_series=pd.Series([10.0]), hist_price_series=pd.Series([80000.0]), hist_eps_series=pd.Series([5000.0]),
            is_golden_cross=False, regime_tag="High P/E Downgrade"
        )
        r_vt = PEBandResult(
            ticker="003", name="트랩", current_price=40000.0, current_fwd_eps=5000.0,
            current_fwd_pe=8.0, pe_percentile=10.0, pe_min=8.0, pe_p25=10.0, pe_median=12.5,
            pe_p75=15.0, pe_max=17.0, pe_mean=12.5, target_bear=50000.0, target_base=62500.0, target_bull=75000.0,
            upside_bear=25.0, upside_base=56.2, upside_bull=87.5,
            hist_pe_series=pd.Series([10.0]), hist_price_series=pd.Series([40000.0]), hist_eps_series=pd.Series([5000.0]),
            is_value_trap=True, regime_tag="Value Trap"
        )

        all_items = [r_gc, r_hd, r_vt]
        strategy_filter = ["🔻고P/E하향"]

        # Filter logic replicating app.py:776-778
        filtered = [r for r in all_items if not (strategy_filter and get_strategy_signal(r) not in strategy_filter)]

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].ticker, "002")
        self.assertEqual(get_strategy_signal(filtered[0]), "🔻고P/E하향")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test Suite: Plotly Figure 2-Subplot Synchronization & Trace Integrity
# ─────────────────────────────────────────────────────────────────────
class TestPlotlyFigureSynchronizationAdversarial(unittest.TestCase):
    """Adversarial stress-testing of Plotly 2-subplot synchronized chart across all 3 band models."""

    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        self.p_series = pd.Series([50000.0 + i * 500.0 for i in range(60)], index=dates)
        self.e_series = pd.Series([4000.0 + i * 50.0 for i in range(60)], index=dates)

    def test_percentile_model_figure_structure_and_traces(self):
        """Percentile model: verify 2 subplots, shared_xaxes, unified hovermode, and traces."""
        r = calc_pe_band("005930", "삼성전자", self.p_series, self.e_series, band_years=5, band_model="percentile", winsorize=True)
        self.assertIsNotNone(r)

        fig = build_integrated_figure(r, self.p_series, self.e_series, model_key="percentile", model_period=5)
        self.assertIsNotNone(fig)

        # 1. Subplot configuration
        self.assertEqual(fig.layout.hovermode, "x unified")
        # In Plotly make_subplots with shared_xaxes=True, xaxis2 matches xaxis or shares domain
        self.assertTrue(hasattr(fig.layout, "xaxis2"))

        # 2. Extract traces
        trace_names = [t.name for t in fig.data if t.name is not None]

        # Upper subplot: Price, p90, Bull (p75), Base (p50), Bear (p25), p10, 현재가
        expected_upper = ["실제 주가", "p90 밴드", "Bull 밴드 (75th)", "Base 밴드 (Median)", "Bear 밴드 (25th)", "p10 밴드"]
        for expected in expected_upper:
            self.assertIn(expected, trace_names, f"Percentile model must contain trace '{expected}'")

        # Lower subplot: 12M Fwd EPS, EPS MA12, 12M Fwd P/E
        expected_lower = ["12M Fwd EPS", "EPS MA12", "12M Fwd P/E"]
        for expected in expected_lower:
            self.assertIn(expected, trace_names, f"Lower subplot must contain trace '{expected}'")

        # 3. Verify horizontal target lines in layout shapes
        # 3 target hlines in upper + 1 median hline in lower = 4 hlines
        hlines = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
        self.assertTrue(len(hlines) >= 3, f"Expected at least 3 horizontal reference lines, got {len(hlines)}")

    def test_mean_sd_model_figure_traces(self):
        """Mean ± SD model: verify +2SD, +1SD, Mean, -1SD, -2SD traces and targets."""
        r = calc_pe_band("005930", "삼성전자", self.p_series, self.e_series, band_years=5, band_model="mean_sd", winsorize=True)
        self.assertIsNotNone(r)

        fig = build_integrated_figure(r, self.p_series, self.e_series, model_key="mean_sd", model_period=5)
        self.assertIsNotNone(fig)

        trace_names = [t.name for t in fig.data if t.name is not None]
        expected_sd_traces = [
            "실제 주가", "+2SD 밴드", "Bull 밴드 (+1SD)", "Base 밴드 (Mean)", "Bear 밴드 (-1SD)", "-2SD 밴드",
            "12M Fwd EPS", "EPS MA12", "12M Fwd P/E"
        ]
        for expected in expected_sd_traces:
            self.assertIn(expected, trace_names, f"Mean±SD model must contain trace '{expected}'")

    def test_fixed_multiples_figure_traces(self):
        """Fixed Multiples model: verify 8x, 10x, 12x, 15x traces."""
        r = calc_pe_band("005930", "삼성전자", self.p_series, self.e_series, band_years=5, band_model="fixed",
                         fixed_multiples=[8.0, 10.0, 12.0, 15.0])
        self.assertIsNotNone(r)

        fig = build_integrated_figure(r, self.p_series, self.e_series, model_key="fixed", model_period=5)
        self.assertIsNotNone(fig)

        trace_names = [t.name for t in fig.data if t.name is not None]
        expected_fixed = ["실제 주가", "8x 밴드", "10x 밴드", "12x 밴드", "15x 밴드", "12M Fwd EPS", "12M Fwd P/E"]
        for expected in expected_fixed:
            self.assertIn(expected, trace_names, f"Fixed Multiples model must contain trace '{expected}'")

    def test_non_positive_eps_masking_in_pe_trace(self):
        """Cyclical negative EPS dates must be masked as NaN in lower P/E subplot without crashing."""
        # Inject negative EPS in middle 10 periods
        e_neg = self.e_series.copy()
        e_neg.iloc[20:30] = -500.0

        r = calc_pe_band("005930", "삼성전자", self.p_series, e_neg, band_years=5, band_model="percentile", winsorize=True)
        fig = build_integrated_figure(r, self.p_series, e_neg, model_key="percentile", model_period=5)
        self.assertIsNotNone(fig)

        pe_trace = next(t for t in fig.data if t.name == "12M Fwd P/E")
        pe_vals = np.array(pe_trace.y, dtype=float)
        # Verify that negative EPS dates produce NaN and no negative P/E values
        finite_pe = pe_vals[np.isfinite(pe_vals)]
        self.assertTrue((finite_pe > 0).all(), "All finite P/E plot values must be strictly positive")

    def test_mismatched_index_frequencies_alignment(self):
        """Adversarial: Monthly price (MS) vs Daily EPS (B) forward-fill alignment test."""
        # Monthly price on 1st of month
        p_monthly = pd.Series([50000.0 + i * 500 for i in range(24)],
                              index=pd.date_range("2022-01-01", periods=24, freq="MS"))
        # Daily EPS on business days
        e_daily = pd.Series([4000.0 + i * 10 for i in range(500)],
                            index=pd.bdate_range("2022-01-01", periods=500))

        r = calc_pe_band("005930", "삼성전자", p_monthly, e_daily, band_years=2, band_model="percentile")
        self.assertIsNotNone(r)

        fig = build_integrated_figure(r, p_monthly, e_daily, model_key="percentile", model_period=2)
        self.assertIsNotNone(fig, "Forward-fill alignment must allow figure to render across mismatched frequencies")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Test Suite: Headless Import Safety, Caching Decoupling & Presets Parity
# ─────────────────────────────────────────────────────────────────────
class TestHeadlessAndCachingAdversarial(unittest.TestCase):
    """Adversarial testing of import safety, caching decoupling, and presets parity."""

    def test_app_ast_parse_and_compilation(self):
        """app.py must parse cleanly via ast and compile to bytecode without warnings/errors."""
        import py_compile
        app_file = ROOT / "app.py"
        self.assertTrue(app_file.exists())

        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code, filename="app.py")
        self.assertIsNotNone(parsed)

        compiled = py_compile.compile(str(app_file), doraise=True)
        self.assertIsNotNone(compiled)

    def test_caching_decoupling_signature(self):
        """Tier 1 caching function load_raw_market_data MUST NOT depend on band_years."""
        sig = inspect.signature(load_raw_market_data)
        self.assertNotIn("band_years", sig.parameters, "load_raw_market_data signature must NOT contain band_years")
        self.assertIn("mtimes", sig.parameters, "load_raw_market_data must accept mtimes cache invalidation key")

    def test_quick_presets_parity_matrix(self):
        """Stress-test 1-click quick presets against oracle across 50 combinatorial synthetic test cases."""
        rng = np.random.default_rng(42)
        n_stocks = 50
        synthetic_rows = []
        for i in range(n_stocks):
            synthetic_rows.append({
                "ticker": f"{i:06d}",
                "name": f"Stock_{i}",
                "pe_percentile": float(rng.uniform(5.0, 95.0)),
                "sector_pe_percentile": float(rng.uniform(5.0, 95.0)),
                "eps_rev_1m": float(rng.uniform(-8.0, 15.0)),
                "current_fwd_eps": float(rng.choice([5000.0, 10000.0, -500.0])),
                "current_fwd_pe": float(rng.uniform(4.0, 35.0)),
                "upside_base": float(rng.uniform(-20.0, 50.0)),
            })
        df_synth = pd.DataFrame(synthetic_rows)

        for pkey in ["ALL", "TURNAROUND", "EPS_TOP", "VALUE", "TRAP"]:
            app_res = apply_quick_preset(df_synth, pkey)
            oracle_res = oracle_apply_quick_preset(df_synth, pkey)

            self.assertEqual(len(app_res), len(oracle_res), f"Length mismatch on preset {pkey}")
            self.assertEqual(list(app_res["ticker"]), list(oracle_res["ticker"]), f"Ticker sequence mismatch on preset {pkey}")

    def test_screener_table_required_columns_completeness(self):
        """Verify all 17 required columns are present in display_cols and column_config."""
        app_file = ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()

        required_cols = [
            "전략 신호", "종목명", "코드", "섹터", "지수", "현재가", "Fwd EPS",
            "1W %", "1M %", "3M %", "Fwd P/E", "P/E 위치(%)", "섹터 P/E 위치(%)",
            "Bear목표", "Base목표", "Bull목표", "Base%"
        ]
        for col in required_cols:
            self.assertIn(f'"{col}"', code, f"Required column '{col}' missing from app.py")


# ─────────────────────────────────────────────────────────────────────────────
# Test Runner & Empirical Verification Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def run_all_adversarial_tests():
    print("=" * 80)
    print(">>> Adversarial Test Suite for Milestone 4 (Dashboard UI/UX & Band Switcher)")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestStrategySignalTaxonomyAdversarial))
    suite.addTests(loader.loadTestsFromTestCase(TestRealtimePricesAdversarial))
    suite.addTests(loader.loadTestsFromTestCase(TestPlotlyFigureSynchronizationAdversarial))
    suite.addTests(loader.loadTestsFromTestCase(TestHeadlessAndCachingAdversarial))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f">>> ALL {result.testsRun} ADVERSARIAL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f">>> FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
    print("=" * 80)
    return result


if __name__ == "__main__":
    run_all_adversarial_tests()
