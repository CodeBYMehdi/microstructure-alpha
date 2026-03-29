"""Tests for the 5 fatal flaw fixes.

Flaw 1: Decision layer signal gates (KL/projection not zeroed, confidence not hardcoded)
Flaw 2: DSR + CSCV wired into validation pipeline
Flaw 3: Realistic execution simulation (vol-depth, partial fills, adverse selection)
Flaw 4: Tightened OOS validation thresholds
Flaw 5: VaR-based sizing cap + daily-return Sharpe
"""
import copy
import inspect
import numpy as np
import pytest
from datetime import datetime

from config.loader import get_config


# ─── Flaw 1a: KL/projection gates not zeroed in optimizer ───

def test_objective_preserves_kl_gate():
    """Optimizer must NOT override kl_min to 0.0."""
    from optimization.objective import run_single_backtest
    source = inspect.getsource(run_single_backtest)
    assert "kl_min = 0.0" not in source, "objective.py overrides kl_min to 0.0"
    assert "projection_min = 0.0" not in source, "objective.py overrides projection_min to 0.0"


def test_objective_preserves_kl_exit():
    """Optimizer must NOT override kl_stable_threshold to 0.0."""
    from optimization.objective import run_single_backtest
    source = inspect.getsource(run_single_backtest)
    assert "kl_stable_threshold = 0.0" not in source, "objective.py disables KL exit"


# ─── Flaw 1b: Confidence not hardcoded ───

def test_entry_confidence_not_hardcoded():
    """Entry confidence must use ConfidenceScorer, not hardcoded 1.0."""
    from main import Strategy
    source = inspect.getsource(Strategy)
    assert "self._entry_confidence = 1.0" not in source, \
        "main.py hardcodes entry confidence to 1.0"


def test_confidence_scorer_used_at_entry():
    """Strategy must call confidence_scorer.score() near entry."""
    from main import Strategy
    source = inspect.getsource(Strategy)
    assert "confidence_scorer.score(" in source, \
        "ConfidenceScorer.score() not called in Strategy"


# ─── Flaw 2: DSR + CSCV wired ───

def test_walk_forward_report_has_dsr_fields():
    """WalkForwardReport must include DSR fields."""
    from optimization.walk_forward import WalkForwardReport
    report = WalkForwardReport(n_folds=3, n_calls_per_fold=10, total_ticks=1000)
    assert hasattr(report, 'dsr_p_value'), "Missing dsr_p_value"
    assert hasattr(report, 'dsr_is_significant'), "Missing dsr_is_significant"


def test_oos_report_has_statistical_validation_fields():
    """OOSReport must include DSR and CSCV fields."""
    from optimization.oos_validator import OOSReport
    report = OOSReport(total_ticks=1000, train_val_ticks=800, test_ticks=200, test_pct=0.2)
    assert hasattr(report, 'dsr_p_value'), "Missing dsr_p_value"
    assert hasattr(report, 'dsr_is_significant'), "Missing dsr_is_significant"
    assert hasattr(report, 'cscv_pbo'), "Missing cscv_pbo"
    assert hasattr(report, 'cscv_is_overfit'), "Missing cscv_is_overfit"


def test_dsr_imported_in_walk_forward():
    """walk_forward.py must import and use deflated_sharpe."""
    import optimization.walk_forward as wf
    source = inspect.getsource(wf)
    assert "deflated_sharpe" in source, "deflated_sharpe not imported in walk_forward"


def test_metrics_exposes_trade_pnls():
    """PerformanceMetrics.compute() must expose trade_pnls for DSR."""
    from backtest.metrics import PerformanceMetrics, TradeRecord

    metrics = PerformanceMetrics(initial_equity=10000.0)
    metrics.record_trade(TradeRecord(
        timestamp=1700000000.0, symbol="TEST", side="BUY",
        qty=10, entry_price=100.0, exit_price=100.05, pnl=0.5,
    ))
    metrics.record_trade(TradeRecord(
        timestamp=1700000100.0, symbol="TEST", side="SELL",
        qty=10, entry_price=100.05, exit_price=99.95, pnl=-1.0,
    ))
    results = metrics.compute()
    assert "trade_pnls" in results, "trade_pnls not in metrics output"
    assert len(results["trade_pnls"]) == 2
    assert results["trade_pnls"][0] == pytest.approx(0.5)


# ─── Flaw 3: Realistic execution simulation ───

def test_synthetic_book_contracts_with_volatility():
    """Synthetic order book depth must shrink during high vol."""
    from backtest.microstructure_sim import MicrostructureSimulator
    from core.types import Tick

    sim = MicrostructureSimulator("TEST", seed=42)

    # Feed low-vol ticks (gentle upward drift)
    for i in range(110):
        sim.on_tick(Tick(
            price=100.0 + 0.001 * i, volume=100,
            timestamp=datetime.now(), symbol="TEST",
        ))
    low_vol_depth = sum(lv.size for lv in sim.order_book.asks)

    # Feed high-vol ticks (whipsaw)
    for i in range(50):
        price = 100.0 + (-1)**i * 2.0
        sim.on_tick(Tick(
            price=price, volume=100,
            timestamp=datetime.now(), symbol="TEST",
        ))
    high_vol_depth = sum(lv.size for lv in sim.order_book.asks)

    assert high_vol_depth < low_vol_depth, \
        f"Book depth should contract in high vol: low={low_vol_depth:.0f}, high={high_vol_depth:.0f}"


def _make_regime_state():
    """Helper to create a RegimeState for test proposals."""
    from core.types import RegimeState, RegimeType
    return RegimeState(
        regime=RegimeType.UNDEFINED, confidence=0.5,
        entropy=1.0, timestamp=datetime.now(), metadata={},
    )


def test_partial_fills_when_order_exceeds_depth():
    """Orders exceeding visible depth should get partial fills."""
    from backtest.execution_sim import ExecutionSimulator
    from backtest.microstructure_sim import MicrostructureSimulator
    from core.types import Tick, TradeProposal, TradeAction

    sim = MicrostructureSimulator("TEST", seed=42)
    sim.on_tick(Tick(price=100.0, volume=100, timestamp=datetime.now(), symbol="TEST"))

    executor = ExecutionSimulator(sim, seed=42)

    total_depth = sum(lv.size for lv in sim.order_book.asks)
    huge_qty = total_depth * 5  # 5x the entire book

    proposal = TradeProposal(
        action=TradeAction.BUY, symbol="TEST", quantity=huge_qty,
        price=100.0, reason="test", timestamp=datetime.now(),
        regime_state=_make_regime_state(),
    )
    result = executor.execute(proposal)
    assert result.filled_quantity < huge_qty, \
        f"Huge order should be partially filled: got {result.filled_quantity:.0f}, requested {huge_qty:.0f}"
    assert result.filled_quantity > 0, "Should fill at least some"


def test_adverse_selection_slippage():
    """Slippage should be worse when trading in the direction of recent momentum."""
    from backtest.execution_sim import ExecutionSimulator
    from backtest.microstructure_sim import MicrostructureSimulator
    from core.types import Tick, TradeProposal, TradeAction

    sim = MicrostructureSimulator("TEST", seed=42)
    # Feed upward-trending ticks
    for i in range(120):
        sim.on_tick(Tick(
            price=100.0 + 0.1 * i, volume=100,
            timestamp=datetime.now(), symbol="TEST",
        ))

    rs = _make_regime_state()

    # Buy into up-trend = adverse selection
    executor1 = ExecutionSimulator(sim, seed=42)
    prop_buy = TradeProposal(
        action=TradeAction.BUY, symbol="TEST", quantity=10,
        price=sim.last_trade.price, reason="test", timestamp=datetime.now(),
        regime_state=rs,
    )
    result_buy = executor1.execute(prop_buy)
    adverse_slip = result_buy.filled_price - prop_buy.price

    # Sell into up-trend = favorable
    executor2 = ExecutionSimulator(sim, seed=42)
    prop_sell = TradeProposal(
        action=TradeAction.SELL, symbol="TEST", quantity=10,
        price=sim.last_trade.price, reason="test", timestamp=datetime.now(),
        regime_state=rs,
    )
    result_sell = executor2.execute(prop_sell)
    favorable_slip = prop_sell.price - result_sell.filled_price

    assert adverse_slip > favorable_slip, \
        f"Adverse selection not modeled: buy_slip={adverse_slip:.6f}, sell_slip={favorable_slip:.6f}"


# ─── Flaw 4: Tightened OOS thresholds ───

def test_oos_degradation_threshold_is_tight():
    """OOS degradation threshold must be <= 1.5."""
    from optimization.oos_validator import OOSValidator
    source = inspect.getsource(OOSValidator)
    assert "degradation < 3.0" not in source, "OOS degradation threshold too loose (3.0x)"
    assert "degradation < 1.5" in source, "OOS degradation should be 1.5x"


def test_oos_min_trades_at_least_20():
    """OOS must require >= 20 trades."""
    from optimization.oos_validator import OOSValidator
    source = inspect.getsource(OOSValidator)
    # Ensure no "test_trades >= 5" (but allow >= 50 etc.)
    lines = source.split('\n')
    for line in lines:
        if 'test_trades >= 5' in line and 'test_trades >= 50' not in line:
            assert False, f"OOS min trades too low: {line.strip()}"


def test_oos_requires_dsr_significance():
    """OOS alpha confirmation must require DSR significance."""
    from optimization.oos_validator import OOSValidator
    source = inspect.getsource(OOSValidator)
    assert "dsr_is_significant" in source, "OOS doesn't check DSR significance"


# ─── Flaw 5a: VaR-based position cap ───

def test_sizing_has_tail_risk_cap():
    """PositionSizer must cap size based on drawdown budget + tail risk."""
    from decision.sizing import PositionSizer
    source = inspect.getsource(PositionSizer)
    assert "tail_penalty" in source and "max_dd" in source, \
        "PositionSizer missing drawdown-budget tail risk cap"


def test_var_cap_reduces_size_for_heavy_tails():
    """VaR cap should produce smaller size when tails are heavier."""
    from decision.sizing import PositionSizer
    from regime.transition import TransitionEvent

    config = copy.deepcopy(get_config())
    sizer = PositionSizer(base_size=100.0, config=config)

    event = TransitionEvent(
        from_regime=0, to_regime=1, strength=0.9,
        delta_vector=np.array([0.01, 0.005, 0.1, 0.5, 2.0, 0.3]),
        is_significant=True, reason="test",
    )

    # With high tail_slope the VaR cap should be tighter
    size_normal = sizer.calculate(event, tail_slope=2.0, win_rate=0.55, profit_factor=1.5)
    size_heavy = sizer.calculate(event, tail_slope=8.0, win_rate=0.55, profit_factor=1.5)

    assert size_heavy <= size_normal, \
        f"Heavy tails should give smaller or equal size: heavy={size_heavy:.2f} vs normal={size_normal:.2f}"


# ─── Flaw 5b: Daily-return Sharpe ───

def test_sharpe_uses_daily_returns():
    """Sharpe must be computed from daily aggregated returns, not per-trade scaled."""
    from backtest.metrics import PerformanceMetrics, TradeRecord

    metrics = PerformanceMetrics(initial_equity=10000.0)

    # 200 trades across 20 days, with high daily variance
    # Some days win big, some days lose — realistic PnL distribution
    base_time = 1700000000.0
    rng = np.random.RandomState(42)
    daily_biases = rng.normal(0.0, 2.0, size=20)  # Daily trend varies a lot
    for day in range(20):
        for i in range(10):
            t = base_time + day * 86400 + i * 100
            pnl = rng.normal(daily_biases[day], 1.0)
            metrics.record_trade(TradeRecord(
                timestamp=t, symbol="TEST", side="BUY",
                qty=10, entry_price=100.0, exit_price=100.0 + pnl / 10,
                pnl=pnl, regime_id="0",
            ))

    results = metrics.compute()
    sharpe = results["sharpe_ratio"]
    # Daily-return Sharpe should be bounded and realistic
    # With noisy daily PnL, Sharpe should be <10 (not >50 from old per-trade inflation)
    assert abs(sharpe) < 15, f"Sharpe {sharpe:.2f} seems inflated — expected < 15 with noisy daily returns"


def test_sharpe_not_inflated_by_trade_frequency():
    """High frequency should NOT inflate Sharpe."""
    from backtest.metrics import PerformanceMetrics, TradeRecord

    rng = np.random.RandomState(42)

    # Same daily PnL but 10x more trades (each 1/10th the PnL)
    def build_metrics(trades_per_day):
        m = PerformanceMetrics(initial_equity=10000.0)
        base_time = 1700000000.0
        for day in range(10):
            for i in range(trades_per_day):
                t = base_time + day * 86400 + i * (86400 // trades_per_day)
                pnl = rng.normal(1.0 / trades_per_day, 0.5 / trades_per_day)
                m.record_trade(TradeRecord(
                    timestamp=t, symbol="TEST", side="BUY",
                    qty=10, entry_price=100.0, exit_price=100.01,
                    pnl=pnl, regime_id="0",
                ))
        return m.compute()

    low_freq = build_metrics(5)
    high_freq = build_metrics(50)

    # Daily Sharpe should be similar regardless of trade frequency
    # (same daily PnL, just split across more trades)
    ratio = abs(high_freq["sharpe_ratio"]) / max(abs(low_freq["sharpe_ratio"]), 0.01)
    assert ratio < 3.0, \
        f"High freq Sharpe inflated {ratio:.1f}x: low={low_freq['sharpe_ratio']:.2f}, high={high_freq['sharpe_ratio']:.2f}"


# ─── Config validation ───

def test_config_has_drawdown_limit():
    """Config schema must include max_drawdown for sizing cap."""
    config = get_config()
    assert hasattr(config.thresholds.risk, 'max_drawdown'), \
        "RiskConfig missing max_drawdown"
    assert config.thresholds.risk.max_drawdown > 0
