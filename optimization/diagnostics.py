# attention aux degats
# le cerveau de l'operation

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from config.schema import AppConfig
from optimization.search_space import apply_params
from optimization.objective import run_single_backtest, _compute_sharpe

logger = logging.getLogger(__name__)


@dataclass
class TailRiskDiagnostics:
    # attention aux degats
    # simu pour pas pleurer en live
    total_trades: int
    win_rate: float
    profit_factor: float
    pnl_skewness: float
    worst_loss: float
    avg_win: float
    avg_loss: float
    # Derived
    payoff_ratio: float = 0.0    # |avg_win / avg_loss|
    kelly_fraction: float = 0.0  # win_rate - (1 - win_rate) / payoff_ratio


@dataclass
class RegimeStabilityDiagnostics:
    # l'usine a gaz
    n_regimes_detected: int
    total_transitions: int
    avg_dwell_time: float       # Mean time in each regime
    churn_rate: float           # transitions per 1000 ticks
    noise_trade_fraction: float # Fraction of trades in regime -1
    regime_pnl_concentration: float  # Gini coefficient of PnL across regimes


@dataclass
class DrawdownDiagnostics:
    # l'usine a gaz
    max_drawdown: float
    final_equity: float
    total_pnl: float
    return_pct: float           # total_pnl / initial_equity


@dataclass
class OverfitDiagnostics:
    # l'usine a gaz
    train_sharpe: float
    test_sharpe: float
    ratio: float                # train / test (target ≤ 1.5)
    assessment: str             # "good" / "moderate" / "overfit"


@dataclass
class DiagnosticsReport:
    # l'usine a gaz
    params: Dict[str, Any]
    tail_risk: TailRiskDiagnostics
    regime_stability: RegimeStabilityDiagnostics
    drawdown: DrawdownDiagnostics
    overfit: OverfitDiagnostics
    sharpe_ratio: float
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_gini(values: List[float]) -> float:
    # la calculette
    if not values or len(values) < 2:
        return 0.0
    arr = np.array(sorted(np.abs(values)))
    n = len(arr)
    if np.sum(arr) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def generate_diagnostics(
    optimal_params: Dict[str, Any],
    base_config: AppConfig,
    train_ticks: List[Any],
    test_ticks: List[Any],
    tick_limit: Optional[int] = None,
    symbol: str = "SPY",
    initial_equity: float = 10000.0,
) -> DiagnosticsReport:
    # attention aux degats
    # la tuyauterie de donnees
    from optimization.objective import _compute_churn_penalty

    config = apply_params(base_config, optimal_params)

    # Run on train and test
    train_results = run_single_backtest(config, train_ticks, tick_limit, symbol)
    test_results = run_single_backtest(config, test_ticks, tick_limit, symbol)

    train_sharpe = _compute_sharpe(train_results)
    test_sharpe = _compute_sharpe(test_results)

    # --- Tail Risk ---
    avg_win = test_results.get("avg_win", 0.0)
    avg_loss = test_results.get("avg_loss", 0.0)
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    win_rate = test_results.get("win_rate", 0.0)

    kelly = 0.0
    if payoff_ratio > 0 and np.isfinite(payoff_ratio):
        kelly = win_rate - (1 - win_rate) / payoff_ratio

    tail_risk = TailRiskDiagnostics(
        total_trades=test_results.get("total_trades", 0),
        win_rate=win_rate,
        profit_factor=test_results.get("profit_factor", 0.0),
        pnl_skewness=test_results.get("pnl_skewness", 0.0),
        worst_loss=test_results.get("worst_transition_loss", 0.0),
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        kelly_fraction=kelly,
    )

    # --- Regime Stability ---
    regime_analytics = test_results.get("regime_analytics", {})
    n_regimes = len([k for k in regime_analytics if str(k) != "-1"])
    transitions = test_results.get("transitions", 0)

    # Avg dwell time
    dwell_times = [v.get("dwell_time", 0.0) for v in regime_analytics.values()]
    avg_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0

    # Churn rate per 1000 ticks (approximate)
    total_trades = test_results.get("total_trades", 0)
    churn_rate = (transitions / max(1, total_trades)) * 1000

    # Noise fraction
    noise_count = regime_analytics.get("-1", {}).get("count", 0)
    noise_count += regime_analytics.get(-1, {}).get("count", 0)
    noise_fraction = noise_count / max(1, total_trades)

    # PnL concentration (Gini)
    regime_pnls = [v.get("total_pnl", 0.0) for k, v in regime_analytics.items() if str(k) != "-1"]
    pnl_gini = _compute_gini(regime_pnls)

    regime_stability = RegimeStabilityDiagnostics(
        n_regimes_detected=n_regimes,
        total_transitions=transitions,
        avg_dwell_time=avg_dwell,
        churn_rate=churn_rate,
        noise_trade_fraction=noise_fraction,
        regime_pnl_concentration=pnl_gini,
    )

    # --- Drawdown ---
    total_pnl = test_results.get("total_pnl", 0.0)
    final_equity = test_results.get("final_equity", initial_equity)
    max_dd = abs(test_results.get("max_drawdown", 0.0))

    drawdown = DrawdownDiagnostics(
        max_drawdown=max_dd,
        final_equity=final_equity,
        total_pnl=total_pnl,
        return_pct=total_pnl / initial_equity if initial_equity > 0 else 0.0,
    )

    # --- Overfitting ---
    if abs(test_sharpe) > 0.01:
        overfit_ratio = abs(train_sharpe / test_sharpe)
    else:
        overfit_ratio = float("inf") if abs(train_sharpe) > 0.1 else 1.0

    if overfit_ratio <= 1.5:
        assessment = "good"
    elif overfit_ratio <= 3.0:
        assessment = "moderate"
    else:
        assessment = "overfit"

    overfit = OverfitDiagnostics(
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe,
        ratio=overfit_ratio,
        assessment=assessment,
    )

    # --- Composite ---
    composite = -test_sharpe  # Matches objective convention (negated)

    report = DiagnosticsReport(
        params=optimal_params,
        tail_risk=tail_risk,
        regime_stability=regime_stability,
        drawdown=drawdown,
        overfit=overfit,
        sharpe_ratio=test_sharpe,
        composite_score=composite,
    )

    logger.info(
        f"\n=== Diagnostics Report ===\n"
        f"  Sharpe (OOS): {test_sharpe:.4f}\n"
        f"  Max Drawdown: {max_dd:.2%}\n"
        f"  Win Rate: {win_rate:.2%}\n"
        f"  Regimes: {n_regimes}, Transitions: {transitions}\n"
        f"  Overfit: {assessment} (ratio={overfit_ratio:.2f})\n"
        f"  Kelly: {kelly:.4f}"
    )

    return report
