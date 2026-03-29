# simu pour pas pleurer en live
# verif rapide

import logging
import itertools
import copy
from typing import Callable, List, Any, Optional, Dict

import numpy as np

from config.loader import get_config
from config.schema import AppConfig
from optimization.search_space import apply_params

logger = logging.getLogger(__name__)


# Penalty weights
DRAWDOWN_PENALTY_LAMBDA = 10.0
CHURN_PENALTY_LAMBDA = 1.0
MARKET_IMPACT_PENALTY_LAMBDA = 1.0  # penalize oversized trades vs top-of-book capacity
EXCEPTION_PENALTY = 100.0


def _compute_sharpe(results: Dict[str, Any], annualization_factor: float = 1.0) -> float:
    """Compute Sharpe ratio from backtest results.

    Uses the pre-computed sharpe_ratio from PerformanceMetrics (computed from the
    exact per-trade PnL sequence) when available. Falls back to a moment-based
    approximation only when the exact series isn't available.
    """
    n_trades = results.get("total_trades", 0)
    if n_trades < 2:
        return 0.0

    # Prefer the exact Sharpe computed from the full PnL sequence in metrics.compute()
    exact_sharpe = results.get("sharpe_ratio")
    if exact_sharpe is not None and np.isfinite(exact_sharpe):
        return float(np.clip(exact_sharpe, -10.0, 10.0))

    # Fallback: moment-based approximation (kept for backward compatibility
    # when sharpe_ratio is missing from results, e.g. partial/failed runs)
    avg_win = results.get("avg_win", 0.0)
    avg_loss = results.get("avg_loss", 0.0)
    win_rate = results.get("win_rate", 0.0)

    expected_pnl = win_rate * avg_win + (1 - win_rate) * avg_loss
    e_x2 = win_rate * (avg_win ** 2) + (1 - win_rate) * (avg_loss ** 2)
    variance = e_x2 - expected_pnl ** 2

    if variance <= 0 or not np.isfinite(variance):
        return 0.0 if expected_pnl <= 0 else 1.0

    std = np.sqrt(variance)
    sharpe = (expected_pnl / std) * np.sqrt(annualization_factor)
    return float(np.clip(sharpe, -10.0, 10.0))


def _compute_churn_penalty(results: Dict[str, Any]) -> float:
    # la calculette
    transitions = results.get("transitions", 0)
    n_trades = results.get("total_trades", 0)

    if n_trades == 0:
        return 0.0

    # on gueule si y a trop de transitions
    churn_ratio = transitions / max(1, n_trades)
    # ca douille au dela de 2.0
    return max(0.0, churn_ratio - 2.0)


def _compute_market_impact_penalty(results: Dict[str, Any]) -> float:
    """Penalize when average trade size would decimate the order book.

    If the optimizer proposes average sizes that exceed the synthetic TWAP
    capacity (based on SPY top-of-book depth assumptions), we penalize the
    Sharpe to prevent the optimizer from selecting parameters that produce
    illusory PnL by crossing unrealistic liquidity.

    The base capacity of 500 shares approximates 5% of the top-3 levels
    on SPY during regular trading hours (~3000-5000 shares visible depth
    per side at each tick level).
    """
    avg_size = results.get("avg_trade_size", 0.0)
    max_safe_liquidity = 500.0  # Base capacity assumption for SPY top-of-book slicing

    if avg_size > max_safe_liquidity:
        overstep = avg_size / max_safe_liquidity
        return max(0.0, (overstep - 1.0) * 2.0)
    return 0.0


def run_single_backtest(
    config: AppConfig,
    tick_list: List[Any],
    tick_limit: Optional[int] = None,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    # on ramene les datas
    # simu pour pas pleurer en live
    from backtest.backtest_agent import BacktestAgent
    from monitoring.event_bus import bus

    # on repart a zero sur les ticks
    if tick_limit:
        stream = itertools.islice(iter(tick_list), tick_limit)
    else:
        stream = iter(tick_list)

    # on force la config pour le backtest
    config = copy.deepcopy(config)
    config.thresholds.risk.slippage_tolerance = config.thresholds.backtest.relaxed_slippage_tolerance
    config.thresholds.risk.max_consecutive_errors = 100

    # on force pour l'instrument
    config.instruments.instruments[0].symbol = symbol
    config.instruments.instruments[0].tick_size = 0.01
    config.instruments.instruments[0].lot_size = 1.0
    config.instruments.instruments[0].min_notional = 100.0

    # on copie pour l'equity SPY
    config.thresholds.regime.transition_strength_min = 0.2
    config.thresholds.regime.window_size = 200
    config.thresholds.regime.update_frequency = 50
    config.thresholds.regime.min_cluster_size = 10
    config.thresholds.regime.min_samples = 3
    # Use config-defined gates — do NOT zero them out (disables signal filtering)
    # kl_min and projection_min provide meaningful information gradient gating
    config.thresholds.risk.confidence_floor = 0.15  # Low floor for backtest but not zero
    config.thresholds.risk.max_consecutive_errors = 100
    config.thresholds.decision.long.skew_min = 0.01
    config.thresholds.decision.long.tail_slope_min = 0.5
    config.thresholds.decision.short.volatility_min = 0.00001
    config.thresholds.decision.short.skew_max = -0.01
    config.thresholds.decision.short.kurtosis_min = 0.1
    config.thresholds.decision.exit.entropy_acceleration_threshold = 2.0
    config.thresholds.decision.exit.fallback_strength_threshold = 0.30
    # Keep KL exit threshold from config — signals edge decay
    config.thresholds.decision.liquidity.spread_max = 0.05  # target spread sur SPY
    config.thresholds.decision.liquidity.depth_slope_min = 0.0
    config.thresholds.decision.sizing.base_size = 10.0      # base 10 actions

    agent = BacktestAgent(data_stream=stream, config=config)
    agent.symbol = symbol
    agent.microstructure.symbol = symbol
    agent.strategy.min_persistence = 0

    try:
        results = agent.run()
    except Exception as e:
        logger.warning(f"Backtest run failed: {e}")
        results = agent.metrics.compute()

    return results


def objective(
    params: Dict[str, Any],
    base_config: AppConfig,
    tick_list: List[Any],
    tick_limit: Optional[int] = None,
    symbol: str = "SPY",
    dd_target: float = 0.05,
) -> float:
    # on ramene les datas
    # simu pour pas pleurer en live
    try:
        # on injecte les params
        config = apply_params(base_config, params)

        # on lance le test
        results = run_single_backtest(config, tick_list, tick_limit, symbol)

        # on calcule les scores
        sharpe = _compute_sharpe(results)
        max_dd = abs(results.get("max_drawdown", 0.0))
        churn_penalty = _compute_churn_penalty(results)
        impact_penalty = _compute_market_impact_penalty(results)

        # on punit les gros drawdowns
        dd_excess = max(0.0, max_dd - dd_target)

        # note globale (au plus haut au mieux)
        score = (
            sharpe
            - DRAWDOWN_PENALTY_LAMBDA * dd_excess
            - CHURN_PENALTY_LAMBDA * churn_penalty
            - MARKET_IMPACT_PENALTY_LAMBDA * impact_penalty
        )

        # on inverse pour minimiser
        return -score

    except Exception as e:
        logger.error(f"Objective function failed: {e}", exc_info=True)
        return EXCEPTION_PENALTY


def create_objective(
    base_config: AppConfig,
    tick_list: List[Any],
    tick_limit: Optional[int] = None,
    symbol: str = "SPY",
    dd_target: float = 0.05,
) -> Callable:
    """Return a callable that accepts a params dict. For sensitivity/diagnostics."""
    def _obj(params: Dict[str, Any]) -> float:
        return objective(params, base_config, tick_list, tick_limit, symbol, dd_target)
    return _obj


def create_optuna_objective(
    base_config: AppConfig,
    tick_list: List[Any],
    tick_limit: Optional[int] = None,
    symbol: str = "SPY",
    dd_target: float = 0.05,
) -> Callable:
    """Return an Optuna-compatible objective for study.optimize()."""
    import optuna
    from optimization.search_space import suggest_params

    def _obj(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        config = apply_params(base_config, params)
        results = run_single_backtest(config, tick_list, tick_limit, symbol)

        sharpe = _compute_sharpe(results)
        max_dd = abs(results.get("max_drawdown", 0.0))
        churn_penalty = _compute_churn_penalty(results)
        impact_penalty = _compute_market_impact_penalty(results)

        dd_excess = max(0.0, max_dd - dd_target)

        score = (
            sharpe
            - DRAWDOWN_PENALTY_LAMBDA * dd_excess
            - CHURN_PENALTY_LAMBDA * churn_penalty
            - MARKET_IMPACT_PENALTY_LAMBDA * impact_penalty
        )
        return -score

    return _obj
