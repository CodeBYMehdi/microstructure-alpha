# verif rapide
# entrainement intensif

import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt
import optuna

from statistics.deflated_sharpe import deflated_sharpe

from config.schema import AppConfig
from optimization.search_space import (
    get_defaults,
    get_param_names,
    suggest_params,
    apply_params,
)
from optimization.objective import (
    create_optuna_objective,
    run_single_backtest,
    _compute_sharpe,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging (we log our own summaries)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class FoldResult:
    # l'usine a gaz
    fold_idx: int
    train_size: int
    test_size: int
    best_params: Dict[str, Any]
    train_score: float          # Negated objective (lower = better)
    train_sharpe: float
    test_sharpe: float
    test_max_drawdown: float
    test_total_trades: int
    test_total_pnl: float
    test_transitions: int
    optimization_time_sec: float
    overfitting_ratio: float    # train_sharpe / test_sharpe (target <= 1.5)


@dataclass
class WalkForwardReport:
    # l'usine a gaz
    n_folds: int
    n_calls_per_fold: int
    total_ticks: int
    folds: List[FoldResult] = field(default_factory=list)
    # Aggregates
    mean_oos_sharpe: float = 0.0
    std_oos_sharpe: float = 0.0
    mean_overfitting_ratio: float = 0.0
    best_overall_params: Dict[str, Any] = field(default_factory=dict)
    total_time_sec: float = 0.0
    # Deflated Sharpe Ratio -- multiple testing correction
    dsr_p_value: float = 1.0
    dsr_is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WalkForwardValidator:
    # verif rapide
    # entrainement intensif

    def __init__(
        self,
        base_config: AppConfig,
        tick_list: List[Any],
        n_folds: int = 5,
        n_calls: int = 30,
        min_train_pct: float = 0.4,
        tick_limit_per_run: Optional[int] = None,
        symbol: str = "SPY",
        dd_target: float = 0.05,
        random_state: int = 42,
    ):
        self.base_config = base_config
        self.tick_list = tick_list
        self.n_folds = n_folds
        self.n_calls = n_calls
        self.min_train_pct = min_train_pct
        self.tick_limit = tick_limit_per_run
        self.symbol = symbol
        self.dd_target = dd_target
        self.random_state = random_state

        self._validate_inputs()

    def _validate_inputs(self):
        n = len(self.tick_list)
        if n < 100:
            raise ValueError(f"Need at least 100 ticks, got {n}")
        if self.n_folds < 2:
            raise ValueError(f"Need at least 2 folds, got {self.n_folds}")
        if self.min_train_pct < 0.2 or self.min_train_pct >= 1.0:
            raise ValueError(f"min_train_pct must be in [0.2, 1.0), got {self.min_train_pct}")

    def get_fold_splits(self) -> List[Tuple[int, int, int]]:
        # verif rapide
        # la calculette
        n = len(self.tick_list)
        test_pct = (1.0 - self.min_train_pct) / self.n_folds
        splits = []

        for i in range(self.n_folds):
            train_end = int(n * (self.min_train_pct + i * test_pct))
            test_start = train_end
            test_end = int(n * (self.min_train_pct + (i + 1) * test_pct))

            # Last fold takes everything remaining
            if i == self.n_folds - 1:
                test_end = n

            splits.append((train_end, test_start, test_end))

        return splits

    def run(self) -> WalkForwardReport:
        # le bif
        total_start = time.time()
        splits = self.get_fold_splits()
        defaults = get_defaults()

        report = WalkForwardReport(
            n_folds=self.n_folds,
            n_calls_per_fold=self.n_calls,
            total_ticks=len(self.tick_list),
        )

        oos_sharpes = []
        best_global_score = float("inf")
        best_global_params = defaults

        for fold_idx, (train_end, test_start, test_end) in enumerate(splits):
            fold_start = time.time()
            logger.info(
                f"=== Fold {fold_idx + 1}/{self.n_folds}: "
                f"Train [0:{train_end}] ({train_end} ticks), "
                f"Test [{test_start}:{test_end}] ({test_end - test_start} ticks) ==="
            )

            # Slice data
            train_ticks = self.tick_list[:train_end]
            test_ticks = self.tick_list[test_start:test_end]

            # Create Optuna objective for this fold's training data
            optuna_obj = create_optuna_objective(
                self.base_config, train_ticks, self.tick_limit, self.symbol, self.dd_target,
            )

            # Run TPE optimization on training fold
            try:
                sampler = optuna.samplers.TPESampler(
                    seed=self.random_state + fold_idx,
                    n_startup_trials=max(5, self.n_calls // 3),
                )
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                )

                # Enqueue defaults as first trial
                study.enqueue_trial(defaults)
                study.optimize(optuna_obj, n_trials=self.n_calls, show_progress_bar=False)

                best_params = study.best_params
                best_train_score = study.best_value

                # Log parameter importance if enough trials
                if len(study.trials) >= 10:
                    try:
                        importance = optuna.importance.get_param_importances(study)
                        top_3 = list(importance.items())[:3]
                        logger.info(
                            f"  Top params: {', '.join(f'{k}={v:.3f}' for k, v in top_3)}"
                        )
                    except Exception:
                        pass  # Importance computation can fail with few completed trials

            except Exception as e:
                logger.error(f"Optimization failed on fold {fold_idx}: {e}")
                best_params = defaults
                best_train_score = 100.0

            # Evaluate on training set (for overfitting detection)
            train_config = apply_params(self.base_config, best_params)
            train_results = run_single_backtest(
                train_config, train_ticks, self.tick_limit, self.symbol,
            )
            train_sharpe = _compute_sharpe(train_results)

            # Evaluate on test set
            test_config = apply_params(self.base_config, best_params)
            test_results = run_single_backtest(
                test_config, test_ticks, self.tick_limit, self.symbol,
            )
            test_sharpe = _compute_sharpe(test_results)

            # Overfitting ratio
            if abs(test_sharpe) > 0.01:
                overfitting_ratio = abs(train_sharpe / test_sharpe)
            else:
                overfitting_ratio = float("inf") if abs(train_sharpe) > 0.1 else 1.0

            fold_time = time.time() - fold_start

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_size=train_end,
                test_size=test_end - test_start,
                best_params=best_params,
                train_score=float(best_train_score),
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                test_max_drawdown=abs(test_results.get("max_drawdown", 0.0)),
                test_total_trades=test_results.get("total_trades", 0),
                test_total_pnl=test_results.get("total_pnl", 0.0),
                test_transitions=test_results.get("transitions", 0),
                optimization_time_sec=fold_time,
                overfitting_ratio=overfitting_ratio,
            )

            report.folds.append(fold_result)
            oos_sharpes.append(test_sharpe)

            # Track best global
            if best_train_score < best_global_score:
                best_global_score = best_train_score
                best_global_params = best_params

            logger.info(
                f"  Fold {fold_idx + 1} done: "
                f"Train Sharpe={train_sharpe:.4f}, "
                f"Test Sharpe={test_sharpe:.4f}, "
                f"Overfit Ratio={overfitting_ratio:.2f}, "
                f"Time={fold_time:.1f}s"
            )

        # Aggregates
        report.mean_oos_sharpe = float(np.mean(oos_sharpes))
        report.std_oos_sharpe = float(np.std(oos_sharpes))
        report.mean_overfitting_ratio = float(np.mean([
            f.overfitting_ratio for f in report.folds
            if np.isfinite(f.overfitting_ratio)
        ]) if report.folds else 0.0)
        report.best_overall_params = best_global_params
        report.total_time_sec = time.time() - total_start

        # Deflated Sharpe Ratio -- correct for multiple testing across folds + param combos
        n_trials = self.n_folds * self.n_calls
        # Use total OOS trades as observation count (not tick count -- that inflates significance)
        n_oos_trades = sum(f.test_total_trades for f in report.folds)
        if report.mean_oos_sharpe != 0.0 and n_oos_trades >= 2:
            oos_arr = np.array(oos_sharpes)
            dsr_result = deflated_sharpe(
                observed_sharpe=report.mean_oos_sharpe,
                n_trials=n_trials,
                n_observations=max(n_oos_trades, 2),
                skewness=float(_scipy_skew(oos_arr)) if len(oos_arr) > 2 else 0.0,
                kurtosis=float(_scipy_kurt(oos_arr)) if len(oos_arr) > 2 else 0.0,
            )
            report.dsr_p_value = dsr_result.p_value
            report.dsr_is_significant = dsr_result.is_significant
            logger.info(
                f"  DSR: p-value={dsr_result.p_value:.4f}, "
                f"significant={dsr_result.is_significant} "
                f"(n_trials={n_trials}, n_obs={n_oos_trades})"
            )

        logger.info(
            f"\n=== Walk-Forward Complete ===\n"
            f"  Mean OOS Sharpe: {report.mean_oos_sharpe:.4f} +/- {report.std_oos_sharpe:.4f}\n"
            f"  Mean Overfit Ratio: {report.mean_overfitting_ratio:.2f}\n"
            f"  DSR p-value: {report.dsr_p_value:.4f} (significant={report.dsr_is_significant})\n"
            f"  Total Time: {report.total_time_sec:.1f}s"
        )

        return report
