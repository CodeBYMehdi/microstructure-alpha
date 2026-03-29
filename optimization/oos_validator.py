# on cherche les pepites
# la tuyauterie de donnees

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt

from config.schema import AppConfig
from statistics.deflated_sharpe import deflated_sharpe, cscv
from optimization.search_space import (
    get_defaults,
    get_param_names,
    apply_params,
)
from optimization.objective import run_single_backtest, _compute_sharpe
from optimization.walk_forward import WalkForwardValidator, WalkForwardReport

logger = logging.getLogger(__name__)


@dataclass
class OOSResult:
    # verif rapide
    test_sharpe: float
    test_total_trades: int
    test_total_pnl: float
    test_win_rate: float
    test_max_drawdown: float
    test_profit_factor: float
    test_expectancy: float
    # Comparison with walk-forward results
    wf_mean_oos_sharpe: float
    degradation_ratio: float  # test_sharpe / wf_mean_oos_sharpe (target >= 0.7)
    is_alpha_real: bool       # True if test_sharpe > 0 and degradation < 3x


@dataclass
class OOSReport:
    # l'usine a gaz
    total_ticks: int
    train_val_ticks: int
    test_ticks: int
    test_pct: float
    # Walk-forward results on train+val
    walk_forward_report: Optional[Dict[str, Any]] = None
    # Sacred test results
    oos_result: Optional[OOSResult] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    total_time_sec: float = 0.0
    # Statistical validation
    dsr_p_value: float = 1.0
    dsr_is_significant: bool = False
    cscv_pbo: float = 0.5       # Probability of Backtest Overfitting [0, 1]
    cscv_is_overfit: bool = True
    # Verdict
    verdict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.walk_forward_report is not None:
            d['walk_forward_report'] = self.walk_forward_report
        return d


class OOSValidator:
    # on cherche les pepites
    # la tuyauterie de donnees

    def __init__(
        self,
        base_config: AppConfig,
        tick_list: List[Any],
        test_pct: float = 0.20,
        n_wf_folds: int = 5,
        n_calls: int = 30,
        min_train_pct: float = 0.4,
        tick_limit_per_run: Optional[int] = None,
        symbol: str = "SPY",
        dd_target: float = 0.05,
        random_state: int = 42,
    ):
        # le cerveau de l'operation
        # la tuyauterie de donnees
        self.base_config = base_config
        self.tick_list = tick_list
        self.test_pct = test_pct
        self.n_wf_folds = n_wf_folds
        self.n_calls = n_calls
        self.min_train_pct = min_train_pct
        self.tick_limit = tick_limit_per_run
        self.symbol = symbol
        self.dd_target = dd_target
        self.random_state = random_state

        self._validate_inputs()

        # Split data
        n = len(tick_list)
        self._split_idx = int(n * (1.0 - test_pct))
        self._train_val_ticks = tick_list[:self._split_idx]
        self._test_ticks = tick_list[self._split_idx:]

        logger.info(
            f"OOS Validator initialized: "
            f"{len(self._train_val_ticks)} train+val ticks, "
            f"{len(self._test_ticks)} test ticks ({test_pct:.0%} hold-out)"
        )

    def _validate_inputs(self):
        n = len(self.tick_list)
        if n < 200:
            raise ValueError(f"Need at least 200 ticks for OOS validation, got {n}")
        if self.test_pct < 0.10 or self.test_pct > 0.40:
            raise ValueError(f"test_pct must be in [0.10, 0.40], got {self.test_pct}")

        test_size = int(n * self.test_pct)
        train_val_size = n - test_size
        if train_val_size < 100:
            raise ValueError(
                f"train+val portion too small ({train_val_size} ticks). "
                f"Reduce test_pct or provide more data."
            )

    def run(self) -> OOSReport:
        # on cherche les pepites
        # la tuyauterie de donnees
        total_start = time.time()

        report = OOSReport(
            total_ticks=len(self.tick_list),
            train_val_ticks=len(self._train_val_ticks),
            test_ticks=len(self._test_ticks),
            test_pct=self.test_pct,
        )

        # ── Step 1: Walk-forward on train+val ──
        logger.info("=" * 60)
        logger.info("STEP 1: Walk-Forward Optimization on Train+Val data")
        logger.info("=" * 60)

        wf_validator = WalkForwardValidator(
            base_config=self.base_config,
            tick_list=self._train_val_ticks,
            n_folds=self.n_wf_folds,
            n_calls=self.n_calls,
            min_train_pct=self.min_train_pct,
            tick_limit_per_run=self.tick_limit,
            symbol=self.symbol,
            dd_target=self.dd_target,
            random_state=self.random_state,
        )

        wf_report = wf_validator.run()
        report.walk_forward_report = wf_report.to_dict()
        report.best_params = wf_report.best_overall_params

        logger.info(
            f"Walk-forward complete: "
            f"Mean OOS Sharpe={wf_report.mean_oos_sharpe:.4f}, "
            f"Overfit Ratio={wf_report.mean_overfitting_ratio:.2f}"
        )

        # ── Step 2: Sacred Test Set Evaluation ──
        logger.info("=" * 60)
        logger.info("STEP 2: SACRED TEST SET — Final evaluation (NEVER TOUCHED before)")
        logger.info(f"Test set: {len(self._test_ticks)} ticks")
        logger.info("=" * 60)

        # Apply best parameters from walk-forward
        test_config = apply_params(self.base_config, report.best_params)

        # Run backtest on test set
        test_results = run_single_backtest(
            test_config, self._test_ticks, self.tick_limit, self.symbol,
        )

        test_sharpe = _compute_sharpe(test_results)
        test_trades = test_results.get("total_trades", 0)
        test_pnl = test_results.get("total_pnl", 0.0)
        test_win_rate = test_results.get("win_rate", 0.0)
        test_max_dd = abs(test_results.get("max_drawdown", 0.0))
        test_pf = test_results.get("profit_factor", 0.0)
        test_expectancy = test_results.get("expectancy", 0.0)

        # ── Step 3: Statistical Validation ──
        wf_sharpe = wf_report.mean_oos_sharpe
        if abs(wf_sharpe) > 0.01:
            degradation = abs(wf_sharpe / test_sharpe) if abs(test_sharpe) > 0.01 else float("inf")
        else:
            degradation = 1.0 if abs(test_sharpe) < 0.01 else float("inf")

        # DSR on sacred test set — correct for all trials run during optimization
        test_pnls = test_results.get("trade_pnls", [])
        n_total_trials = self.n_wf_folds * self.n_calls
        if len(test_pnls) >= 2:
            pnl_arr = np.array(test_pnls)
            dsr_result = deflated_sharpe(
                observed_sharpe=test_sharpe,
                n_trials=n_total_trials,
                n_observations=len(pnl_arr),
                skewness=float(_scipy_skew(pnl_arr)),
                kurtosis=float(_scipy_kurt(pnl_arr)),
            )
            report.dsr_p_value = dsr_result.p_value
            report.dsr_is_significant = dsr_result.is_significant
            logger.info(
                f"  Sacred Test DSR: p-value={dsr_result.p_value:.4f}, "
                f"significant={dsr_result.is_significant}"
            )

        # CSCV — if we have per-fold OOS trade PnLs, run combinatory cross-validation
        # Use walk-forward fold Sharpes as a proxy for CSCV input
        wf_folds = wf_report.folds if hasattr(wf_report, 'folds') else []
        if len(wf_folds) >= 4:
            # PBO approximation: fraction of folds where IS-optimal degrades OOS
            overfit_count = sum(
                1 for f in wf_folds
                if f.overfitting_ratio > 2.0 or f.test_sharpe < 0
            )
            report.cscv_pbo = overfit_count / len(wf_folds)
            report.cscv_is_overfit = report.cscv_pbo > 0.5
            logger.info(
                f"  CSCV proxy: PBO={report.cscv_pbo:.2f}, "
                f"overfit={report.cscv_is_overfit}"
            )

        # Alpha is "real" if ALL conditions met (institutional standard):
        # 1. Test Sharpe > 0 (positive alpha on unseen data)
        # 2. Degradation < 1.5x (tightened from 3x)
        # 3. At least 20 trades (tightened from 5)
        # 4. DSR significant (multiple-testing correction passes)
        # 5. CSCV PBO < 0.5 (not overfit)
        is_alpha_real = (
            test_sharpe > 0.0 and
            degradation < 1.5 and
            test_trades >= 20 and
            report.dsr_is_significant and
            not report.cscv_is_overfit
        )

        oos_result = OOSResult(
            test_sharpe=test_sharpe,
            test_total_trades=test_trades,
            test_total_pnl=test_pnl,
            test_win_rate=test_win_rate,
            test_max_drawdown=test_max_dd,
            test_profit_factor=test_pf,
            test_expectancy=test_expectancy,
            wf_mean_oos_sharpe=wf_sharpe,
            degradation_ratio=degradation,
            is_alpha_real=is_alpha_real,
        )
        report.oos_result = oos_result
        report.total_time_sec = time.time() - total_start

        # Generate verdict
        if is_alpha_real:
            report.verdict = (
                f"✅ ALPHA CONFIRMED — Test Sharpe={test_sharpe:.4f} "
                f"(WF OOS={wf_sharpe:.4f}, degradation={degradation:.2f}x). "
                f"DSR p={report.dsr_p_value:.4f}, PBO={report.cscv_pbo:.2f}. "
                f"{test_trades} trades, PnL={test_pnl:.4f}, WR={test_win_rate:.1%}"
            )
        elif test_trades < 20:
            report.verdict = (
                f"⚠️ INSUFFICIENT TRADES — Only {test_trades} trades on test set. "
                f"Need at least 20. Cannot draw conclusions."
            )
        elif not report.dsr_is_significant:
            report.verdict = (
                f"⚠️ DSR FAILED — Test Sharpe={test_sharpe:.4f} but DSR p={report.dsr_p_value:.4f} "
                f"(not significant after {n_total_trials} trials). "
                f"Observed Sharpe is likely noise from multiple testing."
            )
        elif report.cscv_is_overfit:
            report.verdict = (
                f"⚠️ OVERFIT DETECTED — PBO={report.cscv_pbo:.2f}. "
                f"Test Sharpe={test_sharpe:.4f} but {int(report.cscv_pbo*100)}% of folds show overfitting."
            )
        elif test_sharpe > 0 and degradation >= 1.5:
            report.verdict = (
                f"⚠️ POSSIBLE OVERFITTING — Test Sharpe={test_sharpe:.4f} is positive "
                f"but {degradation:.1f}x worse than WF OOS={wf_sharpe:.4f}. "
                f"Alpha may be partially real but inflated by optimization."
            )
        else:
            report.verdict = (
                f"❌ NO ALPHA — Test Sharpe={test_sharpe:.4f}. "
                f"Strategy does not generate positive returns on unseen data. "
                f"Previous results were likely overfit."
            )

        logger.info("\n" + "=" * 60)
        logger.info("OOS VALIDATION VERDICT")
        logger.info("=" * 60)
        logger.info(report.verdict)
        logger.info(f"Total time: {report.total_time_sec:.1f}s")

        return report

    def save_report(self, report: OOSReport, filepath: str = "oos_validation_report.json"):
        # on garde une trace
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            logger.info(f"OOS report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save OOS report: {e}")
