# le cerveau de l'operation
# verif rapide

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    # l'usine a gaz
    statistic: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    is_significant: bool  # CI doesn't contain 0


@dataclass
class PermutationTestResult:
    # le cerveau de l'operation
    # verif rapide
    observed_stat: float
    p_value: float
    n_permutations: int
    is_significant: bool
    significance_level: float


@dataclass
class ValidationReport:
    # le cerveau de l'operation
    # le bif
    total_return: float
    sharpe_ratio: float
    bootstrap_ci: BootstrapResult
    permutation_test: PermutationTestResult
    cost_sensitivity: Dict[str, float]
    n_trades: int
    is_statistically_significant: bool


class StrategyValidator:
    # stop le massacre
    # le cerveau de l'operation

    def __init__(self, n_bootstrap: int = 5000, n_permutations: int = 5000,
                 significance_level: float = 0.05, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.rng = np.random.RandomState(random_state)
        logger.info(
            f"StrategyValidator initialized: bootstrap={n_bootstrap}, "
            f"permutations={n_permutations}, alpha={significance_level}"
        )

    def validate(self, trade_pnls: List[float],
                 cost_multipliers: Optional[List[float]] = None) -> ValidationReport:
        # le cerveau de l'operation
        # verif rapide
        if not trade_pnls or len(trade_pnls) < 5:
            return ValidationReport(
                total_return=0.0, sharpe_ratio=0.0,
                bootstrap_ci=BootstrapResult("total_return", 0.0, 0.0, 0.0, 0.95, 0, False),
                permutation_test=PermutationTestResult(0.0, 1.0, 0, False, self.significance_level),
                cost_sensitivity={}, n_trades=len(trade_pnls),
                is_statistically_significant=False,
            )

        pnls = np.array(trade_pnls, dtype=float)

        total_return = float(np.sum(pnls))
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        sharpe = mean_pnl / std_pnl if std_pnl > 1e-10 else 0.0

        # 1. Bootstrap CI
        bootstrap_ci = self.bootstrap_confidence_interval(pnls, "total_return")

        # 2. Permutation test
        perm_test = self.permutation_test(pnls)

        # 3. Cost sensitivity
        if cost_multipliers is None:
            cost_multipliers = [1.0, 2.0, 5.0, 10.0]
        cost_sens = self.cost_sensitivity_analysis(pnls, cost_multipliers)

        is_sig = bootstrap_ci.is_significant and perm_test.is_significant

        return ValidationReport(
            total_return=total_return,
            sharpe_ratio=sharpe,
            bootstrap_ci=bootstrap_ci,
            permutation_test=perm_test,
            cost_sensitivity=cost_sens,
            n_trades=len(pnls),
            is_statistically_significant=is_sig,
        )

    def bootstrap_confidence_interval(
        self, pnls: np.ndarray, statistic: str = "total_return",
        confidence: float = 0.95,
    ) -> BootstrapResult:
        # la calculette
        n = len(pnls)

        if statistic == "total_return":
            observed = float(np.sum(pnls))
            stat_fn = np.sum
        elif statistic == "sharpe":
            mean = np.mean(pnls)
            std = np.std(pnls)
            observed = float(mean / std) if std > 1e-10 else 0.0
            def stat_fn(x):
                m, s = np.mean(x), np.std(x)
                return m / s if s > 1e-10 else 0.0
        else:
            observed = float(np.mean(pnls))
            stat_fn = np.mean

        # Bootstrap resampling
        boot_stats = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self.rng.choice(pnls, size=n, replace=True)
            boot_stats[i] = stat_fn(sample)

        alpha = 1 - confidence
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        # Significant if CI doesn't contain 0
        is_sig = (ci_lower > 0) or (ci_upper < 0)

        return BootstrapResult(
            statistic=statistic,
            point_estimate=observed,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence,
            n_bootstrap=self.n_bootstrap,
            is_significant=is_sig,
        )

    def permutation_test(self, pnls: np.ndarray) -> PermutationTestResult:
        # le cerveau de l'operation
        # verif rapide
        observed = float(np.sum(pnls))
        n = len(pnls)

        # Generate null distribution by randomly flipping PnL signs
        null_stats = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            signs = self.rng.choice([-1, 1], size=n)
            null_stats[i] = np.sum(pnls * signs)

        # Two-sided p-value
        p_value = float(np.mean(np.abs(null_stats) >= abs(observed)))

        return PermutationTestResult(
            observed_stat=observed,
            p_value=p_value,
            n_permutations=self.n_permutations,
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )

    def cost_sensitivity_analysis(
        self, pnls: np.ndarray, multipliers: List[float],
    ) -> Dict[str, float]:
        # le cerveau de l'operation
        # verif rapide
        result = {}
        losses = pnls[pnls < 0]
        if len(losses) == 0:
            for m in multipliers:
                result[f"{m}x_costs"] = float(np.sum(pnls))
            return result

        avg_loss_magnitude = float(np.mean(np.abs(losses)))

        for m in multipliers:
            additional_cost = avg_loss_magnitude * (m - 1) * (len(losses) / len(pnls))
            adjusted_pnls = pnls - additional_cost
            result[f"{m}x_costs"] = float(np.sum(adjusted_pnls))

        return result

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        # verif rapide
        n = len(p_values)
        adjusted_alpha = alpha / n
        return [p < adjusted_alpha for p in p_values]

    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        # l'usine a gaz
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # BH thresholds
        thresholds = alpha * np.arange(1, n + 1) / n

        # Find largest k where p(k) <= threshold(k)
        significant = np.zeros(n, dtype=bool)
        max_k = -1
        for k in range(n):
            if sorted_p[k] <= thresholds[k]:
                max_k = k

        if max_k >= 0:
            significant[sorted_indices[:max_k + 1]] = True

        return significant.tolist()
