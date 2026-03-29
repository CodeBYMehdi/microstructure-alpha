"""Deflated Sharpe Ratio and Combinatory Symmetric Cross-Validation.

Implements:
1. Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014)
   Corrects observed Sharpe for multiple testing bias.
   
2. CSCV — Combinatory Symmetric Cross-Validation  
   Detects overfitting by comparing in-sample vs out-of-sample
   across all possible train/test split combinations.

3. Annualized Sharpe (proper, time-weighted)

Usage:
    result = deflated_sharpe(
        observed_sharpe=1.5,
        n_trials=100,         # How many backtests you've run
        n_observations=252,   # How many returns in the backtest
        skewness=-0.5,
        kurtosis=4.0,
    )
    print(f"DSR p-value: {result.p_value}")
    # p_value > 0.05 means your Sharpe is likely noise
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio calculation."""
    observed_sharpe: float      # The Sharpe you measured
    expected_max_sharpe: float  # Expected maximum Sharpe under null (E[max(SR)])
    deflated_sharpe: float      # Adjusted Sharpe after multiple testing correction
    p_value: float              # Probability this Sharpe occurred by chance
    is_significant: bool        # p_value < significance_level
    n_trials: int               # Number of backtests/parameter sets tried
    n_observations: int         # Number of return observations
    significance_level: float   # Alpha level used


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,      # Excess kurtosis (normal = 0, total = 3)
    significance_level: float = 0.05,
    sharpe_std_trials: float = 1.0,  # Assumed std of Sharpe across trials
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    
    The key insight: if you've tried N parameter combinations and picked the
    best Sharpe, the expected maximum Sharpe under the null hypothesis (no alpha)
    grows with N. The DSR corrects for this.
    
    Args:
        observed_sharpe: The Sharpe ratio you measured
        n_trials: Number of backtests/parameter combinations tried
        n_observations: Number of return observations in the backtest
        skewness: Return distribution skewness
        kurtosis: Return distribution EXCESS kurtosis (normal = 0)
        significance_level: Alpha for significance test
        sharpe_std_trials: Assumed std of Sharpe ratios across your trials
    
    Returns:
        DeflatedSharpeResult with p-value and significance determination
    """
    if n_trials < 1:
        n_trials = 1
    if n_observations < 2:
        n_observations = 2

    # ── Step 1: Expected maximum Sharpe under null ──
    # E[max(SR)] ≈ std(SR) * [(1-γ)*Φ⁻¹(1-1/N) + γ*Φ⁻¹(1-1/(N*e))]
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    euler_gamma = 0.5772156649
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max = sharpe_std_trials * ((1 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        expected_max = 0.0

    # ── Step 2: Standard error of the Sharpe estimate ──
    # SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR²) / (T-1))
    # where kurt is excess kurtosis
    sr = observed_sharpe
    variance_term = max(0.0, 1 - skewness * sr + (kurtosis) / 4.0 * sr ** 2)
    se_sr = np.sqrt(variance_term / max(n_observations - 1, 1))

    if se_sr < 1e-10:
        se_sr = 1e-10

    # ── Step 3: Deflated Sharpe = test statistic ──
    # DSR = (SR_observed - E[max(SR)]) / SE(SR)
    dsr = (observed_sharpe - expected_max) / se_sr

    # ── Step 4: p-value (one-sided test: is SR significantly > expected max?) ──
    p_value = 1.0 - stats.norm.cdf(dsr)

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max,
        deflated_sharpe=dsr,
        p_value=p_value,
        is_significant=p_value < significance_level,
        n_trials=n_trials,
        n_observations=n_observations,
        significance_level=significance_level,
    )


def annualized_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Compute properly annualized Sharpe ratio.
    
    Args:
        returns: Array of periodic returns (daily, per-trade, etc.)
        risk_free_rate: Risk-free rate per period
        periods_per_year: Number of periods per year (252 for daily, etc.)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess < 1e-15:
        return 0.0

    return float(mean_excess / std_excess * np.sqrt(periods_per_year))


def probabilistic_sharpe(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(SR > SR*).
    
    What is the probability that the true Sharpe exceeds a benchmark?
    
    Returns:
        Probability [0, 1] that true Sharpe exceeds benchmark_sharpe
    """
    if n_observations < 2:
        return 0.5

    se_sr = np.sqrt(
        (1 - skewness * observed_sharpe + (kurtosis) / 4.0 * observed_sharpe ** 2)
        / max(n_observations - 1, 1)
    )

    if se_sr < 1e-10:
        return 1.0 if observed_sharpe > benchmark_sharpe else 0.0

    z = (observed_sharpe - benchmark_sharpe) / se_sr
    return float(stats.norm.cdf(z))


@dataclass
class CSCVResult:
    """Result of Combinatory Symmetric Cross-Validation."""
    probability_of_overfitting: float  # PBO: [0, 1], higher = more overfit
    avg_oos_sharpe: float
    std_oos_sharpe: float
    avg_is_sharpe: float
    avg_logit_lambda: float  # Average logit (rank of OOS optimal / N)
    n_combinations: int
    is_overfit: bool  # PBO > 0.5


def cscv(
    trial_returns: np.ndarray,
    n_partitions: int = 16,
) -> CSCVResult:
    """Combinatory Symmetric Cross-Validation (Bailey et al., 2015).
    
    Detects overfitting by checking if the best in-sample strategy
    performs well out-of-sample across all possible IS/OOS splits.
    
    Args:
        trial_returns: Matrix of shape (n_observations, n_trials)
            Each column is the return series from a different parameter set.
        n_partitions: Number of partitions (S) to split observations into.
            Must be even.
    
    Returns:
        CSCVResult with probability of overfitting (PBO)
    """
    n_obs, n_trials = trial_returns.shape
    if n_partitions % 2 != 0:
        n_partitions += 1

    # Split observations into S groups
    group_size = n_obs // n_partitions
    if group_size < 2:
        return CSCVResult(
            probability_of_overfitting=0.5,
            avg_oos_sharpe=0.0, std_oos_sharpe=0.0,
            avg_is_sharpe=0.0, avg_logit_lambda=0.0,
            n_combinations=0, is_overfit=True,
        )

    groups = []
    for i in range(n_partitions):
        start = i * group_size
        end = start + group_size if i < n_partitions - 1 else n_obs
        groups.append(trial_returns[start:end, :])

    # Generate all C(S, S/2) combinations
    from itertools import combinations
    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))

    logit_lambdas = []
    oos_sharpes = []
    is_sharpes = []
    n_combos = 0

    for is_indices in all_combos:
        oos_indices = tuple(i for i in range(n_partitions) if i not in is_indices)

        # Build IS and OOS return matrices
        is_returns = np.vstack([groups[i] for i in is_indices])
        oos_returns = np.vstack([groups[i] for i in oos_indices])

        # Compute Sharpe for each trial, IS and OOS
        is_sharpes_trial = np.array([_sharpe(is_returns[:, j]) for j in range(n_trials)])
        oos_sharpes_trial = np.array([_sharpe(oos_returns[:, j]) for j in range(n_trials)])

        # Find the best IS trial
        best_is_idx = np.argmax(is_sharpes_trial)
        best_is_sharpe = is_sharpes_trial[best_is_idx]
        best_oos_sharpe = oos_sharpes_trial[best_is_idx]

        is_sharpes.append(best_is_sharpe)
        oos_sharpes.append(best_oos_sharpe)

        # Rank of the IS-optimal trial in OOS
        oos_ranks = np.argsort(np.argsort(oos_sharpes_trial))  # rank array
        oos_rank_of_best = oos_ranks[best_is_idx]
        relative_rank = oos_rank_of_best / max(n_trials - 1, 1)

        # Logit transformation: logit(rank / N)
        # Clip to avoid log(0)
        relative_rank = np.clip(relative_rank, 0.01, 0.99)
        logit_lambda = np.log(relative_rank / (1 - relative_rank))
        logit_lambdas.append(logit_lambda)

        n_combos += 1

    # PBO = probability that logit(lambda) < 0
    # i.e., the IS-optimal strategy ranks below median OOS
    logit_array = np.array(logit_lambdas)
    pbo = float(np.mean(logit_array < 0))

    return CSCVResult(
        probability_of_overfitting=pbo,
        avg_oos_sharpe=float(np.mean(oos_sharpes)),
        std_oos_sharpe=float(np.std(oos_sharpes)),
        avg_is_sharpe=float(np.mean(is_sharpes)),
        avg_logit_lambda=float(np.mean(logit_lambdas)),
        n_combinations=n_combos,
        is_overfit=pbo > 0.5,
    )


def _sharpe(returns: np.ndarray) -> float:
    """Simple Sharpe ratio for internal use."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(returns) / std)


def alpha_half_life(
    pnl_series: np.ndarray,
    window: int = 50,
) -> Tuple[float, bool]:
    """Estimate the half-life of alpha decay.
    
    Uses rolling Sharpe ratio and fits exponential decay to detect
    whether alpha is decaying over time.
    
    Returns:
        (half_life_windows, is_decaying) — half_life < 0 means no decay
    """
    if len(pnl_series) < window * 3:
        return -1.0, False

    # Compute rolling Sharpe
    rolling_sharpes = []
    for i in range(window, len(pnl_series)):
        chunk = pnl_series[i - window:i]
        std = np.std(chunk)
        if std > 1e-10:
            rolling_sharpes.append(np.mean(chunk) / std)
        else:
            rolling_sharpes.append(0.0)

    if len(rolling_sharpes) < 20:
        return -1.0, False

    # Fit linear trend to log(abs(sharpe)) to detect exponential decay
    sharpe_arr = np.array(rolling_sharpes)
    abs_sharpe = np.abs(sharpe_arr)
    abs_sharpe = np.maximum(abs_sharpe, 1e-10)
    log_sharpe = np.log(abs_sharpe)

    x = np.arange(len(log_sharpe))
    # Linear regression on log(|SR|) vs time
    slope, _, r_value, p_value, _ = stats.linregress(x, log_sharpe)

    is_decaying = slope < 0 and p_value < 0.10  # Even 10% significance

    # Half-life = ln(2) / |decay_rate|
    if slope < -1e-10:
        half_life = -np.log(2) / slope
    else:
        half_life = -1.0  # No decay / growing

    return float(half_life), is_decaying
