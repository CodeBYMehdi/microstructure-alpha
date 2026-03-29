# le cerveau de l'operation
# on cherche les pepites

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    # le cerveau de l'operation
    benchmark_name: str
    benchmark_return: float
    strategy_return: float
    excess_return: float         # strategy - benchmark
    information_ratio: float     # excess / tracking_error
    outperformance_pct: float    # % of periods where strategy > benchmark
    max_drawdown_benchmark: float
    max_drawdown_strategy: float


class BenchmarkComparison:
    # le cerveau de l'operation

    def __init__(self, initial_capital: float = 100000.0, n_random_trials: int = 1000,
                 random_state: int = 42):
        self.initial_capital = initial_capital
        self.n_random_trials = n_random_trials
        self.rng = np.random.RandomState(random_state)

    def compare_all(
        self,
        strategy_trades: List[Dict],
        price_series: np.ndarray,
    ) -> Dict[str, BenchmarkResult]:
        # le cerveau de l'operation
        # simu pour pas pleurer en live
        results = {}

        strategy_pnls = [t.get("pnl", 0.0) for t in strategy_trades]
        strategy_return = sum(strategy_pnls)

        # 1. Buy-and-Hold
        bh = self._buy_and_hold(price_series)
        results["buy_and_hold"] = self._compare(strategy_return, strategy_pnls, bh, "Buy & Hold")

        # 2. Random Entry
        random_ret, random_pnls = self._random_entry(strategy_trades, price_series)
        results["random_entry"] = self._compare(strategy_return, strategy_pnls, random_pnls, "Random Entry")

        # 3. Inverse Strategy
        inverse_pnls = [-p for p in strategy_pnls]
        inverse_return = sum(inverse_pnls)
        results["inverse"] = self._compare(strategy_return, strategy_pnls, inverse_pnls, "Inverse Strategy")

        return results

    def _buy_and_hold(self, price_series: np.ndarray) -> List[float]:
        # la calculette
        # ca fait bim
        if len(price_series) < 2:
            return [0.0]

        # Buy at first price, hold through
        returns = np.diff(np.log(price_series))
        # Scale to same capital as strategy
        position_size = self.initial_capital / price_series[0]
        pnls = returns * position_size * price_series[:-1]

        return pnls.tolist()

    def _random_entry(self, strategy_trades: List[Dict],
                      price_series: np.ndarray) -> tuple:
        # l'usine a gaz
        n_trades = len(strategy_trades)
        if n_trades == 0 or len(price_series) < 10:
            return 0.0, [0.0]

        # Average holding period
        avg_hold = np.mean([t.get("hold_duration_windows", 10) for t in strategy_trades])
        avg_hold = max(5, int(avg_hold))

        all_trial_pnls = []

        for _ in range(min(self.n_random_trials, 100)):  # Limit computation
            trial_pnl = 0.0
            for _ in range(n_trades):
                # Random entry point
                entry_idx = self.rng.randint(0, max(1, len(price_series) - avg_hold - 1))
                exit_idx = min(entry_idx + avg_hold, len(price_series) - 1)

                entry_price = price_series[entry_idx]
                exit_price = price_series[exit_idx]

                # Random direction
                direction = self.rng.choice([-1, 1])
                pnl = direction * (exit_price - entry_price)
                trial_pnl += pnl

            all_trial_pnls.append(trial_pnl)

        avg_random_return = float(np.mean(all_trial_pnls))
        return avg_random_return, all_trial_pnls

    def _compare(self, strategy_return: float, strategy_pnls: List[float],
                 benchmark_pnls, name: str) -> BenchmarkResult:
        # le cerveau de l'operation
        # ca fait bim
        if isinstance(benchmark_pnls, list):
            bench_return = sum(benchmark_pnls)
            bench_array = np.array(benchmark_pnls)
        else:
            bench_return = float(np.sum(benchmark_pnls))
            bench_array = np.array(benchmark_pnls)

        strategy_array = np.array(strategy_pnls)

        excess = strategy_return - bench_return

        # Information Ratio: annualized excess / tracking error
        if len(strategy_array) > 1 and len(bench_array) > 1:
            # Align lengths
            min_len = min(len(strategy_array), len(bench_array))
            diff = strategy_array[:min_len] - bench_array[:min_len]
            tracking_error = float(np.std(diff))
            ir = float(np.mean(diff) / tracking_error) if tracking_error > 1e-10 else 0.0
        else:
            ir = 0.0

        # Outperformance pct
        if len(strategy_array) > 0 and len(bench_array) > 0:
            min_len = min(len(strategy_array), len(bench_array))
            outperf = float(np.mean(
                np.cumsum(strategy_array[:min_len]) > np.cumsum(bench_array[:min_len])
            ))
        else:
            outperf = 0.0

        return BenchmarkResult(
            benchmark_name=name,
            benchmark_return=bench_return,
            strategy_return=strategy_return,
            excess_return=excess,
            information_ratio=ir,
            outperformance_pct=outperf,
            max_drawdown_benchmark=self._max_drawdown(bench_array),
            max_drawdown_strategy=self._max_drawdown(strategy_array),
        )

    @staticmethod
    def _max_drawdown(pnls: np.ndarray) -> float:
        # la calculette
        # ca fait bim
        if len(pnls) == 0:
            return 0.0
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(np.abs(peak), 1e-10)
        return float(np.max(dd))

    def summary_string(self, results: Dict[str, BenchmarkResult]) -> str:
        # l'usine a gaz
        lines = ["=== Benchmark Comparison ===\n"]
        for name, r in results.items():
            verdict = "✅ OUTPERFORMS" if r.excess_return > 0 else "❌ UNDERPERFORMS"
            lines.append(f"  vs {r.benchmark_name}: {verdict}")
            lines.append(f"    Strategy Return: {r.strategy_return:+.2f}")
            lines.append(f"    Benchmark Return: {r.benchmark_return:+.2f}")
            lines.append(f"    Excess: {r.excess_return:+.2f}")
            lines.append(f"    Information Ratio: {r.information_ratio:.3f}")
            lines.append(f"    Outperformance: {r.outperformance_pct:.1%}")
            lines.append("")
        return "\n".join(lines)
