# test de resistance en boucle
# on balance pleins de seeds pr voir si on survit

import copy
import numpy as np
import os
import matplotlib
if os.environ.get("INTERACTIVE_MC", "0") == "0":
    matplotlib.use('Agg')  # Non-interactive backend for headless
import matplotlib.pyplot as plt
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    # le moteur qui fait tourner les simus en lot

    def __init__(self, n_simulations: int = 100, duration_seconds: int = 60):
        self.n_simulations = n_simulations
        self.duration_seconds = duration_seconds
        self.results: List[Dict] = []

    def run_generator(self):
        # refile les resultats un par un pour le dash board live
        # Deferred imports to avoid circular dependency
        from main import Strategy
        from data.tick_stream import SyntheticTickStream

        logger.info(f"Starting Monte Carlo simulation with {self.n_simulations} runs...")

        for i in range(self.n_simulations):
            seed = 42 + i
            # Edge of precision: randomize environment explicitly
            rng = np.random.RandomState(seed)
            drift = rng.uniform(-0.05, 0.05)
            vol_multiplier = rng.uniform(0.5, 3.5)
            fat_tail_prob = rng.uniform(0.0, 0.05)
            drop_tick_prob = rng.uniform(0.0, 0.1)

            try:
                # Deep copy config to avoid mutating the global singleton across simulations
                from config.loader import get_config
                perturbed_config = copy.deepcopy(get_config())

                # Perturb system thresholds slightly to test parameter robustness
                if hasattr(perturbed_config, 'thresholds'):
                    if hasattr(perturbed_config.thresholds, 'regime'):
                        perturbed_config.thresholds.regime.transition_strength_min *= rng.uniform(0.8, 1.2)
                        perturbed_config.thresholds.regime.transition_strength_min = min(1.0, perturbed_config.thresholds.regime.transition_strength_min)
                    if hasattr(perturbed_config.thresholds, 'alpha'):
                        perturbed_config.thresholds.alpha.signal_actionable_threshold *= rng.uniform(0.8, 1.2)
                        perturbed_config.thresholds.alpha.signal_actionable_threshold = min(1.0, perturbed_config.thresholds.alpha.signal_actionable_threshold)

                strategy = Strategy(config=perturbed_config, enable_viz=False)
                stream = SyntheticTickStream(
                    symbol=strategy.config.instruments.instruments[0].symbol,
                    duration_seconds=self.duration_seconds,
                    seed=seed,
                    drift=drift,
                    vol_multiplier=vol_multiplier,
                    fat_tail_prob=fat_tail_prob,
                    drop_tick_prob=drop_tick_prob
                )
                strategy.run(duration_seconds=self.duration_seconds, stream=stream)

                result = {
                    "id": i + 1,
                    "seed": seed,
                    "final_equity": strategy.risk_manager.current_equity,
                    "min_equity": min(strategy.risk_manager.equity_history),
                    "max_equity": strategy.risk_manager.peak_equity,
                    "equity_curve": strategy.risk_manager.equity_history,
                    "drawdown": strategy.risk_manager.current_drawdown_pct,
                    "kill_switch_triggered": strategy.risk_manager.kill_switch.triggered,
                    "kill_reason": strategy.risk_manager.kill_switch.reason,
                    "env_drift": drift,
                    "env_vol": vol_multiplier,
                    "env_fat_tails": fat_tail_prob
                }
                self.results.append(result)
                yield result

            except Exception as e:
                logger.error(f"Simulation {i+1} failed: {e}")

    def run(self) -> None:
        # crack c'est parti pour le run final
        # Consume the generator
        for _ in self.run_generator():
            pass
        self._analyze_results()
        self._plot_results()

    def _analyze_results(self) -> None:
        # petit resumé des degats
        if not self.results:
            logger.warning("No results to analyze.")
            return

        final_equities = [r["final_equity"] for r in self.results]
        drawdowns = [r["drawdown"] for r in self.results]
        survived = [not r["kill_switch_triggered"] for r in self.results]

        mean_equity = np.mean(final_equities)
        median_equity = np.median(final_equities)
        std_equity = np.std(final_equities)
        survival_rate = sum(survived) / len(survived)
        mean_drawdown = np.mean(drawdowns)

        kill_reasons: Dict[str, int] = {}
        for r in self.results:
            if r["kill_switch_triggered"]:
                reason = r["kill_reason"]
                kill_reasons[reason] = kill_reasons.get(reason, 0) + 1

        print("\n=== Monte Carlo Simulation Results (Edge Precision) ===")
        print(f"Simulations: {self.n_simulations}")
        print(f"Survival Rate: {survival_rate:.2%}")
        print(f"Mean Final Equity: ${mean_equity:,.2f}")
        print(f"Median Final Equity: ${median_equity:,.2f}")
        print(f"Std Dev Equity: ${std_equity:,.2f}")
        print(f"Mean Max Drawdown: {mean_drawdown:.2%}")
        
        worst_runs = sorted(self.results, key=lambda x: x["final_equity"])[:3]
        print("\nWorst Runs Context (Edge Conditions Assessment):")
        for w in worst_runs:
            status = "KILLED" if w["kill_switch_triggered"] else "SURVIVED"
            reason = f' ({w.get("kill_reason", "")})' if w["kill_switch_triggered"] else ""
            print(f"  - Run {w['id']} {status}{reason}: Equity=${w['final_equity']:.2f}, Volatility={w.get('env_vol', 1.0):.2f}x, Fat Tails={w.get('env_fat_tails', 0.0):.3f}")

        if kill_reasons:
            print("\nKill Switch Reasons:")
            for reason, count in kill_reasons.items():
                print(f"  - {reason}: {count}")
        print("======================================\n")

    def _plot_results(self) -> None:
        # on sauvegarde le joli graph des courbes
        if not self.results:
            return

        plt.figure(figsize=(12, 8))
        for r in self.results:
            curve = r["equity_curve"]
            color = 'red' if r["kill_switch_triggered"] else 'green'
            alpha = 0.3 if r["kill_switch_triggered"] else 0.5
            plt.plot(curve, color=color, alpha=alpha, linewidth=1)

        plt.title(f"Monte Carlo Simulation ({self.n_simulations} Runs)\nGreen: Survived, Red: Killed")
        plt.xlabel("Trade/Update Steps")
        plt.ylabel("Equity ($)")
        plt.grid(True, alpha=0.3)

        output_path = "monte_carlo_results.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Simulation plots saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = MonteCarloSimulator(n_simulations=5, duration_seconds=10)
    sim.run()
