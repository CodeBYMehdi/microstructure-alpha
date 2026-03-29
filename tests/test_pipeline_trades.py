# simu pour pas pleurer en live
# verif rapide

import unittest
import numpy as np
from datetime import datetime, timedelta

from core.types import Tick
from config.loader import get_config
from config.schema import ExitConfig


class TestPipelineProducesTrades(unittest.TestCase):
    # simu pour pas pleurer en live
    # verif rapide

    @classmethod
    def setUpClass(cls):
        # l'usine a gaz
        cls.ticks = []
        base_time = datetime(2024, 1, 2, 9, 30)
        rng = np.random.RandomState(42)
        price = 450.0

        # Create regime-like dynamics: trending + mean-reverting phases
        for i in range(30_000):
            # Switch between trending and mean-reverting every ~2000 ticks
            phase = (i // 2000) % 3
            if phase == 0:
                # Uptrend
                drift = 0.002
                vol = 0.03
            elif phase == 1:
                # Downtrend
                drift = -0.002
                vol = 0.05
            else:
                # Mean-reverting
                drift = -0.001 * (price - 450.0)
                vol = 0.02

            price += rng.normal(drift, vol)
            price = max(price, 400.0)  # Floor

            delta_ms = rng.randint(10, 500)
            ts = base_time + timedelta(milliseconds=int(i * 100 + delta_ms))

            cls.ticks.append(Tick(
                timestamp=ts,
                symbol="SPY",
                price=round(price, 2),
                volume=abs(rng.normal(100.0, 30.0)),
                bid=round(price - 0.01, 2),
                ask=round(price + 0.01, 2),
                bid_size=abs(rng.normal(50.0, 15.0)),
                ask_size=abs(rng.normal(50.0, 15.0)),
            ))

    def test_fallback_strength_threshold_accessible(self):
        # verif rapide
        # aie aie aie
        config = get_config()
        threshold = config.thresholds.decision.exit.fallback_strength_threshold
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_fallback_strength_threshold_default(self):
        # verif rapide
        # les petits reglages
        ec = ExitConfig()
        self.assertAlmostEqual(ec.fallback_strength_threshold, 0.30, places=2)

    def test_run_single_backtest_produces_trades(self):
        # simu pour pas pleurer en live
        # verif rapide
        from optimization.objective import run_single_backtest

        config = get_config()

        results = run_single_backtest(
            config=config,
            tick_list=self.ticks,
            tick_limit=30_000,
            symbol="SPY",
        )

        total_trades = results.get("total_trades", 0)
        transitions = results.get("transitions", 0)

        print(f"\n=== Pipeline Test Results ===")
        print(f"  Total trades:    {total_trades}")
        print(f"  Transitions:     {transitions}")
        print(f"  Total PnL:       {results.get('total_pnl', 0.0):.4f}")
        print(f"  Sharpe ratio:    {results.get('sharpe_ratio', 0.0):.4f}")
        print(f"  Max drawdown:    {results.get('max_drawdown', 0.0):.4f}")
        print(f"  Win rate:        {results.get('win_rate', 0.0):.2%}")
        print(f"  Final equity:    {results.get('final_equity', 0.0):.2f}")
        print(f"============================\n")

        self.assertGreater(total_trades, 0,
                           f"Expected > 0 trades but got {total_trades}. "
                           f"Transitions: {transitions}")

    def test_config_propagation_to_transition_detector(self):
        # verif rapide
        # les petits reglages
        from regime.transition import TransitionDetector
        import copy

        config = get_config()
        modified_config = copy.deepcopy(config)
        modified_config.thresholds.regime.transition_strength_min = 0.1

        detector = TransitionDetector(config=modified_config)
        self.assertAlmostEqual(detector.strength_threshold, 0.1, places=2,
                               msg="TransitionDetector should use config passed to it")


if __name__ == "__main__":
    unittest.main()
