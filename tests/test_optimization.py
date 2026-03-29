# verif rapide

import unittest
import copy
import numpy as np
from datetime import datetime, timedelta

from config.loader import get_config
from core.types import Tick
from optimization.search_space import (
    get_param_names,
    get_defaults,
    get_param_groups,
    get_param_bounds,
    apply_params,
)


class TestSearchSpace(unittest.TestCase):
    # verif rapide

    def test_param_names_count(self):
        names = get_param_names()
        self.assertEqual(len(names), 22)

    def test_defaults_count(self):
        defaults = get_defaults()
        names = get_param_names()
        self.assertEqual(len(defaults), len(names))

    def test_defaults_are_dict(self):
        defaults = get_defaults()
        self.assertIsInstance(defaults, dict)

    def test_defaults_within_bounds(self):
        defaults = get_defaults()
        bounds = get_param_bounds()
        for name, val in defaults.items():
            ptype, low, high = bounds[name]
            self.assertGreaterEqual(val, low, f"{name}: {val} < {low}")
            self.assertLessEqual(val, high, f"{name}: {val} > {high}")

    def test_param_groups(self):
        groups = get_param_groups()
        self.assertEqual(len(groups), 5)
        self.assertIn("regime", groups)
        self.assertIn("decision", groups)
        self.assertIn("sizing", groups)
        self.assertIn("risk", groups)
        self.assertIn("calibration", groups)

    def test_apply_params_modifies_config(self):
        config = get_config()
        defaults = get_defaults()

        modified = dict(defaults)
        modified["min_cluster_size"] = 77

        new_config = apply_params(config, modified)
        self.assertEqual(new_config.thresholds.regime.min_cluster_size, 77)

        # Original config should be unchanged (deep copy)
        self.assertNotEqual(config.thresholds.regime.min_cluster_size, 77)

    def test_suggest_params_returns_dict(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from optimization.search_space import suggest_params

        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(trial)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 22)


class TestObjective(unittest.TestCase):
    # verif rapide

    @classmethod
    def setUpClass(cls):
        cls.ticks = []
        base_time = datetime(2024, 1, 1, 9, 30)
        price = 450.0
        for i in range(500):
            price += np.random.normal(0, 0.05)
            cls.ticks.append(Tick(
                timestamp=base_time + timedelta(seconds=i),
                symbol="TEST",
                price=price,
                volume=100.0,
            ))

    def test_objective_returns_finite(self):
        from optimization.objective import objective
        config = get_config()
        defaults = get_defaults()

        result = objective(defaults, config, self.ticks, tick_limit=200, symbol="TEST")
        self.assertTrue(np.isfinite(result), f"Got non-finite result: {result}")

    def test_objective_with_modified_params(self):
        from optimization.objective import objective
        config = get_config()
        defaults = get_defaults()

        modified = dict(defaults)
        modified["min_cluster_size"] = 5
        modified["min_samples"] = 2

        result = objective(modified, config, self.ticks, tick_limit=200, symbol="TEST")
        self.assertTrue(np.isfinite(result))

    def test_sharpe_computation(self):
        from optimization.objective import _compute_sharpe

        # No trades
        self.assertEqual(_compute_sharpe({"total_trades": 0}), 0.0)

        # Normal case
        result = _compute_sharpe({
            "total_trades": 100,
            "win_rate": 0.55,
            "avg_win": 10.0,
            "avg_loss": -8.0,
        })
        self.assertTrue(np.isfinite(result))


class TestWalkForward(unittest.TestCase):
    # verif rapide

    def test_fold_splits_non_overlapping(self):
        from optimization.walk_forward import WalkForwardValidator
        config = get_config()

        ticks = [Tick(
            timestamp=datetime(2024, 1, 1) + timedelta(seconds=i),
            symbol="TEST", price=100.0, volume=1.0,
        ) for i in range(1000)]

        wf = WalkForwardValidator(
            base_config=config, tick_list=ticks,
            n_folds=3, n_calls=2,
        )

        splits = wf.get_fold_splits()
        self.assertEqual(len(splits), 3)

        for i in range(len(splits) - 1):
            _, _, test_end_i = splits[i]
            _, test_start_next, _ = splits[i + 1]
            self.assertEqual(test_end_i, test_start_next,
                             f"Gap between fold {i} and {i+1}")

        self.assertEqual(splits[-1][2], 1000)

    def test_minimum_ticks_validation(self):
        from optimization.walk_forward import WalkForwardValidator
        config = get_config()

        ticks = [Tick(
            timestamp=datetime(2024, 1, 1), symbol="TEST",
            price=100.0, volume=1.0,
        ) for _ in range(50)]

        with self.assertRaises(ValueError):
            WalkForwardValidator(config, ticks, n_folds=3, n_calls=2)


class TestSensitivity(unittest.TestCase):
    # verif rapide

    def test_perturbation_respects_bounds(self):
        from optimization.sensitivity import SensitivityAnalyzer

        # Integer
        result = SensitivityAnalyzer._perturb_value(10, 0.5, "int", 2, 20)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 20)

        # Float
        result = SensitivityAnalyzer._perturb_value(0.5, -0.5, "float", 0.1, 1.0)
        self.assertGreaterEqual(result, 0.1)
        self.assertLessEqual(result, 1.0)


class TestDiagnostics(unittest.TestCase):
    # verif rapide

    def test_gini_computation(self):
        from optimization.diagnostics import _compute_gini

        gini_equal = _compute_gini([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(gini_equal, 0.0, places=5)

        gini_unequal = _compute_gini([0.0, 0.0, 0.0, 100.0])
        self.assertGreater(gini_unequal, 0.5)

    def test_empty_gini(self):
        from optimization.diagnostics import _compute_gini
        self.assertEqual(_compute_gini([]), 0.0)


class TestMetricsEnhancement(unittest.TestCase):
    # verif rapide

    def test_sharpe_in_results(self):
        from backtest.metrics import PerformanceMetrics, TradeRecord

        metrics = PerformanceMetrics()
        result = metrics.compute()
        self.assertIn("sharpe_ratio", result)
        self.assertIn("calmar_ratio", result)
        self.assertIn("regime_accuracy", result)
        self.assertIn("avg_dwell_time", result)
        self.assertIn("churn_rate", result)

    def test_sharpe_with_trades(self):
        from backtest.metrics import PerformanceMetrics, TradeRecord

        metrics = PerformanceMetrics(initial_equity=10000.0)
        base_ts = 1700000000.0
        for i in range(20):
            pnl = 10.0 if i % 3 != 0 else -5.0
            metrics.record_trade(TradeRecord(
                timestamp=base_ts + (i // 5) * 86400.0 + (i % 5) * 100.0,
                symbol="TEST", side="BUY",
                qty=1.0, entry_price=100.0, pnl=pnl,
            ))

        result = metrics.compute()
        self.assertTrue(np.isfinite(result["sharpe_ratio"]))
        self.assertGreater(result["sharpe_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
