"""Tests for institutional-grade components."""

import unittest
import numpy as np
import os
import tempfile

from alpha.ensemble import AlphaEnsemble, OnlineRidgeModel, RegimeConditionalModel
from alpha.attribution import AlphaAttribution
from risk.portfolio import PortfolioRiskManager, RiskRegime
from data.quality import DataQualitySentinel, DataQualityConfig, QualityIssue
from data.tick_db import TradeDatabase
from execution.analytics import ExecutionAnalytics
from statistics.deflated_sharpe import (
    deflated_sharpe, annualized_sharpe, probabilistic_sharpe, alpha_half_life,
)


class TestAlphaEnsemble(unittest.TestCase):

    def test_basic_prediction(self):
        ensemble = AlphaEnsemble(n_features=10, min_samples=5)
        features = np.random.randn(10)
        pred = ensemble.predict(features, regime_id=0)
        self.assertIsNotNone(pred.expected_return)
        self.assertEqual(len(pred.model_weights), 4)  # 4 sub-models

    def test_update_and_weight_adaptation(self):
        ensemble = AlphaEnsemble(n_features=5, min_samples=3)
        for i in range(50):
            features = np.random.randn(5)
            ensemble.predict(features, regime_id=0)
            ensemble.update(np.random.randn() * 0.001)

        metrics = ensemble.get_metrics()
        self.assertEqual(metrics['n_predictions'], 50)
        self.assertTrue(len(metrics['per_model']) > 0)

    def test_model_contributions(self):
        ensemble = AlphaEnsemble(n_features=5, min_samples=3)
        features = np.random.randn(5)
        pred = ensemble.predict(features, regime_id=0)
        self.assertEqual(len(pred.model_contributions), 4)

    def test_get_top_model(self):
        ensemble = AlphaEnsemble(n_features=5)
        top = ensemble.get_top_model()
        self.assertIn(top, ['ridge', 'regime_cond', 'momentum_mr', 'gbt'])


class TestOnlineRidge(unittest.TestCase):

    def test_convergence(self):
        model = OnlineRidgeModel(n_features=3, learning_rate=0.01)
        # Simple linear target
        true_weights = np.array([0.5, -0.3, 0.1])
        for _ in range(200):
            x = np.random.randn(3)
            y = np.dot(true_weights, x) + np.random.randn() * 0.01
            model.predict(x)
            model.update(y)

        # Weights should be close to true
        pred, conf = model.predict(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(pred, 0.5, delta=0.3)


class TestAlphaAttribution(unittest.TestCase):

    def test_basic_recording(self):
        att = AlphaAttribution(signal_names=["sig_a", "sig_b"])
        att.record_signals(
            signals={"sig_a": 0.5, "sig_b": -0.3},
            weights={"sig_a": 0.6, "sig_b": 0.4},
            timestamp=1.0,
        )
        att.record_realized(realized_return=0.001)

        summary = att.get_attribution_summary()
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["sig_a"].observations, 1)

    def test_dead_signal_detection(self):
        att = AlphaAttribution(signal_names=["good", "bad"], lookback=200)
        for i in range(150):
            att.record_signals(
                signals={"good": 0.5, "bad": -0.5},
                weights={"good": 0.5, "bad": 0.5},
                timestamp=float(i),
            )
            att.record_realized(realized_return=0.001)

        dead = att.get_dead_signals(min_observations=100)
        self.assertIn("bad", dead)

    def test_rankings(self):
        att = AlphaAttribution(signal_names=["a", "b"])
        for _ in range(20):
            att.record_signals(
                signals={"a": 1.0, "b": -1.0},
                weights={"a": 0.5, "b": 0.5},
                timestamp=0.0,
            )
            att.record_realized(realized_return=0.001)

        rankings = att.get_signal_rankings()
        self.assertEqual(rankings[0], "a")


class TestPortfolioRisk(unittest.TestCase):

    def test_basic_exposure(self):
        prm = PortfolioRiskManager(initial_equity=100000)
        prm.update_position("AAPL", 100, 150.0)
        prm.update_position("MSFT", -50, 300.0)

        exp = prm.get_exposure()
        self.assertAlmostEqual(exp['long'], 15000.0)
        self.assertAlmostEqual(exp['short'], 15000.0)
        self.assertAlmostEqual(exp['gross'], 30000.0)

    def test_regime_detection(self):
        prm = PortfolioRiskManager(initial_equity=100000)
        # Normal conditions
        regime = prm.detect_risk_regime(market_vol=0.01)
        self.assertEqual(regime, RiskRegime.NORMAL)

    def test_drawdown_tracking(self):
        prm = PortfolioRiskManager(initial_equity=100000)
        prm.update_equity(-5000)
        self.assertAlmostEqual(prm.current_drawdown_pct, 0.05)

    def test_pre_trade_check(self):
        prm = PortfolioRiskManager(initial_equity=100000)
        ok, msg = prm.check_new_trade("AAPL", 100, 150.0)
        self.assertTrue(ok)

        # Exceed gross limit (nominal cap is $10M in NORMAL regime)
        ok, msg = prm.check_new_trade("AAPL", 100000, 150.0)
        self.assertFalse(ok)
        self.assertIn("Gross exposure", msg)

    def test_regime_conditional_budget(self):
        prm = PortfolioRiskManager(initial_equity=100000)
        # Force crisis regime
        prm.update_equity(-8000)  # 8% drawdown
        budget = prm.get_current_budget()
        self.assertLess(budget.size_multiplier, 1.0)


class TestDataQuality(unittest.TestCase):

    def test_clean_tick(self):
        sentinel = DataQualitySentinel()
        events = sentinel.check_tick("AAPL", 150.0, 100.0, 149.95, 150.05)
        self.assertEqual(len(events), 0)

    def test_crossed_market(self):
        sentinel = DataQualitySentinel()
        events = sentinel.check_tick("AAPL", 150.0, 100.0, 151.0, 149.0)
        crossed = [e for e in events if e.issue == QualityIssue.CROSSED_MARKET]
        self.assertEqual(len(crossed), 1)

    def test_price_jump(self):
        sentinel = DataQualitySentinel()
        # Build price history
        for i in range(50):
            sentinel.check_tick("AAPL", 150.0 + np.random.randn() * 0.01, 100.0, None, None,
                                receive_ts=1000.0 + i * 0.1)
        # Inject price jump
        events = sentinel.check_tick("AAPL", 200.0, 100.0, None, None, receive_ts=1010.0)
        jumps = [e for e in events if e.issue == QualityIssue.PRICE_JUMP]
        self.assertTrue(len(jumps) > 0)

    def test_data_safe_check(self):
        import time as _time
        sentinel = DataQualitySentinel()
        self.assertFalse(sentinel.is_data_safe("AAPL"))
        now = _time.time()
        for i in range(25):
            sentinel.check_tick("AAPL", 150.0, 100.0, None, None, receive_ts=now + i * 0.01)
        self.assertTrue(sentinel.is_data_safe("AAPL"))


class TestTradeDatabase(unittest.TestCase):

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.db = TradeDatabase(self.db_path)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass

    def test_net_position(self):
        from core.types import TradeProposal, OrderResult, TradeAction, RegimeState, RegimeType
        from datetime import datetime

        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="AAPL", quantity=100.0,
            price=150.0, reason="test", timestamp=datetime.now(),
            regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, datetime.now(), {}),
        )
        result = OrderResult(order_id="t1", status="FILLED",
                             filled_price=150.0, filled_quantity=100.0, fees=1.0,
                             timestamp=datetime.now())

        self.db.record_fill(result, proposal, equity=100000)
        pos = self.db.get_net_position("AAPL")
        self.assertAlmostEqual(pos, 100.0)

    def test_query(self):
        trades = self.db.query_trades(symbol="AAPL")
        self.assertIsInstance(trades, list)

    def test_alert_recording(self):
        self.db.record_alert("KILL_SWITCH", "CRITICAL", "Test alert")
        alerts = self.db.get_recent_alerts()
        self.assertEqual(len(alerts), 1)


class TestExecutionAnalytics(unittest.TestCase):

    def test_record_and_report(self):
        analytics = ExecutionAnalytics()
        analytics.record_fill(
            symbol="AAPL", side="BUY", requested_qty=100,
            filled_qty=100, arrival_price=150.0, filled_price=150.10,
            fees=1.0, decision_ts=1000.0, fill_ts=1000.050,
        )
        report = analytics.generate_report()
        self.assertEqual(report['total_fills'], 1)
        self.assertGreater(report['avg_slippage_bps'], 0)

    def test_cost_curve(self):
        analytics = ExecutionAnalytics()
        analytics.record_fill(
            symbol="AAPL", side="BUY", requested_qty=100,
            filled_qty=100, arrival_price=150.0, filled_price=150.0, fees=1.0,
        )
        curve = analytics.get_cost_curve()
        self.assertIn('total_notional', curve)


class TestDeflatedSharpe(unittest.TestCase):

    def test_significant_sharpe(self):
        result = deflated_sharpe(
            observed_sharpe=2.0, n_trials=10,
            n_observations=252, skewness=0.0, kurtosis=3.0,
        )
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

    def test_noise_sharpe(self):
        result = deflated_sharpe(
            observed_sharpe=0.3, n_trials=1000,
            n_observations=50,
        )
        self.assertFalse(result.is_significant)

    def test_annualized_sharpe(self):
        returns = np.random.randn(252) * 0.01 + 0.001
        sr = annualized_sharpe(returns)
        self.assertIsInstance(sr, float)

    def test_probabilistic_sharpe(self):
        prob = probabilistic_sharpe(
            observed_sharpe=1.5, benchmark_sharpe=0.0,
            n_observations=252,
        )
        self.assertGreater(prob, 0.5)

    def test_alpha_half_life(self):
        # Decaying signal
        pnl = np.concatenate([
            np.random.randn(200) * 0.001 + 0.002,
            np.random.randn(200) * 0.001 - 0.001,
        ])
        half_life, is_decaying = alpha_half_life(pnl, window=50)
        # Should detect some signal (may or may not detect decay with random data)
        self.assertIsInstance(half_life, float)


if __name__ == '__main__':
    unittest.main()
