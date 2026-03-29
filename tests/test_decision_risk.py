# attention aux degats
# verif rapide

import unittest
import numpy as np
from regime.state_vector import StateVector
from regime.transition import TransitionEvent
from core.types import TradeAction, LiquidityState, TradeProposal, RegimeState, RegimeType
from decision.entry_conditions import EntryConditions
from decision.exits import ExitConditions
from decision.sizing import PositionSizer
from decision.eligibility import TradeEligibility, EligibilityResult
from decision.confidence import ConfidenceScorer, ConfidenceBreakdown
from risk.kill_switch import RiskManager, KillSwitch
from risk.exposure import ExposureTracker
from risk.tail_risk import TailRiskAnalyzer
from risk.calibration import RiskCalibrationModel
from datetime import datetime


def _make_transition(**overrides):
    # l'usine a gaz
    defaults = dict(
        from_regime=0, to_regime=1, strength=0.8,
        delta_vector=np.array([0.001, 0.0, 0.1, 0.0, 0.0, 0.0]),
        is_significant=True, kl_divergence=0.2,
        projection_magnitude=2.0, mu_velocity=0.0001,
        mu_acceleration=0.00001, entropy_acceleration=0.0,
    )
    defaults.update(overrides)
    return TransitionEvent(**defaults)


def _make_state(**overrides):
    # dans quel etat j'erre
    defaults = dict(mu=0.001, sigma=0.01, skew=0.5, kurtosis=1.0, tail_slope=3.0, entropy=-5.0)
    defaults.update(overrides)
    return StateVector(**defaults)


class TestEntryConditions(unittest.TestCase):

    def setUp(self):
        self.entry_logic = EntryConditions()

    def test_long_entry(self):
        # Mean-reversion: +vel/+acc (price overshooting up) → SELL to revert
        state = _make_state()
        transition = _make_transition()
        side, reason = self.entry_logic.evaluate(transition, state)
        self.assertEqual(side, TradeAction.SELL)
        self.assertIn("Mean Revert SELL", reason)

    def test_short_entry(self):
        # Mean-reversion: -vel (price overshooting down), decel → BUY to revert
        state = _make_state(mu=-0.001, sigma=0.002, skew=-0.5, kurtosis=2.0)
        transition = _make_transition(
            delta_vector=np.zeros(6),
            mu_velocity=-0.0001, mu_acceleration=-0.00001,
            entropy_acceleration=0.0,
        )
        side, reason = self.entry_logic.evaluate(transition, state)
        self.assertEqual(side, TradeAction.BUY)
        self.assertIn("Mean Revert BUY", reason)

    def test_liquidity_gate(self):
        state = _make_state()
        transition = _make_transition()
        # Spread must exceed config spread_max to trigger gate
        bad_liq = LiquidityState(spread=10.0, depth_imbalance=0.0, depth_slope=0.0, trade_intensity=0.0)
        side, reason = self.entry_logic.evaluate(transition, state, bad_liq)
        self.assertIsNone(side)
        self.assertIn("Liquidity Check Failed", reason)

    def test_kl_gate(self):
        from copy import deepcopy
        from config.loader import get_config
        config = deepcopy(get_config())
        config.thresholds.regime.kl_min = 0.05  # Explicit gate threshold
        entry = EntryConditions(config)
        state = _make_state()
        transition = _make_transition(kl_divergence=0.001)
        side, reason = entry.evaluate(transition, state)
        self.assertIsNone(side)
        self.assertIn("Information Gradient", reason)

    def test_projection_gate(self):
        from copy import deepcopy
        from config.loader import get_config
        config = deepcopy(get_config())
        config.thresholds.regime.projection_min = 1.0  # Explicit gate threshold
        entry = EntryConditions(config)
        state = _make_state()
        transition = _make_transition(projection_magnitude=0.01)
        side, reason = entry.evaluate(transition, state)
        self.assertIsNone(side)
        self.assertIn("Regime Projection", reason)

    def test_invalid_delta_vector(self):
        state = _make_state()
        transition = _make_transition(delta_vector=np.array([1.0, 2.0]))
        side, reason = self.entry_logic.evaluate(transition, state)
        self.assertIsNone(side)
        self.assertIn("Invalid delta vector", reason)

    def test_flat_no_edge(self):
        state = _make_state()
        transition = _make_transition(mu_velocity=0.0, mu_acceleration=0.0)
        side, reason = self.entry_logic.evaluate(transition, state)
        self.assertIsNone(side)
        self.assertIn("No directional edge", reason)


class TestExitConditions(unittest.TestCase):

    def setUp(self):
        self.exits = ExitConditions()

    def test_kl_collapse_exit(self):
        transition = _make_transition(kl_divergence=0.001)
        state = _make_state()
        should_exit, reason = self.exits.check_exit(TradeAction.BUY, transition, state)
        self.assertTrue(should_exit)
        self.assertIn("Information Gradient Collapsed", reason)

    def test_reacceleration_down_exit(self):
        # BUY position exits when price re-accelerates downward (vel < 0, acc < 0)
        transition = _make_transition(kl_divergence=0.1, mu_velocity=-0.001, mu_acceleration=-0.001)
        state = _make_state()
        should_exit, reason = self.exits.check_exit(TradeAction.BUY, transition, state)
        self.assertTrue(should_exit)
        self.assertIn("Re-acceleration down", reason)

    def test_reacceleration_up_exit(self):
        # SELL position exits when price re-accelerates upward (vel > 0, acc > 0)
        transition = _make_transition(kl_divergence=0.1, mu_velocity=0.001, mu_acceleration=0.001)
        state = _make_state()
        should_exit, reason = self.exits.check_exit(TradeAction.SELL, transition, state)
        self.assertTrue(should_exit)
        self.assertIn("Re-acceleration up", reason)

    def test_no_exit(self):
        transition = _make_transition(kl_divergence=0.5, mu_acceleration=0.001, entropy_velocity=-0.1)
        state = _make_state()
        should_exit, _ = self.exits.check_exit(TradeAction.BUY, transition, state)
        self.assertFalse(should_exit)

    def test_none_transition(self):
        state = _make_state()
        should_exit, _ = self.exits.check_exit(TradeAction.BUY, None, state)
        self.assertFalse(should_exit)


class TestPositionSizer(unittest.TestCase):

    def test_basic_sizing(self):
        sizer = PositionSizer()
        transition = _make_transition(strength=0.8)
        size = sizer.calculate(transition, tail_slope=2.0)
        self.assertGreater(size, 0.0)

    def test_tail_penalty(self):
        sizer = PositionSizer()
        transition = _make_transition(strength=1.0)
        size_normal = sizer.calculate(transition, tail_slope=2.0)
        size_fat = sizer.calculate(transition, tail_slope=4.0)
        self.assertLess(size_fat, size_normal)

    def test_zero_strength_returns_zero(self):
        # Zero strength transition → zero expected PnL → zero size
        sizer = PositionSizer()
        transition = _make_transition(strength=0.0)
        self.assertEqual(sizer.calculate(transition, tail_slope=2.0), 0.0)

    def test_size_capped(self):
        sizer = PositionSizer()
        transition = _make_transition(strength=1.0)
        size = sizer.calculate(transition, tail_slope=0.1)
        self.assertLessEqual(size, sizer.base_size * sizer.max_multiplier)


class TestEligibility(unittest.TestCase):

    def test_no_transition(self):
        elig = TradeEligibility()
        result = elig.check(None, _make_state(), True)
        self.assertFalse(result.is_eligible)

    def test_valid_transition_eligible(self):
        # Eligibility no longer gates on is_significant (caller handles that).
        # Any valid transition with risk cleared should be eligible.
        elig = TradeEligibility()
        transition = _make_transition(is_significant=False, strength=0.01)
        result = elig.check(transition, _make_state(), True)
        self.assertTrue(result.is_eligible)

    def test_risk_block(self):
        elig = TradeEligibility()
        transition = _make_transition()
        result = elig.check(transition, _make_state(), False)
        self.assertFalse(result.is_eligible)

    def test_invalid_state(self):
        elig = TradeEligibility()
        transition = _make_transition()
        state = StateVector(0.0, -1.0, 0.0, 0.0, 0.0, -float('inf'))
        result = elig.check(transition, state, True)
        self.assertFalse(result.is_eligible)

    def test_eligible(self):
        elig = TradeEligibility()
        transition = _make_transition()
        result = elig.check(transition, _make_state(), True)
        self.assertTrue(result.is_eligible)


class TestConfidenceScorer(unittest.TestCase):

    def test_basic_scoring(self):
        scorer = ConfidenceScorer()
        result = scorer.score(0.8, _make_transition(), _make_state())
        self.assertGreater(result.composite, 0.0)
        self.assertLessEqual(result.composite, 1.0)

    def test_no_transition(self):
        scorer = ConfidenceScorer()
        result = scorer.score(0.5, None)
        self.assertGreater(result.regime_confidence, 0.0)
        self.assertEqual(result.transition_strength, 0.0)


class TestKillSwitch(unittest.TestCase):

    def test_confidence_trigger(self):
        ks = KillSwitch()
        triggered = ks.check(confidence=0.0, slippage=0.0, drawdown=0.0)
        self.assertTrue(triggered)

    def test_drawdown_trigger(self):
        ks = KillSwitch()
        triggered = ks.check(confidence=1.0, slippage=0.0, drawdown=0.99)
        self.assertTrue(triggered)

    def test_reset_requires_admin(self):
        ks = KillSwitch()
        ks.trigger("test")
        self.assertFalse(ks.reset())  # No admin override
        self.assertTrue(ks.triggered)
        self.assertTrue(ks.reset(admin_override=True))
        self.assertFalse(ks.triggered)


class TestExposureTracker(unittest.TestCase):

    def test_basic_update(self):
        tracker = ExposureTracker(capital=100000.0, max_ratio=2.0)
        state = tracker.update("BTC", 1.0, 50000.0, is_buy=True)
        self.assertAlmostEqual(state.gross_exposure, 50000.0, places=2)

    def test_leverage_limit(self):
        tracker = ExposureTracker(capital=100000.0, max_ratio=2.0)
        self.assertTrue(tracker.can_add("BTC", 1.0, 100000.0, is_buy=True))
        tracker.update("BTC", 1.0, 100000.0, is_buy=True)
        self.assertFalse(tracker.can_add("BTC", 1.0, 200000.0, is_buy=True))

    def test_net_effect(self):
        tracker = ExposureTracker(capital=100000.0, max_ratio=2.0)
        tracker.update("BTC", 1.0, 50000.0, is_buy=True)
        # Selling reduces exposure
        state = tracker.update("BTC", 1.0, 50000.0, is_buy=False)
        self.assertAlmostEqual(state.gross_exposure, 0.0, places=2)

    def test_reset(self):
        tracker = ExposureTracker()
        tracker.update("BTC", 1.0, 100.0, is_buy=True)
        tracker.reset()
        self.assertEqual(tracker.get_state().gross_exposure, 0.0)


class TestTailRiskAnalyzer(unittest.TestCase):

    def test_insufficient_data(self):
        analyzer = TailRiskAnalyzer(min_points=20)
        for i in range(10):
            result = analyzer.update(0.01 * (i - 5))
        self.assertIsNone(result)

    def test_compute_metrics(self):
        np.random.seed(42)
        analyzer = TailRiskAnalyzer(window=200, min_points=20)
        for r in np.random.normal(0, 0.01, 200):
            analyzer.update(r)
        metrics = analyzer.compute()
        self.assertIsNotNone(metrics)
        self.assertLess(metrics.var, 0)  # VaR is negative for left tail
        self.assertLess(metrics.cvar, metrics.var)  # CVaR worse than VaR

    def test_fat_tail_detection(self):
        np.random.seed(42)
        analyzer = TailRiskAnalyzer(window=200, min_points=20)
        # Student-t with df=3 has fat tails
        data = np.random.standard_t(3, 200) * 0.01
        for r in data:
            analyzer.update(r)
        # Just check it doesn't crash
        _ = analyzer.is_fat_tail()

    def test_reset(self):
        analyzer = TailRiskAnalyzer()
        analyzer.update(0.01)
        analyzer.reset()
        self.assertEqual(len(analyzer.returns), 0)

    def test_nan_inf_handling(self):
        analyzer = TailRiskAnalyzer()
        self.assertIsNone(analyzer.update(np.nan))
        self.assertIsNone(analyzer.update(np.inf))
        self.assertIsNone(analyzer.update(None))


class TestRiskCalibration(unittest.TestCase):

    def test_normal_state(self):
        model = RiskCalibrationModel()
        state = _make_state(tail_slope=1.0, kurtosis=1.0, sigma=0.001, entropy=-6.0)
        adj = model.predict(state)
        self.assertTrue(adj.valid)
        self.assertAlmostEqual(adj.stop_multiplier, 1.0)

    def test_risky_state(self):
        model = RiskCalibrationModel()
        state = _make_state(tail_slope=3.0, kurtosis=4.0, sigma=0.02, entropy=5.0)
        adj = model.predict(state, volatility_trend=0.05)
        self.assertLess(adj.stop_multiplier, 1.0)
        self.assertLess(adj.size_scaler, 1.0)
        self.assertLess(adj.max_exposure, 1.0)


if __name__ == '__main__':
    unittest.main()
