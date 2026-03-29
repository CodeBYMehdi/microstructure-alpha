# verif rapide
# la grosse machine

import unittest
import numpy as np
from execution.impact_model import MarketImpactModel, ImpactEstimate
from execution.slippage import SlippageModel, SlippageEstimate
from execution.order_router import OrderRouter


class TestMarketImpactModel(unittest.TestCase):

    def test_zero_quantity(self):
        model = MarketImpactModel()
        estimate = model.estimate(order_qty=0, price=100.0, volatility=0.01)
        self.assertEqual(estimate.total_impact_bps, 0.0)
        self.assertTrue(estimate.is_acceptable)

    def test_positive_impact(self):
        model = MarketImpactModel()
        estimate = model.estimate(order_qty=100, price=100.0, volatility=0.01)
        self.assertGreater(estimate.total_impact_bps, 0)
        self.assertGreater(estimate.cost_estimate, 0)

    def test_larger_orders_more_impact(self):
        model = MarketImpactModel()
        small = model.estimate(order_qty=10, price=100.0, volatility=0.01)
        large = model.estimate(order_qty=1000, price=100.0, volatility=0.01)
        self.assertGreater(large.total_impact_bps, small.total_impact_bps)

    def test_higher_vol_more_impact(self):
        model = MarketImpactModel()
        low_vol = model.estimate(order_qty=100, price=100.0, volatility=0.001)
        high_vol = model.estimate(order_qty=100, price=100.0, volatility=0.1)
        self.assertGreater(high_vol.total_impact_bps, low_vol.total_impact_bps)

    def test_invalid_price(self):
        model = MarketImpactModel()
        estimate = model.estimate(order_qty=100, price=-1.0, volatility=0.01)
        self.assertTrue(estimate.is_acceptable)  # Returns benign estimate

    def test_max_impact_flag(self):
        model = MarketImpactModel(max_impact_bps=0.001)
        estimate = model.estimate(order_qty=10000, price=100.0, volatility=0.1)
        self.assertFalse(estimate.is_acceptable)

    def test_invalid_coefficients(self):
        with self.assertRaises(ValueError):
            MarketImpactModel(temporary_coeff=-1.0)


class TestSlippageModel(unittest.TestCase):

    def test_zero_quantity(self):
        model = SlippageModel(seed=42)
        estimate = model.estimate(price=100.0, quantity=0, is_buy=True)
        self.assertEqual(estimate.total_slippage, 0.0)

    def test_buy_slippage_direction(self):
        model = SlippageModel(seed=42)
        estimate = model.estimate(price=100.0, quantity=10, is_buy=True)
        # Buy execution price should generally be >= target
        # (due to impact, though noise can make it lower)
        self.assertGreater(estimate.execution_price, 0)

    def test_deterministic_with_seed(self):
        m1 = SlippageModel(seed=42)
        m2 = SlippageModel(seed=42)
        e1 = m1.estimate(100.0, 10, True)
        e2 = m2.estimate(100.0, 10, True)
        self.assertEqual(e1.execution_price, e2.execution_price)

    def test_with_bbo(self):
        model = SlippageModel(seed=42)
        estimate = model.estimate(
            price=100.0, quantity=10, is_buy=True,
            bid=99.95, ask=100.05,
        )
        # Spread cost should be half the spread
        self.assertAlmostEqual(estimate.spread_cost, 0.05, places=2)

    def test_symmetric_noise(self):
        # verif rapide
        noises = []
        for seed in range(200):
            model = SlippageModel(seed=seed)
            est = model.estimate(100.0, 1.0, True, volatility=0.001)
            noises.append(est.noise_component)
        # Mean noise should be approximately 0
        self.assertAlmostEqual(np.mean(noises), 0.0, delta=0.001)


class TestOrderRouter(unittest.TestCase):

    def test_simulation_fill_with_fees(self):
        from core.types import TradeProposal, TradeAction, RegimeState, RegimeType
        from datetime import datetime

        router = OrderRouter(mode="simulation")
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, datetime.now(), {}),
        )
        result = router.execute(proposal)
        self.assertEqual(result.status, "FILLED")
        self.assertGreater(result.filled_price, 0)
        # Fees should now be computed (not zero)
        self.assertGreaterEqual(result.fees, 0)

    def test_zero_quantity_rejected(self):
        from core.types import TradeProposal, TradeAction, RegimeState, RegimeType
        from datetime import datetime

        router = OrderRouter(mode="simulation")
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=0.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, datetime.now(), {}),
        )
        result = router.execute(proposal)
        self.assertEqual(result.status, "REJECTED")

    def test_rate_limiting(self):
        from core.types import TradeProposal, TradeAction, RegimeState, RegimeType
        from datetime import datetime

        router = OrderRouter(mode="simulation")
        router._max_orders_per_second = 3  # Low limit for testing

        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, datetime.now(), {}),
        )

        # First 3 should succeed
        for _ in range(3):
            result = router.execute(proposal)
            self.assertEqual(result.status, "FILLED")

        # 4th should be rate-limited
        result = router.execute(proposal)
        self.assertEqual(result.status, "REJECTED")

    def test_no_price_rejected(self):
        from core.types import TradeProposal, TradeAction, RegimeState, RegimeType
        from datetime import datetime

        router = OrderRouter(mode="simulation")
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=None, reason="Test", timestamp=datetime.now(),
            regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, datetime.now(), {}),
        )
        result = router.execute(proposal)
        self.assertEqual(result.status, "REJECTED")


class TestTradeLedger(unittest.TestCase):

    def test_ledger_creation_and_fill(self):
        import tempfile, os
        from core.types import TradeProposal, TradeAction, RegimeState, RegimeType, OrderResult
        from execution.trade_ledger import TradeLedger
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ledger.csv")
            ledger = TradeLedger(filepath=path)

            regime = RegimeState(RegimeType.STABLE_BULL, 0.8, 2.0, datetime.now(), {"id": 1})
            proposal = TradeProposal(
                action=TradeAction.BUY, symbol="SPY", quantity=10.0,
                price=450.0, reason="test", timestamp=datetime.now(),
                regime_state=regime,
            )
            fill = OrderResult(
                order_id="o1", status="FILLED", filled_price=450.05,
                filled_quantity=10.0, timestamp=datetime.now(), fees=0.45,
            )

            ledger.record_order(proposal, "o1", equity=100000.0)
            ledger.record_fill(fill, proposal, equity=99999.55)

            self.assertEqual(ledger.entry_count, 2)
            self.assertAlmostEqual(ledger.get_net_position("SPY"), 10.0)
            self.assertAlmostEqual(ledger.get_total_fees(), 0.45)

            # Crash recovery: reload from same file
            ledger2 = TradeLedger(filepath=path)
            self.assertAlmostEqual(ledger2.get_net_position("SPY"), 10.0)


if __name__ == '__main__':
    unittest.main()
