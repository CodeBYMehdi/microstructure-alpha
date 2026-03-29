# simu pour pas pleurer en live
# verif rapide

import unittest
from datetime import datetime
import numpy as np

from backtest.event_engine import BacktestEngine, EventEngine, Event, EventType
from backtest.microstructure_sim import MicrostructureSimulator
from backtest.execution_sim import ExecutionSimulator
from backtest.metrics import PerformanceMetrics, TradeRecord
from core.types import TradeProposal, TradeAction, RegimeState, RegimeType, Tick
from data.tick_stream import SyntheticTickStream


class MockStrategy:
    def __init__(self):
        self.router = None
        self.ticks_processed = 0

    def on_tick(self, tick):
        self.ticks_processed += 1
        if self.ticks_processed == 1:
            proposal = TradeProposal(
                action=TradeAction.BUY,
                symbol="BTC/USD",
                quantity=1.0,
                price=tick.price,
                reason="Test Buy",
                timestamp=tick.timestamp,
                regime_state=RegimeState(RegimeType.UNDEFINED, 1.0, 0.0, tick.timestamp, {}),
            )
            self.router.execute(proposal)


class TestMicrostructureSimulator(unittest.TestCase):

    def test_tick_processing(self):
        sim = MicrostructureSimulator(seed=42)
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        sim.on_tick(tick)
        self.assertIsNotNone(sim.last_trade)
        self.assertEqual(sim.last_trade.price, 100.0)

    def test_volatility_estimation(self):
        sim = MicrostructureSimulator(seed=42)
        for i in range(50):
            tick = Tick(
                timestamp=datetime.now(), symbol="TEST",
                price=100.0 + np.random.normal(0, 0.5), volume=10.0,
            )
            sim.on_tick(tick)
        self.assertGreater(sim.current_volatility, 0)

    def test_synthetic_book(self):
        sim = MicrostructureSimulator(seed=42)
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        sim.on_tick(tick)
        self.assertTrue(len(sim.order_book.bids) > 0)
        self.assertTrue(len(sim.order_book.asks) > 0)

    def test_execution(self):
        sim = MicrostructureSimulator(seed=42)
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        sim.on_tick(tick)
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=None,
        )
        report = sim.execute(proposal, current_price=100.0, current_volatility=0.01)
        self.assertEqual(report.filled_qty, 1.0)
        self.assertNotEqual(report.filled_price, 100.0)
        self.assertTrue(report.fee > 0)


class TestExecutionSimulator(unittest.TestCase):

    def test_basic_execution(self):
        micro = MicrostructureSimulator(seed=42)
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        micro.on_tick(tick)
        exec_sim = ExecutionSimulator(micro, seed=42)
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=None,
        )
        result = exec_sim.execute(proposal)
        self.assertEqual(result.status, "FILLED")
        self.assertGreater(result.filled_price, 0)
        self.assertGreater(result.fees, 0)

    def test_rejected_no_market_data(self):
        micro = MicrostructureSimulator(seed=42)
        exec_sim = ExecutionSimulator(micro, seed=42)
        proposal = TradeProposal(
            action=TradeAction.BUY, symbol="TEST", quantity=1.0,
            price=100.0, reason="Test", timestamp=datetime.now(),
            regime_state=None,
        )
        result = exec_sim.execute(proposal)
        self.assertEqual(result.status, "REJECTED")

    def test_symmetric_slippage(self):
        # verif rapide
        micro = MicrostructureSimulator(seed=42)
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        micro.on_tick(tick)

        buy_slippages = []
        sell_slippages = []
        for seed in range(100):
            exec_sim = ExecutionSimulator(micro, seed=seed)
            buy_proposal = TradeProposal(
                action=TradeAction.BUY, symbol="TEST", quantity=1.0,
                price=100.0, reason="Test", timestamp=datetime.now(),
                regime_state=None,
            )
            sell_proposal = TradeProposal(
                action=TradeAction.SELL, symbol="TEST", quantity=1.0,
                price=100.0, reason="Test", timestamp=datetime.now(),
                regime_state=None,
            )
            buy_result = exec_sim.execute(buy_proposal)
            exec_sim2 = ExecutionSimulator(micro, seed=seed)
            sell_result = exec_sim2.execute(sell_proposal)

            buy_slippages.append(buy_result.filled_price - 100.0)
            sell_slippages.append(100.0 - sell_result.filled_price)

        # Mean slippage should be similar for buy and sell
        mean_buy = np.mean(buy_slippages)
        mean_sell = np.mean(sell_slippages)
        self.assertAlmostEqual(mean_buy, mean_sell, delta=0.01)


class TestPerformanceMetrics(unittest.TestCase):

    def test_empty_metrics(self):
        metrics = PerformanceMetrics()
        result = metrics.compute()
        self.assertEqual(result['total_trades'], 0)

    def test_trade_recording(self):
        metrics = PerformanceMetrics(initial_equity=10000.0)
        metrics.record_trade(TradeRecord(
            timestamp=1.0, symbol="TEST", side="BUY",
            qty=1.0, entry_price=100.0, pnl=50.0,
        ))
        result = metrics.compute()
        self.assertEqual(result['total_trades'], 1)
        self.assertEqual(result['total_pnl'], 50.0)

    def test_regime_tracking(self):
        metrics = PerformanceMetrics()
        metrics.update_regime("0", 1.0)
        metrics.update_regime("1", 2.0)
        self.assertEqual(metrics.transitions, 2)


class TestEventEngine(unittest.TestCase):

    def test_event_ordering(self):
        engine = EventEngine()
        processed = []
        engine.register_handler(
            EventType.MARKET_TICK,
            lambda e: processed.append(e.payload),
        )
        engine.put(Event(timestamp=datetime(2024, 1, 1, 0, 0, 2), type=EventType.MARKET_TICK, payload="second"))
        engine.put(Event(timestamp=datetime(2024, 1, 1, 0, 0, 1), type=EventType.MARKET_TICK, payload="first"))
        engine.run()
        self.assertEqual(processed, ["first", "second"])


class TestBacktestLoop(unittest.TestCase):

    def test_backtest_engine(self):
        stream = SyntheticTickStream("BTC", duration_seconds=1)
        strategy = MockStrategy()
        engine = BacktestEngine(strategy, stream)
        metrics = engine.run()
        self.assertTrue(strategy.ticks_processed > 0)
        self.assertEqual(len(metrics.trades), 1)
        self.assertEqual(metrics.trades[0].side, "BUY")


if __name__ == '__main__':
    unittest.main()
