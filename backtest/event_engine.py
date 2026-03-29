from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from datetime import datetime
import queue

from backtest.metrics import PerformanceMetrics, TradeRecord
from backtest.microstructure_sim import MicrostructureSimulator
from backtest.execution_sim import ExecutionSimulator

class RecordingRouter:
    def __init__(self, base_router, on_fill):
        self._base_router = base_router
        self._on_fill = on_fill

    def execute(self, proposal, current_tick=None):
        result = self._base_router.execute(proposal)
        if result.status == "FILLED":
            self._on_fill(proposal, result)
        return result

class EventType(Enum):
    MARKET_TICK = "MARKET_TICK"
    MARKET_L2 = "MARKET_L2"
    SIGNAL = "SIGNAL"
    ORDER_REQUEST = "ORDER_REQUEST"
    ORDER_FILL = "ORDER_FILL"
    REGIME_CHANGE = "REGIME_CHANGE"
    KILL_SWITCH = "KILL_SWITCH"
    SYSTEM_STATUS = "SYSTEM_STATUS"

@dataclass(order=True)
class Event:
    timestamp: datetime
    priority: int = field(default=10) # Lower number = higher priority (market data=0 < orders=10)
    type: EventType = field(compare=False, default=EventType.SYSTEM_STATUS)
    payload: Any = field(compare=False, default=None)

class EventEngine:
    # simu pour pas pleurer en live
    # verif rapide
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.active = False
        self.handlers = {event_type: [] for event_type in EventType}
        self.current_time = datetime.min

    def put(self, event: Event):
        # l'usine a gaz
        if event.timestamp < self.current_time:
            # En réel, poss latence, 
            # mais backtest = viol causalité sauf si expl. autorisé.
            # Pour backtest strict, warn.
            pass 
        self.queue.put(event)

    def register_handler(self, event_type: EventType, handler):
        # l'usine a gaz
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def run(self):
        # l'usine a gaz
        self.active = True
        while self.active and not self.queue.empty():
            try:
                event = self.queue.get(block=False)
                self.current_time = event.timestamp
                self._dispatch(event)
            except queue.Empty:
                break
        self.active = False

    def _dispatch(self, event: Event):
        # l'usine a gaz
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                handler(event)

    def clear(self):
        self.queue = queue.PriorityQueue()
        self.current_time = datetime.min

class BacktestEngine(EventEngine):
    def __init__(self, strategy, stream):
        super().__init__()
        self.strategy = strategy
        self.stream = stream
        symbol = getattr(stream, "symbol", "TEST")
        self.microstructure = MicrostructureSimulator(symbol=symbol)
        self.metrics = PerformanceMetrics()
        base_router = ExecutionSimulator(self.microstructure)
        self.strategy.router = RecordingRouter(base_router, self.record_trade)

    def run(self):
        for tick in self.stream:
            self.microstructure.on_tick(tick)
            self.strategy.on_tick(tick)
        return self.metrics

    def record_trade(self, proposal, result):
        regime_id = "UNDEFINED"
        if proposal.regime_state is not None:
            regime_id = str(proposal.regime_state.metadata.get("id", "UNDEFINED"))
        self.metrics.record_trade(TradeRecord(
            timestamp=result.timestamp.timestamp(),
            symbol=proposal.symbol,
            side=proposal.action.value,
            qty=result.filled_quantity,
            entry_price=result.filled_price,
            exit_price=0.0,
            pnl=-result.fees,
            regime_id=regime_id,
            commission=result.fees
        ))
