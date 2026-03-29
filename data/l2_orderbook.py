from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import deque
import numpy as np

from data.event_bus import event_bus, MarketEvent, EventType

logger = logging.getLogger(__name__)

@dataclass
class OrderBookState:
    bids: Dict[int, Tuple[float, float]] = field(default_factory=dict) # position -> (price, size)
    asks: Dict[int, Tuple[float, float]] = field(default_factory=dict) # position -> (price, size)
    timestamp: Optional[datetime] = None
    
    def get_best_bid(self) -> Tuple[float, float]:
        if not self.bids: return (0.0, 0.0)
        if 0 in self.bids: return self.bids[0]
        return (0.0, 0.0)

    def get_best_ask(self) -> Tuple[float, float]:
        if not self.asks: return (float('inf'), 0.0)
        if 0 in self.asks: return self.asks[0]
        return (float('inf'), 0.0)

    def get_mid_price(self) -> float:
        bb = self.get_best_bid()[0]
        ba = self.get_best_ask()[0]
        if bb > 0 and ba < float('inf'):
            return (bb + ba) / 2.0
        return 0.0

class L2OrderBook:
    # la calculette
    # on passe a la caisse

    def __init__(self, instrument_id: str, config=None):
        from config.loader import get_config
        _cfg = (config or get_config()).thresholds.decision.microstructure
        self.instrument_id = instrument_id
        self.state = OrderBookState()
        self.max_depth = 10

        # Config-driven thresholds
        self._liq_pull_threshold = _cfg.liq_pull_threshold
        self._spoofing_size_mult = _cfg.spoofing_size_mult

        # Historical metrics for event detection
        self._last_total_bid_vol = 0.0
        self._last_total_ask_vol = 0.0

        # Liquidity pull tracking (exponentially decaying score)
        self._liq_pull_bid_score = 0.0
        self._liq_pull_ask_score = 0.0
        self._liq_pull_decay = 0.95

        # Spoofing detection: track sizes at top-3 levels
        self._top_level_sizes: deque = deque(maxlen=100)
        self._spoofing_score = 0.0
        self._spoofing_decay = 0.98

        # Book pressure history
        self._pressure_history: deque = deque(maxlen=50)

        # Subscribe to events
        event_bus.subscribe(EventType.L2_UPDATE, self.on_l2_update)
        logger.info(f"L2OrderBook initialized for {instrument_id}")

    def on_l2_update(self, event: MarketEvent):
        if event.instrument_id != self.instrument_id:
            return
            
        self.state.timestamp = event.timestamp_exchange
        
        operation = event.metadata.get('operation')
        side_map = self.state.bids if event.side == 'BID' else self.state.asks
        position = event.depth_level
        
        # 0 = insert, 1 = update, 2 = delete
        if operation == 0:  # Insert
            for i in range(self.max_depth - 1, position, -1):
                if i-1 in side_map:
                    side_map[i] = side_map[i-1]
            side_map[position] = (event.price, event.size)
            
        elif operation == 1:  # Update
            side_map[position] = (event.price, event.size)
            
        elif operation == 2:  # Delete
            for i in range(position, self.max_depth - 1):
                if i+1 in side_map:
                    side_map[i] = side_map[i+1]
                else:
                    if i in side_map: del side_map[i]
            if self.max_depth - 1 in side_map:
                del side_map[self.max_depth - 1]

        # Analyze microstructure dynamics
        self._analyze_book_dynamics(event)

    def _analyze_book_dynamics(self, event: MarketEvent):
        current_bid_vol = sum(v[1] for v in self.state.bids.values())
        current_ask_vol = sum(v[1] for v in self.state.asks.values())
        
        # --- Liquidity Pull Detection ---
        # Decay existing scores
        self._liq_pull_bid_score *= self._liq_pull_decay
        self._liq_pull_ask_score *= self._liq_pull_decay

        if self._last_total_bid_vol > 0:
            delta_bid = (current_bid_vol - self._last_total_bid_vol) / self._last_total_bid_vol
            if delta_bid < self._liq_pull_threshold:
                # Score proportional to severity
                self._liq_pull_bid_score += abs(delta_bid)
                self._emit_microstructure_event("LIQUIDITY_PULL_BID", delta_bid)
                
        if self._last_total_ask_vol > 0:
            delta_ask = (current_ask_vol - self._last_total_ask_vol) / self._last_total_ask_vol
            if delta_ask < self._liq_pull_threshold:
                self._liq_pull_ask_score += abs(delta_ask)
                self._emit_microstructure_event("LIQUIDITY_PULL_ASK", delta_ask)

        # --- Spoofing Detection ---
        # Track sizes at top-3 levels; detect flash-size (>5x median)
        self._spoofing_score *= self._spoofing_decay
        if event.metadata.get('operation') in [0, 1] and event.depth_level < 3:
            size = event.size if event.size else 0.0
            self._top_level_sizes.append(size)
            if len(self._top_level_sizes) >= 20:
                median_size = float(np.median(list(self._top_level_sizes)))
                if median_size > 0 and size > self._spoofing_size_mult * median_size:
                    # Flash size detected — potential spoofing
                    self._spoofing_score += min(1.0, size / (10.0 * median_size))
                    self._emit_microstructure_event("SPOOFING_SUSPECTED", size / median_size)

        self._last_total_bid_vol = current_bid_vol
        self._last_total_ask_vol = current_ask_vol

    def _emit_microstructure_event(self, event_name: str, value: float):
        logger.debug(f"Microstructure Event {event_name}: {value:.2%}")

    def get_features(self) -> Dict[str, float]:
        # on cherche les pepites
        # la calculette
        # 1. Depth imbalance
        total_bid_vol = sum(v[1] for v in self.state.bids.values()) if self.state.bids else 0.0
        total_ask_vol = sum(v[1] for v in self.state.asks.values()) if self.state.asks else 0.0
        total_depth = total_bid_vol + total_ask_vol
        depth_imbalance = (total_bid_vol - total_ask_vol) / total_depth if total_depth > 0 else 0.0

        # 2. Liquidity pull score (net: bid pull is bearish signal, ask pull is bullish)
        # Positive = ask-side pulls (bullish), Negative = bid-side pulls (bearish)
        liquidity_pull_score = self._liq_pull_ask_score - self._liq_pull_bid_score

        # 3. Book pressure: cumulative depth-weighted price asymmetry
        # Measures how much "weight" is supporting bid vs ask side
        book_pressure = 0.0
        mid = self.state.get_mid_price()
        if mid > 0:
            bid_pressure = 0.0
            ask_pressure = 0.0
            for pos, (price, size) in self.state.bids.items():
                if price > 0:
                    # Closer bids get more weight (linear decay by level)
                    level_weight = 1.0 / (1.0 + pos)
                    bid_pressure += size * level_weight
            for pos, (price, size) in self.state.asks.items():
                if price < float('inf'):
                    level_weight = 1.0 / (1.0 + pos)
                    ask_pressure += size * level_weight
            total_pressure = bid_pressure + ask_pressure
            if total_pressure > 0:
                book_pressure = (bid_pressure - ask_pressure) / total_pressure

        # 4. Spoofing score 
        spoofing_score = float(np.clip(self._spoofing_score, 0.0, 5.0))

        # 5. Spread in basis points
        bb_price = self.state.get_best_bid()[0]
        ba_price = self.state.get_best_ask()[0]
        spread_bps = 0.0
        if bb_price > 0 and ba_price < float('inf') and mid > 0:
            spread_bps = ((ba_price - bb_price) / mid) * 10000.0

        return {
            'depth_imbalance': float(np.clip(depth_imbalance, -1.0, 1.0)),
            'liquidity_pull_score': float(np.clip(liquidity_pull_score, -5.0, 5.0)),
            'book_pressure': float(np.clip(book_pressure, -1.0, 1.0)),
            'spoofing_score': spoofing_score,
            'spread_bps': spread_bps,
        }

    def get_snapshot(self) -> Dict:
        return {
            'bids': list(self.state.bids.values()),
            'asks': list(self.state.asks.values()),
            'mid_price': self.state.get_mid_price(),
            'timestamp': self.state.timestamp,
            'features': self.get_features(),
        }
