import numpy as np
from collections import deque
from typing import Deque, Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from data.event_bus import event_bus, MarketEvent, EventType
from data.l2_orderbook import L2OrderBook

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    timestamp: datetime
    mid_price: float
    log_return: float
    volatility: float
    drift: float
    order_book_imbalance: float
    spread: float
    trade_intensity: float
    
class DataNormalizer:
    def __init__(self, window_size_seconds: int = 60, min_history: int = 30):
        self.window_size = timedelta(seconds=window_size_seconds)
        self.min_history = min_history
        
        # État par instrument
        self.price_history: Dict[str, Deque[MarketEvent]] = {}
        self.trade_history: Dict[str, Deque[MarketEvent]] = {}
        self.l2_books: Dict[str, L2OrderBook] = {} # Reference to books (or updated via events)
        
        # S'abonner
        event_bus.subscribe(EventType.TRADE, self.on_trade)
        event_bus.subscribe(EventType.QUOTE, self.on_quote)
        event_bus.subscribe(EventType.L2_UPDATE, self.on_l2_update)
        
        logger.info("DataNormalizer initialized")

    def register_book(self, instrument_id: str, book: L2OrderBook):
        self.l2_books[instrument_id] = book

    def on_trade(self, event: MarketEvent):
        self._update_history(self.trade_history, event)
        # We might want to compute features on every trade or periodically
        # For high-frequency, maybe on every trade or throttled.
        # Here we just maintain state.
        
    def on_quote(self, event: MarketEvent):
        # MAJ cotation (L1) utilisable pour prix médian si L2 indispo
        pass

    def on_l2_update(self, event: MarketEvent):
        # MAJ L2 déclenche recalcul carnet, peut déclencher MAJ caractéristiques
        pass

    def _update_history(self, history_map: Dict, event: MarketEvent):
        if event.instrument_id not in history_map:
            history_map[event.instrument_id] = deque()
        
        queue = history_map[event.instrument_id]
        queue.append(event)
        
        # Prune old events
        cutoff = event.timestamp_received - self.window_size
        while queue and queue[0].timestamp_received < cutoff:
            queue.popleft()

    def compute_features(self, instrument_id: str) -> Optional[MarketFeatures]:
        # Besoin accès OrderBook pour déséquilibre et spread
        book = self.l2_books.get(instrument_id)
        if not book:
            return None
            
        mid_price = book.state.get_mid_price()
        if mid_price == 0: return None
        
        # Calc rendements et volatilité depuis histo récent trade/mid
        # Utiliser histo prix médian L2 serait mieux, mais faut le suivre.
        # Supposons snapshot prix médian à chaque MAJ L2 ? 
        # Pour l'instant, utiliser prix trade ou besoin histo prix médian.
        
        # Simplification: utiliser mid actuel et dernier stocké (faut le stocker)
        # Mieux: maintenir deque de (timestamp, mid_price)
        
        # Ajout suivi histo prix médian
        if not hasattr(self, 'mid_price_history'):
            self.mid_price_history = {}
        if instrument_id not in self.mid_price_history:
            self.mid_price_history[instrument_id] = deque()
            
        # Add current mid
        now = datetime.now() # or event time
        self.mid_price_history[instrument_id].append((now, mid_price))
        
        # Prune
        mp_queue = self.mid_price_history[instrument_id]
        cutoff = now - self.window_size
        while mp_queue and mp_queue[0][0] < cutoff:
            mp_queue.popleft()
            
        if len(mp_queue) < self.min_history:
            return None
            
        prices = np.array([p[1] for p in mp_queue])
        log_returns = np.diff(np.log(prices))
        
        if len(log_returns) == 0:
            vol = 0.0
            drift = 0.0
            curr_ret = 0.0
        else:
            vol = np.std(log_returns)
            drift = np.mean(log_returns)
            curr_ret = log_returns[-1]
            
        # Order Book Imbalance
        # (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
        bid_vol = sum(v[1] for v in book.state.bids.values())
        ask_vol = sum(v[1] for v in book.state.asks.values())
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
        
        # Spread
        best_bid = book.state.get_best_bid()[0]
        best_ask = book.state.get_best_ask()[0]
        spread = best_ask - best_bid
        
        # Trade Intensity
        trades = self.trade_history.get(instrument_id, [])
        trade_intensity = len(trades) / self.window_size.total_seconds()
        
        return MarketFeatures(
            timestamp=now,
            mid_price=mid_price,
            log_return=curr_ret,
            volatility=vol,
            drift=drift,
            order_book_imbalance=imbalance,
            spread=spread,
            trade_intensity=trade_intensity
        )
