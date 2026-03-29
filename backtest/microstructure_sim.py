from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from core.types import Tick

@dataclass
class L2Level:
    price: float
    size: float
    orders: int = 1

@dataclass
class OrderBook:
    bids: List[L2Level] = field(default_factory=list)
    asks: List[L2Level] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    mid_price: float = 0.0
    spread: float = 0.0

@dataclass
class ExecutionReport:
    filled_qty: float
    filled_price: float
    slippage: float
    fee: float
    timestamp: datetime
    status: str = "FILLED"

class MicrostructureSimulator:
    # l'usine a gaz
    def __init__(self, symbol: str = "TEST", seed: int = 42):
        self.symbol = symbol
        self.rng = np.random.RandomState(seed)
        self.order_book = OrderBook()
        self.last_trade: Optional[Tick] = None
        self.current_volatility = 0.0001
        self.vol_window: List[float] = []
        self.window_size = 100
        
        # Param simu pour prof synthétique si pas de L2
        self.synthetic_depth = True
        self.base_spread_bps = 0.0002 # 2 bps
        
    def on_tick(self, tick: Tick):
        # l'usine a gaz
        self.last_trade = tick
        
        # MAJ estim vol
        if len(self.vol_window) > 0:
            ret = np.log(tick.price / self.vol_window[-1])
            # MAJ récursive simple ou append fenêtre?
            # Simple pour l'instant, append fenêtre
        
        self.vol_window.append(tick.price)
        if len(self.vol_window) > self.window_size:
            self.vol_window.pop(0)
            
        if len(self.vol_window) > 2:
            prices = np.array(self.vol_window)
            returns = np.diff(np.log(prices))
            self.current_volatility = np.std(returns)
            
        # Si génération L2 synthétique, MAJ mtn
        if self.synthetic_depth:
            self._generate_synthetic_book(tick.price)
            
    def on_l2_update(self, bids: List[L2Level], asks: List[L2Level], timestamp: datetime):
        # l'usine a gaz
        self.synthetic_depth = False
        self.order_book.bids = bids
        self.order_book.asks = asks
        self.order_book.timestamp = timestamp
        
        best_bid = bids[0].price if bids else 0
        best_ask = asks[0].price if asks else float('inf')
        
        if best_bid > 0 and best_ask < float('inf'):
            self.order_book.mid_price = (best_bid + best_ask) / 2
            self.order_book.spread = best_ask - best_bid
            
    def _generate_synthetic_book(self, mid_price: float):
        # l'usine a gaz
        spread = mid_price * self.base_spread_bps * (1 + self.current_volatility * 100)
        half_spread = spread / 2
        
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Génère 5 niveaux
        self.order_book.bids = []
        self.order_book.asks = []
        
        # Depth contracts during high vol — models market maker withdrawal
        # Base 1000 shares at level 1; during crisis vol (>0.01), drops to ~50
        vol_depth_scale = max(0.05, 1.0 - self.current_volatility * 50)

        for i in range(5):
            base_depth = 1000 * (i + 1) * (1 + self.rng.rand())
            size = base_depth * vol_depth_scale

            self.order_book.bids.append(L2Level(
                price=best_bid - i * self.base_spread_bps * mid_price,
                size=size
            ))

            self.order_book.asks.append(L2Level(
                price=best_ask + i * self.base_spread_bps * mid_price,
                size=size
            ))
            
        self.order_book.mid_price = mid_price
        self.order_book.spread = spread

    def get_liquidity_at_price(self, price: float, side: str) -> float:
        # l'usine a gaz
        # Recherche simple dans carnet
        total_size = 0.0
        if side == "BUY":
            # Regarde asks
            for level in self.order_book.asks:
                if level.price <= price:
                    total_size += level.size
        else:
            # Regarde bids
            for level in self.order_book.bids:
                if level.price >= price:
                    total_size += level.size
        return total_size

    def execute(self, proposal, current_price: float, current_volatility: float) -> ExecutionReport:
        impact = 0.00001 * np.sqrt(proposal.quantity)
        vol_penalty = max(1.0, current_volatility * 1000)
        noise = self.rng.normal(0, 0.0001 * vol_penalty)
        slippage = impact + abs(noise)
        if proposal.action.value == "BUY":
            exec_price = current_price + slippage
        else:
            exec_price = current_price - slippage
        fee = exec_price * proposal.quantity * 0.0001
        return ExecutionReport(
            filled_qty=proposal.quantity,
            filled_price=exec_price,
            slippage=slippage,
            fee=fee,
            timestamp=proposal.timestamp
        )
