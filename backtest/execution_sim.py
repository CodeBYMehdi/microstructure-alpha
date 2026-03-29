# simu pour pas pleurer en live
# verif rapide

from dataclasses import dataclass
import numpy as np
import uuid
from datetime import datetime
from core.types import TradeProposal, TradeAction, OrderResult
from backtest.microstructure_sim import MicrostructureSimulator
from config.loader import get_config


@dataclass
class ExecutionReport:
    # execution sans pitie
    filled_qty: float
    filled_price: float
    slippage: float
    fee: float
    timestamp: datetime
    status: str = "FILLED"


class ExecutionSimulator:
    # la calculette
    # on passe a la caisse

    def __init__(self, microstructure: MicrostructureSimulator, seed: int = 42):
        self.microstructure = microstructure
        self.rng = np.random.RandomState(seed)

        config = get_config()
        sim_cfg = config.thresholds.execution_sim
        self.base_fee = sim_cfg.base_fee_bps / 10000.0     # Convert bps to decimal
        self.slippage_std = sim_cfg.slippage_std_bps / 10000.0
        self.impact_coeff = sim_cfg.impact_coefficient

    def execute(self, proposal: TradeProposal) -> OrderResult:
        # l'usine a gaz
        market_state = self.microstructure
        current_price = market_state.last_trade.price if market_state.last_trade else 0.0

        if current_price <= 0.0:
            return OrderResult(
                order_id=str(uuid.uuid4()),
                status="REJECTED",
                filled_price=0.0,
                filled_quantity=0.0,
                timestamp=proposal.timestamp,
                fees=0.0,
            )

        # Validate quantity
        if proposal.quantity <= 0:
            return OrderResult(
                order_id=str(uuid.uuid4()),
                status="REJECTED",
                filled_price=0.0,
                filled_quantity=0.0,
                timestamp=proposal.timestamp,
                fees=0.0,
            )

        # 1. Base price
        target_price = proposal.price if proposal.price else current_price

        # 2. Market impact: L2-depth-aware when order book is available
        #    Walk the book to estimate how much depth the order consumes,
        #    then apply remaining qty through the square-root model.
        book = market_state.order_book
        is_buy = proposal.action == TradeAction.BUY
        levels = book.asks if is_buy else book.bids

        remaining_qty = proposal.quantity
        depth_cost = 0.0  # volume-weighted price deviation from mid
        mid = book.mid_price if book.mid_price > 0 else target_price

        if levels and mid > 0:
            for level in levels:
                if remaining_qty <= 0:
                    break
                fill_at_level = min(remaining_qty, level.size)
                depth_cost += fill_at_level * abs(level.price - mid)
                remaining_qty -= fill_at_level

        # depth_cost is total dollar-deviation; normalize to per-share bps
        if proposal.quantity > 0 and mid > 0:
            depth_impact = depth_cost / (proposal.quantity * mid)
        else:
            depth_impact = 0.0

        # Square-root impact for any residual qty that exceeded visible book depth
        residual_impact = self.impact_coeff * np.sqrt(max(0, remaining_qty)) if remaining_qty > 0 else 0.0

        # 2b. Partial fill: cap at total visible depth + 20% hidden liquidity
        total_visible_depth = sum(lv.size for lv in levels) if levels else float('inf')
        max_fillable = total_visible_depth * 1.2
        filled_qty = min(proposal.quantity, max_fillable)
        if filled_qty < 1e-10:
            return OrderResult(
                order_id=str(uuid.uuid4()),
                status="REJECTED",
                filled_price=0.0,
                filled_quantity=0.0,
                timestamp=proposal.timestamp,
                fees=0.0,
            )

        # 2c. Adverse selection: trading WITH recent momentum gets worse fills
        adverse_penalty = 0.0
        recent_prices = market_state.vol_window[-20:] if len(market_state.vol_window) >= 20 else market_state.vol_window
        if len(recent_prices) >= 5:
            recent_momentum = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 1e-10)
            # Buying into up-trend or selling into down-trend = adverse
            if (is_buy and recent_momentum > 0) or (not is_buy and recent_momentum < 0):
                adverse_penalty = abs(recent_momentum) * 0.5

        impact = depth_impact + residual_impact + adverse_penalty

        # 3. Volatility penalty
        vol_penalty = max(1.0, market_state.current_volatility * 1000)

        # 4. Random noise -- symmetric
        noise = self.rng.normal(0, self.slippage_std * vol_penalty)

        # 5. Directional slippage
        if is_buy:
            slippage = impact + noise
            exec_price = target_price + slippage
        else:
            slippage = impact - noise
            exec_price = target_price - slippage

        # 6. Fees (on filled quantity, not requested)
        fee = exec_price * filled_qty * self.base_fee

        return OrderResult(
            order_id=str(uuid.uuid4()),
            status="FILLED",
            filled_price=exec_price,
            filled_quantity=filled_qty,
            timestamp=proposal.timestamp,
            fees=fee,
        )
