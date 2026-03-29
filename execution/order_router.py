from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
import threading
from core.interfaces import IExecutionHandler
from core.types import TradeProposal, OrderResult, TradeAction, Tick
from config.loader import get_config
from execution.slippage import SlippageModel
from execution.impact_model import MarketImpactModel
import numpy as np
import logging
import csv
import datetime
import os
import uuid

logger = logging.getLogger(__name__)


class OrderStateTracker:
    """SUBMITTED → FILLED/PARTIAL/CANCELLED — état thread-safe"""

    def __init__(self):
        self._lock = threading.Lock()
        self.confirmed_position: float = 0.0
        self.pending_delta: float = 0.0
        self._pending_orders: dict = {}

    def on_submit(self, order_id: str, signed_qty: float) -> None:
        with self._lock:
            self._pending_orders[order_id] = signed_qty
            self.pending_delta += signed_qty

    def on_fill(self, order_id: str, filled_signed_qty: float) -> None:
        with self._lock:
            self.confirmed_position += filled_signed_qty
            if order_id in self._pending_orders:
                self.pending_delta -= self._pending_orders.pop(order_id)

    def on_cancel(self, order_id: str) -> None:
        with self._lock:
            if order_id in self._pending_orders:
                self.pending_delta -= self._pending_orders.pop(order_id)

    def get_confirmed_position(self) -> float:
        with self._lock:
            return self.confirmed_position

    def get_expected_position(self) -> float:
        with self._lock:
            return self.confirmed_position + self.pending_delta


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"


@dataclass
class Order:
    symbol: str
    side: TradeAction
    qty: float
    order_type: OrderType
    price: Optional[float] = None
    urgency: str = "normal"


class PaperTradeLogger:
    """CSV append-only — fills paper uniquement"""

    COLUMNS = [
        "timestamp", "symbol", "side", "qty", "fill_price",
        "slippage_bps", "regime", "ic", "snr",
    ]

    def __init__(self, path: str = "paper_trades.csv"):
        self._path = path
        self._lock = threading.Lock()
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, timestamp, symbol, side, qty, fill_price, slippage_bps,
            regime="", ic=0.0, snr=0.0) -> None:
        with self._lock:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow([
                    timestamp, symbol, side, qty, f"{fill_price:.6f}",
                    f"{slippage_bps:.2f}", regime, f"{ic:.4f}", f"{snr:.4f}",
                ])


class OrderRouter(IExecutionHandler):

    def __init__(self, mode: str = "simulation"):
        config = get_config()
        self.mode = mode or config.execution.mode
        self.active_limit_orders = {}

        compliance = config.thresholds.compliance
        self._price_collar_pct = compliance.price_collar_pct
        self._max_order_notional = compliance.max_order_notional
        self._last_known_price: float = 0.0

        sim_cfg = config.thresholds.execution_sim
        self._commission_bps = sim_cfg.base_fee_bps
        self.slippage_model = SlippageModel(
            base_spread_bps=sim_cfg.base_fee_bps,
            impact_coeff=sim_cfg.impact_coefficient,
            noise_std_bps=sim_cfg.slippage_std_bps,
        )
        self.impact_model = MarketImpactModel()

        # rate limiting: 10/s, 100/min — protection contre les boucles
        self._order_timestamps: list = []
        self._rate_lock = threading.Lock()
        self._max_orders_per_second = 10
        self._max_orders_per_minute = 100

        self._paper_logger = PaperTradeLogger() if self.mode == "paper" else None

        logger.info(f"Initialized OrderRouter in {self.mode} mode")

    def _check_rate_limit(self) -> bool:
        with self._rate_lock:
            now = time.time()
            self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
            recent_1s = sum(1 for t in self._order_timestamps if now - t < 1.0)
            if recent_1s >= self._max_orders_per_second:
                logger.warning(f"Rate limit: {recent_1s} orders in last second (max {self._max_orders_per_second})")
                return False
            if len(self._order_timestamps) >= self._max_orders_per_minute:
                logger.warning(f"Rate limit: {len(self._order_timestamps)} orders in last minute (max {self._max_orders_per_minute})")
                return False
            return True

    def execute(self, proposal: TradeProposal, current_tick: Optional[Tick] = None) -> OrderResult:
        order_id = str(uuid.uuid4())
        now = datetime.datetime.now()

        if proposal.quantity <= 0:
            logger.warning(f"Invalid order quantity: {proposal.quantity}")
            return OrderResult(
                order_id=order_id, status="REJECTED",
                filled_price=0.0, filled_quantity=0.0, timestamp=now, fees=0.0,
            )

        if not self._check_rate_limit():
            return OrderResult(
                order_id=order_id, status="REJECTED",
                filled_price=0.0, filled_quantity=0.0, timestamp=now, fees=0.0,
            )

        order = self._proposal_to_order(proposal)
        base_price = proposal.price if proposal.price and proposal.price > 0 else 0.0

        if base_price <= 0:
            logger.warning("Order has no valid price")
            return OrderResult(
                order_id=order_id, status="REJECTED",
                filled_price=0.0, filled_quantity=0.0, timestamp=now, fees=0.0,
            )

        # collar: rejette les prix trop éloignés du dernier tick connu
        if self._last_known_price > 0:
            deviation = abs(base_price - self._last_known_price) / self._last_known_price
            if deviation > self._price_collar_pct:
                logger.warning(
                    f"Price collar rejection: price={base_price:.4f} deviates "
                    f"{deviation:.2%} from last={self._last_known_price:.4f} (max {self._price_collar_pct:.0%})"
                )
                return OrderResult(
                    order_id=order_id, status="REJECTED",
                    filled_price=0.0, filled_quantity=0.0, timestamp=now, fees=0.0,
                )
        self._last_known_price = base_price

        notional = base_price * proposal.quantity
        if notional > self._max_order_notional:
            logger.warning(
                f"Max notional rejection: ${notional:.2f} > ${self._max_order_notional:.2f}"
            )
            return OrderResult(
                order_id=order_id, status="REJECTED",
                filled_price=0.0, filled_quantity=0.0, timestamp=now, fees=0.0,
            )

        is_buy = (order.side == TradeAction.BUY)
        bid = current_tick.bid if current_tick and current_tick.bid else None
        ask = current_tick.ask if current_tick and current_tick.ask else None
        volatility = 0.0001
        avg_volume = 1000.0

        slip_est = self.slippage_model.estimate(
            price=base_price, quantity=proposal.quantity, is_buy=is_buy,
            volatility=volatility, bid=bid, ask=ask,
        )
        impact_est = self.impact_model.estimate(
            order_qty=proposal.quantity, price=base_price,
            volatility=volatility, avg_volume=avg_volume,
        )

        filled_price = slip_est.execution_price
        filled_qty = proposal.quantity

        # simulation fill partiel sur limit — proportionnel au volume dispo
        if order.order_type == OrderType.LIMIT and current_tick is not None and self.mode == "simulation":
            queue_ahead = 0.0
            if is_buy and current_tick.bid == order.price:
                queue_ahead = current_tick.bid_size or 0.0
            elif not is_buy and current_tick.ask == order.price:
                queue_ahead = current_tick.ask_size or 0.0

            if queue_ahead > 0:
                available_volume = current_tick.volume or 0.0
                prob_fill = min(1.0, available_volume / max(1e-6, queue_ahead))
                if np.random.random() > prob_fill:
                    fill_ratio = max(0.0, min(1.0, prob_fill * np.random.uniform(0.3, 1.0)))
                    if fill_ratio < 0.1:
                        # miss complet — traverse le spread
                        if is_buy and ask:
                            filled_price = ask + impact_est.temporary_impact * base_price
                        elif not is_buy and bid:
                            filled_price = bid - impact_est.temporary_impact * base_price
                    else:
                        filled_qty = proposal.quantity * fill_ratio
                        logger.info(f"Partial fill: {fill_ratio:.0%} of {proposal.quantity:.2f}")

        # commission séparée du slippage pour éviter double-comptage du spread
        fees = slip_est.total_slippage + (filled_price * filled_qty * self._commission_bps / 10000.0)

        if self.mode == "paper":
            slippage_bps = slip_est.total_slippage / max(base_price * filled_qty, 1e-10) * 10000
            regime = getattr(proposal.regime_state, 'regime_type', '') if proposal.regime_state else ''
            self._paper_logger.log(
                timestamp=now.isoformat(),
                symbol=proposal.symbol,
                side=proposal.action.value,
                qty=filled_qty,
                fill_price=filled_price,
                slippage_bps=slippage_bps,
                regime=str(regime),
            )
            with self._rate_lock:
                self._order_timestamps.append(time.time())
            return OrderResult(
                order_id=order_id, status="FILLED",
                filled_price=filled_price, filled_quantity=filled_qty,
                timestamp=now, fees=fees,
            )

        success = self.send(order)
        status = "FILLED" if success else "FAILED"

        with self._rate_lock:
            self._order_timestamps.append(time.time())

        return OrderResult(
            order_id=order_id,
            status=status,
            filled_price=filled_price,
            filled_quantity=filled_qty,
            timestamp=now,
            fees=fees,
        )

    def send(self, order: Order) -> bool:
        if order.side not in (TradeAction.BUY, TradeAction.SELL):
            logger.warning(f"Invalid order side: {order.side}")
            return False

        logger.info(
            f"[{self.mode.upper()} EXECUTION] {order.side.value} "
            f"{order.qty} {order.symbol} @ {order.order_type.value}"
        )
        if order.price:
            logger.info(f"  Price: {order.price}")
        return True

    def cancel_all(self) -> None:
        """annule tous les ordres limit actifs — appelé sur kill switch"""
        n = len(self.active_limit_orders)
        self.active_limit_orders.clear()
        if n > 0:
            logger.critical(f"Cancelled {n} active limit orders")

    def _proposal_to_order(self, proposal: TradeProposal) -> Order:
        return Order(
            symbol=proposal.symbol,
            side=proposal.action,
            qty=proposal.quantity,
            order_type=OrderType.LIMIT if proposal.price else OrderType.MARKET,
            price=proposal.price,
            urgency="high",
        )
