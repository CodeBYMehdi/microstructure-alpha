"""IBKR Live Order Router — places real orders through TWS/Gateway.

For paper trading: connect to port 7497 (TWS) or 4002 (Gateway).
For live trading: connect to port 7496 (TWS) or 4001 (Gateway).

This router submits orders via the IB API and tracks fill status.
It integrates with the trade ledger for audit trail.
"""

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Optional, Dict

from ibapi.contract import Contract
from ibapi.order import Order as IBOrder
from core.interfaces import IExecutionHandler
from core.types import TradeProposal, OrderResult, TradeAction, Tick
from data.ib_client import IBClient

logger = logging.getLogger(__name__)


class IBKROrderRouter(IExecutionHandler):
    """Routes orders to IBKR TWS/Gateway for execution."""

    def __init__(self, ib_client: IBClient, account_id: str = "", mode: str = "paper"):
        self.ib = ib_client
        self.account_id = account_id
        self.mode = mode

        # Order tracking
        self._pending_orders: Dict[int, Dict] = {}  # ib_order_id -> {proposal, result_event, result}
        self._orphaned_orders: Dict[int, Dict] = {}  # orders that failed to cancel
        self._next_order_id: int = 0
        self._order_id_lock = threading.Lock()
        self._order_id_ready = threading.Event()

        # Rate limiting
        self._order_timestamps: list = []
        self._rate_lock = threading.Lock()
        self._max_orders_per_second = 5   # Conservative for paper
        self._max_orders_per_minute = 50

        # Register fill callback on the IB client
        self._patch_ib_callbacks()

        logger.info(f"IBKROrderRouter initialized: mode={mode}, account={account_id or 'default'}")

    def _patch_ib_callbacks(self):
        """Override IB client callbacks to capture order events."""
        original_next_valid_id = self.ib.nextValidId
        router = self

        def patched_next_valid_id(orderId):
            router._next_order_id = orderId
            router._order_id_ready.set()
            if original_next_valid_id:
                original_next_valid_id(orderId)

        def patched_order_status(orderId, status, filled, remaining, avgFillPrice,
                                  permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
            logger.info(
                f"[IBKR] OrderStatus: id={orderId}, status={status}, "
                f"filled={filled}, remaining={remaining}, avgPrice={avgFillPrice}"
            )
            if orderId in router._pending_orders:
                order_info = router._pending_orders[orderId]
                if status in ("Filled", "ApiCancelled", "Cancelled", "Inactive"):
                    result_status = "FILLED" if status == "Filled" else "CANCELLED"
                    order_info['result'] = OrderResult(
                        order_id=str(orderId),
                        status=result_status,
                        filled_price=avgFillPrice if avgFillPrice > 0 else 0.0,
                        filled_quantity=filled if filled > 0 else 0.0,
                        timestamp=datetime.now(),
                        fees=0.0,  # Updated by execDetails
                    )
                    order_info['fill_event'].set()
                elif status == "PreSubmitted" or status == "Submitted":
                    logger.info(f"[IBKR] Order {orderId} acknowledged: {status}")

        def patched_exec_details(reqId, contract, execution):
            logger.info(
                f"[IBKR] Execution: orderId={execution.orderId}, "
                f"side={execution.side}, qty={execution.shares}, "
                f"price={execution.price}, execId={execution.execId}"
            )
            oid = execution.orderId
            if oid in router._pending_orders:
                order_info = router._pending_orders[oid]
                # Store execId for commission matching
                order_info['exec_id'] = execution.execId
                order_info['result'] = OrderResult(
                    order_id=str(oid),
                    status="FILLED",
                    filled_price=execution.price,
                    filled_quantity=execution.shares,
                    timestamp=datetime.now(),
                    fees=0.0,  # Updated by commissionReport
                )
                # orderStatus(Filled) is the primary signal; execDetails
                # provides exact price.  Don't set fill_event here to give
                # commissionReport a chance to arrive with fees first.
                # orderStatus callback already sets fill_event as backup.

        def patched_commission_report(commissionReport):
            exec_id = commissionReport.execId
            commission = commissionReport.commission
            logger.info(f"[IBKR] Commission: execId={exec_id}, commission={commission:.4f}")
            # Find the order that matches this execution
            for oid, order_info in router._pending_orders.items():
                if order_info.get('exec_id') == exec_id and order_info.get('result'):
                    order_info['result'] = OrderResult(
                        order_id=order_info['result'].order_id,
                        status=order_info['result'].status,
                        filled_price=order_info['result'].filled_price,
                        filled_quantity=order_info['result'].filled_quantity,
                        timestamp=order_info['result'].timestamp,
                        fees=commission if commission < 1e9 else 0.0,  # IB sends 1e10 for "not yet known"
                    )
                    order_info['fill_event'].set()
                    return
            # Commission arrived but no matching order (race) — signal any pending fill
            logger.debug(f"[IBKR] CommissionReport for unknown execId={exec_id}")

        self.ib.nextValidId = patched_next_valid_id
        self.ib.orderStatus = patched_order_status
        self.ib.execDetails = patched_exec_details
        self.ib.commissionReport = patched_commission_report

    def _get_next_order_id(self) -> int:
        with self._order_id_lock:
            if self._next_order_id == 0:
                # Request valid order ID from TWS
                self.ib.reqIds(-1)
                if not self._order_id_ready.wait(timeout=5.0):
                    raise TimeoutError("Failed to get next valid order ID from TWS")
            oid = self._next_order_id
            self._next_order_id += 1
            return oid

    def _check_rate_limit(self) -> bool:
        with self._rate_lock:
            now = time.time()
            self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
            recent_1s = sum(1 for t in self._order_timestamps if now - t < 1.0)
            if recent_1s >= self._max_orders_per_second:
                return False
            if len(self._order_timestamps) >= self._max_orders_per_minute:
                return False
            return True

    def _make_contract(self, symbol: str, primary_exchange: str = "") -> Contract:
        """Create an IBKR Contract for a US equity.

        primary_exchange disambiguates SMART routing.  Common values:
        NASDAQ, NYSE, ARCA, BATS.  If blank, IBKR resolves automatically
        (may be slower or match the wrong listing for dual-listed names).
        """
        # Map common symbols to their actual primary exchange
        _PRIMARY_EXCHANGE = {
            "QQQ": "NASDAQ", "AAPL": "NASDAQ", "MSFT": "NASDAQ",
            "NVDA": "NASDAQ", "AMZN": "NASDAQ", "META": "NASDAQ",
            "TSLA": "NASDAQ", "GOOGL": "NASDAQ", "GOOG": "NASDAQ",
            "SPY": "ARCA", "IWM": "ARCA", "EEM": "ARCA",
            "TLT": "NASDAQ", "GLD": "ARCA", "XLF": "ARCA", "XLE": "ARCA",
            "HYG": "ARCA", "LQD": "ARCA",
        }
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.primaryExchange = primary_exchange or _PRIMARY_EXCHANGE.get(symbol, "")
        contract.currency = "USD"
        return contract

    def _make_order(self, proposal: TradeProposal) -> IBOrder:
        """Convert a TradeProposal to an IBKR Order."""
        order = IBOrder()
        order.action = "BUY" if proposal.action == TradeAction.BUY else "SELL"
        order.totalQuantity = int(proposal.quantity)  # IBKR requires integer shares for stocks
        order.account = self.account_id if self.account_id else ""

        if proposal.price and proposal.price > 0:
            # Limit order
            order.orderType = "LMT"
            order.lmtPrice = round(proposal.price, 2)
            # Time in force: good till cancelled or day
            order.tif = "DAY"
        else:
            # Market order
            order.orderType = "MKT"
            order.tif = "DAY"

        # Adaptive algo for better fills (IBKR paper supports this)
        # order.algoStrategy = "Adaptive"
        # order.algoParams = [TagValue("adaptivePriority", "Normal")]

        return order

    def execute(self, proposal: TradeProposal, current_tick: Optional[Tick] = None) -> OrderResult:
        """Submit order to IBKR and wait for fill (with timeout)."""
        now = datetime.now()
        str_id = str(uuid.uuid4())[:8]

        # Validate — reject sub-1 share orders (don't round up to 1)
        qty_int = int(proposal.quantity)
        if qty_int < 1:
            logger.debug(f"Order rejected: quantity {proposal.quantity:.4f} rounds to {qty_int} shares")
            return OrderResult(str_id, "REJECTED", 0.0, 0.0, now, 0.0)

        # Rate limit
        if not self._check_rate_limit():
            logger.warning("Order rate limited")
            return OrderResult(str_id, "REJECTED", 0.0, 0.0, now, 0.0)

        # Check IB connection
        if not self.ib._connected:
            logger.error("IBKR not connected — order rejected")
            return OrderResult(str_id, "REJECTED", 0.0, 0.0, now, 0.0)

        try:
            ib_order_id = self._get_next_order_id()
        except TimeoutError as e:
            logger.error(f"Order ID timeout: {e}")
            return OrderResult(str_id, "REJECTED", 0.0, 0.0, now, 0.0)

        contract = self._make_contract(proposal.symbol)
        ib_order = self._make_order(proposal)

        # Override quantity to integer
        ib_order.totalQuantity = qty_int

        # Track pending
        fill_event = threading.Event()
        self._pending_orders[ib_order_id] = {
            'proposal': proposal,
            'fill_event': fill_event,
            'result': None,
        }

        # Submit
        logger.info(
            f"[IBKR SUBMIT] OrderId={ib_order_id}, {ib_order.action} "
            f"{ib_order.totalQuantity} {proposal.symbol} @ "
            f"{ib_order.orderType} {getattr(ib_order, 'lmtPrice', 'MKT')}"
        )
        self.ib.placeOrder(ib_order_id, contract, ib_order)
        with self._rate_lock:
            self._order_timestamps.append(time.time())

        # Wait for fill (timeout: 30s for limit, 10s for market)
        timeout = 30.0 if ib_order.orderType == "LMT" else 10.0
        filled = fill_event.wait(timeout=timeout)

        order_info = self._pending_orders.pop(ib_order_id, {})
        result = order_info.get('result')

        if filled and result and result.status == "FILLED":
            logger.info(
                f"[IBKR FILL] {result.filled_quantity} @ {result.filled_price}"
            )
            return result
        elif filled and result:
            logger.warning(f"[IBKR] Order {ib_order_id} ended with status: {result.status}")
            return result
        else:
            # Timeout — cancel the order
            logger.warning(f"[IBKR] Order {ib_order_id} timed out after {timeout}s, cancelling")
            try:
                self.ib.cancelOrder(ib_order_id, "")
            except Exception as e:
                # Track orphaned order — cancel failed, order may still be live at broker
                orphan_info = self._pending_orders.get(ib_order_id, {})
                self._orphaned_orders[ib_order_id] = {
                    'proposal': orphan_info.get('proposal'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                }
                logger.critical(
                    f"[IBKR] ORPHANED ORDER {ib_order_id}: cancel failed ({e}). "
                    f"Order may still be live at broker! Orphans tracked: {len(self._orphaned_orders)}"
                )
            return OrderResult(str(ib_order_id), "TIMEOUT", 0.0, 0.0, datetime.now(), 0.0)

    def cancel_all(self):
        """Emergency: cancel all open orders."""
        logger.critical("[IBKR] CANCELLING ALL OPEN ORDERS")
        self.ib.reqGlobalCancel()

    def query_positions(self) -> Dict[str, float]:
        """Query current positions from broker. Returns {symbol: quantity}."""
        positions: Dict[str, float] = {}
        done_event = threading.Event()

        original_position = getattr(self.ib, 'position', None)
        original_position_end = getattr(self.ib, 'positionEnd', None)

        def on_position(account, contract, pos, avgCost):
            if pos != 0:
                positions[contract.symbol] = float(pos)

        def on_position_end():
            done_event.set()

        self.ib.position = on_position
        self.ib.positionEnd = on_position_end

        try:
            self.ib.reqPositions()
            if not done_event.wait(timeout=5.0):
                logger.warning("[IBKR] Position query timed out after 5s")
        except Exception as e:
            logger.error(f"[IBKR] Failed to query positions: {e}")
        finally:
            # Restore original callbacks
            if original_position is not None:
                self.ib.position = original_position
            if original_position_end is not None:
                self.ib.positionEnd = original_position_end

        logger.info(f"[IBKR] Positions: {positions}")
        return positions
