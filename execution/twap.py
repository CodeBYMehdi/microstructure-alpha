"""TWAP/VWAP execution algorithms — slices large orders over time.

Institutional-grade execution that minimizes market impact by splitting
orders into smaller child orders spread across a time window.

Usage:
    twap = TWAPExecutor(router, duration_seconds=60, n_slices=10)
    result = await twap.execute(proposal, current_price=100.0)
"""

import asyncio
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
from collections import deque

from core.types import TradeProposal, OrderResult, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    """Result of a single TWAP/VWAP slice."""
    slice_idx: int
    timestamp: float
    target_qty: float
    filled_qty: float
    filled_price: float
    fees: float
    status: str            # FILLED, PARTIAL, REJECTED


@dataclass
class AlgoExecutionResult:
    """Aggregate result from an algorithmic execution."""
    order_id: str
    algo_type: str             # "TWAP" or "VWAP"
    symbol: str
    side: str
    total_requested_qty: float
    total_filled_qty: float
    vwap_price: float          # Volume-weighted average fill price
    total_fees: float
    n_slices: int
    n_filled: int
    n_rejected: int
    duration_seconds: float
    slices: List[SliceResult] = field(default_factory=list)
    arrival_price: float = 0.0     # Price at decision time
    implementation_shortfall: float = 0.0  # (VWAP - arrival) / arrival
    status: str = "COMPLETED"


class TWAPExecutor:
    """Time-Weighted Average Price execution algorithm.
    
    Splits a large order into N equal slices spread evenly across
    a time window. Each slice is submitted as a limit order at the
    current market price with a small offset.
    """

    def __init__(
        self,
        router,                         # IExecutionHandler
        duration_seconds: float = 60,   # Total execution window
        n_slices: int = 10,             # Number of child orders
        limit_offset_bps: float = 2.0,  # Limit price offset from mid
        max_retry_per_slice: int = 2,
        urgency_factor: float = 1.0,    # 1.0 = normal, <1 = patient, >1 = aggressive
    ):
        self.router = router
        self.duration_seconds = duration_seconds
        self.n_slices = max(1, n_slices)
        self.limit_offset_bps = limit_offset_bps
        self.max_retry = max_retry_per_slice
        self.urgency_factor = urgency_factor

        self._active = False
        self._cancel_requested = False

    def execute_sync(
        self,
        proposal: TradeProposal,
        current_price: float,
        get_price_fn: Optional[Callable[[], float]] = None,
    ) -> AlgoExecutionResult:
        """Synchronous TWAP execution (for backtest or simple use)."""
        start_time = time.time()
        arrival_price = current_price
        self._active = True
        self._cancel_requested = False

        # Calculate per-slice parameters
        slice_qty = proposal.quantity / self.n_slices
        slice_interval = self.duration_seconds / self.n_slices

        slices: List[SliceResult] = []
        total_filled_qty = 0.0
        total_notional = 0.0
        total_fees = 0.0
        remaining = proposal.quantity
        n_filled = 0
        n_rejected = 0

        for i in range(self.n_slices):
            if self._cancel_requested or remaining <= 0:
                break

            # Adjust quantity for last slice to avoid overfill
            qty = min(slice_qty, remaining)

            # Get current price (allow price discovery between slices)
            price = get_price_fn() if get_price_fn else current_price

            # Create child order
            child_proposal = TradeProposal(
                action=proposal.action,
                symbol=proposal.symbol,
                quantity=qty,
                price=price,
                reason=f"TWAP slice {i + 1}/{self.n_slices}",
                timestamp=proposal.timestamp,
                regime_state=proposal.regime_state,
            )

            # Execute child order (with retries)
            slice_result = None
            for attempt in range(self.max_retry + 1):
                try:
                    result = self.router.execute(child_proposal)
                    if result.status == "FILLED":
                        slice_result = SliceResult(
                            slice_idx=i, timestamp=time.time(),
                            target_qty=qty, filled_qty=result.filled_quantity,
                            filled_price=result.filled_price, fees=result.fees,
                            status="FILLED",
                        )
                        total_filled_qty += result.filled_quantity
                        total_notional += result.filled_quantity * result.filled_price
                        total_fees += result.fees
                        remaining -= result.filled_quantity
                        n_filled += 1
                        break
                    elif result.status == "PARTIAL":
                        slice_result = SliceResult(
                            slice_idx=i, timestamp=time.time(),
                            target_qty=qty, filled_qty=result.filled_quantity,
                            filled_price=result.filled_price, fees=result.fees,
                            status="PARTIAL",
                        )
                        total_filled_qty += result.filled_quantity
                        total_notional += result.filled_quantity * result.filled_price
                        total_fees += result.fees
                        remaining -= result.filled_quantity
                        n_filled += 1
                        break
                except Exception as e:
                    logger.warning(f"TWAP slice {i + 1} attempt {attempt + 1} failed: {e}")

            if slice_result is None:
                slice_result = SliceResult(
                    slice_idx=i, timestamp=time.time(),
                    target_qty=qty, filled_qty=0.0,
                    filled_price=0.0, fees=0.0,
                    status="REJECTED",
                )
                n_rejected += 1

            slices.append(slice_result)

            # Wait between slices (skip for last slice)
            if i < self.n_slices - 1 and not self._cancel_requested:
                time.sleep(slice_interval * self.urgency_factor)

        # Compute VWAP
        vwap = total_notional / total_filled_qty if total_filled_qty > 0 else 0.0
        duration = time.time() - start_time

        # Implementation shortfall: how much worse than arrival price
        impl_shortfall = 0.0
        if arrival_price > 0 and vwap > 0:
            if proposal.action == TradeAction.BUY:
                impl_shortfall = (vwap - arrival_price) / arrival_price
            else:
                impl_shortfall = (arrival_price - vwap) / arrival_price

        self._active = False

        status = "COMPLETED" if total_filled_qty >= proposal.quantity * 0.95 else "PARTIAL"
        if total_filled_qty == 0:
            status = "FAILED"

        return AlgoExecutionResult(
            order_id=f"TWAP_{int(start_time)}",
            algo_type="TWAP",
            symbol=proposal.symbol,
            side=proposal.action.value,
            total_requested_qty=proposal.quantity,
            total_filled_qty=total_filled_qty,
            vwap_price=vwap,
            total_fees=total_fees,
            n_slices=self.n_slices,
            n_filled=n_filled,
            n_rejected=n_rejected,
            duration_seconds=duration,
            slices=slices,
            arrival_price=arrival_price,
            implementation_shortfall=impl_shortfall,
            status=status,
        )

    def cancel(self) -> None:
        """Request cancellation of the running TWAP."""
        self._cancel_requested = True
        logger.info("TWAP cancellation requested")


class AdaptiveExecutor:
    """Selects the best execution method based on order characteristics.
    
    - Small orders (< threshold): Direct market/limit order
    - Large orders: TWAP with duration proportional to size
    - Urgent orders: Market order
    """

    def __init__(
        self,
        router,
        small_order_threshold: float = 100.0,    # Below this qty, use direct
        large_order_threshold: float = 1000.0,    # Above this, use aggressive TWAP
        base_twap_duration: float = 30.0,         # Base duration for TWAP
    ):
        self.router = router
        self.small_threshold = small_order_threshold
        self.large_threshold = large_order_threshold
        self.base_duration = base_twap_duration

    def execute(
        self,
        proposal: TradeProposal,
        current_price: float,
        urgency: str = "medium",
        get_price_fn: Optional[Callable[[], float]] = None,
    ) -> Any:
        """Execute using the best method for the order."""
        qty = proposal.quantity

        if urgency == "high" or qty <= self.small_threshold:
            # Direct execution
            logger.info(f"Direct execution: qty={qty}, urgency={urgency}")
            return self.router.execute(proposal)

        # TWAP for larger orders
        n_slices = max(3, int(qty / self.small_threshold))
        n_slices = min(n_slices, 20)  # Cap at 20 slices

        duration = self.base_duration
        if qty > self.large_threshold:
            duration = self.base_duration * 2
        if urgency == "low":
            duration *= 1.5

        urgency_factor = {"low": 1.2, "medium": 1.0, "high": 0.5}.get(urgency, 1.0)

        twap = TWAPExecutor(
            router=self.router,
            duration_seconds=duration,
            n_slices=n_slices,
            urgency_factor=urgency_factor,
        )

        logger.info(
            f"TWAP execution: qty={qty}, slices={n_slices}, duration={duration:.0f}s, "
            f"urgency={urgency}"
        )
        return twap.execute_sync(proposal, current_price, get_price_fn)
