from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    TRADE = auto()
    QUOTE = auto()
    L2_UPDATE = auto()
    BAR = auto()
    STATUS = auto()

@dataclass
class MarketEvent:
    timestamp_exchange: datetime
    timestamp_received: datetime
    instrument_id: str
    event_type: EventType
    price: Optional[float] = None
    size: Optional[float] = None
    side: Optional[str] = None  # 'BID', 'ASK', or None
    depth_level: Optional[int] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    raw_source: str = "IBKR"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[MarketEvent], None]]] = {}
        self._async_subscribers: Dict[EventType, List[Callable[[MarketEvent], Any]]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[MarketEvent], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_async(self, event_type: EventType, callback: Callable[[MarketEvent], Any]):
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = []
        self._async_subscribers[event_type].append(callback)

    async def publish(self, event: MarketEvent):
        # Sync subscribers — called directly
        self._dispatch_sync(event)

        # Async subscribers
        if event.event_type in self._async_subscribers:
            tasks = []
            for callback in self._async_subscribers[event.event_type]:
                tasks.append(asyncio.create_task(self._safe_async_call(callback, event)))
            if tasks:
                await asyncio.gather(*tasks)

    async def _safe_async_call(self, callback, event):
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Err dans abonné async pour {event.event_type}: {e}")

    def _dispatch_sync(self, event: MarketEvent) -> None:
        """Dispatch to synchronous subscribers directly — never loses events."""
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Err dans abonné sync pour {event.event_type}: {e}")

    def publish_sync(self, event: MarketEvent):
        """Synchronous publish — dispatches sync subscribers directly, schedules async if loop exists."""
        # Always dispatch sync subscribers immediately (never drop events)
        self._dispatch_sync(event)

        # Best-effort async subscribers
        if event.event_type in self._async_subscribers and self._async_subscribers[event.event_type]:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self._dispatch_async(event))
                    return
            except RuntimeError:
                pass

            try:
                asyncio.run(self._dispatch_async(event))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._dispatch_async(event))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

    async def _dispatch_async(self, event: MarketEvent) -> None:
        """Dispatch to async subscribers."""
        if event.event_type in self._async_subscribers:
            tasks = []
            for callback in self._async_subscribers[event.event_type]:
                tasks.append(asyncio.create_task(self._safe_async_call(callback, event)))
            if tasks:
                await asyncio.gather(*tasks)

# Instance globale
event_bus = EventBus()
