import threading
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import TickerId, BarData
from ibapi.ticktype import TickTypeEnum

from data.event_bus import event_bus, MarketEvent, EventType

logger = logging.getLogger(__name__)

class IBClient(EWrapper, EClient):
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Gestion ID req
        self._next_req_id = 1
        self._req_id_lock = threading.Lock()
        
        # Suivi abonnements actifs pour restauration sur reconnexion
        self._subscriptions: Dict[int, Dict[str, Any]] = {}
        
    def connect_and_start(self, loop: asyncio.AbstractEventLoop, max_retries: int = 5):
        self._loop = loop
        self._max_retries = max_retries
        self._do_connect()

    def _do_connect(self):
        """Connect with exponential backoff retry logic."""
        for attempt in range(self._max_retries):
            try:
                self.connect(self.host, self.port, self.client_id)

                self._thread = threading.Thread(target=self.run, daemon=True)
                self._thread.start()

                # Wait for connection
                start_time = time.time()
                while not self._connected and time.time() - start_time < 10:
                    time.sleep(0.1)

                if self._connected:
                    logger.info(f"Connected to IBKR (attempt {attempt + 1})")
                    return
                else:
                    logger.warning(f"Connection timeout (attempt {attempt + 1}/{self._max_retries})")
            except Exception as e:
                logger.error(f"Connection failed (attempt {attempt + 1}/{self._max_retries}): {e}")

            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            backoff = min(2 ** attempt, 30)
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)

        logger.critical(f"Failed to connect to IBKR after {self._max_retries} attempts")

    def get_req_id(self) -> int:
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            return req_id

    # --- EWrapper Overrides ---

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        # Filtrer msgs inoffensifs
        if errorCode in [2104, 2106, 2158]: # Connexion ferme données marché OK
            logger.info(f"Notif IBKR: {errorCode} - {errorString}")
            return
            
        logger.error(f"Err IBKR. ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")
        
        if errorCode == 1100:  # Connectivity lost
            self._connected = False
            logger.warning("Connectivity lost. Starting auto-reconnect...")
            self._schedule_reconnect()
        elif errorCode == 1102:  # Connectivity restored
            self._connected = True
            logger.info("Connectivity restored. Resubscribing...")
            self._resubscribe_all()

    def connectAck(self):
        logger.info("IBKR Connect Ack")

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self._connected = True
        logger.info(f"Connecté. Prochain ID Ordre Valide: {orderId}")

    def connectionClosed(self):
        self._connected = False
        logger.warning("IBKR connection closed")
        self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Auto-reconnect in a background thread with exponential backoff."""
        def _reconnect():
            for attempt in range(getattr(self, '_max_retries', 5)):
                backoff = min(2 ** attempt, 30)
                logger.info(f"Auto-reconnect attempt {attempt + 1} in {backoff}s...")
                time.sleep(backoff)
                if self._connected:
                    return  # Someone else reconnected
                try:
                    self.disconnect()
                    time.sleep(0.5)
                    self._do_connect()
                    if self._connected:
                        self._resubscribe_all()
                        return
                except Exception as e:
                    logger.error(f"Reconnect attempt {attempt + 1} failed: {e}")
            logger.critical("Auto-reconnect exhausted all retries")

        thread = threading.Thread(target=_reconnect, daemon=True, name="ib-reconnect")
        thread.start()

    # --- Data Callbacks ---

    def tickPrice(self, reqId: TickerId, tickType: TickTypeEnum, price: float, attrib):
        if price == -1.0: return # Prix invalide
        
        # Map tickType vers sens
        # 1=Bid, 2=Ask, 4=Last, 6=High, 7=Low, 9=Close
        event_type = EventType.QUOTE
        side = None
        
        if tickType == 1: # Bid
            side = 'BID'
        elif tickType == 2: # Ask
            side = 'ASK'
        elif tickType == 4: # Last
            event_type = EventType.TRADE
        else:
            return # Ignore autres types pour l'instant

        self._emit_event(MarketEvent(
            timestamp_exchange=datetime.now(timezone.utc), # IB donne pas tjrs temps échange dans tickPrice, approx réception ou tickString si dispo
            timestamp_received=datetime.now(timezone.utc),
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=event_type,
            price=price,
            side=side,
            raw_source="IBKR_TICK_PRICE"
        ))

    def tickSize(self, reqId: TickerId, tickType: TickTypeEnum, size: float):
        # 0=Bid Size, 3=Ask Size, 5=Last Size, 8=Volume
        event_type = EventType.QUOTE
        side = None
        
        if tickType == 0:
            side = 'BID'
        elif tickType == 3:
            side = 'ASK'
        elif tickType == 5:
            event_type = EventType.TRADE
        elif tickType == 8:
            # MAJ volume, peut-être pas trade spécifique
            return
        else:
            return

        self._emit_event(MarketEvent(
            timestamp_exchange=datetime.now(timezone.utc),
            timestamp_received=datetime.now(timezone.utc),
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=event_type,
            size=size,
            side=side,
            raw_source="IBKR_TICK_SIZE"
        ))
        
    def tickString(self, reqId: TickerId, tickType: TickTypeEnum, value: str):
        # Gère ticks timestamp (45)
        if tickType == 45: # Last Timestamp
            # value est string timestamp
            pass

    def tickByTickAllLast(self, reqId: int, tickType: int, time: int, price: float, size: float, tickAttribLast, exchange: str, specialConditions: str):
        self._emit_event(MarketEvent(
            timestamp_exchange=datetime.fromtimestamp(time, timezone.utc),
            timestamp_received=datetime.now(timezone.utc),
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=EventType.TRADE,
            price=price,
            size=size,
            raw_source="IBKR_TICK_BY_TICK",
            metadata={'exchange': exchange, 'conditions': specialConditions}
        ))

    def tickByTickBidAsk(self, reqId: int, time: int, bidPrice: float, askPrice: float, bidSize: float, askSize: float, tickAttribBidAsk):
        ts = datetime.fromtimestamp(time, timezone.utc)
        now = datetime.now(timezone.utc)
        
        # Emit Bid
        self._emit_event(MarketEvent(
            timestamp_exchange=ts,
            timestamp_received=now,
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=EventType.QUOTE,
            price=bidPrice,
            size=bidSize,
            side='BID',
            raw_source="IBKR_TICK_BY_TICK_BA"
        ))
        
        # Emit Ask
        self._emit_event(MarketEvent(
            timestamp_exchange=ts,
            timestamp_received=now,
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=EventType.QUOTE,
            price=askPrice,
            size=askSize,
            side='ASK',
            raw_source="IBKR_TICK_BY_TICK_BA"
        ))

    def updateMktDepth(self, reqId: TickerId, position: int, operation: int, side: int, price: float, size: float):
        # l'usine a gaz
        # Map side
        side_str = 'BID' if side == 1 else 'ASK'
        
        self._emit_event(MarketEvent(
            timestamp_exchange=datetime.now(timezone.utc),
            timestamp_received=datetime.now(timezone.utc),
            instrument_id=self._resolve_instrument_id(reqId),
            event_type=EventType.L2_UPDATE,
            price=price,
            size=size,
            side=side_str,
            depth_level=position,
            metadata={'operation': operation},
            raw_source="IBKR_MKT_DEPTH"
        ))

    def historicalData(self, reqId: int, bar: BarData):
        # Parse date
        try:
            # Format IB: yyyyMMdd  HH:mm:ss
            ts = datetime.strptime(bar.date, "%Y%m%d  %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
             # Fallback pour barres journalières qui sont yyyyMMdd
            try:
                ts = datetime.strptime(bar.date, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                logger.warning("Unparseable IB bar date: %s — using current time", bar.date)
                ts = datetime.now(timezone.utc)

        self._emit_event(MarketEvent(
            timestamp_exchange=ts,
            timestamp_received=datetime.now(timezone.utc),
            instrument_id=str(reqId),
            event_type=EventType.BAR,
            price=bar.close,
            size=bar.volume, # ou bar.count
            metadata={
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'bar_count': bar.barCount,
                'wap': bar.wap
            },
            raw_source="IBKR_HISTORICAL"
        ))

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logger.info(f"Fin données histo pour ReqId {reqId}")

    def historicalTicks(self, reqId: int, ticks: List[Any], done: bool):
        for tick in ticks:
            self._emit_event(MarketEvent(
                timestamp_exchange=datetime.fromtimestamp(tick.time, timezone.utc),
                timestamp_received=datetime.now(timezone.utc),
                instrument_id=self._resolve_instrument_id(reqId),
                event_type=EventType.TRADE,
                price=tick.price,
                size=tick.size,
                raw_source="IBKR_HIST_TICKS"
            ))

    def historicalTicksBidAsk(self, reqId: int, ticks: List[Any], done: bool):
        for tick in ticks:
            # Emit Bid
            self._emit_event(MarketEvent(
                timestamp_exchange=datetime.fromtimestamp(tick.time, timezone.utc),
                timestamp_received=datetime.now(timezone.utc),
                instrument_id=self._resolve_instrument_id(reqId),
                event_type=EventType.QUOTE,
                price=tick.priceBid,
                size=tick.sizeBid,
                side='BID',
                raw_source="IBKR_HIST_TICKS_BA"
            ))
            # Emit Ask
            self._emit_event(MarketEvent(
                timestamp_exchange=datetime.fromtimestamp(tick.time, timezone.utc),
                timestamp_received=datetime.now(timezone.utc),
                instrument_id=self._resolve_instrument_id(reqId),
                event_type=EventType.QUOTE,
                price=tick.priceAsk,
                size=tick.sizeAsk,
                side='ASK',
                raw_source="IBKR_HIST_TICKS_BA"
            ))

    def historicalTicksLast(self, reqId: int, ticks: List[Any], done: bool):
        for tick in ticks:
             self._emit_event(MarketEvent(
                timestamp_exchange=datetime.fromtimestamp(tick.time, timezone.utc),
                timestamp_received=datetime.now(timezone.utc),
                instrument_id=self._resolve_instrument_id(reqId),
                event_type=EventType.TRADE,
                price=tick.price,
                size=tick.size,
                raw_source="IBKR_HIST_TICKS_LAST"
            ))

    # --- Helpers ---

    def _emit_event(self, event: MarketEvent):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(event_bus.publish(event), self._loop)
        else:
            # No async loop available — use synchronous dispatch so events are never dropped
            event_bus.publish_sync(event)

    def _resolve_instrument_id(self, req_id: int) -> str:
        sub = self._subscriptions.get(req_id)
        if sub and 'contract' in sub:
            contract = sub.get('contract')
            if contract and getattr(contract, 'symbol', None):
                return contract.symbol
        return str(req_id)

    def _resubscribe_all(self):
        """Re-subscribe to all active data feeds after reconnect.

        Iterates self._subscriptions (populated by RealTimeStream) and
        replays each subscription request. This is the default implementation;
        patch_ib_client_resubscribe() in realtime_stream.py replaces it with
        a version that also handles tick-by-tick feeds.
        """
        if not self._subscriptions:
            logger.info("No subscriptions to restore after reconnect.")
            return

        logger.info(f"Resubscribing {len(self._subscriptions)} data feeds...")
        for req_id, sub in self._subscriptions.items():
            try:
                sub_type = sub.get('type', '')
                contract = sub.get('contract')
                if not contract:
                    continue
                if sub_type == 'mktData':
                    self.reqMktData(req_id, contract, sub.get('generic_ticks', ''), sub.get('snapshot', False), False, [])
                    logger.info(f"Resubscribed MktData req_id={req_id}")
                elif sub_type == 'mktDepth':
                    self.reqMktDepth(req_id, contract, sub.get('num_rows', 10), sub.get('is_smart', True), [])
                    logger.info(f"Resubscribed MktDepth req_id={req_id}")
                elif sub_type == 'tickByTick':
                    self.reqTickByTickData(req_id, contract, sub.get('tick_type', 'AllLast'), sub.get('number_of_ticks', 0), sub.get('ignore_size', False))
                    logger.info(f"Resubscribed TickByTick req_id={req_id}")
            except Exception as e:
                logger.error(f"Failed to resubscribe req_id={req_id}: {e}")

# Placeholder instance globale
ib_client = None
