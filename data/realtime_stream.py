import asyncio
import logging
from typing import Dict, Optional

from ibapi.contract import Contract
from data.ib_client import IBClient
from data.l2_orderbook import L2OrderBook

logger = logging.getLogger(__name__)

class RealTimeStream:
    def __init__(self, ib_client_instance: IBClient):
        self.ib = ib_client_instance
        self.active_subscriptions: Dict[str, Dict[str, int]] = {} # symbol -> {type: req_id}
        self.order_books: Dict[str, L2OrderBook] = {}
        self.instrument_map: Dict[int, str] = {} # req_id -> symbol

    async def start(self):
        # Ensure IB client is connected
        if not self.ib.isConnected():
            logger.info("Connecting IB Client...")
            self.ib.connect_and_start(asyncio.get_running_loop())

        # Request live (streaming) market data type
        # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
        # Paper accounts need market data subscriptions for live data.
        # If no subscription, set to 3 (delayed) as fallback.
        self.ib.reqMarketDataType(1)
        logger.info("Requested live market data type (1)")

    def subscribe_instrument(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD", use_tick_by_tick: bool = True):
        # l'usine a gaz
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        logger.info(f"Subscribing to {symbol}...")

        # 1. Données Niv 1 (Trades, Offre/Demande, Vol)
        # reqMktData(reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions)
        # Ticks génériques: 233 (RTVol), 221 (PrixMark), 225 (Enchère)
        req_id_l1 = self.ib.get_req_id()
        self.instrument_map[req_id_l1] = symbol

        # Stocker info abo pour reco
        self.ib._subscriptions[req_id_l1] = {
            'type': 'mktData',
            'contract': contract,
            'generic_ticks': "233,221",
            'snapshot': False
        }

        self.ib.reqMktData(req_id_l1, contract, "233,221", False, False, [])

        # Données Tick-par-Tick (Granularité sup)
        req_id_tbt = None
        if use_tick_by_tick:
            req_id_tbt = self.ib.get_req_id()
            self.instrument_map[req_id_tbt] = symbol

            self.ib._subscriptions[req_id_tbt] = {
                'type': 'tickByTick',
                'contract': contract,
                'tick_type': 'AllLast', # Trades
                'number_of_ticks': 0,
                'ignore_size': False
            }
            # Abo aux Trades (AllLast)
            self.ib.reqTickByTickData(req_id_tbt, contract, "AllLast", 0, False)

            # Note: Pour Offre/Demande tick-par-tick, peut nécessiter autre req "BidAsk"
            # Mais IB limite svt req tick-par-tick concurrentes. "AllLast" critique pour trades.
            # "BidAsk" critique pour cotations.
            # Demandons BidAsk si nécessaire, mais pour l'instant AllLast + MktData pour cotations.

        # 2. Données Niv 2 (Carnet Ordres)
        req_id_l2 = self.ib.get_req_id()
        self.instrument_map[req_id_l2] = symbol

        # Init Carnet Ordres
        self.order_books[symbol] = L2OrderBook(instrument_id=symbol)

        self.ib._subscriptions[req_id_l2] = {
            'type': 'mktDepth',
            'contract': contract,
            'num_rows': 10,
            'is_smart': True
        }

        # reqMktDepth(reqId, contract, numRows, isSmartDepth, mktDepthOptions)
        self.ib.reqMktDepth(req_id_l2, contract, 10, True, [])

        self.active_subscriptions[symbol] = {
            'l1': req_id_l1,
            'l2': req_id_l2
        }
        logger.info(f"Subscribed to {symbol}: L1 (Id: {req_id_l1}), L2 (Id: {req_id_l2})")

    def unsubscribe_instrument(self, symbol: str):
        if symbol in self.active_subscriptions:
            ids = self.active_subscriptions[symbol]
            self.ib.cancelMktData(ids['l1'])
            self.ib.cancelMktDepth(ids['l2'], True)
            del self.active_subscriptions[symbol]
            # Clean up instrument map (optional, strictly speaking)

    def get_order_book(self, symbol: str) -> Optional[L2OrderBook]:
        return self.order_books.get(symbol)

    def _handle_reconnect(self):
        # Logic to resubscribe is partly in IBClient._resubscribe_all
        # But we might need higher level logic here.
        # For now, IBClient stores the raw params to replay.
        pass

# Helper to modify IBClient to actually use the stored subscriptions
def patch_ib_client_resubscribe(client: IBClient):
    def new_resubscribe():
        logger.info("Executing custom resubscribe logic...")
        for req_id, sub in client._subscriptions.items():
            if sub['type'] == 'mktData':
                logger.info(f"Resubscribing MktData {req_id}")
                client.reqMktData(req_id, sub['contract'], sub['generic_ticks'], sub['snapshot'], False, [])
            elif sub['type'] == 'mktDepth':
                logger.info(f"Resubscribing MktDepth {req_id}")
                client.reqMktDepth(req_id, sub['contract'], sub['num_rows'], sub['is_smart'], [])
            elif sub['type'] == 'tickByTick':
                logger.info(f"Resubscribing TickByTick {req_id}")
                client.reqTickByTickData(req_id, sub['contract'], sub['tick_type'], sub['number_of_ticks'], sub['ignore_size'])
            elif sub['type'] == 'historical':
                 # Usually don't auto-resubscribe historical unless it's a keep-up-to-date
                 pass

    client._resubscribe_all = new_resubscribe

# NOTE: Do NOT patch at module level — ib_client is None at import time.
# Call patch_ib_client_resubscribe(ib_instance) in _run_live() after creating the IBClient.
