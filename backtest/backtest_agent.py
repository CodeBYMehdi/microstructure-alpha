import copy
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import numpy as np

from backtest.event_engine import EventEngine, Event, EventType
from backtest.microstructure_sim import MicrostructureSimulator
from backtest.execution_sim import ExecutionSimulator
from backtest.metrics import PerformanceMetrics, TradeRecord

from core.types import Tick, TradeProposal, OrderResult
from core.interfaces import IExecutionHandler
from config.loader import get_config
from main import Strategy # Import the modified Strategy
from monitoring.event_bus import bus
# L2 book updates are done via direct state assignment for performance

logger = logging.getLogger(__name__)

class BacktestRouter(IExecutionHandler):
    # simu pour pas pleurer en live
    # verif rapide
    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine

    def execute(self, proposal: TradeProposal, current_tick=None) -> OrderResult:
        # Gen ID ordre prov
        order_id = str(uuid.uuid4())
        
        # Créer Event Req Ordre
        # Simu latence réseau ici si besoin, mais traité étape suiv.
        
        event = Event(
            timestamp=proposal.timestamp,
            priority=10,  # Orders AFTER market data (priority 0) to prevent look-ahead bias
            type=EventType.ORDER_REQUEST,
            payload={
                "order_id": order_id,
                "proposal": proposal
            },
        )
        self.event_engine.put(event)
        
        # Retour statut SUBMITTED imméd.
        return OrderResult(
            order_id=order_id,
            status="SUBMITTED",
            filled_price=0.0,
            filled_quantity=0.0,
            timestamp=proposal.timestamp,
            fees=0.0
        )

class BacktestAgent:
    # simu pour pas pleurer en live
    # verif rapide
    def __init__(self, data_stream, start_time: Optional[datetime] = None, config=None):
        self.stream = data_stream
        self.event_engine = EventEngine()
        self.metrics = PerformanceMetrics()
        
        # Composants Simu
        self.symbol = "SPY"
        self.microstructure = MicrostructureSimulator(self.symbol)
        self.execution = ExecutionSimulator(self.microstructure)
        
        # Router
        self.router = BacktestRouter(self.event_engine)
        
        # Deep copy config to avoid mutating the global singleton
        config = copy.deepcopy(config or get_config())
        # Relax constraints for backtesting
        config.thresholds.risk.confidence_floor = config.thresholds.backtest.relaxed_confidence_floor
        config.thresholds.risk.slippage_tolerance = config.thresholds.backtest.relaxed_slippage_tolerance
        
        # Strat
        # Init avec router et sans viz interne -> Modifié pour désactiver la viz pour la vitesse
        self.strategy = Strategy(config=config, router=self.router, enable_viz=False)
        
        # Simu Latence (sec)
        self.latency_model = {
            "order_entry": 0.005, # 5ms
            "market_data": 0.001  # 1ms
        }
        
        # Sub updates régime pour métriques
        bus.clear() # État propre
        bus.subscribe("regime_update", self._on_regime_update)
        
        self._register_handlers()

    def _register_handlers(self):
        self.event_engine.register_handler(EventType.MARKET_TICK, self._handle_tick)
        self.event_engine.register_handler(EventType.ORDER_REQUEST, self._handle_order_request)
        self.event_engine.register_handler(EventType.ORDER_FILL, self._handle_order_fill)
        self.event_engine.register_handler(EventType.MARKET_L2, self._handle_l2)

    def run(self):
        logger.info("Démarrage Backtest...")
        
        # Approche hybride:
        # Itérer flux. Pour chaque tick:
        # 1. Mettre tick dans file (avec latence MD).
        # 2. Exéc moteur jusqu'à vide (traite effets sec).
        # Garantit pas de look-ahead.
        
        count = 0
        for tick in self.stream:
            # Ajout Latence MD
            # timestamp = tick.timestamp + timedelta(seconds=self.latency_model["market_data"])
            # En fait, timestamp tick = moment occurence.
            # Strat le reçoit 'mtn' dans temps simu.
            
            # Mettre event tick
            self.event_engine.put(Event(
                timestamp=tick.timestamp,
                priority=0,  # Market data has highest priority (processed first)
                type=EventType.MARKET_TICK,
                payload=tick,
            ))
            
            # Exéc moteur jusqu'à traitement csq ce tick
            self.event_engine.run()
            
            count += 1
            if count % 100 == 0:
                print(f"Traité {count} ticks...", end='\r', flush=True)
                
        logger.info(f"Backtest Terminé. Traité {count} ticks.")
        
        # Calc métriques finales
        results = self.metrics.compute()
        return results

    def _handle_tick(self, event: Event):
        tick = event.payload

        # 1. Update microstructure state (always, including quotes)
        self.microstructure.on_tick(tick)

        # 2. Feed L2 book data into L2OrderBook via event bus
        #    When ticks carry full book snapshots (Databento MBP-10),
        #    publish L2 events so the order book and liquidity features update.
        if tick.bids or tick.asks:
            self._feed_l2_book(tick)

        # 3. Only feed trade ticks (volume > 0) to the strategy
        #    Quote ticks (volume=0) from Databento book updates would
        #    inject limit-order prices as trade prices, adding noise.
        #    FIX: volume=None ticks still pass through for warmup.
        if tick.volume is None or tick.volume > 0:
            self.strategy.on_tick(tick)

    def _feed_l2_book(self, tick: Tick) -> None:
        """Directly update L2OrderBook state from tick's book snapshot.

        Bypasses the event bus for performance — Databento MBP-10 snapshots
        provide the full book at once, so per-level event publishing is wasteful.
        Direct dict assignment is ~100x faster than 20 event bus calls per tick.
        """
        symbol = tick.symbol

        # Get or create L2OrderBook for this symbol
        if not hasattr(self, '_l2_books'):
            self._l2_books = {}
        if symbol not in self._l2_books:
            from data.l2_orderbook import L2OrderBook
            book = L2OrderBook(instrument_id=symbol)
            self._l2_books[symbol] = book
            # Also register it in the Strategy's data_processor
            if hasattr(self.strategy, 'data_processor'):
                dp = self.strategy.data_processor
                if hasattr(dp, '_order_books'):
                    dp._order_books[symbol] = book

        book = self._l2_books[symbol]
        book.state.timestamp = tick.timestamp

        # Bulk update: overwrite all levels at once (snapshot semantics)
        if tick.bids:
            book.state.bids = {i: (p, s) for i, (p, s) in enumerate(tick.bids) if p > 0 and s > 0}
        if tick.asks:
            book.state.asks = {i: (p, s) for i, (p, s) in enumerate(tick.asks) if p > 0 and s > 0}

    def _handle_l2(self, event: Event):
        # Si données L2 dispo
        data = event.payload
        self.microstructure.on_l2_update(data['bids'], data['asks'], event.timestamp)

    def _handle_order_request(self, event: Event):
        payload = event.payload
        proposal = payload['proposal']
        order_id = payload['order_id']
        
        # Simu Latence Réseau + Moteur Matching
        # Planif exéc future
        execution_time = event.timestamp + timedelta(seconds=self.latency_model["order_entry"])
        
        # Modèle simplifié: exéc immédiate à temps futur
        # Mais besoin état marché À CE MOMENT.
        # Microstructure MAJ par ticks.
        
        # Compromis: Exéc vs état ACTUEL, mais modèle "slippage latence".
        # i.e. estim mvmt prix en 5ms selon vol actuelle.
        # Géré dans `ExecutionSimulator`.
        
        # Donc exéc immédiate dans boucle (temps logique), 
        # mais `ExecutionSimulator` ajoute bruit/slippage pour délai.
        
        result = self.execution.execute(proposal)
        
        # Override order_id pour match req
        result = OrderResult(
            order_id=order_id,
            status=result.status,
            filled_price=result.filled_price,
            filled_quantity=result.filled_quantity,
            timestamp=result.timestamp,
            fees=result.fees
        )
        
        if result.status == "FILLED":
            # Planif Event Fill
            # Retour vers strat
            self.event_engine.put(Event(
                timestamp=result.timestamp,
                priority=5,  # Fills after market data (0) but before new orders (10)
                type=EventType.ORDER_FILL,
                payload=result,
            ))

    def _handle_order_fill(self, event: Event):
        result = event.payload
        _strategy_notified = False

        # Get proposal info
        proposal = self.strategy.open_orders.get(result.order_id)
        if proposal:
            side = proposal.action.value
            qty = result.filled_quantity
            price = result.filled_price
            fees = result.fees
            regime_id = str(proposal.regime_state.metadata.get("id", "UNDEFINED"))
            
            # Track position for PnL computation
            if not hasattr(self, '_position_tracker'):
                self._position_tracker = {
                    'entry_price': 0.0,
                    'entry_side': None,
                    'entry_qty': 0.0,
                    'net_position': 0.0,
                    'total_fees': 0.0,
                    'entry_timestamp': 0.0,
                    'entry_regime': 'UNDEFINED',
                }
            
            tracker = self._position_tracker
            prev_net = tracker['net_position']
            
            if side == 'BUY':
                tracker['net_position'] += qty
            else:
                tracker['net_position'] -= qty
            
            new_net = tracker['net_position']
            tracker['total_fees'] += fees
            
            # Opening a new position from flat
            if abs(prev_net) < 1e-10 and abs(new_net) > 1e-10:
                tracker['entry_price'] = price
                tracker['entry_side'] = side
                tracker['entry_qty'] = qty
                tracker['entry_timestamp'] = result.timestamp.timestamp()
                tracker['entry_regime'] = regime_id
                tracker['total_fees'] = fees
            
            # Position closed (or flipped) — record actual PnL
            if abs(new_net) < 1e-10 or (prev_net * new_net < 0):
                entry_p = tracker['entry_price']
                close_qty = min(abs(prev_net), qty) if abs(prev_net) > 1e-10 else qty

                if tracker['entry_side'] == 'BUY':
                    trade_pnl = (price - entry_p) * close_qty - tracker['total_fees']
                elif tracker['entry_side'] == 'SELL':
                    trade_pnl = (entry_p - price) * close_qty - tracker['total_fees']
                else:
                    trade_pnl = -tracker['total_fees']

                # Notify strategy FIRST so _last_closed_worst/best_unrealized
                # are populated before we read them for TradeRecord
                self.strategy.on_fill(result)
                _strategy_notified = True

                # Read MAE/MFE from strategy's post-close snapshot
                _mae_pct = 0.0
                _mfe_pct = 0.0
                _worst = getattr(self.strategy, '_last_closed_worst_unrealized', 0.0)
                _best = getattr(self.strategy, '_last_closed_best_unrealized', 0.0)
                if _worst < 0:
                    _mae_pct = abs(_worst)
                if _best > 0:
                    _mfe_pct = _best

                self.metrics.record_trade(TradeRecord(
                    timestamp=tracker['entry_timestamp'],
                    symbol=proposal.symbol,
                    side=tracker['entry_side'] or side,
                    qty=close_qty,
                    entry_price=entry_p,
                    exit_price=price,
                    pnl=trade_pnl,
                    regime_id=tracker['entry_regime'],
                    commission=tracker['total_fees'],
                    mae_pct=_mae_pct,
                    mfe_pct=_mfe_pct,
                ))

                # Reset tracker for flipped position
                if abs(new_net) > 1e-10:
                    tracker['entry_price'] = price
                    tracker['entry_side'] = side
                    tracker['entry_qty'] = abs(new_net)
                    tracker['entry_timestamp'] = result.timestamp.timestamp()
                    tracker['entry_regime'] = regime_id
                    tracker['total_fees'] = 0.0
                else:
                    tracker['entry_price'] = 0.0
                    tracker['entry_side'] = None
                    tracker['entry_qty'] = 0.0
                    tracker['total_fees'] = 0.0
            else:
                _strategy_notified = False

        # Notif Strat (only if not already notified above for position close)
        if not _strategy_notified:
            self.strategy.on_fill(result)

    def _on_regime_update(self, data: Dict):
        regime_id = data.get("regime_id", -1)
        timestamp = data.get("timestamp", 0.0)
        self.metrics.update_regime(str(regime_id), timestamp)

