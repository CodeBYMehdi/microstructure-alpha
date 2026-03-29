# attention aux degats
# le cerveau de l'operation

import numpy as np
import time
import csv
import threading
import concurrent.futures
from collections import deque
from datetime import datetime
from typing import Optional, Dict
import logging
import logging.handlers
import sys
import asyncio

from config.loader import get_config
from core.types import (
    Tick, TradeAction, TradeProposal, RegimeState, RegimeType,
    OrderResult, DecisionLog, LiquidityState,
)
from data.tick_stream import SyntheticTickStream, RealTickStream, CsvTickStream
from data.event_bus import event_bus, EventType, MarketEvent
from data.l2_orderbook import L2OrderBook
from data.realtime_stream import RealTimeStream
from data.ib_client import IBClient
from microstructure.returns import ReturnCalculator
from microstructure.pdf.normalizing_flow import GMMDensityEstimator
from microstructure.moments import MomentsCalculator
from microstructure.entropy import EntropyCalculator
from regime.state_vector import StateVector
from regime.hmm_adapter import HMMRegimeAdapter
from regime.transition import TransitionDetector, TransitionEvent
from microstructure.surface_analytics import SurfaceAnalytics
from regime.labels import RegimeLabelManager
from risk.calibration import RiskCalibrationModel

from decision.eligibility import TradeEligibility
from decision.sizing import PositionSizer
from decision.entry_conditions import EntryConditions
from decision.exits import ExitConditions
from decision.confidence import ConfidenceScorer
from decision.adaptive_exits import AdaptiveExitEngine, ExitParameters
from decision.trade_journal import PerformanceTracker, TradeJournalEntry
from risk.kill_switch import RiskManager
from execution.order_router import OrderRouter, OrderStateTracker
from execution.ibkr_router import IBKROrderRouter
from execution.trade_ledger import TradeLedger
from config.credentials import load_credentials, vault_exists
from monitoring.alerts import AlertManager, AlertLevel
from monitoring.watchdog import Watchdog
from risk.compliance import ComplianceGuard
from alpha.signal_quality import SignalQualityTracker

from monitoring.regime_drift import RegimeDriftMonitor
from monitoring.model_health import ModelHealthMonitor
from monitoring.event_bus import bus

# Alpha layer
from alpha.feature_engine import FeatureEngine
from alpha.return_predictor import ReturnPredictor
from alpha.alpha_decay import AlphaDecayModel
from alpha.signal_combiner import SignalCombiner

# ── Institutional-grade additions ──
from alpha.ensemble import AlphaEnsemble
from alpha.attribution import AlphaAttribution
from risk.portfolio import PortfolioRiskManager
from data.quality import DataQualitySentinel
from data.tick_db import TradeDatabase
from execution.analytics import ExecutionAnalytics
from execution.twap import AdaptiveExecutor
from monitoring.dashboard import MonitoringDashboard
from monitoring.webhook_alerts import WebhookAlerter
from storage.sqlite_store import SQLiteStore
from collections import Counter
import os

# console = warnings+, fichier = tout (10MB x5 rotating)
_log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_formatter)
_console_handler.setLevel(logging.WARNING)
_file_handler = logging.handlers.RotatingFileHandler(
    "system.log", maxBytes=10 * 1024 * 1024, backupCount=5
)
_file_handler.setFormatter(_log_formatter)
_file_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
)
logger = logging.getLogger(__name__)


class RegimeValidator:
    """t-test sur rendements futurs par régime — bannit les non-prédictifs"""

    def __init__(self, p_threshold: float = 0.05, min_samples: int = 30):
        self.p_threshold = p_threshold
        self.min_samples = min_samples
        self.banned_regimes: set = set()
        self._regime_returns: Dict[int, list] = {}

    def record_return(self, regime_id: int, forward_return: float) -> None:
        if regime_id not in self._regime_returns:
            self._regime_returns[regime_id] = []
        buf = self._regime_returns[regime_id]
        buf.append(forward_return)
        if len(buf) > 2000:  # cap mémoire
            self._regime_returns[regime_id] = buf[-1000:]

    def validate_regimes(self) -> Dict[int, dict]:
        from scipy.stats import ttest_1samp
        results = {}
        for regime_id, returns in self._regime_returns.items():
            if len(returns) < self.min_samples:
                results[regime_id] = {"p_value": 1.0, "n": len(returns), "banned": regime_id in self.banned_regimes, "reason": "insufficient_data"}
                continue
            arr = np.array(returns)
            t_stat, p_value = ttest_1samp(arr, popmean=0)
            p_value = float(p_value) if np.isfinite(p_value) else 1.0
            if p_value >= self.p_threshold:
                self.banned_regimes.add(regime_id)
                reason = "not_predictive"
            else:
                self.banned_regimes.discard(regime_id)
                reason = "predictive"
            results[regime_id] = {"p_value": p_value, "t_stat": float(t_stat), "n": len(returns), "banned": regime_id in self.banned_regimes, "reason": reason}
        return results

    def is_banned(self, regime_id: int) -> bool:
        return regime_id in self.banned_regimes


class RejectionRateMonitor:
    """taux de rejet glissant — alerte si le modèle se dégrade"""

    def __init__(self, window: int = 100, critical_rate: float = 0.80):
        self._window = window
        self._critical_rate = critical_rate
        self._outcomes: deque = deque(maxlen=window)
        self._rejection_reasons: Counter = Counter()

    def record(self, rejected: bool, reason: str = "") -> None:
        self._outcomes.append(1 if rejected else 0)
        if rejected and reason:
            self._rejection_reasons[reason] += 1

    @property
    def rejection_rate(self) -> float:
        if len(self._outcomes) == 0:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def is_degraded(self) -> bool:
        # min 20 obs avant d'alarmer
        return len(self._outcomes) >= 20 and self.rejection_rate > self._critical_rate

    def get_top_reasons(self, n: int = 5) -> list:
        return self._rejection_reasons.most_common(n)


class DataProcessor:

    def __init__(self, config):
        self.config = config
        self._quote_state: Dict[str, Dict] = {}
        self._order_books: Dict[str, L2OrderBook] = {}
        self._event_subscribed = False
        self._instrument_symbols = [
            instr.symbol for instr in config.instruments.instruments
        ]

    def ensure_subscriptions(self, callback) -> None:
        if self._event_subscribed:
            return
        event_bus.subscribe(EventType.TRADE, callback)
        event_bus.subscribe(EventType.QUOTE, callback)
        self._event_subscribed = True
        for symbol in self._instrument_symbols:
            if symbol not in self._order_books:
                # L2OrderBook s'auto-subscribe à L2_UPDATE sur l'event_bus global
                self._order_books[symbol] = L2OrderBook(instrument_id=symbol)

    def create_stream(self, duration_seconds: int):
        if self.config.execution.data_source == "hdf5":
            return RealTickStream(
                storage_path=self.config.execution.data_path,
                symbol=self.config.instruments.instruments[0].symbol,
            )
        elif self.config.execution.data_source == "csv":
            return CsvTickStream(
                filepath=self.config.execution.data_path,
                symbol=self.config.instruments.instruments[0].symbol,
            )
        else:
            return SyntheticTickStream(
                symbol=self.config.instruments.instruments[0].symbol,
                duration_seconds=duration_seconds,
                seed=42,
            )

    def publish_tick(self, tick: Tick) -> None:
        event = MarketEvent(
            timestamp_exchange=tick.timestamp,
            timestamp_received=tick.timestamp,
            instrument_id=tick.symbol,
            event_type=EventType.TRADE,
            price=tick.price,
            size=tick.volume,
            bid_price=tick.bid,
            ask_price=tick.ask,
            bid_size=tick.bid_size,
            ask_size=tick.ask_size,
            raw_source="STREAM",
        )
        event_bus.publish_sync(event)

    def process_quote(self, event: MarketEvent) -> None:
        symbol = event.instrument_id
        if not symbol:
            return
        state = self._quote_state.setdefault(symbol, {
            "bid": None, "ask": None, "bid_size": None, "ask_size": None,
        })

        if event.event_type == EventType.QUOTE:
            if event.side == "BID":
                if event.price is not None:
                    state["bid"] = event.price
                if event.size is not None:
                    state["bid_size"] = event.size
            elif event.side == "ASK":
                if event.price is not None:
                    state["ask"] = event.price
                if event.size is not None:
                    state["ask_size"] = event.size
            if event.bid_price is not None:
                state["bid"] = event.bid_price
            if event.ask_price is not None:
                state["ask"] = event.ask_price
            if event.bid_size is not None:
                state["bid_size"] = event.bid_size
            if event.ask_size is not None:
                state["ask_size"] = event.ask_size

        # trade events contiennent parfois le BBO — on sync
        for field, key in [("bid_price", "bid"), ("ask_price", "ask"),
                           ("bid_size", "bid_size"), ("ask_size", "ask_size")]:
            val = getattr(event, field, None)
            if val is not None:
                state[key] = val

    def make_tick(self, event: MarketEvent) -> Optional[Tick]:
        symbol = event.instrument_id
        if event.price is None:
            return None
        state = self._quote_state.get(symbol, {})
        return Tick(
            timestamp=event.timestamp_exchange,
            symbol=symbol,
            price=event.price,
            volume=event.size or 0.0,
            bid=state.get("bid"),
            ask=state.get("ask"),
            bid_size=state.get("bid_size"),
            ask_size=state.get("ask_size"),
            exchange=event.raw_source,
        )


class DecisionEngine:

    def __init__(self, config):
        self.eligibility = TradeEligibility(config)
        self.entry_logic = EntryConditions(config)
        self.exits = ExitConditions(config)
        self.sizer = PositionSizer(config=config)
        self.confidence_scorer = ConfidenceScorer()
        self.adaptive_exits = AdaptiveExitEngine(config)
        self.journal = PerformanceTracker(lookback=50)

        # gate coût/edge — priors conservateurs si non calibré
        _sim_cfg = config.thresholds.execution_sim
        _regime_slip = getattr(_sim_cfg, 'regime_slippage', None)
        _impact_cal = getattr(_sim_cfg, 'impact_calibration', None)
        from execution.slippage import SlippageModel
        self.slippage_model = SlippageModel(
            base_spread_bps=_sim_cfg.base_fee_bps,
            impact_coeff=_impact_cal.temporary_coeff if _impact_cal else 0.5,
            noise_std_bps=_sim_cfg.slippage_std_bps,
            regime_slippage_config=_regime_slip,
        )
        if _impact_cal and not _impact_cal.is_calibrated:
            logger.warning("Impact model UNCALIBRATED — using conservative priors. Do not trade real capital.")

        # 30 features: 23 base + 5 L2 + 2 surface 3D
        self.feature_engine = FeatureEngine(
            window=config.thresholds.regime.window_size,
        )
        vol_cfg = config.thresholds.volatility
        try:
            self.moments_calculator = MomentsCalculator(
                garch_refit_interval=vol_cfg.garch_refit_interval,
                garch_min_obs=vol_cfg.garch_min_obs,
            )
        except TypeError:
            # Rust backend doesn't accept GARCH kwargs
            self.moments_calculator = MomentsCalculator()
        self.return_predictor = ReturnPredictor(n_features=38)
        self.alpha_decay = AlphaDecayModel(max_lookback=50)
        self.signal_combiner = SignalCombiner(
            signal_names=[
                "regime_transition",
                "return_prediction",
                "order_flow",
                "momentum",
                "mean_reversion",
                "orderbook",
            ],
        )

        self.alpha_ensemble = AlphaEnsemble(
            n_features=38,
            min_samples=config.thresholds.ensemble.min_samples if hasattr(config.thresholds, 'ensemble') else 50,
        )

        self.alpha_attribution = AlphaAttribution(
            signal_names=[
                "regime_transition", "return_prediction",
                "order_flow", "momentum",
                "mean_reversion", "orderbook",
            ],
        )


class Strategy:

    def __init__(self, config=None, router=None, risk_manager=None, enable_viz=True):
        self.config = config or get_config()

        self.window_size = self.config.thresholds.regime.window_size
        self.update_freq = self.config.thresholds.regime.update_frequency
        max_errors = self.config.thresholds.risk.max_consecutive_errors

        # Components
        self.ret_calc = ReturnCalculator(max_window_size=max(2000, self.window_size * 2))
        self.long_term_ret_calc = ReturnCalculator(max_window_size=max(2000, self.window_size * 5))
        self.long_term_is_bullish = True
        self.last_ofi = 0.0
        self.book_slope = 0.0
        # z-score OFI pour entrées microstructure directes
        self._ofi_history: deque = deque(maxlen=200)
        self._ofi_rolling_std: float = 1.0
        self.clustering = HMMRegimeAdapter(config=self.config)
        self.transitions = TransitionDetector(config=self.config)
        self.risk_calibrator = RiskCalibrationModel(self.config)
        self.labels = RegimeLabelManager()
        self.drift_monitor = RegimeDriftMonitor(self.labels)
        self.health_monitor = ModelHealthMonitor()
        self.alert_manager = AlertManager(
            max_history=self.config.thresholds.alerts.max_history,
            cooldown_seconds=self.config.thresholds.alerts.cooldown_seconds,
        )

        # Data processor
        self.data_processor = DataProcessor(self.config)
        self.surface_analytics = SurfaceAnalytics()

        # Moments (GARCH conditional vol when Python backend active)
        vol_cfg = self.config.thresholds.volatility
        try:
            self.moments_calculator = MomentsCalculator(
                garch_refit_interval=vol_cfg.garch_refit_interval,
                garch_min_obs=vol_cfg.garch_min_obs,
            )
        except TypeError:
            # Rust backend doesn't accept GARCH kwargs
            self.moments_calculator = MomentsCalculator()

        # viz lazy import — évite overhead matplotlib en prod
        self.enable_viz = enable_viz
        self.viz_manager = None
        if self.enable_viz:
            from monitoring.viz_manager import VisualizationManager
            self.viz_manager = VisualizationManager()
            self.viz_manager.start()

        # Decision engine
        self.decision_engine = DecisionEngine(self.config)
        self.risk_manager = risk_manager or RiskManager(self.config)
        self.router = router or OrderRouter()

        # SQLite remplace le CSV — requêtes plus rapides en live
        self.trade_db = TradeDatabase("trades.db")
        self._state_store = SQLiteStore(db_path=str(self.trade_db.filepath))
        # fichier tmp en backtest pour éviter pollution entre runs
        if self.config.execution.mode in ("simulation", "replay"):
            import tempfile
            self.trade_ledger = TradeLedger(
                filepath=os.path.join(tempfile.gettempdir(), f"trade_ledger_{id(self)}.csv")
            )
        else:
            self.trade_ledger = TradeLedger()

        self.portfolio_risk = PortfolioRiskManager(
            initial_equity=self.risk_manager.current_equity,
        )

        self.data_quality = DataQualitySentinel()

        self.execution_analytics = ExecutionAnalytics()

        # TWAP pour les ordres importants
        self.adaptive_executor = AdaptiveExecutor(
            router=self.router,
            small_order_threshold=getattr(self.config.thresholds, 'execution_analytics', None)
                and self.config.thresholds.execution_analytics.twap_threshold_qty or 100.0,
        )

        self.webhook_alerter = WebhookAlerter(
            cooldown_seconds=getattr(self.config.thresholds, 'monitoring', None)
                and self.config.thresholds.monitoring.webhook_cooldown_s or 60.0,
        )

        self.dashboard = MonitoringDashboard(
            port=getattr(self.config.thresholds, 'monitoring', None)
                and self.config.thresholds.monitoring.dashboard_port or 8080,
        )
        self.dashboard.register_components(
            risk_manager=self.portfolio_risk,
            execution_analytics=self.execution_analytics,
            alpha_attribution=self.decision_engine.alpha_attribution
                if hasattr(self.decision_engine, 'alpha_attribution') else None,
            alpha_ensemble=self.decision_engine.alpha_ensemble
                if hasattr(self.decision_engine, 'alpha_ensemble') else None,
            data_quality=self.data_quality,
            trade_db=self.trade_db,
            strategy=self,
        )

        # kill switch pipeline vers le router
        self.risk_manager.kill_switch.set_router(self.router)

        self.order_tracker = OrderStateTracker()

        # récup position au démarrage depuis la DB
        symbol = self.config.instruments.instruments[0].symbol
        ledger_pos = self.trade_db.get_net_position(symbol)
        if abs(ledger_pos) < 1e-10:
            ledger_pos = self.trade_ledger.get_net_position(symbol)  # fallback CSV
        if ledger_pos != 0:
            logger.info(f"Recovered position from DB: {symbol}={ledger_pos:.4f}")
            self.current_position = ledger_pos
            self.risk_manager.current_exposure = 0.0  # corrigé au premier tick
            self.order_tracker.confirmed_position = ledger_pos
            self.portfolio_risk.update_position(symbol, ledger_pos, 0.0)

        # ordres orphelins = SUBMITTED mais jamais résolus — vérifier manuellement
        orphaned = self.trade_ledger.get_orphaned_orders()
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned orders from previous session:")
            for o in orphaned:
                logger.warning(f"  Orphaned order {o['order_id']}: {o['side']} {o['requested_qty']} {o['symbol']} @ {o['timestamp']}")
            logger.warning("These orders may still be active on the broker. Check and cancel manually if needed.")

        self.watchdog = Watchdog(
            kill_switch=self.risk_manager.kill_switch,
            alert_callback=self._on_watchdog_alert,
        )

        self.compliance = ComplianceGuard(self.config)
        self.compliance.start_session(self.risk_manager.current_equity)

        self.signal_quality = SignalQualityTracker(lookback=200)

        _sg = self.config.thresholds.decision.signal_gate
        _eg = self.config.thresholds.decision.entry_gate
        self.regime_validator = RegimeValidator(
            p_threshold=_sg.regime_predictiveness_alpha,
            min_samples=_sg.min_regime_trades_for_ban,
        )
        self.rejection_monitor = RejectionRateMonitor(
            window=100,
            critical_rate=_sg.rejection_rate_critical,
        )

        # process_window non-bloquant en live
        self._process_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="process-window")

        self._position_lock = threading.RLock()  # protège current_position en live

        # State
        self.tick_count = 0
        self.last_price = 0.0
        self.last_tick: Optional[Tick] = None
        self.last_mtm_price = 0.0
        self.current_position = ledger_pos
        self.open_orders: Dict[str, TradeProposal] = {}
        self.current_regime_start_tick = 0
        self.last_regime_id = -1
        self.pending_trade = None
        self.min_persistence = 1
        self._max_errors = max_errors
        self._min_hold_periods = _eg.min_hold_periods
        self._hold_periods_remaining = 0
        self._entry_price = 0.0
        self._entry_tick = 0
        self._position_windows = 0
        self._exit_params: Optional[ExitParameters] = None
        self._entry_reason = ""
        self._entry_regime_id = -1
        self._entry_confidence = 0.0
        self._entry_transition_strength = 0.0
        self._entry_volatility = 0.0
        self._worst_unrealized_pct = 0.0
        self._best_unrealized_pct = 0.0
        # MAE/MFE intra-trade
        self._mae_price = 0.0
        self._mfe_price = 0.0
        # survit à la fermeture pour lecture backtest
        self._last_closed_worst_unrealized = 0.0
        self._last_closed_best_unrealized = 0.0
        self._last_closed_mae_price = 0.0
        self._last_closed_mfe_price = 0.0

        self._autocorrelation = 0.0  # lag-1 autocorr des rendements

        self._min_confidence = _eg.min_confidence_gate

        self._last_composite_signal = None

        # décision log — SQLite principal, CSV backup
        self.decision_log_file = "decision_log.csv"
        self._init_decision_log()

        # Start monitoring dashboard in background
        try:
            dashboard_enabled = getattr(self.config.thresholds, 'monitoring', None) and self.config.thresholds.monitoring.dashboard_enabled
            if dashboard_enabled is not False:
                self.dashboard.start()
        except Exception as e:
            logger.warning(f"Failed to start monitoring dashboard: {e}")

    def _on_watchdog_alert(self, kind: str, msg: str) -> None:
        if hasattr(self.alert_manager, 'fire'):
            level = AlertLevel.CRITICAL if 'kill' in kind.lower() or 'critical' in kind.lower() else AlertLevel.WARNING
            self.alert_manager.fire(level, kind, msg)
        logger.warning(f"WATCHDOG [{kind}]: {msg}")
        if 'kill' in kind.lower() or 'critical' in kind.lower():  # webhook sur CRITICAL seulement
            self.webhook_alerter.send_alert(
                title=f"🚨 {kind}",
                message=msg,
                severity="CRITICAL",
                alert_type=kind,
            )
        self.trade_db.record_alert(alert_type=kind, severity="WARNING", message=msg)

    def _init_decision_log(self) -> None:
        # CSV legacy — conservé pour compatibilité rétro
        try:
            with open(self.decision_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "regime_id", "action", "reason",
                    "delta_mu", "delta_sigma", "delta_entropy", "delta_skew",
                    "regime_age", "transition_strength", "tail_risk",
                    "l2_liquidity_slope", "result",
                ])
        except Exception:
            pass

    def _log_decision(self, log: DecisionLog) -> None:
        # SQLite en premier, CSV en fallback
        try:
            self.trade_db.record_decision({
                'timestamp': log.timestamp,
                'regime_id': log.regime_id,
                'action': log.action,
                'reason': log.reason,
                'delta_mu': log.delta_mu,
                'delta_sigma': log.delta_sigma,
                'delta_entropy': log.delta_entropy,
                'delta_skew': log.delta_skew,
                'regime_age': log.regime_age,
                'transition_strength': log.transition_strength,
                'tail_risk': log.tail_risk,
                'l2_liquidity_slope': log.l2_liquidity_slope,
                'result': log.result,
            })
        except Exception:
            pass
        try:
            with open(self.decision_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log.timestamp, log.regime_id, log.action, log.reason,
                    log.delta_mu, log.delta_sigma, log.delta_entropy, log.delta_skew,
                    log.regime_age, log.transition_strength, log.tail_risk,
                    log.l2_liquidity_slope, log.result,
                ])
        except Exception:
            pass

    def run(self, duration_seconds: int = 60, stream=None) -> None:
        _im = getattr(self.router, 'impact_model', None)
        if _im is not None:
            _tag = "CALIBRATED" if _im.is_calibrated else "UNCALIBRATED (conservative priors, 50% size cap)"
            logger.info(f"Impact model status: {_tag}")

        # Config audit: stamp git SHA + config hash before accepting signals
        import hashlib, json as _json
        _git_sha = os.environ.get("GIT_SHA", "unknown")
        if _git_sha == "unknown":
            try:
                import subprocess
                _git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except (FileNotFoundError, subprocess.CalledProcessError):
                _git_sha = "unknown"
        _config_hash = hashlib.sha256(
            _json.dumps(self.config.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()
        self._state_store.upsert_config_audit(_git_sha, _config_hash)
        logger.info(f"Config audit: sha={_git_sha[:8]}, hash={_config_hash[:8]}")

        logger.info(f"Starting strategy loop for {duration_seconds} seconds...")
        self.data_processor.ensure_subscriptions(self._on_market_event)

        if stream is None and self.config.execution.mode == "live":
            self._run_live(duration_seconds)
            return

        if stream is None:
            stream = self.data_processor.create_stream(duration_seconds)

        for tick in stream:
            self.data_processor.publish_tick(tick)

    def _run_live(self, duration_seconds: int) -> None:
        import signal
        from data.realtime_stream import patch_ib_client_resubscribe

        # vault obligatoire pour démarrage non-interactif
        if not vault_exists():
            raise RuntimeError(
                "No credentials vault found. Run:\n"
                "  python -m config.credentials --init --account YOUR_ACCOUNT --port 7497 --mode paper"
            )
        if not os.environ.get("IBKR_VAULT_PASSWORD"):
            logger.warning(
                "IBKR_VAULT_PASSWORD env var not set — will prompt for password. "
                "Set it for non-interactive (daemon/systemd) startup."
            )
        creds = load_credentials()
        logger.info(
            f"IBKR credentials loaded: mode={creds.trading_mode}, "
            f"host={creds.host}:{creds.port}, account={creds.account_id}"
        )

        # refus si config != live — erreur humaine fréquente
        if creds.trading_mode == "live" and self.config.execution.mode != "live":
            raise RuntimeError("Credentials are set to LIVE but config mode is not 'live'. Aborting.")

        if creds.trading_mode == "paper" and creds.port not in (7497, 4002):
            logger.warning(f"Paper mode but port={creds.port} (expected 7497 or 4002)")
        if creds.trading_mode == "live" and creds.port not in (7496, 4001):
            logger.warning(f"LIVE mode but port={creds.port} (expected 7496 or 4001)")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        ib = IBClient(host=creds.host, port=creds.port, client_id=creds.client_id)
        patch_ib_client_resubscribe(ib)

        realtime_stream = RealTimeStream(ib)
        loop.run_until_complete(realtime_stream.start())

        # Verify connection before proceeding
        if not ib._connected:
            raise RuntimeError(
                "Failed to connect to IBKR TWS/Gateway. Check that:\n"
                f"  1. TWS or IB Gateway is running on {creds.host}:{creds.port}\n"
                "  2. API connections are enabled (File > Global Configuration > API > Settings)\n"
                "  3. 'Enable ActiveX and Socket Clients' is checked\n"
                f"  4. Socket port is {creds.port}\n"
                "  5. Trusted IPs includes 127.0.0.1"
            )
        logger.info("IBKR connection confirmed.")

        # router sim → IBKR; re-wire kill switch + executor
        self.router = IBKROrderRouter(
            ib_client=ib,
            account_id=creds.account_id,
            mode=creds.trading_mode,
        )
        self.risk_manager.kill_switch.set_router(self.router)
        self.adaptive_executor.router = self.router

        # broker = source de vérité, une seule requête (pas par symbole)
        try:
            broker_positions = self.router.query_positions()
        except Exception as e:
            logger.warning(f"Broker position query failed, using ledger: {e}")
            broker_positions = {}

        total_exposure = 0.0
        for instrument in self.config.instruments.instruments:
            sym = instrument.symbol
            broker_pos = broker_positions.get(sym, 0.0)
            ledger_pos = self.trade_ledger.get_net_position(sym)
            if abs(broker_pos - ledger_pos) > 1e-6:
                logger.warning(
                    f"Position mismatch {sym}: broker={broker_pos:.4f} vs ledger={ledger_pos:.4f}. Using broker."
                )
            if abs(broker_pos) > 1e-10:
                logger.info(f"Recovered position from broker: {sym}={broker_pos:.4f}")
            self.portfolio_risk.update_position(sym, broker_pos, 0.0)
            total_exposure += broker_pos

        # position primaire — exposure corrigée au premier tick
        primary_sym = self.config.instruments.instruments[0].symbol
        primary_pos = broker_positions.get(primary_sym, 0.0)
        self.current_position = primary_pos
        self.risk_manager.current_exposure = 0.0
        self.order_tracker.confirmed_position = primary_pos

        self.data_processor.ensure_subscriptions(self._on_market_event)

        self.data_quality.set_alert_callback(
            lambda evt: self.alert_manager.fire(
                AlertLevel.CRITICAL if evt.severity == "CRITICAL" else AlertLevel.WARNING,
                evt.issue.value,
                evt.message,
            )
        )

        self.watchdog.start()

        # Subscribe to all configured instruments
        for instrument in self.config.instruments.instruments:
            realtime_stream.subscribe_instrument(
                symbol=instrument.symbol,
                exchange=instrument.exchange,
            )
            logger.info(f"Subscribed: {instrument.symbol} on {instrument.exchange}")

        # SIGINT/SIGTERM pour arrêt propre
        shutdown_requested = asyncio.Event()

        def _signal_handler(sig, frame):
            sig_name = signal.Signals(sig).name
            logger.info(f"Received {sig_name} — initiating graceful shutdown...")
            shutdown_requested.set()

        signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _signal_handler)

        logger.info(f"Live session started. Running for {duration_seconds}s...")
        try:
            # Run until duration expires or shutdown signal received
            async def _wait_for_shutdown():
                shutdown_task = asyncio.ensure_future(shutdown_requested.wait())
                timer_task = asyncio.ensure_future(asyncio.sleep(duration_seconds))
                done, pending = await asyncio.wait(
                    [shutdown_task, timer_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

            loop.run_until_complete(_wait_for_shutdown())
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user (Ctrl+C)")
        finally:
            logger.info("Shutting down live session...")
            self.watchdog.stop()
            if hasattr(self.router, 'cancel_all'):  # annule les ordres ouverts
                self.router.cancel_all()
            # Unsubscribe
            for instrument in self.config.instruments.instruments:
                try:
                    realtime_stream.unsubscribe_instrument(instrument.symbol)
                except Exception:
                    pass
            ib.disconnect()
            loop.close()
            asyncio.set_event_loop(None)
            logger.info("Live session ended. IB disconnected.")

    def _on_market_event(self, event: MarketEvent) -> None:
        self.data_processor.process_quote(event)
        if event.event_type == EventType.QUOTE:
            return
        if event.event_type != EventType.TRADE:
            return

        tick = self.data_processor.make_tick(event)
        if tick:
            self.on_tick(tick)

    def on_tick(self, tick: Tick) -> None:
        start_time = time.time()
        self.tick_count += 1
        self.last_price = tick.price
        self.last_tick = tick

        if self.tick_count % max(1, int(5 / 0.1)) == 0:  # ~every 5s at 10 ticks/s
            self._checkpoint_strategy_state()

        # Update RiskManager's last known price
        if hasattr(self.risk_manager, 'update_price'):
            self.risk_manager.update_price(tick.price)

        # Watchdog heartbeat
        self.watchdog.heartbeat_tick()

        symbol = self.config.instruments.instruments[0].symbol
        dq_events = self.data_quality.check_tick(
            symbol=symbol,
            price=tick.price,
            volume=tick.volume or 0.0,
            bid=tick.bid,
            ask=tick.ask,
            exchange_ts=tick.timestamp.timestamp() if hasattr(tick.timestamp, 'timestamp') else float(tick.timestamp),
            receive_ts=time.time(),
        )
        if any(e.severity == "CRITICAL" for e in dq_events):  # tick corrompu → drop
            if self.tick_count % 500 == 0:
                logger.warning(f"Tick rejected by data quality sentinel: {[e.issue.value for e in dq_events if e.severity == 'CRITICAL']}")
            return

        if self.risk_manager.kill_switch.triggered:
            # flatten d'urgence sous lock
            with self._position_lock:
                if self.current_position != 0 and self.last_price > 0:
                    symbol = self.config.instruments.instruments[0].symbol
                    close_qty = abs(self.current_position)
                    close_action = TradeAction.SELL if self.current_position > 0 else TradeAction.BUY
                    flatten_proposal = TradeProposal(
                        action=close_action, symbol=symbol, quantity=close_qty,
                        price=self.last_price, reason=f"EMERGENCY FLATTEN: {self.risk_manager.kill_switch.reason}",
                        timestamp=tick.timestamp,
                        regime_state=RegimeState(regime=RegimeType.UNDEFINED, confidence=0.0, entropy=0.0, timestamp=tick.timestamp, metadata={}),
                    )
                    logger.critical(f"EMERGENCY FLATTEN: {close_action.value} {close_qty:.4f} @ {self.last_price:.4f}")
                    result = self.router.execute(flatten_proposal, current_tick=tick)
                    if result.status == "FILLED":
                        self._handle_fill_locked(result, close_action)
            if self.tick_count % 1000 == 0:
                logger.warning(f"Kill switch active ({self.risk_manager.kill_switch.reason}), dropping ticks")
            return

        try:
            ret = self.ret_calc.update(tick.price)
            self.long_term_ret_calc.update(tick.price)

            # Feed return to tail risk analyzer for VaR/CVaR tracking
            if ret is not None:
                self.risk_manager.update_tail_risk(ret, regime_id=self.last_regime_id)

            self.decision_engine.adaptive_exits.update_price(tick.price)

            # stop tick-level: évite les pertes entre deux process_window
            if self.current_position != 0 and self._entry_price > 0 and self._exit_params:
                if self.current_position > 0:
                    _tick_unr = (tick.price - self._entry_price) / self._entry_price
                else:
                    _tick_unr = (self._entry_price - tick.price) / self._entry_price

                # suivi MAE/MFE au tick
                if _tick_unr < getattr(self, '_worst_unrealized_pct', 0.0):
                    self._worst_unrealized_pct = _tick_unr
                if _tick_unr > getattr(self, '_best_unrealized_pct', 0.0):
                    self._best_unrealized_pct = _tick_unr

                _eg = self.config.thresholds.decision.entry_gate
                _stop = self._exit_params.stop_loss_pct
                _hard_stop = _stop * _eg.hard_stop_multiplier

                # plafond dollar: not. × dist. stop × multiplicateur
                _notional = self._entry_price * abs(self.current_position)
                _dollar_loss = abs(_tick_unr) * _notional
                _max_dollar_loss = max(_eg.min_dollar_stop, _notional * _stop * _eg.hard_stop_multiplier)
                _dollar_triggered = _tick_unr < 0 and _dollar_loss >= _max_dollar_loss

                if _tick_unr <= -_stop or _dollar_triggered:
                    if _dollar_triggered and _tick_unr > -_stop:
                        _reason = f"Dollar Stop: ${_dollar_loss:.2f} >= ${_max_dollar_loss:.2f}"
                    elif _tick_unr > -_hard_stop:
                        _reason = f"Tick Stop: {_tick_unr:.4%} <= -{_stop:.4%}"
                    else:
                        _reason = f"Hard Stop: {_tick_unr:.4%} <= -{_hard_stop:.4%}"
                    logger.warning(f"TICK-LEVEL STOP: {_reason}")
                    close_qty = abs(self.current_position)
                    close_action = TradeAction.SELL if self.current_position > 0 else TradeAction.BUY
                    _symbol = self.config.instruments.instruments[0].symbol
                    emergency_proposal = TradeProposal(
                        action=close_action, symbol=_symbol, quantity=close_qty,
                        price=tick.price,
                        reason=_reason,
                        timestamp=tick.timestamp,
                        regime_state=RegimeState(regime=RegimeType.UNDEFINED, confidence=0.0, entropy=0.0, timestamp=tick.timestamp, metadata={"id": self.last_regime_id}),
                    )
                    result = self.router.execute(emergency_proposal, current_tick=tick)
                    if result.status == "FILLED":
                        self._handle_fill_locked(result, close_action)
                    return

            # OFI and Book Slope Tracking
            prev_bid = getattr(self, "_prev_bid", tick.bid or tick.price)
            prev_ask = getattr(self, "_prev_ask", tick.ask or tick.price)
            prev_bid_size = getattr(self, "_prev_bid_size", 0.0)
            prev_ask_size = getattr(self, "_prev_ask_size", 0.0)

            curr_bid = tick.bid or tick.price
            curr_ask = tick.ask or tick.price
            curr_bid_size = tick.bid_size or 0.0
            curr_ask_size = tick.ask_size or 0.0

            ofi = 0.0
            if curr_bid > prev_bid:
                ofi += curr_bid_size
            elif curr_bid == prev_bid:
                ofi += (curr_bid_size - prev_bid_size)
            else:
                ofi -= prev_bid_size

            if curr_ask < prev_ask:
                ofi -= curr_ask_size
            elif curr_ask == prev_ask:
                ofi -= (curr_ask_size - prev_ask_size)
            else:
                ofi += prev_ask_size

            self.last_ofi = ofi
            spread = max(0.01, curr_ask - curr_bid)
            self.book_slope = (curr_bid_size - curr_ask_size) / spread

            # Track OFI rolling stats for Z-score computation
            self._ofi_history.append(ofi)
            if len(self._ofi_history) >= 20:
                _ofi_arr = np.array(self._ofi_history)
                self._ofi_rolling_std = max(float(np.std(_ofi_arr)), 1e-8)

            self._prev_bid = curr_bid
            self._prev_ask = curr_ask
            self._prev_bid_size = curr_bid_size
            self._prev_ask_size = curr_ask_size

            self.decision_engine.feature_engine.update(
                price=tick.price,
                volume=tick.volume,
                bid=tick.bid,
                ask=tick.ask,
                ofi=ofi,
                timestamp=tick.timestamp.timestamp() if hasattr(tick.timestamp, 'timestamp') else float(tick.timestamp),
            )

            if ret is None:

                return

            if self.ret_calc.count >= self.window_size and self.tick_count % self.update_freq == 0:
                if self.config.execution.mode == "live" and not self.data_quality.is_data_safe(symbol):
                    if self.tick_count % 500 == 0:
                        logger.warning("process_window skipped: data quality sentinel reports unsafe")
                elif self.config.execution.mode == "live":
                    if self._process_lock.acquire(blocking=False):  # skip si précédent tourne encore
                        self._executor.submit(self._safe_process_window, tick.timestamp)
                    else:
                        logger.debug("Skipping process_window: previous still running")
                else:
                    self.process_window(tick.timestamp)

            latency = (time.time() - start_time) * 1000
            self.health_monitor.record_latency(latency)
            self.alert_manager.check_latency(latency, self.config.thresholds.alerts.latency_warning_ms)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            self.health_monitor.record_error()

            if self.health_monitor.error_count > self._max_errors:
                logger.critical("Too many errors, triggering kill switch")
                self.risk_manager.kill_switch.trigger("Too many system errors")

    def _safe_process_window(self, timestamp: datetime) -> None:
        """wrapper thread-safe pour process_window en live"""
        try:
            self.process_window(timestamp)
        except Exception as e:
            logger.error(f"process_window error (async): {e}", exc_info=True)
            self.health_monitor.record_error()
        finally:
            self._process_lock.release()

    def _compute_features(self, data_window: np.ndarray):
        """PDF + moments + entropy + surface. Retourne (state, pdf_model, pdf_values, viz_pdf_values, moments, sigma, surface_state)."""
        pdf_cfg = self.config.thresholds.pdf

        try:
            long_term_window = self.long_term_ret_calc.get_window(self.window_size * 5)
            if len(long_term_window) > 0:
                self.long_term_is_bullish = np.mean(long_term_window) > 0
        except ValueError as e:
            logger.warning("long_term_is_bullish parse failed: %s — defaulting False", e)
            self.long_term_is_bullish = False

        pdf_model = GMMDensityEstimator(self.config)
        pdf_model.fit(data_window)

        mu = np.mean(data_window)
        sigma = np.std(data_window)
        if sigma == 0:
            sigma = 1e-6
        x_grid = np.linspace(mu - pdf_cfg.sigma_range * sigma, mu + pdf_cfg.sigma_range * sigma, pdf_cfg.grid_points)
        pdf_values = pdf_model.evaluate(x_grid)
        dx = x_grid[1] - x_grid[0]

        viz_x_grid = np.linspace(-0.01, 0.01, 100)
        viz_pdf_values = pdf_model.evaluate(viz_x_grid)

        moments = self.moments_calculator.compute(data_window)
        entropy = EntropyCalculator.compute_from_pdf(pdf_values, dx)

        state = StateVector(
            mu=moments.mu, sigma=moments.sigma, skew=moments.skew,
            kurtosis=moments.kurtosis, tail_slope=moments.tail_slope, entropy=entropy,
        )

        if len(data_window) > 2:
            _dw = data_window - np.mean(data_window)
            _var = np.dot(_dw, _dw)
            self._autocorrelation = float(np.dot(_dw[:-1], _dw[1:]) / _var) if _var > 1e-20 else 0.0
        else:
            self._autocorrelation = 0.0

        surface_state = self.surface_analytics.update(pdf_values=pdf_values, mu=moments.mu, sigma=moments.sigma)
        return state, pdf_model, pdf_values, viz_pdf_values, moments, sigma, surface_state

    def _compute_alpha_signals(self, state, surface_state, mu: float):
        """Alpha layer: features → prédiction → combinaison. Retourne (prediction, composite_signal, l2_features, feature_vector)."""
        depth_imbalance = self.book_slope / max(abs(self.book_slope), 1.0) if self.book_slope != 0 else 0.0

        l2_features = None
        for sym, book in self.data_processor._order_books.items():
            l2_features = book.get_features()
            break
        if l2_features is None:
            l2_features = {
                'depth_imbalance': depth_imbalance, 'liquidity_pull_score': 0.0,
                'book_pressure': depth_imbalance * 0.5, 'spoofing_score': 0.0, 'spread_bps': 0.0,
            }

        feature_vector = self.decision_engine.feature_engine.compute(
            state=state, regime_age=self.tick_count - self.current_regime_start_tick,
            regime_confidence=0.0, ticks_since_transition=0, cluster_distance=0.0,
            depth_imbalance=depth_imbalance, l2_features=l2_features, surface_state=surface_state,
        )

        self.decision_engine.return_predictor.update(mu)
        prediction = self.decision_engine.return_predictor.predict(feature_vector.to_array())
        self.decision_engine.alpha_decay.on_window(self.last_price)
        self.decision_engine.signal_combiner.update_accuracy(mu)

        _ofi_norm = feature_vector.ofi / self._ofi_rolling_std if self._ofi_rolling_std > 1e-8 else 0.0
        raw_signals = {
            "regime_transition": 0.0,
            "return_prediction": prediction.expected_return if prediction.is_valid else 0.0,
            "order_flow": np.clip(_ofi_norm * 0.3, -1.0, 1.0),
            "momentum": feature_vector.momentum_short,
            "mean_reversion": -feature_vector.mean_reversion * 0.1,
            "orderbook": np.clip(
                l2_features.get('book_pressure', 0.0) * 0.7 + l2_features.get('depth_imbalance', 0.0) * 0.3,
                -1.0, 1.0),
        }
        composite_signal = self.decision_engine.signal_combiner.combine(raw_signals)
        self._last_composite_signal = composite_signal

        if composite_signal.is_actionable:
            logger.debug(f"ALPHA SIGNAL: dir={composite_signal.direction:+.4f} str={composite_signal.strength:.4f}")

        return prediction, composite_signal, l2_features, feature_vector

    def _assess_regime(self, state, pdf_model, confidence_prev: float, timestamp):
        """Clustering + confiance + transitions. Retourne (regime_id, confidence, regime_stats, event, regime_age)."""
        current_regime_id = -1
        confidence = 0.0
        regime_stats = None

        self.clustering.update(state)
        if self.tick_count > self.window_size + 100:
            try:
                self.clustering.fit()
                current_regime_id = self.clustering.predict_latest()
                confidence = self.clustering.calculate_confidence(state, current_regime_id)
                regime_stats = self.clustering.get_regime_stats(current_regime_id)
            except Exception as e:
                logger.warning(f"Regime clustering failed (degraded mode): {e}")
                current_regime_id = -1
                confidence = 0.0

            with self._position_lock:
                self.risk_manager.current_exposure = self.current_position * self.last_price
                if self.last_mtm_price > 0:
                    price_change = self.last_price - self.last_mtm_price
                    mtm_pnl = self.current_position * price_change
                    self.risk_manager.update_metrics(pnl=mtm_pnl, slippage=0.0, confidence=confidence)
                self.last_mtm_price = self.last_price

            self.alert_manager.check_drawdown(
                self.risk_manager.current_drawdown_pct,
                self.config.thresholds.alerts.drawdown_warning,
                self.config.thresholds.alerts.drawdown_critical,
            )

            if current_regime_id != -1:
                self.labels.update_profile(current_regime_id, np.array(state.to_array()), 1)
                self.drift_monitor.record_observation(current_regime_id, state)
                if self.drift_monitor.detect_structural_break(current_regime_id):
                    logger.warning(f"[!] STRUCTURAL BREAK DETECTED in Regime {current_regime_id}")
                self.regime_validator.record_return(current_regime_id, state.mu)
                if self.tick_count % 200 == 0:
                    rv_results = self.regime_validator.validate_regimes()
                    for rid, rv in rv_results.items():
                        if rv["banned"] and rv["reason"] == "not_predictive":
                            logger.warning(f"REGIME {rid} BANNED: p={rv['p_value']:.4f} (n={rv['n']})")
                    if hasattr(self.clustering, 'get_transition_info'):
                        _hmm_info = self.clustering.get_transition_info()
                        if _hmm_info.get('is_initialized'):
                            _tm = _hmm_info.get('transition_matrix')
                            if _tm is not None:
                                _row_spread = _tm.max(axis=1) - _tm.min(axis=1)
                                if _row_spread.min() < 0.2:  # mémoire de régime perdue
                                    logger.warning("HMM near-uniform rows — regime memory lost, consider recalibration")

        if current_regime_id != self.last_regime_id:
            self.current_regime_start_tick = self.tick_count
            self.last_regime_id = current_regime_id
        regime_age = self.tick_count - self.current_regime_start_tick

        hmm_trans_prob = None
        hmm_weight = 0.0
        if hasattr(self.clustering, 'get_transition_info'):
            _hmm_info = self.clustering.get_transition_info()
            if (_hmm_info['is_initialized']
                    and self.last_regime_id >= 0
                    and current_regime_id >= 0
                    and self.last_regime_id != current_regime_id):
                hmm_trans_prob = _hmm_info['transition_matrix'][self.last_regime_id, current_regime_id]
                hmm_weight = min(1.0, _hmm_info['n_observations'] / 500.0)

        event = self.transitions.update(
            current_regime_id, state, curr_kde=pdf_model,
            hmm_transition_prob=hmm_trans_prob, hmm_weight=hmm_weight,
        )

        _eg = self.config.thresholds.decision.entry_gate
        sigma = state.sigma
        _hmm_ready = (
            hasattr(self.clustering, '_hmm')
            and self.clustering._hmm.is_initialized
            and self.clustering._hmm.n_observations > 100
        )
        if event is None and self.current_position == 0 and sigma > 0 and current_regime_id != -1 and _hmm_ready:
            mu_vel = self.transitions._kf_mu.velocity if self.transitions._kf_mu.is_initialized else 0.0
            if abs(mu_vel) > _eg.bootstrap_mu_vel_threshold * sigma:
                delta = np.zeros(6)
                delta[0] = mu_vel
                event = TransitionEvent(
                    from_regime=current_regime_id, to_regime=current_regime_id,
                    strength=min(1.0, abs(mu_vel) / sigma * _eg.bootstrap_strength_scale),
                    delta_vector=delta, is_significant=True,
                    reason="bootstrap_signal", mu_velocity=mu_vel,
                )
                logger.info(f"Bootstrap signal: mu_vel={mu_vel:.6f}, sigma={sigma:.6f}, strength={event.strength:.4f}")

        return current_regime_id, confidence, regime_stats, event, regime_age

    def process_window(self, timestamp: datetime) -> None:
        """features → alpha → régime → décisions"""
        start_time_pw = time.time()
        data_window = self.ret_calc.get_window(self.window_size)

        state, pdf_model, pdf_values, viz_pdf_values, moments, sigma, surface_state = self._compute_features(data_window)

        prediction, composite_signal, l2_features, feature_vector = self._compute_alpha_signals(state, surface_state, moments.mu)

        mu = moments.mu
        current_regime_id, confidence, regime_stats, event, regime_age = self._assess_regime(state, pdf_model, 0.0, timestamp)

        regime_desc = self.labels.describe(current_regime_id)
        self._publish_regime_viz(moments, pdf_values, viz_pdf_values, state, current_regime_id, timestamp)

        current_regime_state = RegimeState(
            regime=RegimeType.UNDEFINED,
            confidence=confidence,
            entropy=state.entropy,
            timestamp=timestamp,
            metadata={"id": current_regime_id, "description": regime_desc, "vector": state.to_dict()},
        )

        # Trading logic
        delta_vector = event.delta_vector if event else np.zeros(6)
        symbol = self.config.instruments.instruments[0].symbol

        log_entry = DecisionLog(
            timestamp=timestamp.timestamp(),
            regime_id=str(current_regime_id),
            action="FLAT", reason="No Transition",
            delta_mu=delta_vector[0], delta_sigma=delta_vector[1],
            delta_entropy=delta_vector[5], delta_skew=delta_vector[2],
            regime_age=regime_age,
            transition_strength=event.strength if event else 0.0,
            tail_risk=state.tail_slope, l2_liquidity_slope=0.0,
            result="FLAT",
        )

        executed_this_tick = self._evaluate_exits(
            event, state, surface_state, symbol, timestamp, current_regime_state, log_entry,
        )

        if not executed_this_tick:
            executed_this_tick = self._confirm_pending(current_regime_id, log_entry)

        if not executed_this_tick and event and event.is_significant:
            executed_this_tick = self._validate_and_enter_transition(
                event, state, prediction, sigma, current_regime_id, confidence,
                regime_stats, symbol, timestamp, current_regime_state, log_entry,
            )

        if not executed_this_tick:
            self._evaluate_ofi_entry(
                state, prediction, sigma, current_regime_id, regime_stats,
                symbol, timestamp, current_regime_state, log_entry,
            )

        self._track_signals(mu, timestamp, log_entry, start_time_pw)

    def _evaluate_exits(self, event, state, surface_state, symbol, timestamp, regime_state, log_entry) -> bool:
        if self.current_position == 0:
            return False

        self._position_windows += 1
        self._update_unrealized_tracking()

        if self._hold_periods_remaining > 0:
            self._hold_periods_remaining -= 1
            logger.debug(f"Hold period active: {self._hold_periods_remaining} cycles remaining")
            return False

        if not (self._entry_price > 0 and self.last_price > 0 and self._exit_params):
            return False

        side = TradeAction.BUY if self.current_position > 0 else TradeAction.SELL
        should_exit, exit_reason = self.decision_engine.adaptive_exits.check_exit(
            entry_price=self._entry_price, current_price=self.last_price,
            position_side=side, position_windows=self._position_windows,
            exit_params=self._exit_params, transition=event, state=state,
            surface_state=surface_state,
        )
        if not should_exit:
            return False

        logger.info(f"EXIT TRIGGERED: {exit_reason}")
        self._record_exit_journal(symbol, exit_reason)

        close_action = TradeAction.SELL if side == TradeAction.BUY else TradeAction.BUY
        proposal = TradeProposal(
            action=close_action, symbol=symbol, quantity=abs(self.current_position),
            price=self.last_price, reason=exit_reason,
            timestamp=timestamp, regime_state=regime_state,
        )
        self._execute_trade(proposal, log_entry)
        return True

    def _update_unrealized_tracking(self):
        if not (self._entry_price > 0 and self.last_price > 0):
            return
        if self.current_position > 0:
            unr = (self.last_price - self._entry_price) / self._entry_price
        else:
            unr = (self._entry_price - self.last_price) / self._entry_price
        self._worst_unrealized_pct = min(self._worst_unrealized_pct, unr)
        self._best_unrealized_pct = max(self._best_unrealized_pct, unr)

        if self.current_position > 0:
            self._mae_price = min(self._mae_price, self.last_price) if self._mae_price > 0 else self.last_price
            self._mfe_price = max(self._mfe_price, self.last_price)
        else:
            self._mae_price = max(self._mae_price, self.last_price)
            self._mfe_price = min(self._mfe_price, self.last_price) if self._mfe_price > 0 else self.last_price

    def _record_exit_journal(self, symbol, exit_reason):
        if self.current_position > 0:
            gross_pnl = (self.last_price - self._entry_price) * abs(self.current_position)
        else:
            gross_pnl = (self._entry_price - self.last_price) * abs(self.current_position)

        journal_entry = TradeJournalEntry(
            timestamp_entry=self._entry_tick, timestamp_exit=self.tick_count,
            symbol=symbol,
            side="BUY" if self.current_position > 0 else "SELL",
            entry_price=self._entry_price, exit_price=self.last_price,
            quantity=abs(self.current_position),
            gross_pnl=gross_pnl, net_pnl=gross_pnl,
            regime_id=self._entry_regime_id,
            regime_confidence=self._entry_confidence,
            entry_reason=self._entry_reason,
            transition_strength=self._entry_transition_strength,
            volatility_at_entry=self._entry_volatility,
            exit_reason=exit_reason,
            hold_duration_windows=self._position_windows,
            best_unrealized_pct=self._best_unrealized_pct,
            worst_unrealized_pct=self._worst_unrealized_pct,
            mae_pct=abs(self._worst_unrealized_pct) if self._worst_unrealized_pct < 0 else 0.0,
            mfe_pct=self._best_unrealized_pct if self._best_unrealized_pct > 0 else 0.0,
            mae_price=self._mae_price, mfe_price=self._mfe_price,
            stop_loss_pct=self._exit_params.stop_loss_pct,
            take_profit_pct=self._exit_params.take_profit_pct,
            trailing_stop_pct=self._exit_params.trailing_stop_pct,
        )
        self.decision_engine.journal.record_trade(journal_entry)

    def _confirm_pending(self, current_regime_id, log_entry) -> bool:
        """confirmation par persistance du régime"""
        if not self.pending_trade:
            return False
        proposal, count = self.pending_trade
        if current_regime_id == proposal.regime_state.metadata["id"]:
            count += 1
            self.pending_trade = (proposal, count)
            if count >= self.min_persistence:
                self._execute_trade(proposal, log_entry)
                self.pending_trade = None
                return True
        else:
            logger.info("Pending trade ABORTED: Regime changed before confirmation")
            self.pending_trade = None
            log_entry.reason = "Aborted: Regime Unstable"
        return False

    def _check_pre_entry_gates(self, event, current_regime_id, log_entry) -> bool:
        """gates 1-5: bruit, banni, santé, toxique, IC"""
        _sg_cfg = self.config.thresholds.decision.signal_gate

        # GATE 1: régime bruit → skip
        if event.to_regime == -1 or current_regime_id == -1:
            log_entry.reason = f"Noise Regime Entry (current={current_regime_id}, to={event.to_regime})"
            log_entry.result = "REJECTED"
            return False

        # GATE 2: predictivité statistique (t-test)
        if self.regime_validator.is_banned(current_regime_id):
            log_entry.reason = f"Regime {current_regime_id} BANNED (not statistically predictive)"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "REGIME_BANNED")
            return False

        # GATE 3
        health = self.health_monitor.check_health()
        if not health.is_healthy:
            logger.warning("SYSTEM UNHEALTHY - SKIPPING TRADE")
            log_entry.reason = "Unhealthy System"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "UNHEALTHY")
            return False

        # GATE 4: toxicité empirique
        if self._is_regime_toxic(event.to_regime):
            _wr = self.decision_engine.journal._regime_win_rates.get(event.to_regime, 0.5)
            _trades = self.decision_engine.journal._regime_trade_counts.get(event.to_regime, 0)
            _pnl = sum(self.decision_engine.journal._regime_performance.get(event.to_regime, []))
            log_entry.reason = f"Toxic Regime (WR={_wr:.1%}, PnL=${_pnl:.2f} on {_trades} trades)"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "TOXIC_REGIME")
            return False

        # GATE 5: IC glissant
        _rolling_ic = self.signal_quality.get_rolling_ic()
        if _rolling_ic < _sg_cfg.min_rolling_ic and self.signal_quality._n_realized > _sg_cfg.ic_lookback:
            log_entry.reason = f"IC_TOO_LOW: rolling_ic={_rolling_ic:.4f} < {_sg_cfg.min_rolling_ic}"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "IC_TOO_LOW")
            return False

        return True

    def _check_post_entry_gates(self, prediction, sigma, exit_params, event, state, regime_stats, log_entry) -> bool:
        """gates 6-8: R:R, coût/edge, SNR"""
        _sg_cfg = self.config.thresholds.decision.signal_gate

        # GATE 7: coût vs edge
        _predicted_move = abs(prediction.expected_return) if prediction.is_valid else 0.0
        _stop_pct_est = exit_params.stop_loss_pct if exit_params and exit_params.stop_loss_pct > 1e-6 else 0.01
        _est_qty = self.decision_engine.sizer.calculate(
            event, state.tail_slope, regime_stats=regime_stats,
            win_rate=self.decision_engine.journal._rolling_win_rate,
            profit_factor=self.decision_engine.journal._rolling_profit_factor,
            equity=self.risk_manager.current_equity,
            price=self.last_price, stop_pct=_stop_pct_est,
            drawdown_pct=self.risk_manager.current_drawdown_pct,
        )
        _rt_cost = self.decision_engine.slippage_model.estimate_round_trip(
            price=self.last_price, quantity=max(1.0, _est_qty), volatility=sigma,
            bid=self.last_tick.bid if self.last_tick else None,
            ask=self.last_tick.ask if self.last_tick else None,
            regime_vol_mult=self.decision_engine.slippage_model.get_regime_multiplier(sigma),
        )
        if _predicted_move < _sg_cfg.cost_edge_multiplier * _rt_cost and _rt_cost > 0:
            log_entry.reason = f"COST_EXCEEDS_EDGE: predicted={_predicted_move:.6f} < {_sg_cfg.cost_edge_multiplier}x cost={_rt_cost:.6f}"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "COST_EXCEEDS_EDGE")
            return False

        # GATE 8
        _snr = abs(_predicted_move) / max(sigma, 1e-10)
        if _snr < _sg_cfg.min_snr:
            log_entry.reason = f"SNR_TOO_LOW: snr={_snr:.4f} < {_sg_cfg.min_snr}"
            log_entry.result = "REJECTED"
            self.rejection_monitor.record(True, "SNR_TOO_LOW")
            return False

        self.rejection_monitor.record(False)
        if self.rejection_monitor.is_degraded:
            _top = self.rejection_monitor.get_top_reasons(3)
            logger.critical(
                f"SIGNAL_DEGRADED: {self.rejection_monitor.rejection_rate:.0%} of signals rejected "
                f"(top reasons: {_top}) — model review required"
            )
        return True

    def _size_and_enter(self, action, event, state, exit_params, confidence, regime_stats,
                        current_regime_id, symbol, timestamp, regime_state, log_entry, liquidity=None,
                        entry_reason=None, sig_strength=None) -> bool:
        """sizing + mise en attente de confirmation"""
        # pas d'accumulation dans la même direction
        with self._position_lock:
            already_long = self.current_position > 0 and action == TradeAction.BUY
            already_short = self.current_position < 0 and action == TradeAction.SELL
        if already_long or already_short:
            log_entry.result = "FLAT"
            log_entry.reason = "Already positioned same direction"
            return False

        # fermeture position inverse d'abord
        if self.current_position != 0:
            close_action = TradeAction.SELL if self.current_position > 0 else TradeAction.BUY
            close_proposal = TradeProposal(
                action=close_action, symbol=symbol, quantity=abs(self.current_position),
                price=self.last_price, reason="Position reversal: closing existing",
                timestamp=timestamp, regime_state=regime_state,
            )
            self._execute_trade(close_proposal)
            logger.info("Closed existing position before reversal")

        # niveaux L2 pour sizing liquidé-ajusté
        _l2_levels = self._get_l2_levels(action)

        _stop_pct = exit_params.stop_loss_pct if exit_params and exit_params.stop_loss_pct > 1e-6 else 0.01
        size = self.decision_engine.sizer.calculate(
            event, state.tail_slope, regime_stats=regime_stats,
            win_rate=self.decision_engine.journal._rolling_win_rate,
            profit_factor=self.decision_engine.journal._rolling_profit_factor,
            l2_levels=_l2_levels, equity=self.risk_manager.current_equity,
            price=self.last_price, stop_pct=_stop_pct,
            drawdown_pct=self.risk_manager.current_drawdown_pct,
            var_95=self.risk_manager.last_var,
        )
        # cap 50% si impact non calibré — conservateur par défaut
        _im = getattr(self.router, 'impact_model', None)
        if _im is not None and not _im.is_calibrated:
            logger.warning("UNCALIBRATED_IMPACT: position size capped at 50%%")
            size = size * 0.5

        if self.config.execution.mode in ("live", "paper") or isinstance(self.router, IBKROrderRouter):
            size = max(1, int(round(size)))
        else:
            size = max(0.1, size)

        reason = entry_reason or log_entry.reason
        proposal = TradeProposal(
            action=action, symbol=symbol, quantity=size,
            price=self.last_price, reason=reason,
            timestamp=timestamp, regime_state=regime_state,
        )

        self._store_entry_context(
            reason, current_regime_id, confidence, event, state, exit_params,
            liquidity=liquidity, sig_strength=sig_strength,
        )
        self.pending_trade = (proposal, 0)
        self._hold_periods_remaining = self._min_hold_periods
        log_entry.action = action.value
        log_entry.reason = reason
        log_entry.result = "PENDING"
        logger.info(f"Signal generated. Size={size:.2f}. Waiting for confirmation...")
        return True

    def _store_entry_context(self, reason, regime_id, confidence, event, state, exit_params,
                              liquidity=None, sig_strength=None):
        self._entry_reason = reason
        self._entry_regime_id = regime_id
        if liquidity and event:
            _conf = self.decision_engine.confidence_scorer.score(
                regime_confidence=confidence, transition=event, state=state, liquidity=liquidity,
            )
            self._entry_confidence = _conf.composite
        else:
            self._entry_confidence = sig_strength if sig_strength is not None else 0.5
        self._entry_transition_strength = event.strength if event else (sig_strength or 0.0)
        self._entry_volatility = state.sigma
        self._exit_params = exit_params
        self._worst_unrealized_pct = 0.0
        self._best_unrealized_pct = 0.0
        self._mae_price = self.last_price
        self._mfe_price = self.last_price

    def _get_l2_levels(self, action):
        _symbol = self.config.instruments.instruments[0].symbol
        _book = self.data_processor._order_books.get(_symbol)
        if not (_book and _book.state):
            return None
        if action == TradeAction.BUY:
            return sorted(
                [(p, s) for p, s in _book.state.asks.values() if p < float('inf')],
                key=lambda x: x[0],
            )
        return sorted(
            [(p, s) for p, s in _book.state.bids.values() if p > 0],
            key=lambda x: -x[0],
        )

    def _get_transition_volatility(self):
        if hasattr(self.clustering, 'get_transition_info'):
            return self.clustering.get_transition_info().get('transition_volatility', 0.0)
        return 0.0

    def _validate_and_enter_transition(self, event, state, prediction, sigma, current_regime_id,
                                       confidence, regime_stats, symbol, timestamp,
                                       regime_state, log_entry) -> bool:
        """chemin d'entrée transition: gates → éligibilité → sizing"""
        logger.info(f"TRANSITION: {event.from_regime} -> {event.to_regime} (Strength: {event.strength:.4f})")
        self.risk_manager.notify_regime_transition()

        if not self._check_pre_entry_gates(event, current_regime_id, log_entry):
            logger.info(f"--- SKIPPING: {log_entry.reason}")
            self._log_decision(log_entry)
            return False

        risk_health_ok = not self.risk_manager.check_kill_switch()
        el_result = self.decision_engine.eligibility.check(event, state, risk_health_ok)
        if not el_result.is_eligible:
            logger.info(f"--- INELIGIBLE: {el_result.reason}")
            log_entry.reason = el_result.reason
            log_entry.result = "REJECTED"
            self._log_decision(log_entry)
            return False

        liquidity = self._build_liquidity_state()
        action, reason = self.decision_engine.entry_logic.evaluate(
            event, state, liquidity, regime_stats,
            autocorrelation=getattr(self, '_autocorrelation', 0.0),
        )
        log_entry.action = action.value if action else "FLAT"
        log_entry.reason = reason

        if not action:
            log_entry.result = "FLAT"
            self._log_decision(log_entry)
            return False

        # GATE 6: R:R
        _regime_wr = self.decision_engine.journal._regime_win_rates.get(event.to_regime, 0.5)
        exit_params = self.decision_engine.adaptive_exits.compute_exit_params(
            state=state, regime_stats=regime_stats,
            current_drawdown_pct=self.risk_manager.current_drawdown_pct,
            transition_strength=event.strength, regime_win_rate=_regime_wr,
            reference_price=self.last_price,
            autocorrelation=getattr(self, '_autocorrelation', 0.0),
            transition_volatility=self._get_transition_volatility(),
        )
        rr_ok, rr_reason = self.decision_engine.adaptive_exits.check_entry_rr(
            entry_price=self.last_price, exit_params=exit_params, side=action,
        )
        if not rr_ok:
            logger.info(f"--- SKIPPING: R:R check failed — {rr_reason}")
            log_entry.result = "REJECTED"
            log_entry.reason = f"R:R Failed: {rr_reason}"
            self.rejection_monitor.record(True, "RR_FAILED")
            self._log_decision(log_entry)
            return False

        # GATES 7-8
        if not self._check_post_entry_gates(prediction, sigma, exit_params, event, state, regime_stats, log_entry):
            logger.info(f"--- SKIPPING: {log_entry.reason}")
            self._log_decision(log_entry)
            return False

        self._size_and_enter(
            action, event, state, exit_params, confidence, regime_stats,
            current_regime_id, symbol, timestamp, regime_state, log_entry,
            liquidity=liquidity, entry_reason=reason,
        )
        self._log_decision(log_entry)
        return True

    def _evaluate_ofi_entry(self, state, prediction, sigma, current_regime_id, regime_stats,
                            symbol, timestamp, regime_state, log_entry):
        """entrée OFI/alpha continue avec gates complets"""
        _sg_cfg = self.config.thresholds.decision.signal_gate
        if not (self.current_position == 0 and not self.pending_trade):
            return
        if current_regime_id == -1 or self._is_regime_toxic(current_regime_id):
            return
        if self.regime_validator.is_banned(current_regime_id):
            return
        _rolling_ic = self.signal_quality.get_rolling_ic()
        if _rolling_ic < _sg_cfg.min_rolling_ic and self.signal_quality._n_realized > _sg_cfg.ic_lookback:
            return

        alpha_action, _entry_reason, _sig_strength = self._determine_ofi_direction()
        if alpha_action is None:
            return

        alpha_exit_params = self.decision_engine.adaptive_exits.compute_exit_params(
            state=state, regime_stats=regime_stats,
            current_drawdown_pct=self.risk_manager.current_drawdown_pct,
            transition_strength=_sig_strength,
            regime_win_rate=self.decision_engine.journal._regime_win_rates.get(current_regime_id, 0.5),
            reference_price=self.last_price,
            autocorrelation=getattr(self, '_autocorrelation', 0.0),
            transition_volatility=self._get_transition_volatility(),
        )
        rr_ok, rr_reason = self.decision_engine.adaptive_exits.check_entry_rr(
            entry_price=self.last_price, exit_params=alpha_exit_params, side=alpha_action,
        )
        if not rr_ok:
            return

        # mêmes seuils que le chemin transition
        _predicted_move = abs(prediction.expected_return) if prediction.is_valid else 0.0
        _alpha_stop = alpha_exit_params.stop_loss_pct if alpha_exit_params and alpha_exit_params.stop_loss_pct > 1e-6 else 0.01
        _synth_event = TransitionEvent(
            from_regime=-1, to_regime=-1, strength=_sig_strength,
            delta_vector=np.zeros(6), is_significant=True,
            kl_divergence=0.0, projection_magnitude=0.0,
        )
        if not self._check_post_entry_gates(prediction, sigma, alpha_exit_params, _synth_event, state, regime_stats, log_entry):
            return

        self._size_and_enter(
            alpha_action, _synth_event, state, alpha_exit_params, 0.0, regime_stats,
            current_regime_id, symbol, timestamp, regime_state, log_entry,
            entry_reason=_entry_reason, sig_strength=_sig_strength,
        )
        logger.info(f"ALPHA/OFI ENTRY: {alpha_action.value} | {_entry_reason}")

    def _determine_ofi_direction(self):
        """direction OFI/alpha. Retourne (action, reason, strength) ou (None, None, None)"""
        _ofi_z = self.last_ofi / self._ofi_rolling_std if self._ofi_rolling_std > 1e-8 else 0.0
        _ofi_entry = abs(_ofi_z) > 2.0 and len(self._ofi_history) >= 50  # z>2 et historique suffisant

        sig = self._last_composite_signal
        _composite_entry = (
            sig is not None and sig.is_actionable and sig.strength > 0.20
            and sum(1 for c in sig.components
                    if (c.value > 0) == (sig.direction > 0) and abs(c.value) > 0.001) >= 2
        )

        if not (_ofi_entry or _composite_entry):
            return None, None, None

        if _ofi_entry:
            action = TradeAction.SELL if _ofi_z > 0 else TradeAction.BUY
            reason = f"OFI entry: z={_ofi_z:+.2f} (fade extreme imbalance)"
            strength = min(1.0, abs(_ofi_z) / 3.0)  # normalisation empirique
        else:
            action = TradeAction.BUY if sig.direction > 0 else TradeAction.SELL
            _agreeing = sum(1 for c in sig.components
                           if (c.value > 0) == (sig.direction > 0) and abs(c.value) > 0.001)
            reason = f"Alpha signal: dir={sig.direction:+.3f} str={sig.strength:.3f} ({_agreeing} channels)"
            strength = sig.strength

        return action, reason, strength

    def _is_regime_toxic(self, regime_id) -> bool:
        _sg_cfg = self.config.thresholds.decision.signal_gate
        _wr = self.decision_engine.journal._regime_win_rates.get(regime_id, 0.5)
        _trades = self.decision_engine.journal._regime_trade_counts.get(regime_id, 0)
        _pnl = sum(self.decision_engine.journal._regime_performance.get(regime_id, []))
        if _trades < _sg_cfg.min_regime_trades_for_ban:  # pas assez de données
            return False
        return _wr < _sg_cfg.toxic_regime_wr or _pnl < _sg_cfg.toxic_regime_pnl

    def _publish_regime_viz(self, moments, pdf_values, viz_pdf_values, state, regime_id, timestamp):
        bus.publish("regime_update", {
            "drift": moments.mu, "vol": moments.sigma,
            "density": float(np.max(pdf_values)),
            "regime_id": regime_id,
            "timestamp": timestamp.timestamp(),
            "pdf_values": viz_pdf_values,
        })
        if self.enable_viz and self.viz_manager:
            self.viz_manager.q_surface.put({
                'type': 'UPDATE',
                'data': {
                    'drift': moments.mu, 'vol': moments.sigma,
                    'density': float(np.max(pdf_values)),
                    'regime_id': regime_id,
                    'timestamp': timestamp.timestamp(),
                }
            })
            self.viz_manager.q_density.put({
                'type': 'UPDATE',
                'data': {
                    'pdf_values': viz_pdf_values,
                    'timestamp': timestamp.timestamp(),
                    'kurtosis': state.kurtosis,
                    'mu': state.mu, 'sigma': state.sigma,
                }
            })

    def _track_signals(self, mu, timestamp, log_entry, start_time_pw):
        if self._last_composite_signal is not None:
            self.signal_quality.record_prediction(
                direction=self._last_composite_signal.direction,
                strength=self._last_composite_signal.strength,
                components={c.name: c.value for c in self._last_composite_signal.components},
                timestamp=timestamp.timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp),
            )
        self.signal_quality.record_realized(mu)

        if self._last_composite_signal and self.tick_count > self.window_size + 100:
            self.risk_manager.update_metrics(
                pnl=0.0, slippage=0.0,
                confidence=max(self._last_composite_signal.confidence, self.signal_quality.get_hit_rate()),
            )

        self._log_decision(log_entry)
        pw_latency = (time.time() - start_time_pw) * 1000
        self.watchdog.heartbeat_process_window(pw_latency)

    def _build_liquidity_state(self) -> Optional[LiquidityState]:
        if not self.last_tick:
            return None

        symbol = self.last_tick.symbol
        l2_book = self.data_processor._order_books.get(symbol) if hasattr(self.data_processor, '_order_books') else None

        if l2_book and l2_book.state.bids and l2_book.state.asks:
            features = l2_book.get_features()
            bb_price = l2_book.state.get_best_bid()[0]
            ba_price = l2_book.state.get_best_ask()[0]
            mid = l2_book.state.get_mid_price()
            spread = max(0.0, ba_price - bb_price) if bb_price > 0 and ba_price < float('inf') else 0.0

            return LiquidityState(
                spread=spread,
                depth_imbalance=features['depth_imbalance'],
                depth_slope=features['book_pressure'],
                trade_intensity=0.0,
                is_liquid=features['spread_bps'] < 50.0,
            )

        # fallback BBO depuis le tick
        bid = self.last_tick.bid if self.last_tick.bid else self.last_tick.price - 0.01
        ask = self.last_tick.ask if self.last_tick.ask else self.last_tick.price + 0.01
        spread = max(0.0, ask - bid)
        bs = self.last_tick.bid_size or 0
        as_ = self.last_tick.ask_size or 0
        total_depth = bs + as_
        imbalance = (bs - as_) / total_depth if total_depth > 0 else 0
        return LiquidityState(
            spread=spread, depth_imbalance=imbalance,
            depth_slope=0.0, trade_intensity=0.0,
        )

    def _execute_trade(self, proposal: TradeProposal, log_entry: Optional[DecisionLog] = None) -> None:
        with self._position_lock:
            return self._execute_trade_locked(proposal, log_entry)

    def _execute_trade_locked(self, proposal: TradeProposal, log_entry: Optional[DecisionLog] = None) -> None:
        if self.risk_manager.kill_switch.triggered:
            logger.warning(f"Trade blocked: kill switch active ({self.risk_manager.kill_switch.reason})")
            if log_entry:
                log_entry.result = "REJECTED"
                log_entry.reason = f"Kill switch: {self.risk_manager.kill_switch.reason}"
            return

        # Compliance pre-trade check (quote staleness only in live mode)
        is_live = self.config.execution.mode == "live"
        if is_live and self.last_tick and hasattr(self.last_tick.timestamp, 'timestamp'):
            last_tick_ts = self.last_tick.timestamp.timestamp()
        else:
            last_tick_ts = None  # staleness check inutile en sim
        # en backtest, heure de marche = tick (wall-clock sans sens)
        if not is_live and self.last_tick and hasattr(self.last_tick.timestamp, 'timestamp'):
            current_time = self.last_tick.timestamp.timestamp()
        else:
            current_time = None
        pos_notional = abs(self.current_position * self.last_price) if self.last_price > 0 else 0.0
        comp_ok, comp_reason = self.compliance.check_order(
            proposal, self.last_price, last_tick_ts, pos_notional, current_time=current_time,
        )
        if not comp_ok:
            logger.warning(f"COMPLIANCE BLOCKED: {comp_reason}")
            if log_entry:
                log_entry.result = "REJECTED"
                log_entry.reason = f"Compliance: {comp_reason}"
            return

        # Daily loss limit check
        dl_ok, dl_reason = self.compliance.check_daily_loss(self.risk_manager.current_equity)
        if not dl_ok:
            logger.critical(f"DAILY LOSS LIMIT: {dl_reason}")
            self.risk_manager.kill_switch.trigger(dl_reason)
            if log_entry:
                log_entry.result = "REJECTED"
                log_entry.reason = dl_reason
            return

        if proposal.quantity > 0 and self.risk_manager.validate(proposal):
            logger.info(
                f">>> EXECUTE: {proposal.action.value} {proposal.quantity:.4f} "
                f"({proposal.reason})"
            )

            # petits ordres directs, gros ordres via TWAP pour limiter l'impact
            exec_result = self.adaptive_executor.execute(
                proposal,
                current_price=self.last_price,
                urgency="high" if self.config.execution.mode != "live" else "medium",
                get_price_fn=lambda: self.last_price,
            )

            # AdaptiveExecutor retourne OrderResult ou AlgoExecutionResult — normaliser
            from execution.twap import AlgoExecutionResult
            if isinstance(exec_result, AlgoExecutionResult):
                # Synthesize an OrderResult from TWAP aggregate
                result = OrderResult(
                    order_id=exec_result.order_id,
                    status="FILLED" if exec_result.total_filled_qty > 0 else "REJECTED",
                    filled_price=exec_result.vwap_price,
                    filled_quantity=exec_result.total_filled_qty,
                    timestamp=proposal.timestamp,
                    fees=exec_result.total_fees,
                )
                if exec_result.implementation_shortfall > 0:
                    logger.info(
                        f"TWAP completed: {exec_result.n_filled}/{exec_result.n_slices} slices, "
                        f"impl shortfall={exec_result.implementation_shortfall:.4%}"
                    )
            else:
                result = exec_result

            signed_qty = proposal.quantity if proposal.action == TradeAction.BUY else -proposal.quantity
            self.order_tracker.on_submit(result.order_id, signed_qty)
            self.trade_ledger.record_order(
                proposal, result.order_id,
                equity=self.risk_manager.current_equity,
                kill_switch=self.risk_manager.kill_switch.triggered,
            )
            if log_entry:
                log_entry.result = "TRADE"
            if result.status == "FILLED":
                self.trade_ledger.record_fill(
                    result, proposal,
                    equity=self.risk_manager.current_equity,
                    kill_switch=self.risk_manager.kill_switch.triggered,
                )
                self._handle_fill_locked(result, proposal.action)
            elif result.status == "SUBMITTED":
                logger.info(f"Order SUBMITTED: {result.order_id}")
                self.open_orders[result.order_id] = proposal
        else:
            reason = f"Kill switch: {self.risk_manager.kill_switch.reason}" if self.risk_manager.kill_switch.triggered else "Exposure/Size limit"
            logger.info(f"--- REJECTED BY RISK: {reason}")
            self.trade_ledger.record_rejection(
                proposal, reason,
                equity=self.risk_manager.current_equity,
                kill_switch=self.risk_manager.kill_switch.triggered,
            )
            if log_entry:
                log_entry.result = "REJECTED"
                log_entry.reason = reason

    def on_fill(self, result: OrderResult) -> None:
        if result.order_id in self.open_orders:
            proposal = self.open_orders.pop(result.order_id)
            self._handle_fill(result, proposal.action)
        else:
            logger.warning(f"Received fill for unknown order: {result.order_id}")

    def _handle_fill(self, result: OrderResult, side: TradeAction) -> None:
        with self._position_lock:
            self._handle_fill_locked(result, side)

    def _handle_fill_locked(self, result: OrderResult, side: TradeAction) -> None:
        # doit être appelé sous _position_lock
        prev_position = self.current_position
        signed_qty = result.filled_quantity if side == TradeAction.BUY else -result.filled_quantity
        self.current_position += signed_qty

        fill_price = result.filled_price

        self.risk_manager.current_exposure = self.current_position * fill_price  # en notionnel

        self.order_tracker.on_fill(result.order_id, signed_qty)
        confirmed = self.order_tracker.get_confirmed_position()
        if abs(self.current_position - confirmed) > 1e-6:
            logger.warning(
                f"Position divergence: strategy={self.current_position:.4f} vs tracker={confirmed:.4f}"
            )

        expected_price = self.last_price
        slippage = abs(fill_price - expected_price) / expected_price if expected_price > 0 else 0.0

        fee_pnl = -result.fees if result.fees > 0 else 0.0
        signal_conf = self._last_composite_signal.confidence if self._last_composite_signal else self.risk_manager.last_confidence
        self.risk_manager.update_metrics(pnl=fee_pnl, slippage=slippage, confidence=signal_conf)

        symbol = self.config.instruments.instruments[0].symbol
        trade_ts = self.last_tick.timestamp.timestamp() if self.last_tick and hasattr(self.last_tick.timestamp, 'timestamp') else None
        self.compliance.record_trade(symbol, side, pnl=fee_pnl, timestamp=trade_ts)

        logger.info(
            f"FILL CONFIRMED: {side.value} {result.filled_quantity:.4f} @ {fill_price:.4f} "
            f"(slip={slippage:.4%}, fees=${result.fees:.4f})"
        )

        if self.enable_viz and self.viz_manager:
            trade_data = {
                'price': fill_price,
                'size': result.filled_quantity,
                'side': side.value,
                'timestamp': time.time(),
                'drift': getattr(self.ret_calc, 'mu', 0.0),
                'vol': getattr(self.ret_calc, 'sigma', 0.0),
                'density': 0.0,
                'regime_id': self.last_regime_id
            }
            self.viz_manager.q_surface.put({'type': 'TRADE', 'data': trade_data})
            self.viz_manager.q_density.put({'type': 'TRADE', 'data': trade_data})

        # Track entry price and initialize adaptive exits
        if prev_position == 0 and self.current_position != 0:
            self._entry_price = fill_price
            self._entry_tick = self.tick_count
            self._position_windows = 0
            self._worst_unrealized_pct = 0.0
            self._best_unrealized_pct = 0.0
            self._mae_price = fill_price
            self._mfe_price = fill_price
            entry_side = TradeAction.BUY if self.current_position > 0 else TradeAction.SELL
            self.decision_engine.adaptive_exits.reset_trailing(fill_price, entry_side)
        elif self.current_position == 0:
            # sauvegarde MAE/MFE finaux avant reset (lus par backtest agent)
            self._last_closed_worst_unrealized = self._worst_unrealized_pct
            self._last_closed_best_unrealized = self._best_unrealized_pct
            self._last_closed_mae_price = self._mae_price
            self._last_closed_mfe_price = self._mfe_price
            # Reset for next trade
            self._entry_price = 0.0
            self._position_windows = 0
            self._exit_params = None
            self._mae_price = 0.0
            self._mfe_price = 0.0
            self._worst_unrealized_pct = 0.0
            self._best_unrealized_pct = 0.0


    def _compute_pnl_1h(self) -> float:
        """Compute PnL over the last hour from equity snapshots."""
        try:
            rows = self.trade_db._conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 2"
            ).fetchall()
            if len(rows) >= 2:
                return rows[0][0] - rows[-1][0]
        except Exception:
            pass
        return 0.0

    def _checkpoint_strategy_state(self) -> None:
        """Write position + risk metrics to store for independent risk process."""
        if not hasattr(self, '_state_store'):
            return
        try:
            _unrealized = (
                (self.last_price - self._entry_price) * self.current_position
                if self.current_position != 0 else 0.0
            )
            self._state_store.checkpoint("position", {
                "symbol": self.config.instruments.instruments[0].symbol if self.config.instruments.instruments else "UNKNOWN",
                "qty": self.current_position,
                "entry_price": self._entry_price,
                "unrealized_pnl": _unrealized,
            })
            self._state_store.checkpoint("risk_metrics", {
                "drawdown": self.risk_manager.current_drawdown_pct,
                "equity": self.risk_manager.current_equity,
                "pnl_1h": self._compute_pnl_1h() if hasattr(self, '_compute_pnl_1h') else 0.0,
                "kill_switch": self.risk_manager.kill_switch.triggered,
            })
        except Exception as e:
            logger.error(f"Strategy state checkpoint failed: {e}")

    def get_system_status(self) -> Dict:
        health = self.health_monitor.check_health()
        return {
            "tick_count": self.tick_count,
            "kill_switch_active": self.risk_manager.kill_switch.triggered,
            "kill_switch_reason": self.risk_manager.kill_switch.reason,
            "current_equity": self.risk_manager.current_equity,
            "peak_equity": self.risk_manager.peak_equity,
            "current_drawdown_pct": self.risk_manager.current_drawdown_pct,
            "current_position": self.current_position,
            "current_exposure": self.risk_manager.current_exposure,
            "last_regime_id": self.last_regime_id,
            "system_healthy": health.is_healthy,
            "error_rate": health.error_rate,
            "trades": self.trade_ledger.session_fill_count,
            "total_trades": self.trade_ledger.fill_count,
            "ledger_entries": self.trade_ledger.entry_count,
            "total_fees": self.trade_ledger.session_fees,
            "positions": self.trade_ledger.get_all_positions(),
            "watchdog": self.watchdog.get_status(),
            "signal_quality": self.signal_quality.get_metrics(),
            "compliance": self.compliance.get_status(),
        }


def main():
    try:
        strategy = Strategy()
        strategy.run(duration_seconds=30)

        status = strategy.get_system_status()
        logger.info(f"Final status: equity=${status['current_equity']:.2f}, "
                     f"drawdown={status['current_drawdown_pct']:.2%}, "
                     f"trades={status['trades']}, "
                     f"fees=${status['total_fees']:.2f}")
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        if 'strategy' in locals() and strategy.viz_manager:
            strategy.viz_manager.stop()


if __name__ == "__main__":
    main()
