# coupe-circuit et gest. risque pdfff
# stop tout si ça part en vrille (KillSwitch)
# RiskManager gère les positions & potefeuille

import logging
import threading
from core.interfaces import IRiskManager
from core.types import TradeProposal, TradeAction
from config.loader import get_config
from risk.tail_risk import TailRiskAnalyzer, TailRiskMetrics

logger = logging.getLogger(__name__)


class KillSwitch:
    # frein d'urgence global

    def __init__(self, config=None):
        self.config = config or get_config()
        self._triggered_event = threading.Event()
        self.reason: str = ""
        self.confidence_floor: float = self.config.thresholds.risk.confidence_floor
        # Use tighter live tolerance when running in live mode
        if self.config.execution.mode == "live":
            self.slippage_tolerance: float = self.config.thresholds.risk.live_slippage_tolerance
        else:
            self.slippage_tolerance: float = self.config.thresholds.risk.slippage_tolerance
        self.max_drawdown: float = self.config.thresholds.risk.max_drawdown
        self._router = None

    @property
    def triggered(self) -> bool:
        return self._triggered_event.is_set()

    @triggered.setter
    def triggered(self, value: bool) -> None:
        if value:
            self._triggered_event.set()
        else:
            self._triggered_event.clear()

    def set_router(self, router) -> None:
        """Wire the order router so kill switch can cancel all orders on trigger."""
        self._router = router

    def check(self, confidence: float, slippage: float, drawdown: float,
              cvar: float = 0.0) -> bool:
        # verif si on doit tout stoper
        if self.triggered:
            return True

        # 1. Model confidence
        if confidence < self.confidence_floor:
            self.trigger(f"Low model confidence: {confidence:.2f} < {self.confidence_floor}")
            return True

        # 2. Slippage
        if slippage > self.slippage_tolerance:
            self.trigger(f"Excessive slippage: {slippage:.4f} > {self.slippage_tolerance}")
            return True

        # 3. Drawdown
        if drawdown > self.max_drawdown:
            self.trigger(f"Max drawdown breached: {drawdown:.4f} > {self.max_drawdown}")
            return True

        # 4. CVaR breach — severe tail risk (CVaR is negative, more negative = worse)
        severe_cvar = getattr(self.config.thresholds.tail_risk, 'severe_cvar', -0.05)
        if cvar < severe_cvar and cvar < 0:
            self.trigger(f"Severe CVaR breach: {cvar:.4f} < {severe_cvar}")
            return True

        return False

    def trigger(self, reason: str) -> None:
        # boom on declenche le coupe-circuit
        self.triggered = True
        self.reason = reason
        logger.critical(f"!!! KILL SWITCH ACTIVATED: {reason} !!!")
        # Cancel all open orders when kill switch fires
        if self._router is not None and hasattr(self._router, 'cancel_all'):
            try:
                self._router.cancel_all()
                logger.critical("Kill switch: cancelled all open orders")
            except Exception as e:
                logger.error(f"Kill switch: failed to cancel orders: {e}")

    def reset(self, admin_override: bool = False) -> bool:
        # reset avec mdp admin obligatoire
        if not admin_override:
            logger.warning("Kill switch reset rejected: admin_override required")
            return False
        self.triggered = False
        self.reason = ""
        logger.info("Kill switch reset by admin override")
        return True


class RiskManager(IRiskManager):
    # boss final des risques

    def __init__(self, config=None, initial_capital: float = 100000.0):
        self.config = config or get_config()
        self.kill_switch = KillSwitch(self.config)
        self.max_exposure: float = self.config.thresholds.risk.max_position_size
        self.initial_capital: float = initial_capital

        # Position tracking
        self.current_exposure: float = 0.0
        self.current_equity: float = initial_capital
        self.peak_equity: float = initial_capital

        # Tail risk — VaR/CVaR tracking
        self.tail_risk_analyzer = TailRiskAnalyzer(
            window=200, confidence=0.95, config=self.config,
        )
        self.last_var: float = 0.0
        self.last_cvar: float = 0.0

        # Metrics
        self.current_drawdown_pct: float = 0.0
        self.last_slippage: float = 0.0
        self.last_confidence: float = 1.0
        self.last_known_price: float = 0.0
        self.equity_history: list = [initial_capital]

        # Confidence EMA — use faster alpha (0.3) for quicker recovery after dips
        ema_alpha = getattr(self.config.thresholds.risk, 'confidence_ema_alpha', 0.3)
        warmup = getattr(self.config.thresholds.risk, 'confidence_warmup', 20)
        self._confidence_ema: float = 1.0
        self._confidence_samples: int = 0
        self._confidence_alpha: float = ema_alpha
        # Warmup must cover enough process_window calls for HDBSCAN to produce valid clusters.
        min_cluster = self.config.thresholds.regime.min_cluster_size
        cluster_warmup = min_cluster * 3
        self._confidence_warmup: int = max(warmup, cluster_warmup)
        # Transition exemption: suppress kill switch for N ticks after regime transition
        self._transition_exemption_ticks: int = 10
        self._ticks_since_transition: int = 999  # start high = no exemption

    def validate(self, proposal: TradeProposal) -> bool:
        # verif si le trade passe ou casse
        if self.kill_switch.triggered:
            logger.warning(f"Trade rejected: Kill switch active ({self.kill_switch.reason})")
            return False

        # FIX: Use last known price if proposal.price is None, not 0.0
        price = proposal.price if proposal.price else self.last_price_fallback()
        if price <= 0:
            logger.warning("Trade rejected: No valid price available")
            return False

        # FIX: Account for position direction in exposure calculation
        # For BUY, exposure increases; for SELL, it may decrease if reducing position
        notional = proposal.quantity * price
        if proposal.action == TradeAction.BUY:
            new_exposure = self.current_exposure + notional
        elif proposal.action == TradeAction.SELL:
            new_exposure = self.current_exposure - notional
        else:
            return True  # HOLD/FLAT don't change exposure

        if abs(new_exposure) > self.max_exposure:
            logger.warning(f"Risk rejection: Exposure |{new_exposure:.2f}| > Max {self.max_exposure}")
            return False

        return True

    def last_price_fallback(self) -> float:
        # prix de secours si None
        return self.last_known_price

    def update_price(self, price: float) -> None:
        self.last_known_price = price

    def notify_regime_transition(self) -> None:
        """Call when a regime transition is detected to suppress false kill switch triggers."""
        self._ticks_since_transition = 0

    def check_kill_switch(self) -> bool:
        # on est dead ou pas ?
        self._ticks_since_transition += 1
        return self.kill_switch.check(
            confidence=self._effective_confidence(),
            slippage=self.last_slippage,
            drawdown=self.current_drawdown_pct,
            cvar=self.last_cvar,
        )

    def update_tail_risk(self, ret: float, regime_id: int = -1) -> None:
        """Feed return to tail risk analyzer, update VaR/CVaR."""
        metrics = self.tail_risk_analyzer.update(ret)
        if metrics is not None:
            self.last_var = metrics.var
            self.last_cvar = metrics.cvar
        if regime_id >= 0:
            self.tail_risk_analyzer.update_regime(ret, regime_id)

    def update_metrics(self, pnl: float, slippage: float, confidence: float) -> None:
        # maj des stats en direct
        self.current_equity += pnl
        self.equity_history.append(self.current_equity)
        self.last_slippage = slippage
        self.last_confidence = confidence
        self._update_confidence(confidence)

        # Update HWM & drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        drawdown_amount = self.peak_equity - self.current_equity
        if self.peak_equity > 0:
            self.current_drawdown_pct = drawdown_amount / self.peak_equity
        else:
            self.current_drawdown_pct = 0.0

        if self.kill_switch.check(self._effective_confidence(), slippage,
                                  self.current_drawdown_pct, cvar=self.last_cvar):
            logger.critical(f"Kill switch activated. DD: {self.current_drawdown_pct:.2%}")

    def update_position(self, filled_qty: float, side: TradeAction, price: float) -> None:
        # suivi de l'expo au marché
        notional = filled_qty * price
        if side == TradeAction.BUY:
            self.current_exposure += notional
        elif side == TradeAction.SELL:
            self.current_exposure -= notional

    def _update_confidence(self, confidence: float) -> None:
        # recalcul EMA pr confience
        self._confidence_samples += 1
        if self._confidence_samples == 1:
            self._confidence_ema = confidence
            return
        self._confidence_ema = (
            self._confidence_alpha * confidence
            + (1.0 - self._confidence_alpha) * self._confidence_ema
        )

    def _effective_confidence(self) -> float:
        # return 1.0 during warmup or regime transition exemption window
        if self._confidence_samples < self._confidence_warmup:
            return self.kill_switch.confidence_floor
        # Suppress confidence-based kill during regime transitions (legitimate dips)
        if self._ticks_since_transition < self._transition_exemption_ticks:
            return max(self._confidence_ema, self.kill_switch.confidence_floor + 0.05)
        return self._confidence_ema
