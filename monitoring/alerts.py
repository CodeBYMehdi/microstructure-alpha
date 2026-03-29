# l'usine a gaz

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    # l'usine a gaz
    timestamp: float
    level: AlertLevel
    category: str
    message: str
    data: Optional[Dict[str, Any]] = None


class AlertManager:
    # le big boss du truc

    def __init__(
        self,
        max_history: int = 1000,
        cooldown_seconds: float = 60.0,
    ):
        self.max_history = max_history
        self.cooldown_seconds = cooldown_seconds
        self._history: deque[Alert] = deque(maxlen=max_history)
        self._last_fired: Dict[str, float] = {}
        self._handlers: List[Callable[[Alert], None]] = []
        # Register default log handler
        self._handlers.append(self._log_handler)
        logger.info(f"AlertManager initialized: cooldown={cooldown_seconds}s, max_history={max_history}")

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        # l'usine a gaz
        self._handlers.append(handler)

    def fire(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        bypass_cooldown: bool = False,
    ) -> bool:
        # stop le massacre
        # la tuyauterie de donnees
        now = time.time()

        # Check cooldown
        if not bypass_cooldown:
            last = self._last_fired.get(category, 0.0)
            if now - last < self.cooldown_seconds:
                return False

        alert = Alert(
            timestamp=now,
            level=level,
            category=category,
            message=message,
            data=data,
        )

        self._history.append(alert)
        self._last_fired[category] = now

        # Dispatch to all handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        return True

    def get_history(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[Alert]:
        # l'usine a gaz
        alerts = list(self._history)
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        if category is not None:
            alerts = [a for a in alerts if a.category == category]
        return alerts[-limit:]

    def clear_history(self) -> None:
        # dans quel etat j'erre
        self._history.clear()
        self._last_fired.clear()

    # --- Built-in alert helpers ---

    def check_drawdown(self, drawdown_pct: float, warning: float = 0.05, critical: float = 0.10) -> None:
        # l'usine a gaz
        if drawdown_pct >= critical:
            self.fire(AlertLevel.CRITICAL, "drawdown", f"Critical drawdown: {drawdown_pct:.2%}")
        elif drawdown_pct >= warning:
            self.fire(AlertLevel.WARNING, "drawdown", f"Drawdown warning: {drawdown_pct:.2%}")

    def check_latency(self, latency_ms: float, threshold_ms: float = 50.0) -> None:
        # l'usine a gaz
        if latency_ms > threshold_ms:
            self.fire(
                AlertLevel.WARNING,
                "latency",
                f"High latency: {latency_ms:.1f}ms > {threshold_ms:.1f}ms",
                data={"latency_ms": latency_ms},
            )

    def check_error_rate(self, error_rate: float, threshold: float = 0.05) -> None:
        # aie aie aie
        if error_rate > threshold:
            self.fire(
                AlertLevel.WARNING,
                "error_rate",
                f"High error rate: {error_rate:.2%} > {threshold:.2%}",
                data={"error_rate": error_rate},
            )

    @staticmethod
    def _log_handler(alert: Alert) -> None:
        # l'usine a gaz
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"[ALERT:{alert.category}] {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"[ALERT:{alert.category}] {alert.message}")
        else:
            logger.info(f"[ALERT:{alert.category}] {alert.message}")
