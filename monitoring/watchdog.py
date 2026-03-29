"""Watchdog — background daemon that monitors system liveness.

Checks every 5 seconds for:
- Tick staleness (no tick for 30s)
- Kill switch status
- Process window latency
"""

import logging
import threading
import time
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(self, kill_switch=None, alert_callback: Optional[Callable] = None,
                 tick_timeout_s: float = 30.0, check_interval_s: float = 5.0):
        self._kill_switch = kill_switch
        self._alert_callback = alert_callback
        self._tick_timeout_s = tick_timeout_s
        self._check_interval_s = check_interval_s

        self._last_tick_time: float = time.time()
        self._last_process_window_time: float = 0.0
        self._last_process_window_latency_ms: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def heartbeat_tick(self) -> None:
        self._last_tick_time = time.time()

    def heartbeat_process_window(self, latency_ms: float = 0.0) -> None:
        self._last_process_window_time = time.time()
        self._last_process_window_latency_ms = latency_ms

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_tick_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Watchdog stopped")

    def _run(self) -> None:
        while self._running:
            try:
                self._check()
            except Exception as e:
                logger.error(f"Watchdog check error: {e}")
            time.sleep(self._check_interval_s)

    def _check(self) -> None:
        now = time.time()

        # Check tick staleness
        tick_age = now - self._last_tick_time
        if tick_age > self._tick_timeout_s:
            msg = f"Watchdog: no tick for {tick_age:.0f}s (timeout={self._tick_timeout_s}s)"
            logger.warning(msg)
            if self._alert_callback:
                self._alert_callback("TICK_STALE", msg)
            # Trigger kill switch if data stale for 2x timeout (likely data outage)
            if tick_age > self._tick_timeout_s * 2 and self._kill_switch is not None:
                if not self._kill_switch.triggered:
                    kill_msg = f"Data stale for {tick_age:.0f}s (2x timeout={self._tick_timeout_s * 2:.0f}s)"
                    logger.critical(f"Watchdog triggering kill switch: {kill_msg}")
                    self._kill_switch.trigger(kill_msg)

        # Check kill switch
        if self._kill_switch is not None and self._kill_switch.triggered:
            msg = f"Watchdog: kill switch active ({self._kill_switch.reason})"
            logger.warning(msg)
            if self._alert_callback:
                self._alert_callback("KILL_SWITCH", msg)

    def get_status(self) -> Dict:
        now = time.time()
        return {
            "running": self._running,
            "tick_age_s": round(now - self._last_tick_time, 1),
            "tick_stale": (now - self._last_tick_time) > self._tick_timeout_s,
            "last_process_window_latency_ms": round(self._last_process_window_latency_ms, 1),
            "kill_switch_active": self._kill_switch.triggered if self._kill_switch else False,
        }
