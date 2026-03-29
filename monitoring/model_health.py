# aie aie aie

import time
from collections import deque
import numpy as np
from dataclasses import dataclass
from config.loader import get_config

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class HealthStatus:
    # l'usine a gaz
    is_healthy: bool
    latency_ms: float
    memory_usage_mb: float
    error_rate: float


class ModelHealthMonitor:
    # aie aie aie
    # la grosse machine

    def __init__(self, window_size: int = 1000):
        self.config = get_config()
        self.latency_budget = self.config.thresholds.risk.latency_budget_ms
        self.max_error_rate = self.config.thresholds.risk.max_error_rate

        self.latencies: deque = deque(maxlen=window_size)
        self.errors: deque = deque(maxlen=window_size)
        self.start_time = time.time()

    def record_latency(self, latency_ms: float) -> None:
        # l'usine a gaz
        self.latencies.append(latency_ms)
        self.errors.append(0)

    def record_error(self) -> None:
        # aie aie aie
        self.errors.append(1)

    @property
    def error_count(self) -> int:
        # aie aie aie
        return sum(self.errors)

    def check_health(self) -> HealthStatus:
        # l'usine a gaz
        if not self.latencies:
            return HealthStatus(True, 0.0, 0.0, 0.0)

        avg_latency = float(np.mean(self.latencies))
        error_rate = float(np.mean(self.errors))

        is_healthy = True
        if avg_latency > self.latency_budget:
            is_healthy = False
        if error_rate > self.max_error_rate:
            is_healthy = False

        # Memory usage (real if psutil available)
        mem_mb = 0.0
        if _HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        return HealthStatus(
            is_healthy=is_healthy,
            latency_ms=avg_latency,
            memory_usage_mb=mem_mb,
            error_rate=error_rate,
        )
