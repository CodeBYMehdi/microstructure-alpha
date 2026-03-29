import time
import threading
import logging
from typing import Any, Dict

from storage.state_store import StateStore

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Periodically serializes component state to StateStore."""

    INTERVAL_S: float = 30.0

    def __init__(self, store: StateStore, components: Dict[str, Any]):
        self._store = store
        self._components = components
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="checkpoint-mgr",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Graceful shutdown: flush one last time, then stop."""
        self._stop_event.set()
        self._flush()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush()
            self._stop_event.wait(timeout=self.INTERVAL_S)

    def _flush(self) -> None:
        for key, component in self._components.items():
            try:
                self._store.checkpoint(key, component.get_state())
            except Exception as e:
                logger.error("checkpoint_fail: component=%s error=%s", key, e)

    def restore_all(self) -> bool:
        """Restore all components. Returns True if all restored, False if warmup needed."""
        for key, component in self._components.items():
            record = self._store.load(key)
            if not record:
                logger.warning("checkpoint_missing: key=%s", key)
                return False
            age = time.time() - record.updated_at
            if age > 300:
                logger.warning("checkpoint_stale: key=%s age_s=%.0f", key, age)
                return False
            component.restore_state(record.state)
            logger.info("checkpoint_restored: key=%s age_s=%.0f", key, age)
        return True
