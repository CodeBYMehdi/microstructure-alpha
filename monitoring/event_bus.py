
import logging
from typing import Callable, Dict, List, Any
from collections import defaultdict
import queue
import threading

logger = logging.getLogger(__name__)

class EventBus:
    # l'usine a gaz
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._subscribers = defaultdict(list)
            cls._instance._lock = threading.RLock()
        return cls._instance

    def subscribe(self, topic: str, handler: Callable[[Any], None]):
        # l'usine a gaz
        with self._lock:
            self._subscribers[topic].append(handler)
            logger.debug(f"Subscribed to {topic}")

    def publish(self, topic: str, data: Any):
        # l'usine a gaz
        with self._lock:
            handlers = self._subscribers.get(topic, [])
        
        # Notifie hors lock pr éviter deadlock si handlers sub/pub
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error handling event {topic}: {e}", exc_info=True)

    def clear(self):
        # l'usine a gaz
        with self._lock:
            self._subscribers.clear()

# Instance globale
bus = EventBus()
