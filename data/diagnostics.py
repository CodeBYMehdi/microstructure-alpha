import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

from data.event_bus import MarketEvent, EventType
from data.l2_orderbook import L2OrderBook

logger = logging.getLogger(__name__)

class DataDiagnostics:
    def __init__(self):
        self.packet_loss_counter = 0
        self.timestamp_drifts: List[float] = []
        
    def check_drift(self, event: MarketEvent):
        # l'usine a gaz
        if event.timestamp_exchange and event.timestamp_received:
            drift = (event.timestamp_received - event.timestamp_exchange).total_seconds()
            self.timestamp_drifts.append(drift)
            
            if drift > 1.0: # Seuil alerte 1 sec
                logger.warning(f"Haute latence détectée: {drift:.4f}s pour {event.event_type}")
                
    def detect_gaps(self, historical_bars: List[MarketEvent], timeframe_seconds: int):
        # l'usine a gaz
        if not historical_bars: return
        
        sorted_bars = sorted(historical_bars, key=lambda x: x.timestamp_exchange)
        expected_diff = timedelta(seconds=timeframe_seconds)
        
        gaps = []
        for i in range(1, len(sorted_bars)):
            diff = sorted_bars[i].timestamp_exchange - sorted_bars[i-1].timestamp_exchange
            if diff > expected_diff * 1.1: # Tolérance 10%
                gaps.append((sorted_bars[i-1].timestamp_exchange, sorted_bars[i].timestamp_exchange))
                
        if gaps:
            logger.warning(f"Détecté {len(gaps)} trous dans données histo.")
            for start, end in gaps[:5]:
                logger.warning(f"Trou: {start} -> {end}")

    def compare_streams(self, hist_events: List[MarketEvent], real_events: List[MarketEvent]):
        # le flux en scred
        # Convertir en DF pour comparaison facile
        # Suppose périodes chevauchantes
        pass

    def report(self):
        if self.timestamp_drifts:
            avg_drift = sum(self.timestamp_drifts) / len(self.timestamp_drifts)
            max_drift = max(self.timestamp_drifts)
            logger.info(f"Rapport Diag: Drift Moy={avg_drift*1000:.2f}ms, Drift Max={max_drift*1000:.2f}ms")
