import multiprocessing
import logging
import numpy as np
import time
from datetime import datetime
from collections import deque
from typing import Optional

from data.event_bus import event_bus, MarketEvent, EventType
from data.normalizer import DataNormalizer
from microstructure.pdf.kde import AdaptiveKDE

logger = logging.getLogger(__name__)

class VizConnector:
    # l'usine a gaz
    def __init__(self, 
                 normalizer: DataNormalizer, 
                 q_surface: multiprocessing.Queue, 
                 q_density: multiprocessing.Queue,
                 instrument_id: str = "1"):
        self.normalizer = normalizer
        self.q_surface = q_surface
        self.q_density = q_density
        self.instrument_id = instrument_id
        
        # Buffer pr KDE
        self.kde = AdaptiveKDE()
        self.returns_buffer = deque(maxlen=500)
        
        # Abonner
        event_bus.subscribe(EventType.TRADE, self.on_trade)
        event_bus.subscribe(EventType.QUOTE, self.on_quote) # Déclencher MAJ
        
        self.last_update_time = time.time()
        self.update_interval = 0.5 # Limiter MAJ à 2Hz

    def on_trade(self, event: MarketEvent):
        # Transférer trade à Viz
        # Trade attend: {'price': p, 'size': s, 'side': side, 'timestamp': ts}
        trade_data = {
            'price': event.price,
            'size': event.size,
            'side': event.side,
            'timestamp': event.timestamp.timestamp() # Utilise ts event pr compat backtest
        }
        
        self.q_surface.put({'type': 'TRADE', 'data': trade_data})
        self.q_density.put({'type': 'TRADE', 'data': trade_data})
        
        # Aussi MAJ features
        self._process_update(event)

    def on_quote(self, event: MarketEvent):
        self._process_update(event)

    def _process_update(self, event: MarketEvent):
        if event.instrument_id != self.instrument_id:
            return

        now = event.timestamp.timestamp()
        if now - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = now
        
        features = self.normalizer.compute_features(self.instrument_id)
        if not features:
            return
            
        # Collecter rend pr KDE
        if features.log_return != 0:
            self.returns_buffer.append(features.log_return)
            
        if len(self.returns_buffer) < 20:
            return
            
        # Fit KDE
        try:
            data = np.array(self.returns_buffer)
            self.kde.fit(data)
            
            # Évaluer PDF
            x_grid = np.linspace(-0.01, 0.01, 100)
            pdf_values = self.kde.evaluate(x_grid)
            
            # 1. MAJ File Surface
            # Besoin (dérive, vol, densité, id_rég)
            # Densité: Pic PDF
            peak_density = np.max(pdf_values)
            
            surface_data = {
                'drift': features.drift,
                'vol': features.volatility,
                'density': peak_density,
                'regime_id': 0, # Placeholder jusqu'à intégration Modèle Régime
                'timestamp': now
            }
            self.q_surface.put({'type': 'UPDATE', 'data': surface_data})
            
            # 2. MAJ File Densité
            density_data = {
                'pdf_values': pdf_values,
                'timestamp': now
            }
            self.q_density.put({'type': 'UPDATE', 'data': density_data})
            
        except Exception as e:
            logger.error(f"Error in VizConnector: {e}")

