
import os
import logging
from datetime import datetime
from data.tick_stream import SyntheticTickStream
from data.storage import TimeSeriesStorage
from config.loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(filepath: str, duration: int = 600, symbol: str = "BTC-USDT"):
    # la tuyauterie de donnees
    logger.info(f"Generating {duration} seconds of synthetic data for {symbol}...")
    
    # 1. Générer
    stream = SyntheticTickStream(symbol=symbol, duration_seconds=duration, seed=42)
    ticks = list(stream)
    
    logger.info(f"Generated {len(ticks)} ticks.")
    
    # 2. Stocker
    if os.path.exists(filepath):
        os.remove(filepath)
        
    storage = TimeSeriesStorage(filepath, mode='w')
    storage.store_ticks(ticks)
    
    logger.info(f"Data saved to {filepath}")
    
    # Vérifier
    count = storage.get_tick_count()
    logger.info(f"Verified storage count: {count}")

if __name__ == "__main__":
    # Assurer dossier data existe
    os.makedirs("data", exist_ok=True)
    generate_sample_data("data/sample_market_data.h5")
