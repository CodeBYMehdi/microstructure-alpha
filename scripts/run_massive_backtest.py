
import os
import logging
import multiprocessing
from datetime import datetime
import glob

import sys
base_dir = os.path.abspath(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from main import Strategy
from core.types import Tick

try:
    import databento as db
except ImportError:
    db = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SP500_Backtest")

class DatabentoTickStream:
    def __init__(self, file_path: str, symbol: str = "UNKNOWN"):
        if db is None:
            raise ImportError("databento non installé. Lance: pip install databento")
        self.file_path = file_path
        self.symbol = symbol

    def _scale(self, val: float) -> float:
        if val is None:
            return None
        if abs(val) > 1e6:
            return float(val) * 1e-9
        return float(val)

    def __iter__(self):
        store = db.DBNStore.from_file(self.file_path)
        for record in store:
            ts = getattr(record, "ts_event", None)
            if ts is None:
                continue
            timestamp = datetime.fromtimestamp(ts / 1e9)

            price = None
            volume = 0.0
            bid = None
            ask = None
            bid_size = None
            ask_size = None

            if hasattr(record, "price"):
                price = self._scale(record.price)
                if hasattr(record, "size"):
                    volume = float(record.size)

            if hasattr(record, "bid_px_00") and hasattr(record, "ask_px_00"):
                bid = self._scale(record.bid_px_00)
                ask = self._scale(record.ask_px_00)
                if hasattr(record, "bid_sz_00"):
                    bid_size = float(record.bid_sz_00)
                if hasattr(record, "ask_sz_00"):
                    ask_size = float(record.ask_sz_00)
                if price is None and bid is not None and ask is not None:
                    price = (bid + ask) / 2.0

            if price is None:
                continue

            yield Tick(
                timestamp=timestamp,
                symbol=self.symbol,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                exchange="Databento"
            )

def run_single_backtest(file_path: str):
    # simu pour pas pleurer en live
    # verif rapide
    try:
        # Extract symbol from filename (assuming format "symbol_date.dbn" or similar)
        # Fallback to generic if parsing fails
        filename = os.path.basename(file_path)
        symbol = filename.split('.')[0].split('_')[0] # Heuristic
        
        logger.info(f"[{symbol}] Starting backtest process on {file_path}")
        
        # Initialize Stream
        stream = DatabentoTickStream(file_path=file_path, symbol=symbol)
        
        # Initialize Strategy (Disable Viz for massive headless run)
        strategy = Strategy(enable_viz=False)
        
        # Run
        # duration_seconds is huge to cover full file
        strategy.run(duration_seconds=99999999, stream=stream)
        
        # Save Results (Strategy saves decision_log.csv by default, we should rename it)
        # Strategy currently writes to hardcoded "decision_log.csv". 
        # We need to monkey-patch or modify Strategy to support custom output paths
        # For now, we rename the file after run.
        
        default_log = "decision_log.csv"
        new_log = f"results/decision_log_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if os.path.exists(default_log):
            os.makedirs("results", exist_ok=True)
            os.rename(default_log, new_log)
            logger.info(f"[{symbol}] Results saved to {new_log}")
        else:
            logger.warning(f"[{symbol}] No log file generated.")
            
    except Exception as e:
        logger.error(f"[{symbol}] Failed: {e}")

def run_batch_backtest(data_dir: str, max_workers: int = 4):
    # la tuyauterie de donnees
    # simu pour pas pleurer en live
    # Find all DBN files
    files = glob.glob(os.path.join(data_dir, "*.dbn")) + glob.glob(os.path.join(data_dir, "*.dbn.zst"))
    
    if not files:
        logger.error(f"No .dbn files found in {data_dir}")
        return
        
    logger.info(f"Found {len(files)} files. Starting pool with {max_workers} workers.")
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        pool.map(run_single_backtest, files)
        
    logger.info("Batch backtest completed.")

if __name__ == "__main__":
    # Example usage
    # User should point this to their Databento downloads folder
    DATA_DIR = "data/databento_downloads" 
    
    # Ensure dir exists for demo
    os.makedirs(DATA_DIR, exist_ok=True)
    
    run_batch_backtest(DATA_DIR)
