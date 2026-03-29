from typing import Iterator, Optional, List
from datetime import datetime, timedelta
import numpy as np
import logging
from core.interfaces import IDataSource
from core.types import Tick
from config.loader import get_config
from data.storage import TimeSeriesStorage

logger = logging.getLogger(__name__)

class TickStream(IDataSource):
    # la tuyauterie de donnees
    # le flux en scred
    def __iter__(self) -> Iterator[Tick]:
        return self.stream()

class SyntheticTickStream(TickStream):
    # la tuyauterie de donnees
    # verif rapide
    def __init__(self, symbol: str = "BTC-USDT", start_time: datetime = None, duration_seconds: int = 3600, seed: int = 42,
                 drift: float = 0.0, vol_multiplier: float = 1.0,
                 fat_tail_prob: float = 0.0, drop_tick_prob: float = 0.0):
        self.symbol = symbol
        self.start_time = start_time or datetime.now()
        self.duration_seconds = duration_seconds
        self.seed = seed
        self.drift = drift
        self.vol_multiplier = vol_multiplier
        self.fat_tail_prob = fat_tail_prob
        self.drop_tick_prob = drop_tick_prob
        self._rng = np.random.RandomState(seed)
        self._current_time = self.start_time
        self._price = 10000.0
        
        # Load instrument config if available to respect tick size etc.
        self.config = get_config()
        self.tick_size = 0.01 # Default fallback
        if hasattr(self.config, 'instruments') and hasattr(self.config.instruments, 'instruments'):
            for instr in self.config.instruments.instruments:
                if instr.symbol == symbol:
                    self.tick_size = instr.tick_size
                    break
                
        logger.info(f"Initialized SyntheticTickStream for {symbol} with seed {seed}, vol_mult={vol_multiplier}, fat_tail={fat_tail_prob}")

    def stream(self) -> Iterator[Tick]:
        end_time = self.start_time + timedelta(seconds=self.duration_seconds)
        
        while self._current_time < end_time:
            # Simulate time passing (randomly between 10ms and 1s)
            delta_ms = self._rng.randint(10, 1000)
            self._current_time += timedelta(milliseconds=delta_ms)

            # Randomly drop ticks to test robustness against sparse data
            if self._rng.random() < self.drop_tick_prob:
                continue

            # Generate random walk
            volatility = 1.0 * self.vol_multiplier
            if self._rng.random() < self.fat_tail_prob:
                # Extreme shock
                shock = self._rng.standard_cauchy() * volatility * 2.5
            else:
                shock = self._rng.normal(self.drift, volatility)
                
            self._price += shock
            # Enforce price floor — negative prices are invalid
            self._price = max(self.tick_size, self._price)
            self._price = round(self._price / self.tick_size) * self.tick_size
            
            # Generate tick
            tick = Tick(
                timestamp=self._current_time,
                symbol=self.symbol,
                price=self._price,
                volume=abs(self._rng.normal(1.0, 0.5)),
                bid=self._price - self.tick_size,
                ask=self._price + self.tick_size,
                bid_size=abs(self._rng.normal(5.0, 2.0)),
                ask_size=abs(self._rng.normal(5.0, 2.0))
            )
            
            yield tick

class RealTickStream(TickStream):
    # le flux en scred
    def __init__(self, storage_path: str, symbol: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        self.storage = TimeSeriesStorage(storage_path, mode='r')
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        logger.info(f"Initialized RealTickStream from {storage_path} for {symbol}")

    def stream(self) -> Iterator[Tick]:
        logger.info(f"Starting replay for {self.symbol}...")
        count = 0
        for tick in self.storage.load_ticks(symbol=self.symbol, start_time=self.start_time, end_time=self.end_time):
            count += 1
            yield tick
        logger.info(f"Replay finished. Yielded {count} ticks.")

class CsvTickStream(TickStream):
    # le flux en scred
    def __init__(self, filepath: str, symbol: str):
        self.filepath = filepath
        self.symbol = symbol
        import pandas as pd
        self.df = pd.read_csv(filepath)
        # Ensure timestamp is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.sort_values('timestamp', inplace=True)

    def stream(self) -> Iterator[Tick]:
        # Use itertuples for ~10x speedup over iterrows
        cols = set(self.df.columns)
        for row in self.df.itertuples(index=False):
            price = row.price
            # Validate: skip rows with invalid prices
            if price is None or not np.isfinite(price) or price <= 0:
                logger.warning(f"Skipping invalid tick: price={price}")
                continue
            volume = getattr(row, 'volume', 0.0)
            if volume is None or not np.isfinite(volume):
                volume = 0.0
            yield Tick(
                timestamp=row.timestamp,
                symbol=self.symbol,
                price=price,
                volume=volume,
                bid=getattr(row, 'bid', None) if 'bid' in cols else None,
                ask=getattr(row, 'ask', None) if 'ask' in cols else None,
                bid_size=getattr(row, 'bid_size', None) if 'bid_size' in cols else None,
                ask_size=getattr(row, 'ask_size', None) if 'ask_size' in cols else None,
                exchange=getattr(row, 'exchange', 'generic') if 'exchange' in cols else 'generic',
            )
