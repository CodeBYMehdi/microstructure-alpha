# la tuyauterie de donnees
# dans quel etat j'erre

import pandas as pd
import numpy as np
from typing import List, Optional, Iterator
from datetime import datetime
import logging
import os
from core.types import Tick, RegimeState

logger = logging.getLogger(__name__)


class TimeSeriesStorage:
    # la tuyauterie de donnees
    # le flux en scred

    def __init__(self, filepath: str, mode: str = 'a'):
        self.filepath = filepath
        self.mode = mode
        self.key_ticks = 'ticks'
        self.key_regimes = 'regimes'

        # Ensure directory exists (handle bare filename case)
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def store_ticks(self, ticks: List[Tick], append: bool = True) -> None:
        # l'usine a gaz
        if not ticks:
            return

        data = [
            {
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'price': t.price,
                'volume': t.volume,
                'bid': t.bid,
                'ask': t.ask,
                'bid_size': t.bid_size,
                'ask_size': t.ask_size,
                'exchange': t.exchange
            }
            for t in ticks
        ]

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        try:
            with pd.HDFStore(self.filepath, mode=self.mode) as store:
                store.put(self.key_ticks, df, format='table', append=append, data_columns=True)
            logger.info(f"Stored {len(ticks)} ticks to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to store ticks: {e}")
            raise

    def store_regimes(self, regimes: List[RegimeState], append: bool = True) -> None:
        # dans quel etat j'erre
        if not regimes:
            return

        data = []
        for r in regimes:
            entry = {
                'timestamp': r.timestamp,
                'regime_id': r.metadata.get('id', -1),
                'confidence': r.confidence,
                'entropy': r.entropy
            }
            data.append(entry)

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        try:
            with pd.HDFStore(self.filepath, mode=self.mode) as store:
                store.put(self.key_regimes, df, format='table', append=append, data_columns=True)
        except Exception as e:
            logger.error(f"Failed to store regimes: {e}")
            raise

    def load_ticks(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Iterator[Tick]:
        # on ramene les datas
        # le flux en scred
        where_clauses = []
        if symbol:
            where_clauses.append(f'symbol="{symbol}"')
        if start_time:
            # Preserve microsecond precision for HDF5 where clause
            ts_str = start_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            where_clauses.append(f'index >= "{ts_str}"')
        if end_time:
            ts_str = end_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            where_clauses.append(f'index <= "{ts_str}"')

        where = " & ".join(where_clauses) if where_clauses else None

        try:
            chunksize = 10000
            with pd.HDFStore(self.filepath, mode='r') as store:
                if self.key_ticks not in store:
                    return

                iterator = store.select(self.key_ticks, where=where, chunksize=chunksize, iterator=True)

                for chunk in iterator:
                    for timestamp, row in chunk.iterrows():
                        yield Tick(
                            timestamp=timestamp.to_pydatetime(),
                            symbol=row['symbol'],
                            price=row['price'],
                            volume=row['volume'],
                            bid=row['bid'] if not pd.isna(row['bid']) else None,
                            ask=row['ask'] if not pd.isna(row['ask']) else None,
                            bid_size=row['bid_size'] if not pd.isna(row['bid_size']) else None,
                            ask_size=row['ask_size'] if not pd.isna(row['ask_size']) else None,
                            exchange=row['exchange']
                        )

        except (OSError, KeyError) as e:
            logger.error(f"Error loading ticks from {self.filepath}: {e}")
            return

    def get_tick_count(self) -> int:
        # le bif
        try:
            with pd.HDFStore(self.filepath, mode='r') as store:
                if self.key_ticks in store:
                    return store.get_storer(self.key_ticks).nrows
                return 0
        except (OSError, KeyError):
            return 0
