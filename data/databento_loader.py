import logging
import os
import glob
from typing import Iterator, List, Optional, Tuple
from datetime import datetime
import databento as db
import pandas as pd
import numpy as np

from core.types import Tick

logger = logging.getLogger(__name__)

class DatabentoLoader:
    def __init__(self, data_dir: str, symbol_override: str = "TEST", trades_only: bool = False):
        self.data_dir = data_dir
        self.symbol_override = symbol_override
        self.trades_only = trades_only

        # Full 10-level book state: list of (price, size) for each side
        self._bid_levels: List[Tuple[float, float]] = []
        self._ask_levels: List[Tuple[float, float]] = []

        # Databento prices are fixed-precision int64 (1e9 scaling)
        self._price_divisor = 1e9

    @property
    def _best_bid(self) -> float:
        return self._bid_levels[0][0] if self._bid_levels else 0.0

    @property
    def _best_ask(self) -> float:
        return self._ask_levels[0][0] if self._ask_levels else float('inf')

    @property
    def _bid_size(self) -> float:
        return self._bid_levels[0][1] if self._bid_levels else 0.0

    @property
    def _ask_size(self) -> float:
        return self._ask_levels[0][1] if self._ask_levels else 0.0

    def load_files(self) -> Iterator[Tick]:
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.dbn.zst")))
        if not files:
            logger.warning(f"No .dbn.zst files found in {self.data_dir}")
            return

        logger.info(f"Found {len(files)} Databento files.")

        for file_path in files:
            logger.info(f"Processing {file_path}...")
            try:
                stored_data = db.read_dbn(file_path)

                for record in stored_data:
                    tick = self._process_record(record)
                    if tick:
                        yield tick

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    def _process_record(self, record) -> Optional[Tick]:
        action = record.action
        action_char = action.value if hasattr(action, 'value') else str(action)

        try:
            ts_event = record.ts_event
            timestamp = pd.to_datetime(ts_event, unit='ns').to_pydatetime()
        except Exception as e:
            logger.error(f"Error converting timestamp: {e}")
            return None

        price = record.price / self._price_divisor
        size = record.size

        side = record.side
        side_char = side.value if hasattr(side, 'value') else str(side)

        # --- Extract full MBP levels if available ---
        self._update_book_from_record(record)

        # --- TRADE ---
        if action_char == 'T':
            return Tick(
                timestamp=timestamp,
                symbol=self.symbol_override,
                price=price,
                volume=float(size),
                bid=self._best_bid if self._best_bid > 0 else None,
                ask=self._best_ask if self._best_ask < float('inf') else None,
                bid_size=self._bid_size,
                ask_size=self._ask_size,
                bids=list(self._bid_levels) if self._bid_levels else None,
                asks=list(self._ask_levels) if self._ask_levels else None,
                exchange="Databento"
            )

        # --- BOOK UPDATES ---
        if hasattr(record, 'depth') and record.depth == 0:
            # BBO updated via _update_book_from_record above

            if self.trades_only:
                return None

            # Emit quote tick with full L2 snapshot
            return Tick(
                timestamp=timestamp,
                symbol=self.symbol_override,
                price=price,
                volume=0.0,
                bid=self._best_bid if self._best_bid > 0 else None,
                ask=self._best_ask if self._best_ask < float('inf') else None,
                bid_size=self._bid_size,
                ask_size=self._ask_size,
                bids=list(self._bid_levels) if self._bid_levels else None,
                asks=list(self._ask_levels) if self._ask_levels else None,
                exchange="Databento"
            )

        return None

    def _update_book_from_record(self, record) -> None:
        """Extract all MBP levels from a Databento record into local book state.

        Databento MBP-10 records expose a `levels` array with up to 10 BookLevel
        entries, each having bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct.
        If `levels` is not present, fall back to single-level depth/side/price/size.
        """
        if hasattr(record, 'levels') and record.levels:
            bid_levels = []
            ask_levels = []
            for level in record.levels:
                bid_px = getattr(level, 'bid_px', 0)
                ask_px = getattr(level, 'ask_px', 0)
                bid_sz = getattr(level, 'bid_sz', 0)
                ask_sz = getattr(level, 'ask_sz', 0)

                # Databento uses UNDEF_PRICE (int64 max) for empty levels
                if bid_px > 0 and bid_px < 1e18 and bid_sz > 0:
                    bid_levels.append((bid_px / self._price_divisor, float(bid_sz)))
                if ask_px > 0 and ask_px < 1e18 and ask_sz > 0:
                    ask_levels.append((ask_px / self._price_divisor, float(ask_sz)))

            if bid_levels:
                self._bid_levels = bid_levels
            if ask_levels:
                self._ask_levels = ask_levels
        else:
            # Fallback: single-level update from depth/side/price/size fields
            if not hasattr(record, 'depth'):
                return
            action = record.action
            action_char = action.value if hasattr(action, 'value') else str(action)
            if action_char == 'T':
                return

            side = record.side
            side_char = side.value if hasattr(side, 'value') else str(side)
            price = record.price / self._price_divisor
            sz = float(record.size)
            depth = record.depth

            if side_char == 'B':
                while len(self._bid_levels) <= depth:
                    self._bid_levels.append((0.0, 0.0))
                self._bid_levels[depth] = (price, sz)
            elif side_char == 'A':
                while len(self._ask_levels) <= depth:
                    self._ask_levels.append((0.0, 0.0))
                self._ask_levels[depth] = (price, sz)
