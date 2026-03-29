# le feu vert
# ca fait bim

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttributionBucket:
    # ca fait bim
    name: str
    total_pnl: float
    n_trades: int
    win_rate: float
    avg_pnl: float
    pnl_contribution_pct: float  # % of total PnL


class PnLAttribution:
    # la calculette
    # ca fait bim

    def __init__(self):
        self._trades: List[Dict[str, Any]] = []

    def add_trade(self, trade: Dict[str, Any]) -> None:
        # l'usine a gaz
        self._trades.append(trade)

    def add_trades(self, trades: List[Dict[str, Any]]) -> None:
        # l'usine a gaz
        self._trades.extend(trades)

    def by_regime(self) -> List[AttributionBucket]:
        # ca fait bim
        return self._attribute_by_key("regime_id")

    def by_side(self) -> List[AttributionBucket]:
        # ca fait bim
        return self._attribute_by_key("side")

    def by_signal(self) -> List[AttributionBucket]:
        # le feu vert
        # ca fait bim
        return self._attribute_by_key("entry_reason")

    def by_holding_period(self, bins: Optional[List[int]] = None) -> List[AttributionBucket]:
        # ca fait bim
        if bins is None:
            bins = [0, 5, 10, 20, 50, 100, 1000]

        total_pnl = sum(t.get("pnl", 0) for t in self._trades) or 1.0
        buckets = []

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            label = f"{lo}-{hi} windows"
            in_bucket = [t for t in self._trades
                         if lo <= t.get("hold_duration_windows", 0) < hi]

            if not in_bucket:
                continue

            pnls = [t.get("pnl", 0) for t in in_bucket]
            wins = [p for p in pnls if p > 0]

            buckets.append(AttributionBucket(
                name=label,
                total_pnl=sum(pnls),
                n_trades=len(in_bucket),
                win_rate=len(wins) / len(pnls) if pnls else 0.0,
                avg_pnl=float(np.mean(pnls)),
                pnl_contribution_pct=sum(pnls) / total_pnl * 100,
            ))

        return buckets

    def _attribute_by_key(self, key: str) -> List[AttributionBucket]:
        # l'usine a gaz
        groups: Dict[str, List[float]] = defaultdict(list)

        for trade in self._trades:
            group_val = str(trade.get(key, "unknown"))
            pnl = trade.get("pnl", 0.0)
            groups[group_val].append(pnl)

        total_pnl = sum(sum(v) for v in groups.values()) or 1.0
        buckets = []

        for name, pnls in sorted(groups.items()):
            wins = [p for p in pnls if p > 0]
            buckets.append(AttributionBucket(
                name=name,
                total_pnl=sum(pnls),
                n_trades=len(pnls),
                win_rate=len(wins) / len(pnls) if pnls else 0.0,
                avg_pnl=float(np.mean(pnls)),
                pnl_contribution_pct=sum(pnls) / total_pnl * 100,
            ))

        return sorted(buckets, key=lambda b: b.total_pnl, reverse=True)

    def full_report(self) -> Dict[str, List[AttributionBucket]]:
        # l'usine a gaz
        return {
            "by_regime": self.by_regime(),
            "by_side": self.by_side(),
            "by_signal": self.by_signal(),
            "by_holding_period": self.by_holding_period(),
        }

    def summary_string(self) -> str:
        # l'usine a gaz
        report = self.full_report()
        lines = ["=== PnL Attribution Report ===\n"]

        for dimension, buckets in report.items():
            lines.append(f"\n--- {dimension.upper()} ---")
            for b in buckets[:10]:  # Top 10
                lines.append(
                    f"  {b.name}: PnL={b.total_pnl:+.2f} | "
                    f"Trades={b.n_trades} | WR={b.win_rate:.1%} | "
                    f"Avg={b.avg_pnl:+.4f} | Contrib={b.pnl_contribution_pct:.1f}%"
                )

        return "\n".join(lines)
