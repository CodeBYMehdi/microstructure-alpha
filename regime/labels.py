# l'usine a gaz

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class RegimeProfile:
    # l'usine a gaz
    id: int
    centroid: np.ndarray
    sample_count: int
    persistence_score: float = 0.0
    label: str = "Unknown"


class RegimeLabelManager:
    # le big boss du truc
    # la tuyauterie de donnees

    def __init__(self):
        self._profiles: Dict[int, RegimeProfile] = {}

    def update_profile(self, regime_id: int, centroid: np.ndarray, count: int) -> None:
        # l'usine a gaz
        if regime_id == -1:
            return

        if regime_id not in self._profiles:
            self._profiles[regime_id] = RegimeProfile(
                id=regime_id,
                centroid=centroid,
                sample_count=count,
            )
        else:
            # Update existing (moving average centroid could go here)
            self._profiles[regime_id].centroid = centroid
            self._profiles[regime_id].sample_count += count

    def get_profile(self, regime_id: int) -> Optional[RegimeProfile]:
        # l'usine a gaz
        return self._profiles.get(regime_id, None)

    def describe(self, regime_id: int) -> str:
        # l'usine a gaz
        if regime_id == -1:
            return "NOISE/TRANSITION"

        p = self.get_profile(regime_id)
        if not p:
            return f"Regime {regime_id} (Unprofiled)"

        # Auto-labeling based on centroid
        # centroid: [mu, sigma, skew, kurt, tail, entropy]
        c = p.centroid
        desc = []

        # Volatility
        if c[1] > 0.001:
            desc.append("HighVol")
        elif c[1] < 0.0001:
            desc.append("LowVol")
        else:
            desc.append("MedVol")

        # Entropy
        if c[5] > -4:
            desc.append("HighEntropy")

        # Drift
        if c[0] > 0.0001:
            desc.append("Bullish")
        elif c[0] < -0.0001:
            desc.append("Bearish")

        return f"Regime {regime_id}: {' '.join(desc)}"
