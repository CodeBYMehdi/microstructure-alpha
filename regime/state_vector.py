# dans quel etat j'erre

from dataclasses import dataclass, asdict
import json
import numpy as np
from typing import List


@dataclass(frozen=True)
class StateVector:
    # dans quel etat j'erre
    mu: float
    sigma: float
    skew: float
    kurtosis: float
    tail_slope: float
    entropy: float

    def __post_init__(self):
        # on prepare le terrain
        for field_name in ('mu', 'sigma', 'skew', 'kurtosis', 'tail_slope', 'entropy'):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(val)}")

    def to_array(self) -> List[float]:
        # on passe a la caisse
        return [self.mu, self.sigma, self.skew, self.kurtosis, self.tail_slope, self.entropy]

    def to_dict(self) -> dict:
        # l'usine a gaz
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"StateVector(mu={self.mu:.6f}, sigma={self.sigma:.6f}, "
            f"skew={self.skew:.4f}, kurt={self.kurtosis:.4f}, "
            f"tail={self.tail_slope:.4f}, entropy={self.entropy:.4f})"
        )
