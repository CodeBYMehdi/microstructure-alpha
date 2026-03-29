import numpy as np

class Normalizer:
    # l'usine a gaz
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._buffer = []
        self._mean = 0.0
        self._std = 1.0
        
    def update(self, value: float) -> float:
        # l'usine a gaz
        self._buffer.append(value)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
            
        # Recalcul stats (opt: pourrait utiliser algo Welford pour MAJ en ligne)
        if len(self._buffer) > 1:
            self._mean = np.mean(self._buffer)
            self._std = np.std(self._buffer, ddof=1)
            
        if self._std == 0:
            return 0.0
            
        return (value - self._mean) / self._std

    def get_stats(self):
        return {"mean": self._mean, "std": self._std}
