import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveKDE:
    # l'usine a gaz
    
    def __init__(self, kernel: str = 'gaussian', seed: Optional[int] = None):
        self.kernel_type = kernel
        self._kde_model = None
        self._dataset = None
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        logger.info(f"Initialized AdaptiveKDE with kernel={kernel}, seed={seed}")

    def fit(self, data: np.ndarray) -> 'AdaptiveKDE':
        # l'usine a gaz
        if len(data) < 10:
             # logger.warning(f"Insufficient data for KDE fit (min 10 samples), got {len(data)}")
             raise ValueError("Insufficient data for KDE fit (min 10 samples)")
             
        # Nettoyer données (retirer NaNs/Infs)
        clean_data = data[np.isfinite(data)]
        
        if len(clean_data) == 0:
             logger.error("No valid data after cleaning NaNs/Infs")
             raise ValueError("No valid data after cleaning")
             
        if np.std(clean_data) == 0:
             # Gérer cas variance zéro - matrice singulière
             # Ajout bruit minime déterministe
             logger.info("Zero variance detected in data, adding epsilon noise")
             noise = self._rng.normal(0, 1e-8, size=len(clean_data))
             clean_data = clean_data + noise

        self._dataset = clean_data
        try:
            self._kde_model = stats.gaussian_kde(clean_data, bw_method='scott')
        except np.linalg.LinAlgError:
             logger.error("Singular matrix in KDE fit despite noise addition")
             raise ValueError("Singular matrix in KDE fit - data might be constant")
        except Exception as e:
             logger.error(f"KDE fit failed: {e}")
             raise e
             
        return self

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # l'usine a gaz
        if self._kde_model is None:
            raise RuntimeError("Model not fitted")
        return self._kde_model(x)

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        # l'usine a gaz
        if self._kde_model is None:
            raise RuntimeError("Model not fitted")
        
        # Resample Scipy utilise état global numpy ou interne?
        # stats.gaussian_kde.resample prend arg 'seed' versions récentes?
        # Utilise self.random_state si dispo, mais pas exposé facilement vieilles versions.
        # Source: resample utilise self.covariance... et génère random.
        # Idéalement contrôler seed.
        
        # Pr contrôle total, impl manuelle si scipy supporte pas seed facil.
        # Pr l'instant, re-seed global numpy ou RNG interne si exposé.
        # Meilleur pr reprod: échantillonner dataset + bruit (bootstrap lissé).
        
        # Impl bootstrap lissé avec propre RNG:
        indices = self._rng.choice(len(self._dataset), size=n_samples, replace=True)
        samples = self._dataset[indices]
        
        # Ajout bruit selon bandwidth
        bw = self._kde_model.factor
        std = np.std(self._dataset)
        noise = self._rng.normal(0, bw * std, size=n_samples)
        
        return samples + noise

    def get_bandwidth(self) -> float:
        if self._kde_model is None:
            return 0.0
        return self._kde_model.factor

    def get_bounds(self, sigma_mult: float = 5.0) -> Tuple[float, float]:
        # l'usine a gaz
        if self._dataset is None:
            return -0.1, 0.1
        
        mu = np.mean(self._dataset)
        sigma = np.std(self._dataset)
        if sigma == 0: sigma = 1e-4
        
        return mu - sigma_mult * sigma, mu + sigma_mult * sigma
