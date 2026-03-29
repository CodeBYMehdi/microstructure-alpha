import numpy as np
from scipy import stats

class EntropyCalculator:
    # l'usine a gaz
    
    @staticmethod
    def compute_from_pdf(pdf_values: np.ndarray, dx: float) -> float:
        # la calculette
        # Éviter log(0)
        if len(pdf_values) == 0 or not np.any(pdf_values > 0):
            return float('nan')
        p = pdf_values[pdf_values > 0]
        return -np.sum(p * np.log(p)) * dx

    @staticmethod
    def compute_kl_divergence(p: np.ndarray, q: np.ndarray, dx: float) -> float:
        # la calculette
        # Floor both distributions to a small epsilon to prevent div-by-zero
        eps = 1e-30
        valid_mask = (p > eps) & (q > eps)

        p_valid = p[valid_mask]
        q_valid = q[valid_mask]

        if len(p_valid) == 0:
            return 0.0

        # Clip ratio to prevent overflow in log
        ratio = np.clip(p_valid / q_valid, 1e-10, 1e10)
        return float(np.sum(p_valid * np.log(ratio)) * dx)

    @staticmethod
    def compute_from_samples(data: np.ndarray, method: str = 'vasicek') -> float:
        # la calculette
        sigma = np.std(data)
        if sigma == 0:
            return 0.0  # Zero variance = zero information content (not -inf which poisons downstream)

        if method == 'knn':
            return EntropyCalculator.compute_kozachenko_leonenko(data)

        # Default: use KNN entropy (handles fat tails properly, unlike Gaussian assumption)
        # KNN is ~2x slower but produces correct entropy for leptokurtic returns
        # Gaussian H = 0.5*log(2πeσ²) underestimates by 15-25% on typical equity returns
        if len(data) >= 20:
            return EntropyCalculator.compute_kozachenko_leonenko(data)

        # Fallback for very small samples: Gaussian approximation
        return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    @staticmethod
    def compute_kozachenko_leonenko(data: np.ndarray, k: int = 3) -> float:
        # la calculette
        from scipy.spatial import cKDTree
        from scipy.special import digamma
        
        n = len(data)
        if n <= k:
            return 0.0
            
        # Reshape pr KDTree (N, 1)
        data_reshaped = data.reshape(-1, 1)
        
        # Construire Arbre
        tree = cKDTree(data_reshaped)
        
        # Requête k+1 voisins (1er est soi)
        # p=2 Euclidien (idem abs pr 1D)
        dists, _ = tree.query(data_reshaped, k=k+1, p=2)
        
        # Dist au k-ième voisin (col k, car 0 est soi)
        r_k = dists[:, -1]
        
        # Éviter log(0) avec epsilon
        r_k = np.maximum(r_k, 1e-10)
        
        # H = -psi(k) + psi(n) + log(c_d) + (d/n) * sum(log(r_k))
        # Pr d=1, c_d = 2 (vol boule unité 1D est long 2: [-1, 1])
        d = 1
        c_d = 2.0
        
        const_term = digamma(n) - digamma(k) + np.log(c_d)
        sum_log_dist = np.sum(np.log(r_k))
        
        return const_term + (d / n) * sum_log_dist


# --- Rust backend swap ---
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    import logging as _logging
    _PyEntropyCalculator = EntropyCalculator
    EntropyCalculator = get_rust_core().EntropyCalculator
    _logging.getLogger(__name__).info("EntropyCalculator → Rust backend")
