# la tuyauterie de donnees
# la grosse machine

import numpy as np
from typing import Optional, Callable
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
from core.types import PDFData, PDFModelOutput, PDFDiagnostics
from microstructure.entropy import EntropyCalculator
from config.loader import get_config

logger = logging.getLogger(__name__)


class GMMDensityEstimator:
    # la tuyauterie de donnees
    # la grosse machine

    def __init__(self, config=None):
        self.config = config or get_config()
        pdf_cfg = self.config.thresholds.pdf
        self.fitted: bool = False
        self._gmm: Optional[GaussianMixture] = None
        self._data_std: float = 1.0
        self._data_mean: float = 0.0
        self._max_components: int = pdf_cfg.max_gmm_components
        self._min_data: int = pdf_cfg.min_data_points
        self._grid_points: int = pdf_cfg.grid_points
        self._sigma_range: float = pdf_cfg.sigma_range
        self._mode_collapse_threshold: float = pdf_cfg.mode_collapse_threshold
        logger.info("Initialized GMMDensityEstimator")

    def fit(self, data: np.ndarray) -> None:
        # la tuyauterie de donnees
        # la grosse machine
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=float)

        # Input validation
        clean = data[np.isfinite(data)]
        if len(clean) < self._min_data:
            logger.warning(f"Insufficient data for density model ({len(clean)} < {self._min_data})")
            return

        data_2d = clean.reshape(-1, 1)

        try:
            best_gmm = None
            best_bic = np.inf
            max_k = min(self._max_components, len(clean))
            for k in range(1, max_k + 1):
                gmm = GaussianMixture(
                    n_components=k, covariance_type="full", random_state=42
                )
                gmm.fit(data_2d)
                bic = gmm.bic(data_2d)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm

            self._gmm = best_gmm
            self.fitted = self._gmm is not None
            self._data_mean = float(np.mean(clean))
            self._data_std = float(np.std(clean))
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"Density model fit failed: {e}")
            self.fitted = False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # l'usine a gaz
        if not self.fitted or self._gmm is None:
            return stats.norm.pdf(x, loc=self._data_mean, scale=max(self._data_std, 1e-8))

        x_flat = np.asarray(x).flatten()
        x_2d = x_flat.reshape(-1, 1)
        log_prob = self._gmm.score_samples(x_2d)
        return np.exp(log_prob)

    def get_model_output(self, data_window: np.ndarray) -> PDFModelOutput:
        # la grosse machine
        if not self.fitted:
            return PDFModelOutput(
                pdf_callable=lambda x: np.zeros_like(x, dtype=float),
                log_likelihood=-np.inf,
                entropy=0.0,
                tail_slope=0.0,
                diagnostics=PDFDiagnostics(True, True, True),
                valid=False,
            )

        mu = np.mean(data_window)
        sigma = max(np.std(data_window), 1e-8)
        x_grid = np.linspace(
            mu - self._sigma_range * sigma,
            mu + self._sigma_range * sigma,
            self._grid_points,
        )
        dx = x_grid[1] - x_grid[0]
        pdf_values = self.evaluate(x_grid)

        # Entropy
        entropy = EntropyCalculator.compute_from_pdf(pdf_values, dx)

        # Tail slope proxy
        tail_slope = 0.0
        try:
            q95 = np.percentile(data_window, 95)
            d_model = self.evaluate(np.array([q95]))[0]
            d_gauss = stats.norm.pdf(q95, loc=mu, scale=sigma)
            tail_slope = d_model / (d_gauss + 1e-9)
        except (ValueError, IndexError):
            pass

        # Log-likelihood
        eval_data = self.evaluate(data_window)
        log_likelihood = float(np.sum(np.log(eval_data + 1e-10)))

        # Diagnostics
        mode_collapse = sigma < 1e-6
        tail_instability = tail_slope > 10.0 or tail_slope < 0.1
        entropy_jump = False  # Requires history tracking

        return PDFModelOutput(
            pdf_callable=self.evaluate,
            log_likelihood=log_likelihood,
            entropy=entropy,
            tail_slope=tail_slope,
            diagnostics=PDFDiagnostics(
                mode_collapse=mode_collapse,
                tail_instability=tail_instability,
                entropy_jump=entropy_jump,
            ),
            valid=not (mode_collapse or tail_instability),
        )

    def get_bounds(self, sigma_mult: float = 5.0) -> tuple:
        # l'usine a gaz
        mu = self._data_mean
        sigma = self._data_std
        if self._gmm is not None and self.fitted:
            weights = self._gmm.weights_
            means = self._gmm.means_.flatten()
            covs = self._gmm.covariances_.reshape(-1)
            mu = float(np.sum(weights * means))
            var = float(np.sum(weights * (covs + (means - mu) ** 2)))
            sigma = np.sqrt(max(var, 1e-8))
        if sigma == 0:
            sigma = 1e-4
        return mu - sigma_mult * sigma, mu + sigma_mult * sigma

    def get_pdf_data(self, x_grid: np.ndarray) -> PDFData:
        # la tuyauterie de donnees
        y = self.evaluate(x_grid)
        return PDFData(
            x=x_grid,
            y=y,
            method="GMMDensity",
            bandwidth=None,
            metadata={"fitted": self.fitted},
        )

    def get_state(self) -> dict:
        """Serialize GMM state for checkpointing."""
        if not self.fitted or self._gmm is None:
            return {"fitted": False}
        return {
            "fitted":              True,
            "data_mean":           float(self._data_mean),
            "data_std":            float(self._data_std),
            "gmm_weights":         self._gmm.weights_.tolist(),
            "gmm_means":           self._gmm.means_.tolist(),
            "gmm_covariances":     self._gmm.covariances_.tolist(),
            "gmm_precisions_chol": self._gmm.precisions_cholesky_.tolist(),
        }

    def restore_state(self, state: dict) -> None:
        """Restore GMM state from checkpoint."""
        if not state.get("fitted"):
            self.fitted = False
            return
        self._data_mean = state["data_mean"]
        self._data_std  = state["data_std"]
        # Restore sklearn GMM internals
        if self._gmm is None:
            self._gmm = GaussianMixture(
                n_components=len(state["gmm_weights"]),
                covariance_type="full",
            )
        self._gmm.weights_             = np.array(state["gmm_weights"])
        self._gmm.means_               = np.array(state["gmm_means"])
        self._gmm.covariances_         = np.array(state["gmm_covariances"])
        self._gmm.precisions_cholesky_ = np.array(state["gmm_precisions_chol"])
        # Set sklearn fitted markers so score_samples() works without re-fitting
        self._gmm.converged_   = True
        self._gmm.n_iter_      = 1
        self._gmm.lower_bound_ = 0.0
        self.fitted = True


# Backward compatibility alias
NormalizingFlow = GMMDensityEstimator


# --- Rust backend: fast 1D GMM ---
from core.backend import _USE_RUST_CORE, get_rust_core as _get_rust_core
if _USE_RUST_CORE:
    _rc = _get_rust_core()
    _PyGMMDensityEstimator = GMMDensityEstimator

    class GMMDensityEstimator(_PyGMMDensityEstimator):
        """GMM with Rust-accelerated fit/evaluate for 1D data."""
        def __init__(self, config=None):
            super().__init__(config)
            self._rust_gmm = _rc.RustGMM1D(
                max_components=self._max_components,
            )

        def fit(self, data):
            import numpy as _np
            if not isinstance(data, _np.ndarray):
                data = _np.asarray(data, dtype=float)
            clean = data[_np.isfinite(data)]
            if len(clean) < self._min_data:
                return
            self._rust_gmm.fit(clean.astype(_np.float64))
            self.fitted = self._rust_gmm.is_fitted
            self._data_mean = self._rust_gmm.data_mean
            self._data_std = self._rust_gmm.data_std

        def evaluate(self, x):
            import numpy as _np
            if not self.fitted:
                return stats.norm.pdf(x, loc=self._data_mean, scale=max(self._data_std, 1e-8))
            # If Rust GMM is fitted, use it; otherwise fall back to sklearn GMM
            # (the sklearn path is used after restore_state when Rust state cannot
            # be reconstructed from serialized weights/means/variances alone)
            if self._rust_gmm.is_fitted:
                x_arr = _np.asarray(x, dtype=_np.float64).flatten()
                return _np.asarray(self._rust_gmm.evaluate(x_arr))
            # Fallback: use the sklearn GMM restored via restore_state
            return _PyGMMDensityEstimator.evaluate(self, x)

        def get_bounds(self, sigma_mult=5.0):
            if self.fitted and self._rust_gmm.is_fitted:
                return self._rust_gmm.get_bounds(sigma_mult)
            return _PyGMMDensityEstimator.get_bounds(self, sigma_mult)

        def get_state(self) -> dict:
            """Serialize GMM state; uses Rust internals when available."""
            if not self.fitted:
                return {"fitted": False}
            if self._rust_gmm.is_fitted:
                return {
                    "fitted":          True,
                    "data_mean":       float(self._data_mean),
                    "data_std":        float(self._data_std),
                    "gmm_weights":     list(self._rust_gmm.get_weights()),
                    "gmm_means":       list(self._rust_gmm.get_means()),
                    "gmm_variances":   list(self._rust_gmm.get_variances()),
                    "backend":         "rust",
                }
            # Fallback to sklearn state (e.g. after restore from checkpoint)
            return _PyGMMDensityEstimator.get_state(self)

        def restore_state(self, state: dict) -> None:
            """Restore from checkpoint; rebuilds a sklearn GMM for evaluation."""
            if not state.get("fitted"):
                self.fitted = False
                return
            self._data_mean = state["data_mean"]
            self._data_std  = state["data_std"]
            # Rebuild sklearn GMM from stored parameters so evaluate() works
            from sklearn.mixture import GaussianMixture as _GaussianMixture
            weights   = np.array(state["gmm_weights"])
            means_1d  = np.array(state["gmm_means"])
            n         = len(weights)
            if self._gmm is None:
                self._gmm = _GaussianMixture(n_components=n, covariance_type="full")
            self._gmm.weights_ = weights
            self._gmm.means_   = means_1d.reshape(n, 1)
            if "gmm_variances" in state:
                # Rust backend stores variances (1D); build full covariance matrices
                variances = np.array(state["gmm_variances"])
                self._gmm.covariances_         = variances.reshape(n, 1, 1)
                self._gmm.precisions_cholesky_ = (1.0 / np.sqrt(variances)).reshape(n, 1, 1)
            else:
                # sklearn backend stored full covariances/precisions_cholesky
                self._gmm.covariances_         = np.array(state["gmm_covariances"])
                self._gmm.precisions_cholesky_ = np.array(state["gmm_precisions_chol"])
            self._gmm.converged_   = True
            self._gmm.n_iter_      = 1
            self._gmm.lower_bound_ = 0.0
            # _rust_gmm is NOT re-fitted; evaluate() will use sklearn fallback
            self.fitted = True

    NormalizingFlow = GMMDensityEstimator
    logger.info("GMMDensityEstimator → Rust GMM backend")
