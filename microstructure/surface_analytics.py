# la calculette
# dans quel etat j'erre

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class SurfaceState:
    density_velocity: float = 0.0          # How fast PDF max is shifting/deforming
    density_bifurcation: float = 0.0       # Detection of multi-modal clustering over time
    regime_trajectory_z: float = 0.0       # Gradient in z (drift/vol) direction
    surface_curvature: float = 0.0         # 2nd derivative (Hessian approximation)
    is_surface_collapsing: bool = False    # Trigger for early exit


class DensitySurfaceTracker:
    # l'usine a gaz
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        # Store PDF arrays over time
        self._pdf_history = deque(maxlen=history_size)
    
    def update(self, pdf_values: np.ndarray) -> Tuple[float, float]:
        # petit calcul
        self._pdf_history.append(pdf_values)
        
        if len(self._pdf_history) < 2:
            return 0.0, 0.0
            
        prev_pdf = self._pdf_history[-2]
        curr_pdf = self._pdf_history[-1]
        
        # d(PDF)/dt
        density_diff = curr_pdf - prev_pdf
        # Take the mean of the absolute deformations
        density_velocity = float(np.mean(np.abs(density_diff)))
        
        # Bifurcation: Look for multimodal separation
        # A simple proxy is divergence of the tails or presence of multiple local maxima
        curr_max = np.max(curr_pdf)
        # Are there peaks away from the mean?
        # A quick bifurcation approx: standard deviation of the high-density points
        high_density_idx = np.where(curr_pdf > curr_max * 0.5)[0]
        if len(high_density_idx) > 0:
            bifurcation_score = float(np.std(high_density_idx))
        else:
            bifurcation_score = 0.0
            
        return density_velocity, bifurcation_score


class RegimeSurfaceTracker:
    # la calculette
    # la grosse machine
    def __init__(self, history_size: int = 10, curvature_threshold: float = 1.0):
        self.history_size = history_size
        # Curvature threshold to detect surface collapse (falling off the regime)
        self.curvature_threshold = curvature_threshold
        
        self._mu_history = deque(maxlen=history_size)
        self._sigma_history = deque(maxlen=history_size)
        self._density_history = deque(maxlen=history_size)
        
    def update(self, mu: float, sigma: float, max_density: float) -> Tuple[float, float, bool]:
        # le bif
        self._mu_history.append(mu)
        self._sigma_history.append(sigma)
        self._density_history.append(max_density)
        
        if len(self._density_history) < 3:
            return 0.0, 0.0, False
            
        # Using 1D trajectories of max density as a proxy for traversing the 3D surface
        # Compute gradient (1st derivative) of density wrt time
        density_arr = np.array(self._density_history)
        grad_1 = np.gradient(density_arr)
        
        # Compute curvature (2nd derivative)
        grad_2 = np.gradient(grad_1)
        
        current_traj = float(grad_1[-1])
        current_curv = float(grad_2[-1])
        
        # Surface collapses if we are falling off a density ridge rapidly
        # (Negative trajectory + high negative curvature)
        is_collapsing = current_traj < 0 and current_curv < -self.curvature_threshold
        
        return current_traj, abs(current_curv), is_collapsing


class SurfaceAnalytics:
    # l'usine a gaz
    def __init__(self):
        self.density_tracker = DensitySurfaceTracker()
        self.regime_tracker = RegimeSurfaceTracker()
        self.state = SurfaceState()
        
    def update(self, pdf_values: np.ndarray, mu: float, sigma: float) -> SurfaceState:
        if len(pdf_values) == 0:
            return self.state
            
        max_density = float(np.max(pdf_values))
        
        vel, bif = self.density_tracker.update(pdf_values)
        traj, curv, is_collapsing = self.regime_tracker.update(mu, sigma, max_density)
        
        self.state = SurfaceState(
            density_velocity=vel,
            density_bifurcation=bif,
            regime_trajectory_z=traj,
            surface_curvature=curv,
            is_surface_collapsing=is_collapsing
        )
        return self.state


# --- Rust backend swap ---
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    _rc = get_rust_core()
    _PySurfaceAnalytics = SurfaceAnalytics

    class SurfaceAnalytics:
        """Rust-accelerated surface analytics with Python SurfaceState dataclass."""
        def __init__(self):
            self._rust = _rc.RustSurfaceAnalytics()
            self.state = SurfaceState()

        def update(self, pdf_values, mu: float, sigma: float) -> SurfaceState:
            import numpy as _np
            pdf_arr = _np.asarray(pdf_values, dtype=_np.float64)
            if pdf_arr.size == 0:
                return self.state
            vel, bif, traj, curv, collapse = self._rust.update(pdf_arr, mu, sigma)
            self.state = SurfaceState(
                density_velocity=vel,
                density_bifurcation=bif,
                regime_trajectory_z=traj,
                surface_curvature=curv,
                is_surface_collapsing=collapse,
            )
            return self.state

    logger.info("SurfaceAnalytics → Rust backend")
