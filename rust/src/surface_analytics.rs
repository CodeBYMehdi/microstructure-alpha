use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use std::collections::VecDeque;

/// Density surface tracker: computes PDF deformation velocity and bifurcation score.
/// Replaces microstructure/surface_analytics.py DensitySurfaceTracker + RegimeSurfaceTracker.
#[pyclass]
pub struct RustSurfaceAnalytics {
    // DensitySurfaceTracker state
    pdf_history: VecDeque<Vec<f64>>,
    history_size: usize,

    // RegimeSurfaceTracker state
    density_history: VecDeque<f64>,
    curvature_threshold: f64,
}

#[pymethods]
impl RustSurfaceAnalytics {
    #[new]
    #[pyo3(signature = (history_size=10, curvature_threshold=1.0))]
    fn new(history_size: usize, curvature_threshold: f64) -> Self {
        Self {
            pdf_history: VecDeque::with_capacity(history_size),
            history_size,
            density_history: VecDeque::with_capacity(history_size),
            curvature_threshold,
        }
    }

    /// Full update: returns (density_velocity, bifurcation, trajectory_z, curvature, is_collapsing).
    fn update(
        &mut self,
        pdf_values: PyReadonlyArray1<f64>,
        _mu: f64,
        _sigma: f64,
    ) -> (f64, f64, f64, f64, bool) {
        let pdf = pdf_values.as_slice().unwrap();
        if pdf.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, false);
        }

        // --- Density surface ---
        let (density_velocity, bifurcation) = self.update_density(pdf);

        // --- Regime surface ---
        let max_density = pdf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (trajectory_z, curvature, is_collapsing) = self.update_regime(max_density);

        (density_velocity, bifurcation, trajectory_z, curvature, is_collapsing)
    }
}

impl RustSurfaceAnalytics {
    fn update_density(&mut self, pdf: &[f64]) -> (f64, f64) {
        let curr = pdf.to_vec();

        if self.pdf_history.len() >= self.history_size {
            self.pdf_history.pop_front();
        }
        self.pdf_history.push_back(curr.clone());

        if self.pdf_history.len() < 2 {
            return (0.0, 0.0);
        }

        let prev = &self.pdf_history[self.pdf_history.len() - 2];
        let len = curr.len().min(prev.len());

        // d(PDF)/dt: mean absolute deformation
        let mut abs_diff_sum = 0.0f64;
        for i in 0..len {
            abs_diff_sum += (curr[i] - prev[i]).abs();
        }
        let density_velocity = abs_diff_sum / len as f64;

        // Bifurcation: std of high-density indices
        let max_val = curr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_val * 0.5;
        let high_indices: Vec<f64> = curr
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > threshold)
            .map(|(i, _)| i as f64)
            .collect();

        let bifurcation = if !high_indices.is_empty() {
            let mean: f64 = high_indices.iter().sum::<f64>() / high_indices.len() as f64;
            let var: f64 = high_indices.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / high_indices.len() as f64;
            var.sqrt()
        } else {
            0.0
        };

        (density_velocity, bifurcation)
    }

    fn update_regime(&mut self, max_density: f64) -> (f64, f64, bool) {
        if self.density_history.len() >= self.history_size {
            self.density_history.pop_front();
        }
        self.density_history.push_back(max_density);

        if self.density_history.len() < 3 {
            return (0.0, 0.0, false);
        }

        // Gradient (1st derivative) via central differences
        let n = self.density_history.len();
        let vals: Vec<f64> = self.density_history.iter().copied().collect();

        // np.gradient uses central differences internally
        // For last element: forward/backward difference
        let grad1_last = vals[n - 1] - vals[n - 2];
        let _grad1_prev = if n >= 3 {
            (vals[n - 1] - vals[n - 3]) / 2.0
        } else {
            grad1_last
        };

        // 2nd derivative at last point
        let grad2_last = if n >= 3 {
            vals[n - 1] - 2.0 * vals[n - 2] + vals[n - 3]
        } else {
            0.0
        };

        let trajectory_z = grad1_last;
        let curvature = grad2_last.abs();
        let is_collapsing = trajectory_z < 0.0 && grad2_last < -self.curvature_threshold;

        (trajectory_z, curvature, is_collapsing)
    }
}
