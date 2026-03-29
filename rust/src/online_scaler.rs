use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray as _};

/// Online StandardScaler (Welford incremental mean/variance).
/// Replaces sklearn.preprocessing.StandardScaler.partial_fit + transform
/// which is a bottleneck in HMMRegimeAdapter._scale_observation.
///
/// sklearn's partial_fit recalculates from scratch each call;
/// this maintains running stats in O(1) per observation.
#[pyclass]
pub struct RustOnlineScaler {
    n_features: usize,
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>, // Sum of squared differences (Welford)
}

#[pymethods]
impl RustOnlineScaler {
    #[new]
    #[pyo3(signature = (n_features=6))]
    fn new(n_features: usize) -> Self {
        Self {
            n_features,
            count: 0,
            mean: vec![0.0; n_features],
            m2: vec![0.0; n_features],
        }
    }

    /// Partial fit + transform in one call (the hot path).
    /// Returns the scaled observation as a numpy array.
    fn partial_fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        observation: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let obs = observation.as_slice()?;
        if obs.len() != self.n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} features, got {}", self.n_features, obs.len()),
            ));
        }

        // Welford online update
        self.count += 1;
        let n = self.count as f64;

        for i in 0..self.n_features {
            let delta = obs[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = obs[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }

        // Transform: (x - mean) / std
        let result: Vec<f64> = (0..self.n_features)
            .map(|i| {
                let var = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f64
                } else {
                    1.0
                };
                let std = var.sqrt().max(1e-10);
                (obs[i] - self.mean[i]) / std
            })
            .collect();

        Ok(result.into_pyarray_bound(py))
    }

    /// Inverse transform: x_orig = x_scaled * std + mean
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        scaled: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = scaled.as_slice()?;
        let result: Vec<f64> = (0..self.n_features)
            .map(|i| {
                let var = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f64
                } else {
                    1.0
                };
                let std = var.sqrt().max(1e-10);
                x[i] * std + self.mean[i]
            })
            .collect();
        Ok(result.into_pyarray_bound(py))
    }

    /// Get current scale (std) for inverse_scale_std.
    fn get_scale<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let scale: Vec<f64> = (0..self.n_features)
            .map(|i| {
                let var = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f64
                } else {
                    1.0
                };
                var.sqrt().max(1e-10)
            })
            .collect();
        scale.into_pyarray_bound(py)
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.count > 0
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.count
    }
}
