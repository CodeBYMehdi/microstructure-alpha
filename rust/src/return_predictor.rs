use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray as _};
use std::collections::VecDeque;

/// Online SGD linear regressor with exponential forgetting.
/// Replaces alpha/return_predictor.py hot path: predict + update per window.
///
/// All numerical ops in Rust: normalization, dot product, gradient, Welford stats.
#[pyclass]
pub struct RustReturnPredictor {
    n_features: usize,
    lr: f64,
    l2_lambda: f64,
    forgetting: f64,
    min_samples: usize,

    weights: Vec<f64>,
    bias: f64,

    // Running stats (Welford online)
    feature_mean: Vec<f64>,
    feature_var: Vec<f64>,
    target_mean: f64,
    target_var: f64,
    n_updates: usize,

    // Error tracking
    errors: VecDeque<f64>,
    max_history: usize,

    // Pending for update
    pending_features: Option<Vec<f64>>,
    pending_prediction: Option<f64>,
}

#[pymethods]
impl RustReturnPredictor {
    #[new]
    #[pyo3(signature = (n_features=23, learning_rate=0.001, l2_lambda=0.01, forgetting_factor=0.995, min_samples=30, max_history=500))]
    fn new(
        n_features: usize,
        learning_rate: f64,
        l2_lambda: f64,
        forgetting_factor: f64,
        min_samples: usize,
        max_history: usize,
    ) -> Self {
        Self {
            n_features,
            lr: learning_rate,
            l2_lambda,
            forgetting: forgetting_factor,
            min_samples,
            weights: vec![0.0; n_features],
            bias: 0.0,
            feature_mean: vec![0.0; n_features],
            feature_var: vec![1.0; n_features],
            target_mean: 0.0,
            target_var: 1.0,
            n_updates: 0,
            errors: VecDeque::with_capacity(max_history),
            max_history,
            pending_features: None,
            pending_prediction: None,
        }
    }

    /// Predict from feature array. Returns (expected_return, confidence, prediction_std, is_valid).
    fn predict(&mut self, features: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64, bool)> {
        let feat = features.as_slice()?;
        if feat.len() != self.n_features {
            return Ok((0.0, 0.0, 0.0, false));
        }

        // Normalize
        let x: Vec<f64> = (0..self.n_features)
            .map(|i| (feat[i] - self.feature_mean[i]) / (self.feature_var[i] + 1e-8).sqrt())
            .collect();

        // Dot product
        let raw_pred: f64 = x.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum::<f64>() + self.bias;

        // Confidence
        let (confidence, pred_std, is_valid) = if self.n_updates < self.min_samples {
            (0.0, 1.0, false)
        } else if self.errors.len() > 5 {
            let n = self.errors.len() as f64;
            let mean_err: f64 = self.errors.iter().sum::<f64>() / n;
            let var_err: f64 = self.errors.iter().map(|e| (e - mean_err).powi(2)).sum::<f64>() / n;
            let std_err = var_err.sqrt();
            let target_std = self.target_var.max(1e-10).sqrt();
            let error_ratio = if target_std > 0.0 { std_err / target_std } else { 1.0 };
            let conf = (1.0 / (1.0 + error_ratio)).clamp(0.0, 1.0);
            (conf, std_err, true)
        } else {
            (0.0, 1.0, true)
        };

        self.pending_features = Some(x);
        self.pending_prediction = Some(raw_pred);

        Ok((raw_pred, confidence, pred_std, is_valid))
    }

    /// SGD update with realized return.
    fn update(&mut self, actual_return: f64) {
        let (x, pred) = match (self.pending_features.take(), self.pending_prediction.take()) {
            (Some(x), Some(p)) => (x, p),
            _ => return,
        };

        let error = actual_return - pred;

        // Track error
        if self.errors.len() >= self.max_history {
            self.errors.pop_front();
        }
        self.errors.push_back(error);

        // SGD with L2
        let adaptive_lr = self.lr / (1.0 + 0.001 * self.n_updates as f64);
        for i in 0..self.n_features {
            let grad = -error * x[i] + self.l2_lambda * self.weights[i];
            self.weights[i] -= adaptive_lr * grad;
            self.weights[i] *= self.forgetting;
        }
        self.bias -= adaptive_lr * (-error);

        // Update running stats (EMA)
        let alpha = 0.01;
        if self.n_updates == 0 {
            self.feature_mean = x.clone();
            self.feature_var = vec![1.0; self.n_features];
            self.target_mean = actual_return;
            self.target_var = 1.0;
        } else {
            for i in 0..self.n_features {
                let delta = x[i] - self.feature_mean[i];
                self.feature_mean[i] += alpha * delta;
                self.feature_var[i] = (1.0 - alpha) * self.feature_var[i] + alpha * delta * delta;
            }
            let t_delta = actual_return - self.target_mean;
            self.target_mean += alpha * t_delta;
            self.target_var = (1.0 - alpha) * self.target_var + alpha * t_delta * t_delta;
        }

        self.n_updates += 1;
    }

    /// Get |weights| normalized as feature importance.
    fn get_feature_importance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let abs_w: Vec<f64> = self.weights.iter().map(|w| w.abs()).collect();
        let total: f64 = abs_w.iter().sum();
        let importance: Vec<f64> = if total > 0.0 {
            abs_w.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / self.n_features as f64; self.n_features]
        };
        importance.into_pyarray_bound(py)
    }

    #[getter]
    fn n_updates(&self) -> usize {
        self.n_updates
    }

    fn reset(&mut self) {
        self.weights = vec![0.0; self.n_features];
        self.bias = 0.0;
        self.feature_mean = vec![0.0; self.n_features];
        self.feature_var = vec![1.0; self.n_features];
        self.n_updates = 0;
        self.errors.clear();
        self.pending_features = None;
        self.pending_prediction = None;
    }
}
