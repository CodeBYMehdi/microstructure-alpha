use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray as _};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Fast 1D Gaussian Mixture Model with BIC model selection.
/// Replaces sklearn.mixture.GaussianMixture for the 1D density estimation case.
///
/// Key speedup: sklearn GMM is generic N-dim with full covariance matrices.
/// For 1D returns data, we only need scalar mean/variance per component,
/// reducing the per-iteration cost from O(D^3) to O(N*K).
#[pyclass]
pub struct RustGMM1D {
    max_components: usize,
    n_iter: usize,
    tol: f64,
    seed: u64,

    // Fitted model
    n_components: usize,
    weights: Vec<f64>,
    means: Vec<f64>,
    variances: Vec<f64>,
    fitted: bool,

    // Data stats
    data_mean: f64,
    data_std: f64,
}

#[pymethods]
impl RustGMM1D {
    #[new]
    #[pyo3(signature = (max_components=4, n_iter=100, tol=1e-4, seed=42))]
    fn new(max_components: usize, n_iter: usize, tol: f64, seed: u64) -> Self {
        Self {
            max_components,
            n_iter,
            tol,
            seed,
            n_components: 1,
            weights: vec![1.0],
            means: vec![0.0],
            variances: vec![1.0],
            fitted: false,
            data_mean: 0.0,
            data_std: 1.0,
        }
    }

    /// Fit GMM with BIC selection over K=1..max_components.
    fn fit(&mut self, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        let raw = data.as_slice()?;

        // Filter NaN/Inf
        let clean: Vec<f64> = raw.iter().filter(|x| x.is_finite()).copied().collect();
        let n = clean.len();
        if n < 10 {
            return Ok(());
        }

        self.data_mean = clean.iter().sum::<f64>() / n as f64;
        let var: f64 = clean.iter().map(|x| (x - self.data_mean).powi(2)).sum::<f64>() / n as f64;
        self.data_std = var.sqrt().max(1e-10);

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut best_bic = f64::INFINITY;
        let mut best_k = 1;
        let mut best_weights = vec![1.0];
        let mut best_means = vec![self.data_mean];
        let mut best_variances = vec![var.max(1e-10)];

        let max_k = self.max_components.min(n);

        for k in 1..=max_k {
            let (w, m, v, ll) = self.em_fit(&clean, k, &mut rng);

            // BIC = -2*LL + p*ln(n), where p = 3K-1 for 1D GMM
            let p = (3 * k - 1) as f64;
            let bic = -2.0 * ll + p * (n as f64).ln();

            if bic < best_bic {
                best_bic = bic;
                best_k = k;
                best_weights = w;
                best_means = m;
                best_variances = v;
            }
        }

        self.n_components = best_k;
        self.weights = best_weights;
        self.means = best_means;
        self.variances = best_variances;
        self.fitted = true;

        Ok(())
    }

    /// Evaluate PDF at given points.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let pts = x.as_slice().unwrap();
        let result: Vec<f64> = pts.iter().map(|&xi| self.pdf(xi)).collect();
        result.into_pyarray_bound(py)
    }

    /// Score samples: log-probability at each point.
    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let pts = x.as_slice().unwrap();
        let result: Vec<f64> = pts.iter().map(|&xi| self.pdf(xi).max(1e-300).ln()).collect();
        result.into_pyarray_bound(py)
    }

    /// Total log-likelihood of data.
    fn score(&self, data: PyReadonlyArray1<f64>) -> f64 {
        let pts = data.as_slice().unwrap();
        pts.iter().map(|&xi| self.pdf(xi).max(1e-300).ln()).sum()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.n_components
    }

    #[getter]
    fn data_mean(&self) -> f64 {
        self.data_mean
    }

    #[getter]
    fn data_std(&self) -> f64 {
        self.data_std
    }

    /// Get component means.
    fn get_means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.means.clone().into_pyarray_bound(py)
    }

    /// Get component weights.
    fn get_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.weights.clone().into_pyarray_bound(py)
    }

    /// Get component variances.
    fn get_variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.variances.clone().into_pyarray_bound(py)
    }

    /// Get model bounds (mean +/- sigma_mult * sigma).
    fn get_bounds(&self, sigma_mult: f64) -> (f64, f64) {
        if !self.fitted {
            return (
                self.data_mean - sigma_mult * self.data_std,
                self.data_mean + sigma_mult * self.data_std,
            );
        }
        // Mixture mean and variance
        let mu: f64 = self.weights.iter().zip(self.means.iter()).map(|(w, m)| w * m).sum();
        let var: f64 = self
            .weights
            .iter()
            .zip(self.means.iter())
            .zip(self.variances.iter())
            .map(|((w, m), v)| w * (v + (m - mu).powi(2)))
            .sum();
        let sigma = var.sqrt().max(1e-4);
        (mu - sigma_mult * sigma, mu + sigma_mult * sigma)
    }
}

impl RustGMM1D {
    /// EM algorithm for K-component 1D GMM. Returns (weights, means, variances, log_likelihood).
    fn em_fit(
        &self,
        data: &[f64],
        k: usize,
        rng: &mut ChaCha8Rng,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let n = data.len();

        // Initialize: K-means++ style
        let global_mean: f64 = data.iter().sum::<f64>() / n as f64;
        let global_var: f64 =
            data.iter().map(|x| (x - global_mean).powi(2)).sum::<f64>() / n as f64;

        let mut means: Vec<f64> = Vec::with_capacity(k);
        means.push(data[rng.gen_range(0..n)]);

        for _ in 1..k {
            let dists: Vec<f64> = data
                .iter()
                .map(|x| {
                    means
                        .iter()
                        .map(|m| (x - m).powi(2))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let sum_d: f64 = dists.iter().sum();
            if sum_d > 0.0 {
                let r: f64 = rng.gen::<f64>() * sum_d;
                let mut cum = 0.0;
                let mut chosen = 0;
                for (i, &d) in dists.iter().enumerate() {
                    cum += d;
                    if cum >= r {
                        chosen = i;
                        break;
                    }
                }
                means.push(data[chosen]);
            } else {
                means.push(data[rng.gen_range(0..n)]);
            }
        }

        let mut weights = vec![1.0 / k as f64; k];
        let mut variances = vec![global_var.max(1e-10); k];

        // Responsibility matrix
        let mut resp = vec![vec![0.0f64; k]; n];
        let mut ll = f64::NEG_INFINITY;

        for _iter in 0..self.n_iter {
            // E-step: compute responsibilities
            for i in 0..n {
                let mut max_log = f64::NEG_INFINITY;
                let mut log_probs = vec![0.0; k];
                for j in 0..k {
                    let lp = weights[j].max(1e-300).ln() + gaussian_log_pdf(data[i], means[j], variances[j]);
                    log_probs[j] = lp;
                    if lp > max_log {
                        max_log = lp;
                    }
                }
                // Log-sum-exp trick
                let mut sum_exp = 0.0;
                for j in 0..k {
                    resp[i][j] = (log_probs[j] - max_log).exp();
                    sum_exp += resp[i][j];
                }
                if sum_exp > 0.0 {
                    for j in 0..k {
                        resp[i][j] /= sum_exp;
                    }
                }
            }

            // M-step
            for j in 0..k {
                let nk: f64 = resp.iter().map(|r| r[j]).sum();
                if nk < 1e-10 {
                    continue;
                }
                weights[j] = nk / n as f64;
                means[j] = resp.iter().zip(data.iter()).map(|(r, &x)| r[j] * x).sum::<f64>() / nk;
                variances[j] = resp
                    .iter()
                    .zip(data.iter())
                    .map(|(r, &x)| r[j] * (x - means[j]).powi(2))
                    .sum::<f64>()
                    / nk;
                variances[j] = variances[j].max(1e-10); // Floor
            }

            // Log-likelihood
            let new_ll: f64 = (0..n)
                .map(|i| {
                    let mut sum = 0.0f64;
                    for j in 0..k {
                        sum += weights[j] * gaussian_pdf(data[i], means[j], variances[j]);
                    }
                    sum.max(1e-300).ln()
                })
                .sum();

            if (new_ll - ll).abs() < self.tol {
                ll = new_ll;
                break;
            }
            ll = new_ll;
        }

        (weights, means, variances, ll)
    }

    fn pdf(&self, x: f64) -> f64 {
        if !self.fitted {
            return gaussian_pdf(x, self.data_mean, self.data_std * self.data_std);
        }
        let mut sum = 0.0f64;
        for j in 0..self.n_components {
            sum += self.weights[j] * gaussian_pdf(x, self.means[j], self.variances[j]);
        }
        sum.max(1e-300)
    }
}

#[inline]
fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    let v = variance.max(1e-15);
    let diff = x - mean;
    let exponent = -0.5 * diff * diff / v;
    let norm = (2.0 * std::f64::consts::PI * v).sqrt();
    exponent.exp() / norm
}

#[inline]
fn gaussian_log_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    let v = variance.max(1e-15);
    let diff = x - mean;
    -0.5 * (diff * diff / v + v.ln() + (2.0 * std::f64::consts::PI).ln())
}
