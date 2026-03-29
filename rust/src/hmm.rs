use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray as _};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::linalg::{mvn_pdf, normalize_rows};

/// Online Gaussian HMM for regime detection.
/// Replaces scipy.stats.multivariate_normal.pdf with cached Cholesky MVN.
/// ~100x faster filter_step + online_update vs Python/scipy.
#[pyclass]
pub struct GaussianHMM {
    #[pyo3(get)]
    pub n_states: usize,
    #[pyo3(get)]
    pub n_features: usize,

    pub base_learning_rate: f64,
    pub learning_rate: f64,
    emission_reg: f64,

    start_prob: DVector<f64>,
    transition_matrix: DMatrix<f64>,
    means: Vec<DVector<f64>>,
    covariances: Vec<DMatrix<f64>>,

    alpha: Option<DVector<f64>>,
    prev_alpha: Option<DVector<f64>>,
    prev_state: i32,
    n_updates: usize,

    // Adaptive LR
    innovation_ema: f64,
    innovation_alpha: f64,

    // Entropy regularization
    max_row_kl: f64,
    entropy_reg_strength: f64,

    // Initialization
    obs_buffer: Vec<DVector<f64>>,
    initialized: bool,
    warmup_size: usize,

    rng: ChaCha8Rng,
}

#[pymethods]
impl GaussianHMM {
    #[new]
    #[pyo3(signature = (n_states=4, n_features=6, learning_rate=0.01, prior_strength=1.0, emission_reg=1e-4, seed=42))]
    fn new(
        n_states: usize,
        n_features: usize,
        learning_rate: f64,
        prior_strength: f64,
        emission_reg: f64,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Uniform start probability
        let start_prob = DVector::from_element(n_states, 1.0 / n_states as f64);

        // Sticky prior transition matrix
        let mut tm = DMatrix::from_element(n_states, n_states, prior_strength);
        for i in 0..n_states {
            tm[(i, i)] += prior_strength * 4.0;
        }
        normalize_rows(&mut tm);

        // Emission parameters: small random perturbation
        let mut means = Vec::with_capacity(n_states);
        for _ in 0..n_states {
            let v: Vec<f64> = (0..n_features)
                .map(|_| rng.gen::<f64>() * 0.02 - 0.01)
                .collect();
            means.push(DVector::from_vec(v));
        }

        let covariances = vec![DMatrix::identity(n_features, n_features); n_states];

        Self {
            n_states,
            n_features,
            base_learning_rate: learning_rate,
            learning_rate,
            emission_reg,
            start_prob,
            transition_matrix: tm,
            means,
            covariances,
            alpha: None,
            prev_alpha: None,
            prev_state: -1,
            n_updates: 0,
            innovation_ema: 0.0,
            innovation_alpha: 0.05,
            max_row_kl: 0.8 * (n_states.max(2) as f64).ln(),
            entropy_reg_strength: 0.03,
            obs_buffer: Vec::new(),
            initialized: false,
            warmup_size: (n_states * 10).max(50),
            rng,
        }
    }

    /// Single forward-filter step. Returns (most_likely_state, posterior_array).
    fn filter_step<'py>(
        &mut self,
        py: Python<'py>,
        observation: PyReadonlyArray1<f64>,
    ) -> PyResult<(i32, Bound<'py, PyArray1<f64>>)> {
        let obs_slice = observation.as_slice()?;
        let obs = DVector::from_column_slice(obs_slice);

        // NaN/Inf guard
        if !obs.iter().all(|v| v.is_finite()) {
            let posterior = match &self.alpha {
                Some(a) => a.as_slice().to_vec(),
                None => vec![1.0 / self.n_states as f64; self.n_states],
            };
            let state = if self.prev_state >= 0 { self.prev_state } else { 0 };
            return Ok((state, posterior.into_pyarray_bound(py)));
        }

        if !self.initialized {
            self.obs_buffer.push(obs.clone());
            if self.obs_buffer.len() >= self.warmup_size {
                self.initialize_from_data();
            } else {
                let uniform = vec![1.0 / self.n_states as f64; self.n_states];
                return Ok((0, uniform.into_pyarray_bound(py)));
            }
        }

        // Emission probabilities
        let emission_probs = self.compute_emissions(&obs);

        // Store previous alpha
        self.prev_alpha = self.alpha.clone();

        // Forward step
        match &self.alpha {
            None => {
                self.alpha = Some(self.start_prob.component_mul(&emission_probs));
            }
            Some(prev) => {
                let predicted = self.transition_matrix.transpose() * prev;
                self.alpha = Some(emission_probs.component_mul(&predicted));
            }
        }

        // Normalize
        if let Some(ref mut a) = self.alpha {
            let sum = a.sum();
            if sum < 1e-300 {
                a.fill(1.0 / self.n_states as f64);
            } else {
                *a /= sum;
            }
        }

        // Adaptive learning rate (capped at 2x)
        let alpha_ref = self.alpha.as_ref().unwrap();
        let total_emission: f64 = emission_probs.dot(alpha_ref);
        let innovation = -(total_emission.max(1e-300)).ln();
        self.innovation_ema = (1.0 - self.innovation_alpha) * self.innovation_ema
            + self.innovation_alpha * innovation;
        if self.innovation_ema > 1e-10 {
            let ratio = (innovation / self.innovation_ema).clamp(1.0, 2.0);
            self.learning_rate = self.base_learning_rate * ratio;
        }

        let most_likely = alpha_ref
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);

        self.prev_state = most_likely;
        self.n_updates += 1;

        let posterior = alpha_ref.as_slice().to_vec();
        Ok((most_likely, posterior.into_pyarray_bound(py)))
    }

    /// Incremental Baum-Welch parameter update.
    fn online_update(&mut self, observation: PyReadonlyArray1<f64>) -> PyResult<()> {
        if !self.initialized {
            return Ok(());
        }
        let obs_slice = observation.as_slice()?;
        let obs = DVector::from_column_slice(obs_slice);

        if !obs.iter().all(|v| v.is_finite()) {
            return Ok(());
        }

        let gamma = match &self.alpha {
            Some(a) => a.clone(),
            None => return Ok(()),
        };
        let lr = self.learning_rate;

        // Update emission parameters
        for k in 0..self.n_states {
            let g_k = gamma[k];
            if g_k < 1e-10 {
                continue;
            }
            let delta = &obs - &self.means[k];
            self.means[k] += lr * g_k * &delta;

            let outer = &delta * delta.transpose();
            self.covariances[k] =
                (1.0 - lr * g_k) * &self.covariances[k] + lr * g_k * outer;

            // Floor diagonal
            for i in 0..self.n_features {
                if self.covariances[k][(i, i)] < self.emission_reg {
                    self.covariances[k][(i, i)] = self.emission_reg;
                }
            }
        }

        // Update transition matrix
        if let Some(ref prev) = self.prev_alpha {
            let emission_probs = self.compute_emissions(&obs);
            let mut xi = DMatrix::zeros(self.n_states, self.n_states);
            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    xi[(i, j)] = prev[i] * self.transition_matrix[(i, j)] * emission_probs[j];
                }
            }
            let xi_sum = xi.sum();
            if xi_sum > 1e-300 {
                xi /= xi_sum;
            }
            self.transition_matrix =
                (1.0 - lr) * &self.transition_matrix + lr * xi;
            normalize_rows(&mut self.transition_matrix);
        }

        self.regularize_transition_matrix();
        Ok(())
    }

    fn get_transition_prob(&self, from_state: usize, to_state: usize) -> f64 {
        if from_state < self.n_states && to_state < self.n_states {
            self.transition_matrix[(from_state, to_state)]
        } else {
            0.0
        }
    }

    fn get_state_posterior<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        match &self.alpha {
            Some(a) => a.as_slice().to_vec().into_pyarray_bound(py),
            None => vec![1.0 / self.n_states as f64; self.n_states].into_pyarray_bound(py),
        }
    }

    fn get_mean<'py>(&self, py: Python<'py>, state: usize) -> Bound<'py, PyArray1<f64>> {
        self.means[state].as_slice().to_vec().into_pyarray_bound(py)
    }

    fn get_covariance<'py>(&self, _py: Python<'py>, state: usize) -> PyResult<PyObject> {
        // Return as nested list (numpy will auto-convert)
        let cov = &self.covariances[state];
        let py = _py;
        let numpy = py.import_bound("numpy")?;
        let rows: Vec<Vec<f64>> = (0..self.n_features)
            .map(|i| (0..self.n_features).map(|j| cov[(i, j)]).collect())
            .collect();
        let arr = numpy.call_method1("array", (rows,))?;
        Ok(arr.into())
    }

    #[getter]
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    #[getter]
    fn n_observations(&self) -> usize {
        self.n_updates
    }
}

impl GaussianHMM {
    fn compute_emissions(&self, obs: &DVector<f64>) -> DVector<f64> {
        let mut probs = DVector::from_element(self.n_states, 1e-300);
        for k in 0..self.n_states {
            let reg_cov = &self.covariances[k]
                + DMatrix::identity(self.n_features, self.n_features) * self.emission_reg;
            probs[k] = mvn_pdf(obs, &self.means[k], &reg_cov).max(1e-300);
        }
        probs
    }

    fn initialize_from_data(&mut self) {
        let n = self.obs_buffer.len();
        let d = self.n_features;

        // Global covariance
        let mean_global: DVector<f64> = self.obs_buffer.iter().fold(
            DVector::zeros(d),
            |acc, x| acc + x,
        ) / n as f64;

        let mut cov = DMatrix::zeros(d, d);
        for x in &self.obs_buffer {
            let diff = x - &mean_global;
            cov += &diff * diff.transpose();
        }
        cov /= n as f64;
        cov += DMatrix::identity(d, d) * self.emission_reg * 10.0;

        // k-means++ initialization
        let idx0 = self.rng.gen_range(0..n);
        self.means[0] = self.obs_buffer[idx0].clone();

        for k in 1..self.n_states {
            let mut dists: Vec<f64> = (0..n)
                .map(|i| {
                    (0..k)
                        .map(|j| (&self.obs_buffer[i] - &self.means[j]).norm())
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let sum_d2: f64 = dists.iter().map(|d| d * d).sum();
            if sum_d2 > 0.0 {
                for d in &mut dists {
                    *d = (*d * *d) / sum_d2;
                }
                // Weighted random choice
                let r: f64 = self.rng.gen();
                let mut cum = 0.0;
                let mut chosen = 0;
                for (i, &w) in dists.iter().enumerate() {
                    cum += w;
                    if cum >= r {
                        chosen = i;
                        break;
                    }
                }
                self.means[k] = self.obs_buffer[chosen].clone();
            } else {
                let idx = self.rng.gen_range(0..n);
                self.means[k] = self.obs_buffer[idx].clone();
            }
        }

        // Initialize covariances from global
        for k in 0..self.n_states {
            self.covariances[k] = cov.clone();
        }

        self.initialized = true;
        self.obs_buffer.clear();
    }

    fn regularize_transition_matrix(&mut self) {
        let uniform = 1.0 / self.n_states as f64;
        for i in 0..self.n_states {
            let mut kl = 0.0f64;
            for j in 0..self.n_states {
                let p = self.transition_matrix[(i, j)].max(1e-300);
                kl += p * (p * self.n_states as f64).ln();
            }
            if kl > self.max_row_kl {
                let s = self.entropy_reg_strength;
                for j in 0..self.n_states {
                    self.transition_matrix[(i, j)] =
                        (1.0 - s) * self.transition_matrix[(i, j)] + s * uniform;
                }
            }
        }
        normalize_rows(&mut self.transition_matrix);
    }
}
