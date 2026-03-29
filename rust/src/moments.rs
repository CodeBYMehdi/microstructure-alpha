use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

/// Microstructure moments result — matches Python MicrostructureMoments dataclass.
#[pyclass]
#[derive(Clone)]
pub struct MicrostructureMoments {
    #[pyo3(get)]
    pub mu: f64,
    #[pyo3(get)]
    pub sigma: f64,
    #[pyo3(get)]
    pub skew: f64,
    #[pyo3(get)]
    pub kurtosis: f64,
    #[pyo3(get)]
    pub tail_slope: f64,
}

/// High-performance moments calculator.
/// Replaces scipy.stats.skew/kurtosis calls with direct computation.
/// Hill estimator uses partial sort O(n) instead of full sort.
#[pyclass]
pub struct MomentsCalculator;

#[pymethods]
impl MomentsCalculator {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute all 5 microstructure moments from a return array.
    /// Accepts numpy array by reference (zero-copy read).
    #[staticmethod]
    fn compute(data: PyReadonlyArray1<f64>) -> MicrostructureMoments {
        let arr = data.as_slice().unwrap();
        compute_moments(arr)
    }
}

/// Pure Rust computation — no Python/GIL needed.
pub fn compute_moments(data: &[f64]) -> MicrostructureMoments {
    let n = data.len();
    if n < 10 {
        return MicrostructureMoments {
            mu: 0.0,
            sigma: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
            tail_slope: 0.0,
        };
    }

    // Mean
    let sum: f64 = data.iter().sum();
    let mu = sum / n as f64;

    // Variance (ddof=1)
    let ss: f64 = data.iter().map(|x| (x - mu).powi(2)).sum();
    let variance = ss / (n - 1) as f64;
    let sigma = variance.sqrt();

    if sigma == 0.0 {
        return MicrostructureMoments {
            mu,
            sigma: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
            tail_slope: 0.0,
        };
    }

    // Skewness (Fisher, bias-corrected)
    let m3: f64 = data.iter().map(|x| ((x - mu) / sigma).powi(3)).sum();
    let skew = m3 / n as f64;

    // Excess kurtosis (Fisher, normal=0)
    let m4: f64 = data.iter().map(|x| ((x - mu) / sigma).powi(4)).sum();
    let kurtosis = m4 / n as f64 - 3.0;

    // Hill tail estimator on top 10% of absolute returns
    let tail_slope = hill_estimator(data);

    MicrostructureMoments {
        mu,
        sigma,
        skew,
        kurtosis,
        tail_slope,
    }
}

/// Hill estimator for tail index.
/// Uses partial sort (O(n)) to find top 10% of absolute returns.
/// Returns 1/alpha where alpha is the Pareto tail exponent.
fn hill_estimator(data: &[f64]) -> f64 {
    let n = data.len();
    let n_tail = (n / 10).max(5);
    if n_tail >= n {
        return 0.0;
    }

    // Get absolute values
    let mut abs_vals: Vec<f64> = data.iter().map(|x| x.abs()).collect();

    // Partial sort: select the n_tail-th largest element
    // This is O(n) on average via select_nth_unstable
    let pivot_idx = n - n_tail;
    abs_vals.select_nth_unstable_by(pivot_idx, |a, b| a.partial_cmp(b).unwrap());

    let x_min = abs_vals[pivot_idx];
    if x_min < 1e-15 {
        return 0.0;
    }

    // Sum log(x_i / x_min) for the top n_tail values
    let mut sum_log = 0.0f64;
    let mut count = 0usize;
    for &v in &abs_vals[pivot_idx..] {
        if v > 0.0 {
            sum_log += (v / x_min).ln();
            count += 1;
        }
    }

    if sum_log < 1e-10 || count == 0 {
        return 0.0;
    }

    let hill_alpha = count as f64 / sum_log;
    1.0 / hill_alpha.max(0.1)
}
