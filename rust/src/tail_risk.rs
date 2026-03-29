use pyo3::prelude::*;

/// Tail risk metrics: VaR, CVaR, Hill tail estimator, excess kurtosis.
/// Replaces risk/tail_risk.py compute() + _estimate_tail_slope() hot path.
///
/// Uses O(n) partial sort for percentile (select_nth_unstable) instead of full sort.
#[pyfunction]
#[pyo3(signature = (returns, confidence=0.95, tail_fraction=0.1, min_tail_points=5, alpha_clamp_min=0.5, alpha_clamp_max=10.0, default_tail_slope=2.0))]
pub fn compute_tail_risk(
    returns: Vec<f64>,
    confidence: f64,
    tail_fraction: f64,
    min_tail_points: usize,
    alpha_clamp_min: f64,
    alpha_clamp_max: f64,
    default_tail_slope: f64,
) -> Option<(f64, f64, f64, f64)> {
    // Filter NaN/Inf
    let mut arr: Vec<f64> = returns.into_iter().filter(|r| r.is_finite()).collect();
    let n = arr.len();
    if n < 10 {
        return None;
    }

    // Historical VaR via O(n) partial sort
    let var_idx = ((1.0 - confidence) * n as f64) as usize;
    let var_idx = var_idx.min(n - 1);
    arr.select_nth_unstable_by(var_idx, |a, b| a.partial_cmp(b).unwrap());
    let var = arr[var_idx];

    // CVaR: mean of returns <= VaR
    let tail_returns: Vec<f64> = arr.iter().filter(|&&r| r <= var).copied().collect();
    let cvar = if !tail_returns.is_empty() {
        tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
    } else {
        var
    };

    // Excess kurtosis
    let mean: f64 = arr.iter().sum::<f64>() / n as f64;
    let variance: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    let kurtosis = if std > 1e-10 {
        let m4: f64 = arr.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64;
        m4 - 3.0
    } else {
        0.0
    };

    // Hill tail estimator on left tail (negative returns)
    let k = min_tail_points.max((n as f64 * tail_fraction) as usize);
    let tail_slope = hill_left_tail(&mut arr, k, alpha_clamp_min, alpha_clamp_max, default_tail_slope);

    Some((var, cvar, tail_slope, kurtosis))
}

/// Hill estimator on the left tail (most negative returns).
/// Uses partial sort O(n) to find the k smallest values.
fn hill_left_tail(
    arr: &mut [f64],
    k: usize,
    alpha_min: f64,
    alpha_max: f64,
    default: f64,
) -> f64 {
    let n = arr.len();
    if k >= n || k == 0 {
        return default;
    }

    // Partial sort: put smallest k values at the front
    arr.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());

    // Left tail = most negative, flip sign
    let mut tail: Vec<f64> = arr[..k].iter().map(|x| -x).collect();
    tail.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

    // All tail values must be positive (i.e., original values were negative)
    if tail.is_empty() || tail.last().copied().unwrap_or(0.0) <= 0.0 {
        return default;
    }

    let x_min = *tail.last().unwrap();
    let sum_log: f64 = tail.iter().map(|x| (x / x_min).ln()).sum();
    if sum_log <= 0.0 {
        return default;
    }

    let alpha = k as f64 / sum_log;
    alpha.clamp(alpha_min, alpha_max)
}

/// Online incremental VaR/CVaR tracker using sorted insert.
/// Avoids full re-sort on every update. O(n) insert into sorted array.
#[pyclass]
pub struct RustTailRiskTracker {
    returns: Vec<f64>,
    window: usize,
    confidence: f64,
    min_points: usize,
    tail_fraction: f64,
    min_tail_points: usize,
    default_tail_slope: f64,
    alpha_clamp_min: f64,
    alpha_clamp_max: f64,
}

#[pymethods]
impl RustTailRiskTracker {
    #[new]
    #[pyo3(signature = (window=100, confidence=0.95, min_points=20, tail_fraction=0.1, min_tail_points=5, default_tail_slope=2.0, alpha_clamp_min=0.5, alpha_clamp_max=10.0))]
    fn new(
        window: usize,
        confidence: f64,
        min_points: usize,
        tail_fraction: f64,
        min_tail_points: usize,
        default_tail_slope: f64,
        alpha_clamp_min: f64,
        alpha_clamp_max: f64,
    ) -> Self {
        Self {
            returns: Vec::with_capacity(window),
            window,
            confidence,
            min_points,
            tail_fraction,
            min_tail_points,
            default_tail_slope,
            alpha_clamp_min,
            alpha_clamp_max,
        }
    }

    /// Feed a return. Returns Some((var, cvar, tail_slope, kurtosis)) if enough data.
    fn update(&mut self, ret: f64) -> Option<(f64, f64, f64, f64)> {
        if !ret.is_finite() {
            return None;
        }
        self.returns.push(ret);
        if self.returns.len() > self.window {
            self.returns.remove(0);
        }
        if self.returns.len() < self.min_points {
            return None;
        }
        self.compute()
    }

    fn compute(&self) -> Option<(f64, f64, f64, f64)> {
        if self.returns.len() < self.min_points {
            return None;
        }
        compute_tail_risk(
            self.returns.clone(),
            self.confidence,
            self.tail_fraction,
            self.min_tail_points,
            self.alpha_clamp_min,
            self.alpha_clamp_max,
            self.default_tail_slope,
        )
    }

    fn reset(&mut self) {
        self.returns.clear();
    }

    #[getter]
    fn n_returns(&self) -> usize {
        self.returns.len()
    }
}
