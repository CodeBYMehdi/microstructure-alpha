use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use statrs::function::gamma::digamma;

/// High-performance entropy calculator.
/// KNN entropy uses sorted-array nearest-neighbor (O(n log n)) instead of
/// rebuilding a cKDTree on every call. For 1D data, sorting + neighbor lookup
/// is faster than any spatial tree.
#[pyclass]
pub struct EntropyCalculator;

#[pymethods]
impl EntropyCalculator {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Differential entropy from PDF values.
    #[staticmethod]
    fn compute_from_pdf(pdf_values: PyReadonlyArray1<f64>, dx: f64) -> f64 {
        let arr = pdf_values.as_slice().unwrap();
        compute_from_pdf(arr, dx)
    }

    /// KL divergence D_KL(p || q).
    #[staticmethod]
    fn compute_kl_divergence(
        p: PyReadonlyArray1<f64>,
        q: PyReadonlyArray1<f64>,
        dx: f64,
    ) -> f64 {
        let p_arr = p.as_slice().unwrap();
        let q_arr = q.as_slice().unwrap();
        compute_kl_divergence(p_arr, q_arr, dx)
    }

    /// Entropy from samples. Default uses KNN for n>=20, Gaussian fallback otherwise.
    #[staticmethod]
    #[pyo3(signature = (data, method="vasicek"))]
    fn compute_from_samples(data: PyReadonlyArray1<f64>, method: &str) -> f64 {
        let arr = data.as_slice().unwrap();
        let sigma = std_dev(arr);
        if sigma == 0.0 {
            return 0.0;
        }
        if method == "knn" || arr.len() >= 20 {
            return kozachenko_leonenko(arr, 3);
        }
        // Gaussian fallback for very small samples
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * sigma * sigma).ln()
    }

    /// Kozachenko-Leonenko KNN entropy estimator.
    #[staticmethod]
    #[pyo3(signature = (data, k=3))]
    fn compute_kozachenko_leonenko(data: PyReadonlyArray1<f64>, k: usize) -> f64 {
        let arr = data.as_slice().unwrap();
        kozachenko_leonenko(arr, k)
    }
}

fn std_dev(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

pub fn compute_from_pdf(pdf_values: &[f64], dx: f64) -> f64 {
    if pdf_values.is_empty() {
        return f64::NAN;
    }
    let mut has_positive = false;
    let mut sum = 0.0f64;
    for &p in pdf_values {
        if p > 0.0 {
            has_positive = true;
            sum -= p * p.ln();
        }
    }
    if !has_positive {
        return f64::NAN;
    }
    sum * dx
}

pub fn compute_kl_divergence(p: &[f64], q: &[f64], dx: f64) -> f64 {
    let eps = 1e-30;
    let mut sum = 0.0f64;
    let n = p.len().min(q.len());
    for i in 0..n {
        if p[i] > eps && q[i] > eps {
            let ratio = (p[i] / q[i]).clamp(1e-10, 1e10);
            sum += p[i] * ratio.ln();
        }
    }
    sum * dx
}

/// Kozachenko-Leonenko KNN entropy for 1D data.
///
/// For 1D, sorting the data gives O(n log n) nearest-neighbor queries
/// (the k-th nearest neighbor of sorted[i] is at sorted[i-k] or sorted[i+k]).
/// This is 10-50x faster than building a cKDTree per call.
pub fn kozachenko_leonenko(data: &[f64], k: usize) -> f64 {
    let n = data.len();
    if n <= k {
        return 0.0;
    }

    // Sort for O(1) per-point neighbor lookup
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let eps = 1e-10;
    let mut sum_log = 0.0f64;

    for i in 0..n {
        // Find k-th nearest neighbor distance in sorted array
        // Check both directions and take the k-th closest
        let mut dists: Vec<f64> = Vec::with_capacity(2 * k);
        // Left neighbors
        for j in 1..=k.min(i) {
            dists.push((sorted[i] - sorted[i - j]).abs());
        }
        // Right neighbors
        for j in 1..=k.min(n - 1 - i) {
            dists.push((sorted[i + j] - sorted[i]).abs());
        }
        dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let r_k = if dists.len() >= k {
            dists[k - 1].max(eps)
        } else if !dists.is_empty() {
            dists.last().copied().unwrap().max(eps)
        } else {
            eps
        };

        sum_log += r_k.ln();
    }

    // H = -psi(k) + psi(n) + log(c_d) + (d/n) * sum(log(r_k))
    // For d=1, c_d = 2 (volume of unit ball in 1D = length of [-1,1])
    let c_d: f64 = 2.0;
    let d: f64 = 1.0;

    let psi_n = digamma(n as f64);
    let psi_k = digamma(k as f64);

    psi_n - psi_k + c_d.ln() + (d / n as f64) * sum_log
}
