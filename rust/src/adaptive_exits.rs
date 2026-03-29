use pyo3::prelude::*;

/// ATR (Average True Range) tracker.
/// Computes ATR from micro-bars (N-tick bars) with exponential weighting.
#[pyclass]
pub struct RustATRTracker {
    period: usize,
    micro_window: usize,
    prices: Vec<f64>,
    true_ranges: Vec<f64>,
    atr: f64,
    prev_close: f64,
}

#[pymethods]
impl RustATRTracker {
    #[new]
    #[pyo3(signature = (period=14, micro_window=10))]
    fn new(period: usize, micro_window: usize) -> Self {
        Self {
            period,
            micro_window,
            prices: Vec::new(),
            true_ranges: Vec::new(),
            atr: 0.0,
            prev_close: 0.0,
        }
    }

    fn update(&mut self, price: f64) {
        self.prices.push(price);

        if self.prices.len() >= self.micro_window {
            let start = self.prices.len() - self.micro_window;
            let window = &self.prices[start..];
            let high = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let low = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let close = *window.last().unwrap();

            let tr = if self.prev_close > 0.0 {
                (high - low)
                    .max((high - self.prev_close).abs())
                    .max((low - self.prev_close).abs())
            } else {
                high - low
            };

            self.true_ranges.push(tr);
            self.prev_close = close;

            // Keep bounded
            if self.true_ranges.len() > self.period * 3 {
                let keep_from = self.true_ranges.len() - self.period * 2;
                self.true_ranges = self.true_ranges[keep_from..].to_vec();
            }

            // Compute ATR with exponential weighting
            if self.true_ranges.len() >= self.period {
                let n = self.true_ranges.len().min(self.period * 2);
                let recent = &self.true_ranges[self.true_ranges.len() - n..];
                let mut weights: Vec<f64> = (0..n)
                    .map(|i| (-1.0 + i as f64 / (n - 1).max(1) as f64).exp())
                    .collect();
                let w_sum: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= w_sum;
                }
                self.atr = recent
                    .iter()
                    .zip(weights.iter())
                    .map(|(r, w)| r * w)
                    .sum();
            } else if !self.true_ranges.is_empty() {
                self.atr =
                    self.true_ranges.iter().sum::<f64>() / self.true_ranges.len() as f64;
            }

            // Trim price history
            if self.prices.len() > self.micro_window * 5 {
                let keep_from = self.prices.len() - self.micro_window * 3;
                self.prices = self.prices[keep_from..].to_vec();
            }
        }
    }

    #[getter]
    fn atr(&self) -> f64 {
        self.atr
    }

    fn atr_pct(&self, reference_price: f64) -> f64 {
        if reference_price <= 0.0 {
            return 0.0;
        }
        self.atr / reference_price
    }

    #[getter]
    fn is_ready(&self) -> bool {
        self.true_ranges.len() >= self.period
    }
}

/// Compute adaptive exit parameters from state/regime features.
/// Returns (stop_loss_pct, take_profit_pct, trailing_stop_pct, max_hold, min_rr, atr_value).
///
/// This is the pure-math hot path extracted from AdaptiveExitEngine.compute_exit_params.
/// Config values are passed in rather than read from the global singleton.
#[pyfunction]
#[pyo3(signature = (
    sigma, kurtosis, skew, entropy,
    atr_pct, atr_ready,
    current_drawdown_pct, transition_strength, regime_win_rate,
    autocorrelation, transition_volatility,
    stop_sigma_mult, tp_sigma_mult, trail_sigma_mult,
    kurtosis_stop_scale, kurtosis_trail_scale, skew_tp_scale,
    entropy_hold_scale, strength_tp_bonus,
    min_stop_pct, min_tp_pct,
    max_hold_base,
    recent_sigmas,
))]
pub fn compute_exit_params_rust(
    sigma: f64,
    kurtosis: f64,
    skew: f64,
    entropy: f64,
    atr_pct: f64,
    atr_ready: bool,
    current_drawdown_pct: f64,
    transition_strength: f64,
    regime_win_rate: f64,
    autocorrelation: f64,
    transition_volatility: f64,
    // Config params
    stop_sigma_mult: f64,
    tp_sigma_mult: f64,
    trail_sigma_mult: f64,
    kurtosis_stop_scale: f64,
    kurtosis_trail_scale: f64,
    skew_tp_scale: f64,
    entropy_hold_scale: f64,
    strength_tp_bonus: f64,
    min_stop_pct: f64,
    min_tp_pct: f64,
    max_hold_base: i32,
    recent_sigmas: Vec<f64>,
) -> (f64, f64, f64, i32, f64, f64) {
    let sigma_clean = if sigma.is_finite() { sigma.abs().max(1e-9) } else { 1e-6 };
    let kurt_clean = if kurtosis.is_finite() { kurtosis } else { 3.0 };
    let skew_clean = if skew.is_finite() { skew } else { 0.0 };
    let entropy_clean = if entropy.is_finite() { entropy } else { 1.0 };

    // EMA sigma
    let ema_sigma = if !recent_sigmas.is_empty() {
        let n = recent_sigmas.len();
        let mut weights: Vec<f64> = (0..n)
            .map(|i| (-1.0 + i as f64 / (n - 1).max(1) as f64).exp())
            .collect();
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }
        recent_sigmas.iter().zip(weights.iter()).map(|(s, w)| s * w).sum()
    } else {
        sigma_clean
    };

    // Base volatility: ATR preferred, sigma fallback
    let base_vol = if atr_ready && atr_pct > 0.0 {
        let sigma_boost = if atr_pct > 0.0 {
            ((ema_sigma / atr_pct - 1.0).max(0.0)) * 0.3
        } else {
            0.0
        };
        atr_pct * (1.0 + sigma_boost)
    } else {
        ema_sigma
    };

    // Tail risk factor
    let excess_kurt = (kurt_clean - 3.0).max(0.0);
    let tail_factor = 1.0 + excess_kurt * kurtosis_stop_scale;

    // Skew factor
    let skew_tp_factor = 1.0 + skew_clean.abs() * skew_tp_scale;

    // Drawdown tightening
    let dd_factor = if current_drawdown_pct > 0.01 {
        (1.0 - current_drawdown_pct * 5.0).max(0.5)
    } else {
        1.0
    };

    // Transition strength bonus
    let strength_bonus = 1.0 + transition_strength * strength_tp_bonus;

    // Regime-type scaling
    let (regime_stop_scale, regime_tp_scale, mut regime_trail_scale) = if autocorrelation > 0.05 {
        (0.7, 0.8, 0.6)
    } else if autocorrelation < -0.05 {
        (1.2, 1.3, 1.2)
    } else {
        (1.0, 1.0, 1.0)
    };

    let mut regime_stop_scale_final = regime_stop_scale;
    if transition_volatility > 0.15 {
        let instability_scale = (1.0 - transition_volatility * 0.8).max(0.6);
        regime_trail_scale *= instability_scale;
        regime_stop_scale_final *= instability_scale.max(0.7);
    }

    // Stop loss
    let tail_capped = tail_factor.min(1.2);
    let raw_stop = base_vol * stop_sigma_mult * tail_capped * 0.8 * regime_stop_scale_final;
    let stop_loss = min_stop_pct.max(raw_stop * dd_factor);

    // Take profit
    let raw_tp = base_vol * tp_sigma_mult * skew_tp_factor.max(0.8) * strength_bonus * regime_tp_scale;
    let mut take_profit = min_tp_pct.max(raw_tp);

    // Trailing stop
    let trail_factor = 1.0 + excess_kurt * kurtosis_trail_scale;
    let raw_trail = base_vol * trail_sigma_mult * trail_factor * regime_trail_scale;
    let trailing = min_stop_pct.max(raw_trail * dd_factor);

    // Max hold
    let entropy_bonus = (entropy_clean * entropy_hold_scale * 0.5).max(0.0) as i32;
    let max_hold = (5i32).max(max_hold_base + entropy_bonus).min(max_hold_base * 2);

    // Dynamic R:R
    let base_rr = if regime_win_rate >= 0.6 {
        1.0
    } else if regime_win_rate >= 0.45 {
        1.2
    } else {
        1.5
    };
    let mut rr_ratio = base_rr * (1.0 - transition_strength * 0.2).max(0.8);
    if current_drawdown_pct > 0.005 {
        rr_ratio *= 1.0 + current_drawdown_pct * 5.0;
    }
    rr_ratio = rr_ratio.clamp(0.5, 3.0);

    // Enforce minimum TP from R:R
    let min_tp_from_rr = stop_loss * rr_ratio;
    if take_profit < min_tp_from_rr {
        take_profit = min_tp_from_rr;
    }

    (stop_loss, take_profit, trailing, max_hold, rr_ratio, base_vol)
}
