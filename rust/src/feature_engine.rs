use pyo3::prelude::*;
use std::collections::HashMap;

use crate::ring_buffer::RingBuffer;

/// High-performance feature computation engine.
/// Replaces Python FeatureEngine hot-path: autocorrelation, momentum,
/// mean-reversion Z-score, VPIN, vol-of-vol.
#[pyclass]
pub struct RustFeatureEngine {
    window: usize,

    returns: RingBuffer<f64>,
    prices: RingBuffer<f64>,
    volumes: RingBuffer<f64>,
    spreads: RingBuffer<f64>,
    sigma_history: RingBuffer<f64>,
    ofi_history: RingBuffer<f64>,
    timestamps: RingBuffer<f64>,

    // VPIN state
    vpin_buckets: RingBuffer<f64>,
    vpin_current_buy_vol: f64,
    vpin_current_sell_vol: f64,
    vpin_bucket_size: f64,
    vpin_bucket_filled: f64,

    last_price: f64,
    tick_count: usize,
}

#[pymethods]
impl RustFeatureEngine {
    #[new]
    #[pyo3(signature = (window=200, vol_lookback=50))]
    fn new(window: usize, vol_lookback: usize) -> Self {
        Self {
            window,
            returns: RingBuffer::new(window * 2),
            prices: RingBuffer::new(window * 2),
            volumes: RingBuffer::new(window),
            spreads: RingBuffer::new(window),
            sigma_history: RingBuffer::new(vol_lookback),
            ofi_history: RingBuffer::new(window),
            timestamps: RingBuffer::new(window),
            vpin_buckets: RingBuffer::new(50),
            vpin_current_buy_vol: 0.0,
            vpin_current_sell_vol: 0.0,
            vpin_bucket_size: 0.0,
            vpin_bucket_filled: 0.0,
            last_price: 0.0,
            tick_count: 0,
        }
    }

    /// Feed a new tick.
    fn update(
        &mut self,
        price: f64,
        volume: f64,
        bid: f64,
        ask: f64,
        ofi: f64,
        timestamp: f64,
    ) {
        self.tick_count += 1;
        self.prices.push(price);
        self.volumes.push(volume);
        self.timestamps.push(timestamp);
        self.ofi_history.push(ofi);

        // Return
        if self.last_price > 0.0 {
            let ret = (price / self.last_price).ln();
            if ret.is_finite() {
                self.returns.push(ret);
            }
        }
        self.last_price = price;

        // Spread
        if ask > bid && bid > 0.0 {
            let spread_pct = (ask - bid) / ((ask + bid) / 2.0);
            self.spreads.push(spread_pct);
        }

        // VPIN
        self.update_vpin(price, volume);
    }

    /// Compute features. Returns a dict of feature_name -> value.
    /// State vector fields (mu, sigma, skew, etc.) and regime context
    /// are passed as arguments since they come from Python objects.
    #[pyo3(signature = (sigma=0.0))]
    fn compute_derived(&mut self, sigma: f64) -> HashMap<String, f64> {
        let mut features: HashMap<String, f64> = HashMap::new();

        // OFI sum
        features.insert("ofi".into(), self.ofi_history.sum());

        // VPIN
        features.insert(
            "vpin".into(),
            if self.vpin_buckets.is_empty() {
                0.0
            } else {
                self.vpin_buckets.mean()
            },
        );

        // Trade flow toxicity
        let total_vol = if self.volumes.is_empty() {
            1.0
        } else {
            self.volumes.sum().max(1.0)
        };
        features.insert(
            "trade_flow_toxicity".into(),
            self.ofi_history.sum() / total_vol,
        );

        // Autocorrelations (lagged to prevent lookahead)
        let returns: Vec<f64> = self.returns.to_vec();
        let lagged = if returns.len() > 1 {
            &returns[..returns.len() - 1]
        } else {
            &returns[..]
        };

        if lagged.len() >= 20 {
            features.insert("autocorr_1".into(), autocorr(lagged, 1));
            features.insert("autocorr_5".into(), autocorr(lagged, 5));
            features.insert("autocorr_10".into(), autocorr(lagged, 10));
        } else {
            features.insert("autocorr_1".into(), 0.0);
            features.insert("autocorr_5".into(), 0.0);
            features.insert("autocorr_10".into(), 0.0);
        }

        // Momentum (lagged)
        if lagged.len() >= 50 {
            features.insert(
                "momentum_short".into(),
                lagged[lagged.len() - 50..].iter().sum(),
            );
        } else {
            features.insert("momentum_short".into(), 0.0);
        }

        if lagged.len() >= 200 {
            features.insert(
                "momentum_long".into(),
                lagged[lagged.len() - 200..].iter().sum(),
            );
        } else {
            features.insert("momentum_long".into(), 0.0);
        }

        // Mean reversion Z-score
        let prices: Vec<f64> = self.prices.to_vec();
        if prices.len() >= 50 {
            let start = prices.len().saturating_sub(200);
            let window = &prices[start..];
            let p_mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let p_std = {
                let ss: f64 = window.iter().map(|x| (x - p_mean).powi(2)).sum();
                (ss / window.len() as f64).sqrt()
            };
            if p_std > 0.0 {
                features.insert(
                    "mean_reversion".into(),
                    (*prices.last().unwrap() - p_mean) / p_std,
                );
            } else {
                features.insert("mean_reversion".into(), 0.0);
            }
        } else {
            features.insert("mean_reversion".into(), 0.0);
        }

        // Vol of vol
        self.sigma_history.push(sigma);
        if self.sigma_history.len() >= 10 {
            features.insert("vol_of_vol".into(), self.sigma_history.std(0));
        } else {
            features.insert("vol_of_vol".into(), 0.0);
        }

        // Spread
        if !self.spreads.is_empty() {
            let spread_tail: Vec<f64> = self.spreads.tail(20).copied().collect();
            features.insert(
                "spread_pct".into(),
                spread_tail.iter().sum::<f64>() / spread_tail.len() as f64,
            );
        } else {
            features.insert("spread_pct".into(), 0.0);
        }

        // Volume intensity
        let ts_vec = self.timestamps.to_vec();
        if ts_vec.len() >= 2 {
            let dt = ts_vec.last().unwrap() - ts_vec.first().unwrap();
            if dt > 0.0 {
                features.insert("volume_intensity".into(), self.volumes.sum() / dt);
            } else {
                features.insert("volume_intensity".into(), 0.0);
            }
        } else {
            features.insert("volume_intensity".into(), 0.0);
        }

        features
    }

    fn reset(&mut self) {
        self.returns.clear();
        self.prices.clear();
        self.volumes.clear();
        self.spreads.clear();
        self.sigma_history.clear();
        self.ofi_history.clear();
        self.timestamps.clear();
        self.vpin_buckets.clear();
        self.last_price = 0.0;
        self.tick_count = 0;
    }

    #[getter]
    fn tick_count(&self) -> usize {
        self.tick_count
    }
}

impl RustFeatureEngine {
    fn update_vpin(&mut self, price: f64, volume: f64) {
        if self.vpin_bucket_size <= 0.0 {
            if self.volumes.len() >= 100 {
                self.vpin_bucket_size = self.volumes.mean() * 50.0;
            }
            return;
        }

        // Tick rule classification
        if self.prices.len() >= 2 {
            let prev = self.prices.to_vec();
            let prev_price = prev[prev.len() - 2];
            if price > prev_price {
                self.vpin_current_buy_vol += volume;
            } else {
                self.vpin_current_sell_vol += volume;
            }
        }

        self.vpin_bucket_filled += volume;

        if self.vpin_bucket_filled >= self.vpin_bucket_size {
            let total = self.vpin_current_buy_vol + self.vpin_current_sell_vol;
            if total > 0.0 {
                let imbalance =
                    (self.vpin_current_buy_vol - self.vpin_current_sell_vol).abs() / total;
                self.vpin_buckets.push(imbalance);
            }
            self.vpin_current_buy_vol = 0.0;
            self.vpin_current_sell_vol = 0.0;
            self.vpin_bucket_filled = 0.0;
        }
    }
}

/// Autocorrelation at given lag.
fn autocorr(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if n <= lag {
        return 0.0;
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 {
        return 0.0;
    }
    let mut c = 0.0f64;
    for i in lag..n {
        c += (x[i] - mean) * (x[i - lag] - mean);
    }
    c /= n as f64 * var;
    c.clamp(-1.0, 1.0)
}
