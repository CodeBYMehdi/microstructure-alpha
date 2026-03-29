use pyo3::prelude::*;
use std::collections::HashMap;

use crate::ring_buffer::RingBuffer;

/// L2 order book feature computation (hot path).
/// The event handling (insert/update/delete) stays in Python because it
/// interacts with the event bus. This module accelerates get_features()
/// and book dynamics analysis.
#[pyclass]
pub struct RustL2Features {
    max_depth: usize,

    // Liquidity pull tracking
    liq_pull_bid_score: f64,
    liq_pull_ask_score: f64,
    liq_pull_decay: f64,

    // Spoofing detection
    top_level_sizes: RingBuffer<f64>,
    spoofing_score: f64,
    spoofing_decay: f64,

    // Previous volumes
    last_total_bid_vol: f64,
    last_total_ask_vol: f64,
}

#[pymethods]
impl RustL2Features {
    #[new]
    #[pyo3(signature = (max_depth=10))]
    fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            liq_pull_bid_score: 0.0,
            liq_pull_ask_score: 0.0,
            liq_pull_decay: 0.95,
            top_level_sizes: RingBuffer::new(100),
            spoofing_score: 0.0,
            spoofing_decay: 0.98,
            last_total_bid_vol: 0.0,
            last_total_ask_vol: 0.0,
        }
    }

    /// Analyze book dynamics after an L2 update.
    /// Call this from Python after each on_l2_update.
    fn analyze_dynamics(
        &mut self,
        current_bid_vol: f64,
        current_ask_vol: f64,
        operation: i32,  // 0=insert, 1=update, 2=delete
        depth_level: usize,
        size: f64,
    ) {
        // Decay existing scores
        self.liq_pull_bid_score *= self.liq_pull_decay;
        self.liq_pull_ask_score *= self.liq_pull_decay;

        if self.last_total_bid_vol > 0.0 {
            let delta_bid =
                (current_bid_vol - self.last_total_bid_vol) / self.last_total_bid_vol;
            if delta_bid < -0.15 {
                self.liq_pull_bid_score += delta_bid.abs();
            }
        }

        if self.last_total_ask_vol > 0.0 {
            let delta_ask =
                (current_ask_vol - self.last_total_ask_vol) / self.last_total_ask_vol;
            if delta_ask < -0.15 {
                self.liq_pull_ask_score += delta_ask.abs();
            }
        }

        // Spoofing detection
        self.spoofing_score *= self.spoofing_decay;
        if (operation == 0 || operation == 1) && depth_level < 3 {
            self.top_level_sizes.push(size);
            if self.top_level_sizes.len() >= 20 {
                let mut sorted: Vec<f64> = self.top_level_sizes.to_vec();
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2];
                if median > 0.0 && size > 5.0 * median {
                    self.spoofing_score += (size / (10.0 * median)).min(1.0);
                }
            }
        }

        self.last_total_bid_vol = current_bid_vol;
        self.last_total_ask_vol = current_ask_vol;
    }

    /// Compute L2 features from current book state.
    /// bids/asks: list of (price, size) tuples sorted by level.
    fn compute_features(
        &self,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> HashMap<String, f64> {
        let mut features = HashMap::new();

        // Depth imbalance
        let total_bid_vol: f64 = bids.iter().map(|(_, s)| s).sum();
        let total_ask_vol: f64 = asks.iter().map(|(_, s)| s).sum();
        let total_depth = total_bid_vol + total_ask_vol;
        let depth_imbalance = if total_depth > 0.0 {
            ((total_bid_vol - total_ask_vol) / total_depth).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        features.insert("depth_imbalance".into(), depth_imbalance);

        // Liquidity pull score (net: ask pull bullish, bid pull bearish)
        features.insert(
            "liquidity_pull_score".into(),
            (self.liq_pull_ask_score - self.liq_pull_bid_score).clamp(-5.0, 5.0),
        );

        // Book pressure: depth-weighted asymmetry
        let bb_price = bids.first().map_or(0.0, |b| b.0);
        let ba_price = asks.first().map_or(f64::INFINITY, |a| a.0);
        let mid = if bb_price > 0.0 && ba_price < f64::INFINITY {
            (bb_price + ba_price) / 2.0
        } else {
            0.0
        };

        let mut book_pressure = 0.0;
        if mid > 0.0 {
            let mut bid_pressure = 0.0f64;
            let mut ask_pressure = 0.0f64;
            for (pos, (price, size)) in bids.iter().enumerate() {
                if *price > 0.0 {
                    let level_weight = 1.0 / (1.0 + pos as f64);
                    bid_pressure += size * level_weight;
                }
            }
            for (pos, (price, size)) in asks.iter().enumerate() {
                if *price < f64::INFINITY {
                    let level_weight = 1.0 / (1.0 + pos as f64);
                    ask_pressure += size * level_weight;
                }
            }
            let total_pressure = bid_pressure + ask_pressure;
            if total_pressure > 0.0 {
                book_pressure = ((bid_pressure - ask_pressure) / total_pressure).clamp(-1.0, 1.0);
            }
        }
        features.insert("book_pressure".into(), book_pressure);

        // Spoofing score
        features.insert("spoofing_score".into(), self.spoofing_score.clamp(0.0, 5.0));

        // Spread in bps
        let spread_bps = if bb_price > 0.0 && ba_price < f64::INFINITY && mid > 0.0 {
            ((ba_price - bb_price) / mid) * 10000.0
        } else {
            0.0
        };
        features.insert("spread_bps".into(), spread_bps);

        features
    }
}
