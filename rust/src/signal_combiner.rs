use pyo3::prelude::*;
use std::collections::HashMap;

use crate::ring_buffer::RingBuffer;

/// High-performance signal combiner.
/// Adaptive weight updates + rolling accuracy tracking in Rust.
#[pyclass]
pub struct RustSignalCombiner {
    signal_names: Vec<String>,
    weights: HashMap<String, f64>,
    accuracy: HashMap<String, RingBuffer<f64>>,
    last_signals: HashMap<String, f64>,
    min_weight: f64,
    actionable_threshold: f64,
}

#[pymethods]
impl RustSignalCombiner {
    #[new]
    #[pyo3(signature = (signal_names=None, initial_weights=None, min_weight=0.05, actionable_threshold=0.15, lookback=100))]
    fn new(
        signal_names: Option<Vec<String>>,
        initial_weights: Option<HashMap<String, f64>>,
        min_weight: f64,
        actionable_threshold: f64,
        lookback: usize,
    ) -> Self {
        let names = signal_names.unwrap_or_else(|| {
            vec![
                "regime_transition".into(),
                "return_prediction".into(),
                "order_flow".into(),
                "momentum".into(),
                "mean_reversion".into(),
                "orderbook".into(),
            ]
        });

        // Default weights: L2 signals get higher weight
        let default_weights: HashMap<String, f64> = [
            ("regime_transition", 0.10),
            ("return_prediction", 0.10),
            ("order_flow", 0.25),
            ("momentum", 0.10),
            ("mean_reversion", 0.10),
            ("orderbook", 0.35),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

        let n = names.len();
        let mut weights = HashMap::new();
        for name in &names {
            let w = if let Some(ref iw) = initial_weights {
                *iw.get(name).unwrap_or(&(1.0 / n as f64))
            } else {
                *default_weights.get(name.as_str()).unwrap_or(&(1.0 / n as f64))
            };
            weights.insert(name.clone(), w);
        }

        // Normalize
        let total: f64 = weights.values().sum();
        if total > 0.0 {
            for v in weights.values_mut() {
                *v /= total;
            }
        }

        let accuracy = names
            .iter()
            .map(|n| (n.clone(), RingBuffer::new(lookback)))
            .collect();

        Self {
            signal_names: names,
            weights,
            accuracy,
            last_signals: HashMap::new(),
            min_weight,
            actionable_threshold,
        }
    }

    /// Combine signals into (direction, strength, confidence, is_actionable).
    /// Takes dicts of signal values and optional confidences.
    #[pyo3(signature = (signals, confidences=None))]
    fn combine(
        &mut self,
        signals: HashMap<String, f64>,
        confidences: Option<HashMap<String, f64>>,
    ) -> (f64, f64, f64, bool) {
        let confs = confidences.unwrap_or_default();

        let mut weighted_sum = 0.0f64;
        let mut total_weight = 0.0f64;
        let mut conf_sum = 0.0f64;

        for name in &self.signal_names {
            let value = *signals.get(name).unwrap_or(&0.0);
            let conf = *confs.get(name).unwrap_or(&0.5);
            let weight = *self.weights.get(name).unwrap_or(&0.0);

            let effective_weight = weight * conf;
            weighted_sum += value * effective_weight;
            total_weight += effective_weight;
            conf_sum += conf * weight;
        }

        let direction = if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let strength = direction.abs();

        // Store for accuracy update
        self.last_signals = signals;

        (direction, strength, conf_sum, strength >= self.actionable_threshold)
    }

    /// Update accuracy tracking based on realized return.
    fn update_accuracy(&mut self, actual_return: f64) {
        if self.last_signals.is_empty() {
            return;
        }

        for (name, &signal_value) in &self.last_signals.clone() {
            if let Some(acc) = self.accuracy.get_mut(name) {
                let correct = (signal_value > 0.0 && actual_return > 0.0)
                    || (signal_value < 0.0 && actual_return < 0.0)
                    || signal_value.abs() < 1e-8;
                acc.push(if correct { 1.0 } else { 0.0 });
            }
        }

        self.update_weights();
    }

    fn get_weights(&self) -> HashMap<String, f64> {
        self.weights.clone()
    }
}

impl RustSignalCombiner {
    fn update_weights(&mut self) {
        let mut raw_weights: HashMap<String, f64> = HashMap::new();

        for name in &self.signal_names {
            if let Some(acc) = self.accuracy.get(name) {
                if acc.is_empty() {
                    raw_weights.insert(name.clone(), self.min_weight);
                } else {
                    let accuracy = acc.mean();
                    raw_weights.insert(name.clone(), (accuracy - 0.3).max(self.min_weight));
                }
            } else {
                raw_weights.insert(name.clone(), self.min_weight);
            }
        }

        let total: f64 = raw_weights.values().sum();
        if total > 0.0 {
            for name in &self.signal_names {
                if let Some(rw) = raw_weights.get(name) {
                    self.weights.insert(name.clone(), rw / total);
                }
            }
        }
    }
}
