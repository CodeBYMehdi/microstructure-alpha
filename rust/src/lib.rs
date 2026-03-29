use pyo3::prelude::*;

mod ring_buffer;
mod linalg;
mod returns;
mod ofi;
mod moments;
mod entropy;
mod hmm;
mod transition;
mod signal_combiner;
mod feature_engine;
mod adaptive_exits;
mod l2_orderbook;
mod return_predictor;
mod tail_risk;
mod surface_analytics;
mod online_scaler;
mod gmm;

/// _microstructure_core — Rust-accelerated hot path for the microstructure trading system.
#[pymodule]
fn _microstructure_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core data
    m.add_class::<returns::ReturnCalculator>()?;

    // Microstructure
    m.add_class::<moments::MomentsCalculator>()?;
    m.add_class::<moments::MicrostructureMoments>()?;
    m.add_class::<entropy::EntropyCalculator>()?;

    // PDF / Density
    m.add_class::<gmm::RustGMM1D>()?;

    // Regime
    m.add_class::<hmm::GaussianHMM>()?;
    m.add_class::<transition::KalmanDerivativeTracker>()?;
    m.add_class::<online_scaler::RustOnlineScaler>()?;

    // Alpha / Signals
    m.add_class::<signal_combiner::RustSignalCombiner>()?;
    m.add_class::<feature_engine::RustFeatureEngine>()?;
    m.add_class::<return_predictor::RustReturnPredictor>()?;

    // Execution / Exits
    m.add_class::<adaptive_exits::RustATRTracker>()?;
    m.add_function(wrap_pyfunction!(adaptive_exits::compute_exit_params_rust, m)?)?;

    // Risk
    m.add_class::<tail_risk::RustTailRiskTracker>()?;
    m.add_function(wrap_pyfunction!(tail_risk::compute_tail_risk, m)?)?;

    // Surface Analytics
    m.add_class::<surface_analytics::RustSurfaceAnalytics>()?;

    // L2 Order Book
    m.add_class::<l2_orderbook::RustL2Features>()?;

    // Pure functions
    m.add_function(wrap_pyfunction!(ofi::compute_ofi, m)?)?;
    m.add_function(wrap_pyfunction!(ofi::compute_book_slope, m)?)?;

    Ok(())
}
