use pyo3::prelude::*;

/// Compute Order Flow Imbalance from BBO changes.
/// Pure arithmetic — extracted from main.py on_tick() for zero interpreter overhead.
///
/// OFI measures the net pressure from bid/ask price and size changes:
/// - Bid up → aggressive buying (positive OFI)
/// - Ask down → aggressive selling (negative OFI)
#[pyfunction]
pub fn compute_ofi(
    curr_bid: f64,
    prev_bid: f64,
    curr_ask: f64,
    prev_ask: f64,
    curr_bid_size: f64,
    prev_bid_size: f64,
    curr_ask_size: f64,
    prev_ask_size: f64,
) -> f64 {
    let mut ofi = 0.0f64;

    // Bid side
    if curr_bid > prev_bid {
        ofi += curr_bid_size;
    } else if curr_bid == prev_bid {
        ofi += curr_bid_size - prev_bid_size;
    } else {
        ofi -= prev_bid_size;
    }

    // Ask side
    if curr_ask < prev_ask {
        ofi -= curr_ask_size;
    } else if curr_ask == prev_ask {
        ofi -= curr_ask_size - prev_ask_size;
    } else {
        ofi += prev_ask_size;
    }

    ofi
}

/// Compute book slope from bid/ask sizes and spread.
#[pyfunction]
pub fn compute_book_slope(
    bid_size: f64,
    ask_size: f64,
    spread: f64,
) -> f64 {
    let s = if spread > 0.01 { spread } else { 0.01 };
    (bid_size - ask_size) / s
}
