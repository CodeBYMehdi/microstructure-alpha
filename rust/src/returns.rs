use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray as _};
use crate::ring_buffer::RingBuffer;

/// High-performance return calculator using a Rust ring buffer.
/// Drop-in replacement for microstructure.returns.ReturnCalculator.
#[pyclass]
pub struct ReturnCalculator {
    returns: RingBuffer<f64>,
    last_price: Option<f64>,
    count_val: usize,
}

#[pymethods]
impl ReturnCalculator {
    #[new]
    #[pyo3(signature = (max_window_size=5000))]
    fn new(max_window_size: usize) -> PyResult<Self> {
        if max_window_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_window_size must be positive",
            ));
        }
        Ok(Self {
            returns: RingBuffer::new(max_window_size),
            last_price: None,
            count_val: 0,
        })
    }

    /// Feed a new price. Returns the log return if a previous price exists.
    fn update(&mut self, price: f64) -> PyResult<Option<f64>> {
        if !price.is_finite() || price <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Price must be positive and finite, got {}", price),
            ));
        }

        let ret = match self.last_price {
            Some(prev) if prev > 0.0 => {
                let r = (price / prev).ln();
                if r.is_finite() {
                    self.returns.push(r);
                    Some(r)
                } else {
                    self.returns.push(0.0);
                    Some(0.0)
                }
            }
            _ => None,
        };
        self.last_price = Some(price);
        Ok(ret)
    }

    /// Get the last `size` returns as a numpy array.
    fn get_window<'py>(&self, py: Python<'py>, size: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Window size must be positive"));
        }
        if size > self.returns.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Requested window size {} larger than available history {}",
                    size,
                    self.returns.len()
                ),
            ));
        }
        let data = self.returns.tail_vec(size);
        Ok(data.into_pyarray_bound(py))
    }

    #[getter]
    fn count(&self) -> usize {
        self.returns.len()
    }

    fn reset(&mut self) {
        self.returns.clear();
        self.last_price = None;
        self.count_val = 0;
    }
}
