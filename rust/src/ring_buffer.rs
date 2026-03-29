use std::collections::VecDeque;

/// Fixed-capacity ring buffer matching Python's `collections.deque(maxlen=N)`.
/// When full, oldest element is dropped on push.
pub struct RingBuffer<T> {
    buf: VecDeque<T>,
    maxlen: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(maxlen: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(maxlen),
            maxlen,
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if self.buf.len() == self.maxlen {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Get last N elements as a slice-like iterator.
    pub fn tail(&self, n: usize) -> impl Iterator<Item = &T> {
        let skip = self.buf.len().saturating_sub(n);
        self.buf.iter().skip(skip)
    }

    /// Copy all elements to a Vec<f64> (for numpy conversion).
    pub fn to_vec(&self) -> Vec<T> {
        self.buf.iter().cloned().collect()
    }

    /// Get element at index from front.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.buf.get(idx)
    }

    /// Get last element.
    #[inline]
    pub fn back(&self) -> Option<&T> {
        self.buf.back()
    }

    /// Iterate over all elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buf.iter()
    }

    pub fn maxlen(&self) -> usize {
        self.maxlen
    }
}

// f64-specific helpers for numerical computations
impl RingBuffer<f64> {
    pub fn sum(&self) -> f64 {
        self.buf.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        self.sum() / self.buf.len() as f64
    }

    /// Variance with ddof (degrees of freedom correction).
    pub fn variance(&self, ddof: usize) -> f64 {
        let n = self.buf.len();
        if n <= ddof {
            return 0.0;
        }
        let mean = self.mean();
        let ss: f64 = self.buf.iter().map(|x| (x - mean).powi(2)).sum();
        ss / (n - ddof) as f64
    }

    pub fn std(&self, ddof: usize) -> f64 {
        self.variance(ddof).sqrt()
    }

    /// Copy tail of buffer into a new Vec<f64>.
    pub fn tail_vec(&self, n: usize) -> Vec<f64> {
        self.tail(n).copied().collect()
    }
}
