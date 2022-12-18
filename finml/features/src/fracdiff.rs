use pyo3::prelude::*;
use std::collections::VecDeque;

#[pyclass]
#[derive(Debug)]
pub struct FracDiff {
    weights: Vec<f64>,
    value: f64,
    period: usize,
    input_array: VecDeque<f64>,
    initialized: bool,
}

#[pymethods]
impl FracDiff {
    #[new]
    pub fn new(order: f64, threshold: f64, period: usize) -> Self {
        let weights = Self::get_weights(order, threshold, period);
        Self {
            weights,
            value: 0.0,
            period,
            input_array: VecDeque::with_capacity(period),
            initialized: false,
        }
    }

    pub fn update_raw(&mut self, input: f64) {
        if self.input_array.len() >= self.period {
            self.initialized = true;
        }
        self.input_array.push_back(input);
        let dot_product = self.input_array.iter().zip(self.weights.iter())
            .fold(0.0, |acc, (x, y)| acc + x * y);
        self.value = dot_product;
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn get_weights(order: f64, threshold: f64, max_width: usize) -> Vec<f64> {
        let mut weights = vec![1.0];
        let mut k = 1;
        while k < max_width {
            let next_w = -weights[weights.len() - 1] * (order - k as f64 + 1.0) / k as f64;
            if next_w.abs() < threshold {
                break;
            }
            weights.push(next_w);
            k += 1;
        }
        weights.into_iter().rev().collect()
    }
}

#[pymodule]
fn fracdiff(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FracDiff>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fracdiff() {
        let mut fd = FracDiff::new(0.5, 1e-3, 3);
        fd.update_raw(1.0);
        fd.update_raw(2.0);
        fd.update_raw(3.0);
        assert_abs_diff_eq!(fd.value, -0.5, epsilon = 1e-6);
        fd.update_raw(4.0);
        assert_abs_diff_eq!(fd.value, 0.0, epsilon = 1e-6);
        fd.reset();
        assert!(!fd.initialized());
    }
}
