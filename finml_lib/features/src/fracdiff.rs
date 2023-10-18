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

pub fn get_weights(order: f64,  max_width: usize, threshold: f64) -> Vec<f64> {
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

#[pymethods]
impl FracDiff {
    #[new]
    #[args(threshold = "1e-5")]
    pub fn new(order: f64, period: usize, threshold: f64) -> Self {
        let weights = get_weights(order, period, threshold);
        Self {
            weights:weights,
            value: 0.0,
            period: period,
            input_array: VecDeque::with_capacity(period),
            initialized: false,
        }
    }
    
    pub fn update_raw(&mut self, input: f64) {
        self.input_array.push_back(input);
        if self.input_array.len() >= self.period {
            self.initialized = true;
        }
        if self.input_array.len() >=self.period + 1
        {
            self.input_array.pop_front();
            let dot_product = self.input_array.iter().zip(self.weights.iter())
            .fold(0.0, |acc, (x, y)| acc + x * y);
            self.value = dot_product;
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fracdiff() {
        let mut fd = FracDiff::new(0.9, 10, 1e-5);
        for i in 0..10
        {
            println!("the weights are {}",fd.weights[i]);
        }
        for i in 0..100
        {
            fd.update_raw(i as f64);
            println!("The length of input_array is {}",fd.input_array.len());
            println!("{}",fd.value);
            if i == 99
            {
                assert_abs_diff_eq!(fd.value,2.60561581,epsilon=1e-6);
            }
        }
        fd.reset();
        assert!(!fd.initialized());
    }
}
