
use pyo3::prelude::*;
use std::collections::VecDeque;
use crate::sadf;

#[pyclass]
#[derive(Debug)]
pub struct RollRegression {
    period: usize,
    input_array_x: VecDeque<f64>,
    input_array_y: VecDeque<f64>,
    slope: f64,
    intercept: f64, 
    residual: f64,
    initialized: bool,
}

#[pymethods]
impl RollRegression {
    #[new]
    pub fn new(period: usize) -> Self {
        Self {
            period: period,
            input_array_x: VecDeque::with_capacity(period),
            input_array_y: VecDeque::with_capacity(period),
            initialized: false,
            slope: 0.0,
            intercept: 0.0, 
            residual: 0.0,
        }
    }
    
    pub fn update_raw(&mut self, input_x: f64, input_y: f64) {
        self.input_array_x.push_back(input_x);
        self.input_array_y.push_back(input_y);
        if self.input_array_x.len() >= self.period {
            self.initialized = true;
        }
        if self.input_array_x.len() >=self.period + 1
        {
            self.input_array_x.pop_front();
            self.input_array_y.pop_front();
            (self.slope, self.intercept, self.residual) = sadf::ols_linear_regression(&self.input_array_x, &self.input_array_y);
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn slope(&self) -> f64 {
        self.slope
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    pub fn residual(&self) -> f64 {
        self.residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use approx::assert_abs_diff_eq;
    #[test]
    fn test_stats() {
        let mut sta = RollRegression::new(5);
        let values = vec![2,4,6,8,10,12,14,16,18,20];
        for (i, &val) in values.iter().enumerate() 
        {
            sta.update_raw(i as f64, val as f64);
            println!("The length of input_array_x is {}",sta.input_array_x.len());
            println!("{:?}",sta.input_array_x);
            println!("{}",sta.slope());
            println!("{}",sta.intercept());
            println!("{}",sta.residual());
        }
        sta.reset();
        assert!(!sta.initialized());
    }
}

