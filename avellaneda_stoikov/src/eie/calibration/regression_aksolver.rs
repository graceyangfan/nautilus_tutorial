use linreg::linear_regression_of;
use log::debug;
use std::time::Instant;

pub struct RegressionAkSolver {
    pub last_valid_value: (f64, f64),
    pub spread_specification: Vec<f64>,
}


impl RegressionAkSolver {
    pub fn new(spread_specification: &[f64]) -> Self {
        let spread_specification = spread_specification.iter().map(|&val| val.abs()).collect();
        RegressionAkSolver {
            last_valid_value: (0f64, 0f64),
            spread_specification: spread_specification,
        }
    }
    pub fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64) {
        let ins = Instant::now();

        let mut tuples: Vec<(f64, f64)> = Vec::new();
        for i in 0..self.spread_specification.len() {
            tuples.push((self.spread_specification[i], intensities[i].ln()));
        }

        let (slope, intercept): (f64, f64) =
            linear_regression_of(&tuples).unwrap_or_else(|_| self.last_valid_value);

        self.last_valid_value = (slope, intercept);
        debug!("RegressionAkSolver time: {:?}", ins.elapsed());
        return (intercept.exp(), -slope);
    }
    pub fn mean(&self, data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>();
        let count = data.len();

        match count {
            positive if positive > 0 => Some(sum / count as f64),
            _ => None,
        }
    }
}

