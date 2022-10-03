use super::traits::AbstractAkSolver;
use linreg::linear_regression_of;
use log::debug;
use std::time::Instant;

pub struct RegressionAkSolver {
    pub last_valid_value: (f64, f64),
    pub spread_specification: Vec<f64>,
}

impl AbstractAkSolver for RegressionAkSolver {
    fn new(spread_specification: &[f64]) -> Box<dyn AbstractAkSolver> {
        let mut solver = RegressionAkSolver {
            last_valid_value: (0f64, 0f64),
            spread_specification: spread_specification.to_vec(),
        };
        solver.spread_specification = solver.abs_spread(spread_specification);
        Box::new(solver)
    }

    fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64) {
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
}
