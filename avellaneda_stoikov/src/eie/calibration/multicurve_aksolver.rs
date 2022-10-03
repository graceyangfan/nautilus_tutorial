use log::debug;

use super::traits::AbstractAkSolver;
use std::time::Instant;

pub struct MultiCurveAkSolver {
    pub spread_specification: Vec<f64>,
    pub a_estimates: Vec<f64>,
    pub k_estimates: Vec<f64>,
}

impl AbstractAkSolver for MultiCurveAkSolver {
    fn new(spread_specification: &[f64]) -> Box<dyn AbstractAkSolver> {
        let n_estimates = spread_specification.len() * (spread_specification.len() - 1) / 2;
        let mut solver = MultiCurveAkSolver {
            spread_specification: spread_specification.to_vec(),
            a_estimates: vec![0.0; n_estimates],
            k_estimates: vec![0.0; n_estimates],
        };
        solver.spread_specification = solver.abs_spread(spread_specification);
        Box::new(solver)
    }

    fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64) {
        let ins = Instant::now();

        let mut est_idx = 0;
        for i in 0..intensities.len() - 1 {
            for j in i + 1..intensities.len() {
                self.k_estimates[est_idx] = (intensities[j] / intensities[i]).ln()
                    / (self.spread_specification[i] - self.spread_specification[j]);
                self.a_estimates[est_idx] = intensities[i]
                    * (self.k_estimates[est_idx] * self.spread_specification[i]).exp();
                est_idx += 1;
            }
        }

        let a_mean = self.mean(&self.a_estimates).unwrap();
        let k_mean = self.mean(&self.k_estimates).unwrap();
        debug!("MultiCurveAkSolver time: {:?}", ins.elapsed());
        return (a_mean, k_mean);
    }
}
