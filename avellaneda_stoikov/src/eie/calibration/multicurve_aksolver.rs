use log::debug;
use std::time::Instant;

pub struct MultiCurveAkSolver {
    pub spread_specification: Vec<f64>,
    pub a_estimates: Vec<f64>,
    pub k_estimates: Vec<f64>,
}

impl MultiCurveAkSolver{
    pub fn new(spread_specification:&[f64]) -> Self {
        let n_estimates = spread_specification.len() * (spread_specification.len() - 1) / 2;
        let spread_specification = spread_specification.iter().map(|&val| val.abs()).collect();
        MultiCurveAkSolver {
            spread_specification: spread_specification,
            a_estimates: vec![0.0; n_estimates],
            k_estimates: vec![0.0; n_estimates],
        }
    }
    pub fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64) {
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
    pub fn mean(&self, data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>();
        let count = data.len();

        match count {
            positive if positive > 0 => Some(sum / count as f64),
            _ => None,
        }
    }
}