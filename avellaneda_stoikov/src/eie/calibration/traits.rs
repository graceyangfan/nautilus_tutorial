/**
 * Abstract solver of A and k
 */
pub trait AbstractAkSolver {
    fn new(spread_specification: &[f64]) -> Box<dyn AbstractAkSolver>
    where
        Self: Sized;

    fn abs_spread(&mut self, spread_specification: &[f64]) -> Vec<f64> {
        spread_specification.iter().map(|&val| val.abs()).collect()
    }

    fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64);

    fn mean(&self, data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>();
        let count = data.len();

        match count {
            positive if positive > 0 => Some(sum / count as f64),
            _ => None,
        }
    }
}
