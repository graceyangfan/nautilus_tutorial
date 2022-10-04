use super::aksolver_factory::{AkSolverFactory,SolverType};
use super::empirical_intensity_estimator::EmpiricalIntensityEstimator;
use super::multicurve_aksolver::MultiCurveAkSolver;
use super::regression_aksolver::RegressionAkSolver;

pub enum AbstractAkSolver {
    MultiCurve(MultiCurveAkSolver),
    LogRegression(RegressionAkSolver),
}

impl AbstractAkSolver{
    pub fn solve_ak(&mut self, intensities: &[f64]) -> (f64, f64){
        match self {
            AbstractAkSolver::MultiCurve(item) => item.solve_ak(intensities),
            AbstractAkSolver::LogRegression(item) => item.solve_ak(intensities),
        }
    }
}

pub struct SpreadIntensityCurve {
    pub intensity_estimators: Vec<EmpiricalIntensityEstimator>,
    pub intensity_estimates: Vec<f64>,
    pub aksolver: AbstractAkSolver,
}

impl SpreadIntensityCurve {
    pub fn new(
        spread_step: f64,
        n_spreads: usize,
        dt: u64,
        solver_factory: AkSolverFactory,
    ) -> Self {
        let mut intensity_estimators = Vec::with_capacity(n_spreads);
        let mut spread_specification = vec![0.0; n_spreads];
        let intensity_estimates = vec![0.0; n_spreads];

        for i in 0..n_spreads {
            spread_specification[i] = i as f64 * spread_step;
            intensity_estimators.push(EmpiricalIntensityEstimator::new(
                spread_specification[i],
                spread_step.signum(),
                dt,
            ))
        }

        let solver_type = solver_factory.get_solver_type();
        let aksolver = match solver_type {
            SolverType::MultiCurve => AbstractAkSolver::MultiCurve(MultiCurveAkSolver::new(&spread_specification)),
            SolverType::LogRegression => AbstractAkSolver::LogRegression(RegressionAkSolver::new(&spread_specification)),
        };
    

        SpreadIntensityCurve {
            intensity_estimators: intensity_estimators,
            intensity_estimates: intensity_estimates,
            aksolver: aksolver,
        }
    }

    pub fn on_tick(&mut self, ref_price: f64, fill_price: f64, ts: u64, window_start: u64) {
        for est in self.intensity_estimators.iter_mut() {
            est.on_tick(ref_price, fill_price, ts, window_start);
        }
    }

    pub fn estimate_ak(&mut self, ts: u64, window_start: u64) -> (f64, f64) {
        for i in 0..self.intensity_estimates.len() {
            self.intensity_estimates[i] =
                self.intensity_estimators[i].estimate_intensity(ts, window_start)
        }

        return self.aksolver.solve_ak(&self.intensity_estimates);
    }
}
