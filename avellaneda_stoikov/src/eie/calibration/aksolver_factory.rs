use super::multicurve_aksolver::MultiCurveAkSolver;
use super::regression_aksolver::RegressionAkSolver;
use super::traits::AbstractAkSolver;

#[derive(Debug, Copy, Clone)]
pub enum SolverType {
    MultiCurve,
    LogRegression,
}

#[derive(Debug, Copy, Clone)]
pub struct AkSolverFactory {
    solver_type: SolverType,
}

impl AkSolverFactory {
    /**
     * @param intensities Array of intensities (Y axis of Spread - Intensity curve)
     * @return array with estimated A and k (A, k)
     */
    pub fn new(t: &SolverType) -> Self {
        AkSolverFactory { solver_type: *t }
    }

    pub fn get_solver(&self, spread_specification: &[f64]) -> Box<dyn AbstractAkSolver> {
        match self.solver_type {
            SolverType::MultiCurve => MultiCurveAkSolver::new(&spread_specification),
            SolverType::LogRegression => RegressionAkSolver::new(&spread_specification),
        }
    }
}
