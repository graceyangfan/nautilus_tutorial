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
    pub fn get_solver_type(&self) -> SolverType {
        self.solver_type
    }
}
