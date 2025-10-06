use crate::{
    eie::{
        calibration::aksolver_factory::{AkSolverFactory, SolverType},
        intensity_estimator::IntensityEstimator,
        //intensity_info::IntensityInfo,
    },
};
extern crate pyo3;
use pyo3::prelude::*;

#[pyclass]
pub struct AvellanedaStoikov {
    pub tick_size: f64,
    pub n_spreads: usize,
    pub estimate_window: u64,
    pub period: u64,
    pub start_time: u64,  //time in milliseconds (ms)
    pub initialized: bool,
    pub ie: IntensityEstimator,
}

#[pymethods]
impl AvellanedaStoikov {
    #[new] 
    pub fn new(
        tick_size: f64,
        n_spreads: usize,
        estimate_window: u64,
        period: u64,
        start_time: u64, 
    ) -> Self{
        let solver = SolverType::LogRegression;
        let sf = AkSolverFactory::new(&solver);
        let ie = IntensityEstimator::new(
            tick_size.clone(),
            n_spreads.clone(),
            estimate_window.clone(),
            period.clone(),
            sf,
        );
        AvellanedaStoikov{
            tick_size: tick_size,
            n_spreads: n_spreads,
            estimate_window: estimate_window,
            period: period, 
            start_time: start_time,
            initialized: false,
            ie:ie,
        }
    }

    /// Returns whether the estimator has enough data to produce A/k
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Ingest a single L1 tick (best ask/bid) with timestamp in milliseconds.
    /// Returns true if the internal estimator has enough data to estimate A/k.
    pub fn ingest_tick(&mut self, ask: f64, bid: f64, ts: u64) -> bool {
        let ready = self.ie.on_tick(bid, ask, ts);
        // mark as initialized only when window condition also satisfied
        self.initialized = ready && ts >= self.start_time.saturating_add(self.estimate_window);
        self.initialized
    }

    /// Estimate (buy_a, buy_k, sell_a, sell_k) if ready; otherwise returns None
    pub fn estimate_ak(&mut self, ts: u64) -> Option<(f64, f64, f64, f64)> {
        if ts >= self.start_time.saturating_add(self.estimate_window) {
            let ii = self.ie.estimate(ts);
            let (ba, bk, sa, sk) = ii.get_ak();
            // ensure strictly positive values are returned upstream
            if ba.is_finite() && bk.is_finite() && sa.is_finite() && sk.is_finite() {
                self.initialized = true;
                return Some((ba, bk, sa, sk));
            }
        }
        self.initialized = false;
        None
    }

    /// Reset estimator state and start_time
    pub fn reset(&mut self, start_time: u64) {
        // Recreate estimator to drop internal windows safely
        let solver = SolverType::LogRegression;
        let sf = AkSolverFactory::new(&solver);
        self.ie = IntensityEstimator::new(
            self.tick_size,
            self.n_spreads,
            self.estimate_window,
            self.period,
            sf,
        );
        self.start_time = start_time;
        self.initialized = false;
    }

    pub fn calculate_intensity_info(
        &mut self, 
        ask: f64, 
        bid: f64, 
        ts: u64
    ) -> (f64, f64, f64, f64)
    {
        // Backward-compatible API: ingest + try estimate, else return zeros
        let _ = self.ingest_tick(ask, bid, ts);
        if let Some((ba, bk, sa, sk)) = self.estimate_ak(ts) {
            (ba, bk, sa, sk)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    }

}

#[pymodule]
fn avellaneda_stoikov(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AvellanedaStoikov>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use crate::avellaneda_stoikov::AvellanedaStoikov;
    
    #[test]
    fn test_init() {
        let tick_size: f64 = 0.001;
        let n_spreads: usize = 10;
        let estimate_window: u64 = 30;
        let period: u64 = 20;
        let start_time: u64 = 0;
        let mut as_ex = AvellanedaStoikov::new(
            tick_size,
            n_spreads,
            estimate_window,
            period,
            start_time,
        );
        let x = 10_f64.ln();
        println!("{}",x);
        //println!("{}",&as_ex.estimate_window);
        let info = as_ex.calculate_intensity_info(
            0.900,
            1.100,
            10040,
        );
        println!("{:?}",info);
        println!("{has it initlized?}",as_ex.initialized());
    }

}
