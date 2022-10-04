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
            ie:ie,
        }
    }

    pub fn calculate_intensity_info(
        &mut self, 
        ask: f64, 
        bid: f64, 
        ts: u64
    ) -> (f64, f64, f64, f64)
    {
        let can_get = self.ie.on_tick(bid, ask, ts);
        // wait to get more data
        if can_get && ts > self.start_time + self.estimate_window + 1 {
            let ii = self.ie.estimate(ts);
            return ii.get_ak();
        } else {
            (0.0,0.0,0.0,0.0)
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
        //println!("{}",&as_ex.estimate_window);
        let info = as_ex.calculate_intensity_info(
            0.900,
            1.100,
            10040,
        );
        println!("{:?}",info);
    }

}
