use super::{
    calibration::{
        aksolver_factory::AkSolverFactory, spread_intensity_curve::SpreadIntensityCurve,
    },
    intensity_info::IntensityInfo,
};

pub struct IntensityEstimator {
    sell_execution_intensity: SpreadIntensityCurve,
    buy_execution_intensity: SpreadIntensityCurve,
    init_done_ts: Option<u64>,
    is_initializing: bool,
    is_initialized: bool,
    w: u64,
}

impl IntensityEstimator {
    pub fn new(
        spread_step: f64,
        n_spreads: usize,
        w: u64,
        dt: u64,
        solver_factor: AkSolverFactory,
    ) -> Self {
        IntensityEstimator {
            sell_execution_intensity: SpreadIntensityCurve::new(
                spread_step,
                n_spreads,
                dt,
                solver_factor.clone(),
            ),
            buy_execution_intensity: SpreadIntensityCurve::new(
                -spread_step,
                n_spreads,
                dt,
                solver_factor.clone(),
            ),
            init_done_ts: None,
            is_initializing: true,
            is_initialized: false,
            w: w,
        }
    }

    pub fn on_tick(&mut self, bid: f64, ask: f64, ts: u64) -> bool {
        if self.is_initializing {
            self.init(ts);
        }

        let mid_price = (bid + ask) / 2.0;
        let window_start = ts - self.w;
        self.sell_execution_intensity
            .on_tick(mid_price, bid, ts, window_start);
        self.buy_execution_intensity
            .on_tick(mid_price, ask, ts, window_start);
        return self.is_initialized;
    }

    pub fn init(&mut self, ts: u64) {
        match self.init_done_ts {
            Some(ts) => {
                if self.init_done_ts.unwrap() <= ts {
                    self.is_initialized = true;
                    self.is_initializing = false;
                }
            }
            None => {
                self.init_done_ts = Some(ts + self.w);
            }
        }
    }

    pub fn estimate(&mut self, ts: u64) -> IntensityInfo {
        let window_start = ts - self.w;
        return IntensityInfo::new(
            self.buy_execution_intensity.estimate_ak(ts, window_start),
            self.sell_execution_intensity.estimate_ak(ts, window_start),
        );
    }
}
