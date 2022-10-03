use std::cell::RefCell;

#[derive(Debug, Copy, Clone)]
struct Fill;

impl Fill {
    fn is_order_filled(&self, spread_direction: f64, filled_price: f64, order_price: f64) -> bool {
        match spread_direction > 0.0 {
            true => {
                return filled_price > order_price;
            }
            false => {
                return filled_price < order_price;
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct LimitOrderTracker {
    order_price: f64,
    start_ts: u64,
}

impl LimitOrderTracker {
    fn new(order_price: f64, start_ts: u64) -> Self {
        LimitOrderTracker {
            order_price: order_price,
            start_ts: start_ts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmpiricalIntensityEstimator {
    spread: f64,
    spread_direction: f64,
    dt: u64,
    fill_comp: Fill,
    initializing: bool,
    last_price: f64,
    last_limit_order_inserted: u64,
    live_trackers: RefCell<Vec<LimitOrderTracker>>,
    live_trackers_start_time_sum: u64,
    finished_trackers: RefCell<Vec<(u64, u64)>>,
    finished_trackers_wait_time_sum: u64,
}

impl EmpiricalIntensityEstimator {
    pub fn new(spread: f64, spread_direction: f64, dt: u64) -> Self {
        EmpiricalIntensityEstimator {
            spread: spread,
            spread_direction: spread_direction,
            dt: dt,
            fill_comp: Fill,
            initializing: true,
            last_price: f64::NAN,
            last_limit_order_inserted: 0,
            live_trackers: RefCell::new(Vec::new()),
            live_trackers_start_time_sum: 0,
            finished_trackers: RefCell::new(Vec::new()),
            finished_trackers_wait_time_sum: 0,
        }
    }

    pub fn on_tick(&mut self, ref_price: f64, fill_price: f64, ts: u64, window_start: u64) {
        if self.initializing {
            self.initializing = false;
            self.last_limit_order_inserted = ts - self.dt;
        }

        let lt = &mut self.live_trackers.borrow_mut();

        while self.last_limit_order_inserted + self.dt < ts {
            self.last_limit_order_inserted = self.last_limit_order_inserted + self.dt;
            lt.push(LimitOrderTracker::new(
                self.last_price + self.spread,
                self.last_limit_order_inserted,
            ));
            self.live_trackers_start_time_sum += self.last_limit_order_inserted;
        }

        if self.last_limit_order_inserted + self.dt == ts {
            self.last_limit_order_inserted = ts;
            lt.push(LimitOrderTracker::new(ref_price + self.spread, ts));
            self.live_trackers_start_time_sum += ts;
        }

        self.last_price = ref_price;

        for i in 0..lt.len() {
            if let Some(tr) = lt.get(i) {
                if window_start > tr.start_ts {
                    self.live_trackers_start_time_sum -= tr.start_ts;
                    lt.remove(i);
                    continue;
                }

                if self
                    .fill_comp
                    .is_order_filled(self.spread_direction, fill_price, tr.order_price)
                {
                    self.live_trackers_start_time_sum -= tr.start_ts;

                    let duration = ts - tr.start_ts;
                    // add to finished trackers, add duration to sum
                    self.finished_trackers
                        .borrow_mut()
                        .push((tr.start_ts, duration));
                    self.finished_trackers_wait_time_sum += duration;
                    lt.remove(i);
                }
            }
        }
    }

    pub fn estimate_intensity(&mut self, ts: u64, window_start: u64) -> f64 {
        let ft = &mut self.finished_trackers.borrow_mut();

        for i in 0..ft.len() {
            if let Some(&tr) = ft.get(i) {
                if tr.0 < window_start {
                    ft.remove(i);
                    self.finished_trackers_wait_time_sum -= tr.1;
                }
            }
        }

        if !self.live_trackers.borrow().is_empty()
            && (ts != self.live_trackers.borrow().last().unwrap().start_ts)
        {
            let lt = &mut self.live_trackers.borrow_mut();
            for i in 0..lt.len() {
                if let Some(&tr) = lt.get(i) {
                    if window_start > tr.start_ts {
                        lt.remove(i);
                        self.live_trackers_start_time_sum -= tr.start_ts;
                    }
                }
            }
        }

        return self.dt as f64 * ft.len() as f64
            / (self.live_trackers.borrow().len() as f64 * ts as f64
                - self.live_trackers_start_time_sum as f64
                + self.finished_trackers_wait_time_sum as f64);
    }
}
