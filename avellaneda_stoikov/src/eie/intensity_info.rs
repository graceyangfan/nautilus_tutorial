#[derive(Debug, Copy, Clone)]
pub struct IntensityInfo {
    pub buy_a: f64,
    pub buy_k: f64,
    pub sell_a: f64,
    pub sell_k: f64,
}

impl IntensityInfo {
    pub fn new(buy_ak: (f64, f64), sell_ak: (f64, f64)) -> Self {
        IntensityInfo {
            buy_a: buy_ak.0,
            buy_k: buy_ak.1,
            sell_a: sell_ak.0,
            sell_k: sell_ak.1,
        }
    }

    pub fn get_sell_fill_intensity(&self, spread: f64) -> f64 {
        return get_intensity(spread, self.sell_a, self.sell_k);
    }

    pub fn get_buy_fill_intensity(&self, spread: f64) -> f64 {
        return get_intensity(spread, self.buy_a, self.buy_k);
    }

    pub fn get_sell_spread(&self, intensity: f64) -> f64 {
        return get_spread(intensity, self.sell_a, self.sell_k);
    }

    pub fn get_buy_spread(&self, intensity: f64) -> f64 {
        return get_spread(intensity, self.buy_a, self.buy_k);
    }

    pub fn get_ak(&self) -> (f64, f64, f64, f64) {
        (self.buy_a, self.buy_k, self.sell_a, self.sell_k)
    }
}

pub fn get_intensity(target_spread: f64, a: f64, k: f64) -> f64 {
    return a * (-k * target_spread).exp();
}

pub fn get_spread(target_intensity: f64, a: f64, k: f64) -> f64 {
    return -((target_intensity / a).ln()) / k;
}
