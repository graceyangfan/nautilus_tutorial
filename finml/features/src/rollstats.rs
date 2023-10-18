
use pyo3::prelude::*;
use std::collections::VecDeque;

#[pyclass]
#[derive(Debug)]
pub struct RollStats {
    period: usize,
    avg: f64,
    sum_2: f64,
    sum_3: f64,
    sum_4: f64,
    input_array: VecDeque<f64>,
    initialized: bool,
}

#[pymethods]
impl RollStats {
    #[new]
    pub fn new(period: usize) -> Self {
        Self {
            period: period,
            avg: 0.0,
            sum_2: 0.0,
            sum_3: 0.0,
            sum_4:0.0,
            input_array: VecDeque::with_capacity(period),
            initialized: false
        }
    }
    
    pub fn update_raw(&mut self, input: f64) {
        self.input_array.push_back(input);
        if self.input_array.len() >= self.period {
            self.initialized = true;
        }
        if self.input_array.len() >=self.period + 1
        {
            // sum_a = sum_(xi - avg_t-1) = n(avg_t - avg_t-1) = n*delta_n = n*b
            //sum_a**2 = sum([xi - avg_t-1]**2) 
            // a⁴- 4a³b + 6a²b² - 4ab³ + b⁴
            let n = self.period as f64;
            let delta_n = (input - self.input_array[0]) / n;
            self.sum_2 = self.sum_2 - (self.input_array[0] - self.avg).powf(2.0) +  (input - self.avg).powf(2.0);
            self.sum_3 = self.sum_3 - (self.input_array[0] - self.avg).powf(3.0) +  (input - self.avg).powf(3.0);
            self.sum_4 = self.sum_4 - (self.input_array[0] - self.avg).powf(4.0) +  (input - self.avg).powf(4.0);
            
            self.sum_4 = self.sum_4 - 3.0*n*delta_n.powf(4.0) + 6.0 * delta_n.powf(2.0) * self.sum_2 - 4.0 * delta_n * self.sum_3;
            self.sum_3 = self.sum_3 + 2.0*n*delta_n.powf(3.0) - 3.0*delta_n*self.sum_2;
            self.sum_2 = self.sum_2 - delta_n*n*delta_n;
            self.avg = self.avg + delta_n;
            self.input_array.pop_front();
        }
        else
        {
            //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            let delta = input - self.avg;
            let n = self.input_array.len() as f64;
            self.sum_4 +=  delta.powf(4.0)* (n - 1.0) * (n*n -3.0 *n + 3.0)/ n.powf(3.0) + 6.0* delta.powf(2.0)* self.sum_2 / n.powf(2.0) -4.0 * delta *self.sum_3 / n;
            self.sum_3 +=  delta.powf(3.0)* (n - 1.0) * (n - 2.0) / n.powf(2.0) - 3.0* delta * self.sum_2 / n;
            self.sum_2 +=  delta.powf(2.0)* (n - 1.0) / n;
            self.avg += delta / n;
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn mean(&self) -> f64
    {
        self.avg
    }

    pub fn std(&self) -> f64 
    {
        (self.sum_2/ (self.input_array.len() as f64 - 1.0)).sqrt()
    }
    pub fn skewness(&self) -> f64
    {
        (self.input_array.len() as f64).sqrt() * self.sum_3 /self.sum_2.powf(1.5)
    }

    pub fn kurt(&self) -> f64
    {
        (self.input_array.len() as f64) * self.sum_4 / self.sum_2.powf(2.0)  - 3.0
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    //use approx::assert_abs_diff_eq;
    #[test]
    fn test_stats() {
        let mut sta = RollStats::new(5);
        let values = vec![3.,5.,7.,8.,1.,9.,12.,4.,6.,7.];
        for &item in values.iter()
        {
            sta.update_raw(item as f64);
            println!("The length of input_array is {}",sta.input_array.len());
            println!("{:?}",sta.input_array);
            println!("{}",sta.mean());
            println!("{}",sta.std());
            println!("{}",sta.skewness());
            println!("{}",sta.kurt());
        }
        sta.reset();
        assert!(!sta.initialized());
    }
}
