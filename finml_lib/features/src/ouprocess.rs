use pyo3::prelude::*;
use std::collections::VecDeque;
use crate::sadf;


#[pyclass]
#[derive(Debug)]
pub struct OUProcessTransformer{
    x_data: VecDeque<f64>,
    y_data: VecDeque<f64>,
    period: usize,
    t_score: f64,
    initialized: bool,
}

#[pymethods]
impl OUProcessTransformer{
    #[new]
    fn new(period: usize) -> Self {
        OUProcessTransformer{
            x_data: VecDeque::with_capacity(period),
            y_data: VecDeque::with_capacity(period),
            period,
            t_score: 0.0,
            initialized: false,
        }
    }
    
    fn update_raw(&mut self, input_x: f64, input_y: f64) {
        self.x_data.push_back(input_x);
        self.y_data.push_back(input_y);
        if self.y_data.len() >= self.period {
            self.initialized = true;
        }
        if self.y_data.len() >= self.period + 1 {
            self.x_data.pop_front();
            self.y_data.pop_front();
            
            let (slope, _, _) = sadf::ols_linear_regression(&self.x_data, &self.y_data);

            let residuals: VecDeque<f64> = self.y_data.iter()
                .zip(self.x_data.iter())
                .map(|(&s1_i, &s2_i)| s1_i - s2_i * slope)
                .collect();

            let x_t: VecDeque<f64> = residuals.iter().scan(0.0, |state, &x| {
                *state += x;
                Some(*state)
            }).collect();
            
            let mut lag_price: VecDeque<f64> = x_t.clone();
            lag_price.push_front(0.0);
            lag_price.pop_back();
            
            let x_t_slice: VecDeque<f64> = x_t.iter().skip(1).cloned().collect();
            let lag_price_slice: VecDeque<f64> = lag_price.iter().skip(1).cloned().collect();
            
            let (b, a, _) = sadf::ols_linear_regression(&x_t_slice, &lag_price_slice);
            let mu = a / (1.0 - b);
            let x_t_mean = x_t.iter().sum::<f64>() / x_t.len() as f64;
            let sigma = (x_t.iter().map(|&x| (x - x_t_mean).powi(2)).sum::<f64>() / (x_t.len() - 1) as f64).sqrt();                                           
            self.t_score = (x_t[x_t.len() - 1] - mu) / sigma;
        }
    }

    fn reset(&mut self) {
        self.initialized = false;
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn t_score(&self) -> f64 {
        self.t_score
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ou_process_transformer() {
        let mut transformer = OUProcessTransformer::new(5);

        // Test initialization
        assert!(!transformer.initialized());

        // Test update and t_score
        for i in 0..100 {
            let input_x = i as f64;
            let input_y = 2.0*input_x*input_x; // Example input data

            transformer.update_raw(input_x, input_y);

            if i == 90 {
                // Add assertion for the expected t_score value
                println!("{}",transformer.t_score());
                //assert_abs_diff_eq!(transformer.t_score(), expected_t_score, epsilon = 1e-6);
            }
        }
        // Test reset
        transformer.reset();
        assert!(!transformer.initialized());
    }
}

