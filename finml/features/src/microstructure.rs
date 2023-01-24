use pyo3::prelude::*;
use std::collections::VecDeque;
use statrs::statistics::Statistics;
use statrs::distribution::{Normal, Univariate};
use crate::sadf;


#[pyclass]
#[derive(Debug)]
pub struct MicroStructucture {
    period: usize,
    high_array:VecDeque<f64>,
    low_array:VecDeque<f64>,
    close_array: VecDeque<f64>,
    volume_array: VecDeque<f64>,
    close_diff: VecDeque<f64>,
    close_diff_shift: VecDeque<f64>,
    tick_directions: VecDeque<f64>,
    betas: VecDeque<f64>,
    dollor_value_array: VecDeque<f64>,
    log_net_array: VecDeque<f64>,
    volume_imbalance_array: VecDeque<f64>,
    gamma: f64,
    initialized: bool,
}

#[pymethods]
impl MicroStructucture {
    #[new]
    #[args(b0 = "1.0")]
    pub fn new(period: usize,b0: f64) -> Self {
        let mut data = MicroStructucture {
            period: period,
            high_array: VecDeque::with_capacity(period),
            low_array: VecDeque::with_capacity(period),
            close_array: VecDeque::with_capacity(period),
            volume_array: VecDeque::with_capacity(period),
            volume_imbalance_array: VecDeque::with_capacity(period),
            close_diff: VecDeque::with_capacity(period),
            close_diff_shift: VecDeque::with_capacity(period),
            tick_directions: VecDeque::with_capacity(period),
            betas: VecDeque::with_capacity(period),
            dollor_value_array: VecDeque::with_capacity(period),
            log_net_array: VecDeque::with_capacity(period),
            gamma: 0.0,
            initialized: false,
        };
        data.tick_directions.push_back(b0);
        data
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }    

    pub fn update_raw(
        &mut self, 
        high: f64,
        low: f64,
        close: f64,
        volume: f64
    )
    {   
        self.high_array.push_back(high);
        self.low_array.push_back(low);
        self.close_array.push_back(close);
        self.volume_array.push_back(volume);
        self.dollor_value_array.push_back(close*volume);
        if self.close_array.len() >= 2
        {
            let price_change = self.close_array[self.close_array.len() -1] - self.close_array[self.close_array.len() -2];
            self.close_diff.push_back(price_change);
            if price_change.abs() < 1e-20{
                self.tick_directions.push_back(self.tick_directions[self.tick_directions.len() -1]);
            }
            else{
                self.tick_directions.push_back(price_change.abs() / price_change);
            }
            //compute hl 
            let hl = (high/low).ln().powf(2.0);
            let last_hl = (self.high_array[self.high_array.len() -2] / self.low_array[self.low_array.len() -2]).ln().powf(2.0);
            self.betas.push_back(hl+last_hl);
            let h = high.max(self.high_array[self.high_array.len() -2]);
            let l = low.min(self.low_array[self.low_array.len() -2]);
            self.gamma = (h/l).ln().powf(2.0);
            self.log_net_array.push_back((self.close_array[self.close_array.len() - 1]/ self.close_array[self.close_array.len() -2]).ln());
        }
        if self.close_diff.len() >= 2
        {
            self.close_diff_shift.push_back(self.close_diff[self.close_diff.len()-2]);
        }
        if self.close_diff.len() >= self.period
        {
            let dp = &self.close_diff;
            let std = dp.std_dev();
            let norm  = Normal::new(0.0, 1.0).unwrap();
            let buy_volume = self.volume_array[self.volume_array.len() - 1] * norm.cdf(self.close_diff[self.close_diff.len() - 1] / std);
            let sell_volume = self.volume_array[self.volume_array.len() - 1] - buy_volume;
            let volume_imbalance = (buy_volume - sell_volume).abs();
            self.volume_imbalance_array.push_back(volume_imbalance);
        }

        if self.close_diff_shift.len() >= self.period{
            self.initialized = true;
        }
        self.pop_front();
    }

    pub fn pop_front(&mut self)
    {
        if self.high_array.len() >= self.period + 1 {
            self.high_array.pop_front();
        }
        if self.low_array.len() >= self.period + 1 {
            self.low_array.pop_front();
        }
        if self.close_array.len() >= self.period + 1 {
            self.close_array.pop_front();
        }
        if self.volume_array.len() >= self.period + 1 {
            self.volume_array.pop_front();
        }
        if self.close_diff.len() >= self.period + 1 {
            self.close_diff.pop_front();
        }
        if self.close_diff_shift.len() >= self.period + 1 {
            self.close_diff_shift.pop_front();
        }
        if self.tick_directions.len() >= self.period + 1 {
            self.tick_directions.pop_front();
        }
        if self.betas.len() >= self.period + 1 {
            self.betas.pop_front();
        }
        if self.dollor_value_array.len() >= self.period + 1 {
            self.dollor_value_array.pop_front();
        }
        if self.log_net_array.len() >= self.period + 1 {
            self.log_net_array.pop_front();
        }
        if self.volume_imbalance_array.len() >= self.period + 1 {
            self.volume_imbalance_array.pop_front();
        }
    }

    pub fn roll_effective_spread(&self) -> (f64, f64) {
        let dp = &self.close_array;
        let dp_dif = &self.close_diff;
        let dp_dif_shift = &self.close_diff_shift;
        let dp_var = dp.variance();
        let dp_cov = (dp_dif).covariance(dp_dif_shift);
        let c = (-dp_cov).max(0.0).sqrt()*2.0;
        let dm_var = dp_var + 2.0 * dp_cov;
        (c, dm_var)
    }
    

    pub fn high_low_volatility(&self) -> f64 {
        let high_prices = &self.high_array;
        let low_prices = &self.low_array;
        
        let hl_log_rets: f64 = high_prices
            .iter()
            .zip(low_prices)
            .map(|(high, low)| (high.ln() - low.ln()).powf(2.0))
            .sum::<f64>() / (self.period as f64);
        
        (hl_log_rets / (4.0 * 2.0f64.ln())).sqrt()
    }

    pub fn get_alpha(&self, beta: f64) -> f64{
        let den = 3.0 - 2.0 * 2.0f64.sqrt();
        let mut alpha = (2.0f64.sqrt() - 1.0) * (beta.sqrt()) / den;
        alpha -= (self.gamma / den).sqrt();
        if alpha < 0.0 {
            alpha = 0.0 
        }
        alpha 
    }

    pub fn get_beta(&self) -> f64{
        let beta_array = &self.betas;
        beta_array.mean()
    }
    pub fn get_gamma(&self) ->f64{
        self.gamma 
    }

    pub fn corwin_schultz_volatility(&self, beta: f64) -> f64 {
        let k2 = (8.0 / std::f64::consts::PI).sqrt();
        let denom = 3.0 - 2.0 * 2.0_f64.sqrt();
        let sigma = ((0.5_f64).sqrt() - 1.0) * beta.sqrt() / (k2 * denom) + (self.gamma / (k2.powi(2) * denom)).sqrt();
        if sigma < 0.0 {
            0.0
        } else {
            sigma
        }
    }

    pub fn corwin_schultz_spread(&self) -> (f64, f64)
    {
        let beta = self.get_beta();
        let alpha = self.get_alpha(beta);
        let spread = 2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp());
        let volatility = self.corwin_schultz_volatility(beta);
        (spread,volatility)
    }

    pub fn bar_based_kyle_lambda(&self) -> f64 
    {
        let signed_volume = self.volume_array.iter().zip(self.tick_directions.iter()).map(|(x, y)| x*y).collect::<Vec<_>>();
        let array = self.close_diff.iter().zip(signed_volume.iter()).map(|(x, y)| x/y).collect::<Vec<_>>();
        return array.mean();
    }

    pub fn bar_based_amihud_lambda(&self) -> f64
    {
        let array = self.log_net_array.iter().zip(self.dollor_value_array.iter()).map(|(x, y)| x/y).collect::<Vec<_>>();
        return array.mean();
    }

    pub fn bar_based_hasbrouck_lambda(&self) ->f64 
    {
        let input_x = self.dollor_value_array.iter().zip(self.tick_directions.iter()).map(|(x,d)| x.sqrt() * d).collect::<Vec<_>>();
        let array = self.log_net_array.iter().zip(input_x.iter()).map(|(x, y)| x/y).collect::<Vec<_>>();
        return array.mean();
    }
  
    pub fn trades_based_kyle_lambda(&self) ->(f64, f64)
    {
        let signed_volume = self.volume_array.iter().zip(self.tick_directions.iter()).map(|(x, y)| x*y).collect();
        let (_betas,_variances) = sadf::get_betas(&signed_volume,&self.close_diff);
        let _tvalue = _betas / _variances;
        (_betas,_tvalue)
    }

    pub fn trades_based_amihud_lambda(&self) -> (f64, f64) {
        let (_betas,_variances) = sadf::get_betas(&self.dollor_value_array,&self.log_net_array);
        let _tvalue = _betas / _variances;
        (_betas,_tvalue)
    }
    

    pub fn trades_based_hasbrouck_lambda(&self) ->(f64,f64)
    {
        let input_x = self.dollor_value_array
            .iter()
            .zip(self.tick_directions.iter())
            .map(|(x,d)| x.sqrt() * d).collect();
        let (_betas,_variances) = sadf::get_betas(&input_x,&self.log_net_array);
        let _tvalue = _betas / _variances;
        (_betas,_tvalue)
    }

    pub fn vpin(&self) -> f64
    {
        let volume_imbalance_array = &self.volume_imbalance_array;
        (volume_imbalance_array).mean() / self.volume_array[self.volume_array.len() - 1]
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_microstructucture() {
        // Create a new MicroStructucture object with a period of 10 and a window of 5
        let mut ms = MicroStructucture::new(5, 1.0);

        // Update the MicroStructucture object with some dummy data
        ms.update_raw(138.23, 137.41, 138.23, 438.33);
        ms.update_raw(138.49, 137.5, 138.49, 690.89);
        ms.update_raw(138.87, 137.69, 138.87, 1138.94);
        ms.update_raw(139.06, 137.79, 139.06, 1186.38);
        ms.update_raw(139.28, 137.89, 139.28, 1391.53);
        ms.update_raw(139.53, 138.0, 139.53, 1511.03);
        ms.update_raw(139.61, 138.09, 139.61, 1078.97);
        ms.update_raw(139.68, 138.18, 139.68, 1065.64);
        ms.update_raw(139.84, 138.28, 139.84, 1207.42);
        ms.update_raw(139.89, 138.38, 139.89, 1036.27);
        ms.update_raw(140.17, 138.47, 140.17, 1535.06);
        ms.update_raw(140.47, 138.57, 140.47, 1578.72);
        ms.update_raw(140.61, 138.67, 140.61, 1772.86);
        let (c, dm_var) = ms.roll_effective_spread();
        println!("{},{}",c,dm_var);
        ms.update_raw(140.72, 138.76, 140.72, 1727.96);
        let (c, dm_var) = ms.roll_effective_spread();
        println!("{},{}",c,dm_var);
        ms.update_raw(141.05, 138.86, 141.05, 1791.76);
        let (c, dm_var) = ms.roll_effective_spread();
        println!("{},{}",c,dm_var);
        ms.update_raw(141.06, 138.96, 141.06, 1434.92);
    
        // Check that the MicroStructucture object is initialized
        assert!(ms.initialized());
    
        // Check the values of various fields in the MicroStructucture object
        assert_eq!(ms.high_array, vec![140.47, 140.61, 140.72, 141.05, 141.06]);
        assert_eq!(ms.low_array, vec![138.57, 138.67, 138.76, 138.86, 138.96]);
        assert_eq!(ms.close_array, vec![140.47, 140.61, 140.72, 141.05, 141.06]);
        assert_eq!(ms.volume_array, vec![1578.72, 1772.86, 1727.96, 1791.76, 1434.92]);
        // Calculate the effective spread and check the result
        let (c, dm_var) = ms.roll_effective_spread();
        assert_abs_diff_eq!(c, 0.15962455951389545, epsilon=1e-6);
        assert_abs_diff_eq!(dm_var,0.0572299999999844, epsilon=1e-6);
        // Calculate the high-low volatility and check the result
        let volatility = ms.high_low_volatility();
        assert_abs_diff_eq!(volatility,0.008682445296228003, epsilon=1e-6);
        // Calculate the Corwin-Schultz spread and volatility and check the result
        let  (spread,volatility) = ms.corwin_schultz_spread();
        assert_abs_diff_eq!(spread, 0.010504, epsilon=1e-6);
        assert_abs_diff_eq!(volatility, 0.002311,epsilon=1e-6);
        // Calculate the trades-based Kyle lambda and t-value and check the result
        let (betas,tvalue) = ms.trades_based_kyle_lambda();
        assert_abs_diff_eq!(betas,0.0001092478534217924, epsilon=1e-6);
        assert_abs_diff_eq!(tvalue,93509.72002350067,epsilon=1e-6);
        // Calculate the Amihud lambda and t-value and check the result
        let (lambda, tvalue) = ms.trades_based_amihud_lambda();
        assert_abs_diff_eq!(lambda,5.517580116293106e-9, epsilon=1e-12);
        assert_abs_diff_eq!(tvalue,1849193958.488904,epsilon=1e-6);
        // Calculate the trades-based Hasbrouck lambda and t-value and check the result
        let (lambda, tvalue) = ms.trades_based_hasbrouck_lambda();
        assert_abs_diff_eq!(lambda,2.650398114976172e-6, epsilon=1e-9);
        assert_abs_diff_eq!(tvalue,3591027.05015759,epsilon=1e-6);
        // Calculate the VPIN and check the result
        let vpin = ms.vpin();
        assert_abs_diff_eq!(vpin, 0.8605053721383698,epsilon=1e-6);
    }
}

