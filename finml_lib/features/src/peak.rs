use pyo3::prelude::*;
use std::collections::VecDeque;

#[pyclass]
#[derive(Debug)]
pub struct PeakDetector {
    high_array:VecDeque<f64>,
    low_array:VecDeque<f64>,
    indicate_array:VecDeque<f64>,
    peak_values: VecDeque<f64>,
    peak_indicate_values: VecDeque<f64>,
    period: usize,
    order: usize,
    initialized: bool,
}

#[pymethods]
impl PeakDetector {
    #[new]
    pub fn new(period: usize, order: usize) -> Self {
        Self {
                high_array: VecDeque::with_capacity(period),
                low_array: VecDeque::with_capacity(period),
                indicate_array: VecDeque::with_capacity(period),
                peak_values: VecDeque::with_capacity(5),
                peak_indicate_values: VecDeque::with_capacity(5),
                period: period,
                order: order,
                initialized: false,
        }
    }
    
    pub fn update_raw(
        &mut self, 
        high: f64,
        low: f64,
        indicate_value: f64,
    )
    {   
        self.high_array.push_back(high);
        self.low_array.push_back(low);
        self.indicate_array.push_back(indicate_value);
        self.pop_front();
        if self.high_array.len() >= self.period
        {
            self.initialized = true;
        }

        if self.high_array.len() >= 2*self.order + 1
        {
            let high_values = self.high_array.range(self.high_array.len()-2*self.order-1..self.high_array.len()).copied().collect::<Vec<_>>();
            let low_values = self.low_array.range(self.low_array.len()-2*self.order-1..self.low_array.len()).copied().collect::<Vec<_>>();
            let max_high_idx =  high_values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            let min_low_idx = low_values.iter().enumerate().min_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();            
            let real_index = self.high_array.len() - 1 - self.order;
            if max_high_idx  == self.order 
            {
                self.peak_values.push_back(self.high_array[real_index]);
                self.peak_indicate_values.push_back(self.indicate_array[real_index]);
            }
            
            if min_low_idx == self.order 
            {
                self.peak_values.push_back(self.low_array[real_index]);
                self.peak_indicate_values.push_back(self.indicate_array[real_index]);
            }
        }
    }


    pub fn pop_front(&mut self)
    {
        if self.high_array.len() >= self.period + 1 
        {
            self.high_array.pop_front();
            self.low_array.pop_front();
            self.indicate_array.pop_front();
        }        
        if self.peak_values.len() >= 6
        {
            self.peak_values.pop_front();
            self.peak_indicate_values.pop_front();
        }
    }

    pub fn  peak_lengths(&self) -> usize
    {
        self.peak_values.len()
    }


    pub fn get_peak_values(&self) -> (f64,f64,f64,f64,f64)
    {
        if self.peak_lengths() < 5
        {
            return (0.0,0.0,0.0,0.0,0.0);
        }
        else
        {
            return (self.peak_values[0],self.peak_values[1],self.peak_values[2],self.peak_values[3],self.peak_values[4]);
        }
    }
    
    pub fn get_peak_indicate_values(&self) -> (f64,f64,f64,f64,f64)
    {
        if self.peak_lengths() < 5
        {
            return (0.0,0.0,0.0,0.0,0.0);
        }
        else
        {
            return (self.peak_indicate_values[0],self.peak_indicate_values[1],self.peak_indicate_values[2],self.peak_indicate_values[3],self.peak_indicate_values[4]);
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

}


