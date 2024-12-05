use pyo3::prelude::*;
use std::collections::VecDeque;

#[derive(Debug,Clone)]
pub struct OrderBlock
{
    pub is_bull: bool,
    pub top: f64,
    pub bottom: f64,
    pub value: f64,
    pub ts_event: u64,
}

pub fn remove_mitigated(block: &OrderBlock, target: f64) -> bool 
{
    if block.is_bull 
    {
        if target < block.bottom
        {
            return true;
        }
    }
    else
    {
        if target > block.top
        {
            return true;
        }
    }
    false
}

#[pyclass]
#[derive(Debug)]
pub struct OrderBlockDetector {
    high_array:VecDeque<f64>,
    low_array:VecDeque<f64>,
    close_array: VecDeque<f64>,
    volume_array: VecDeque<f64>,
    bull_blocks: VecDeque<OrderBlock>,
    bear_blocks: VecDeque<OrderBlock>,
    period: usize,
    block_nums: usize,
    delta_pct: f64,
    os: usize,
    initialized: bool,
}

#[pymethods]
impl OrderBlockDetector {
    #[new]
    #[args(block_nums = "1")]
    pub fn new(period: usize, delta_pct: f64, block_nums: usize) -> Self {
        Self {
                high_array: VecDeque::with_capacity(period),
                low_array: VecDeque::with_capacity(period),
                close_array: VecDeque::with_capacity(period),
                volume_array: VecDeque::with_capacity(2*period+1),
                bull_blocks: VecDeque::with_capacity(block_nums),
                bear_blocks: VecDeque::with_capacity(block_nums),
                period: period,
                block_nums: block_nums,
                delta_pct: delta_pct,
                os: 0,
                initialized: false,
        }
    }
    
    pub fn update_raw(
        &mut self, 
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        ts_event: u64,
    )
    {   
        let mut h = high;
        let mut l = low;
        let mut hl = (h + l)/2.0;
        if self.volume_array.len() >= 2*self.period + 1
        {
            self.initialized = true;
            h = self.high_array[0];
            l = self.low_array[0];
            hl = (h + l)/2.0;
        }
        self.pop_front();

        self.high_array.push_back(high);
        self.low_array.push_back(low);
        self.close_array.push_back(close);
        self.volume_array.push_back(volume);


        let upper = self.high_array.iter().max_by(|a, b| a.total_cmp(b)).cloned().unwrap();
        let lower = self.low_array.iter().min_by(|a, b| a.total_cmp(b)).cloned().unwrap();
        let target_bull = self.close_array.iter().min_by(|a, b| a.total_cmp(b)).cloned().unwrap();
        let target_bear = self.close_array.iter().max_by(|a, b| a.total_cmp(b)).cloned().unwrap();

        if h > upper
        {
            self.os = 0;
        }
        else
        {
            if l < lower
            {
                self.os = 1;
            }
        }
        
        let max_volume = self.volume_array.iter().max_by(|a, b| a.total_cmp(b)).cloned().unwrap();
        let phv = self.volume_array[(self.volume_array.len()-1)/2] / max_volume > 1.0 - self.delta_pct;

        if phv && self.os == 1 && self.initialized
        {
            self.bull_blocks.push_back(
                    OrderBlock{
                        is_bull: true,
                        top: hl,
                        bottom: l,
                        value: l,
                        ts_event: ts_event,
                    }
            )
        }

        if phv && self.os == 0 && self.initialized
        {
            self.bear_blocks.push_back(
                    OrderBlock{
                        is_bull: false,
                        top: h,
                        bottom: hl,
                        value: h,
                        ts_event: ts_event,
                    }
            )
        }
        
        // remove mitigated block 
        for i in (0..self.bull_blocks.len()).rev()
        {
            if remove_mitigated(&self.bull_blocks[i], target_bull)
            {
                self.bull_blocks.remove(i);
            }
        }
       
        for i in (0..self.bear_blocks.len()).rev()
        {
            if remove_mitigated(&self.bear_blocks[i], target_bear)
            {
                self.bear_blocks.remove(i);
            }
        }
    }

    pub fn pop_front(&mut self)
    {
        if self.high_array.len() >= self.period + 1 
        {
            self.high_array.pop_front();
            self.low_array.pop_front();
            self.close_array.pop_front();
        }
        if self.volume_array.len() >= 2 * self.period + 2 
        {
            self.volume_array.pop_front();
        }
        if self.bull_blocks.len() >= self.block_nums + 1
        {
            self.bull_blocks.pop_front();
        }
        if self.bear_blocks.len() >= self.block_nums + 1 
        {
            self.bear_blocks.pop_front();
        }
    }

    pub fn bull_blocks_num(&self) -> usize {
        self.bull_blocks.len()
    }

    pub fn bear_blocks_num(&self) -> usize {
        self.bear_blocks.len()
    }

    pub fn get_bull_block(&self, idx: usize) -> (f64, f64, f64, u64)
    {
        if idx > self.bull_blocks_num() - 1 
        {
            return (0.0, 0.0, 0.0, 0);
        }
        let block = &self.bull_blocks[self.bull_blocks.len()-1-idx];
        (block.top, block.bottom, block.value, block.ts_event)
    }

    pub fn get_bear_block(&self, idx: usize) -> (f64, f64, f64, u64)
    {
        if idx > self.bear_blocks_num() - 1 
        {
            return (0.0, 0.0, 0.0, 0);
        }
        let block = &self.bear_blocks[self.bear_blocks.len()-1-idx];
        (block.top, block.bottom, block.value, block.ts_event)
    }

    
    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_orderblock_detect()
    {
        // create an instance of OrderBlockDetector with period = 5, delta_pct = 0.01 and block_nums = 1
        let mut detecter =  OrderBlockDetector::new(5, 0.01, 1);
        // use some sample data for high, low, close, volume and timestamp
        let high = vec![10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10350.0];
        let low = vec![9900.0, 10000.0, 10100.0, 10200.0, 10300.0, 10250.0];
        let close = vec![10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10300.0];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0];
        let ts_event = vec![1u64, 2u64, 3u64 ,4u64 ,5u64 ,6u64];
    
        // loop through the data and update the detecter
        for i in 0..high.len() {
            detecter.update_raw(high[i], low[i], close[i], volume[i], ts_event[i]);
        }
    
        // check the number of bull blocks
        assert_eq!(detecter.bull_blocks_num(), 1);
        // check the number of bear blocks
        assert_eq!(detecter.bear_blocks_num(), 1);
        // check the details of the bull block
        let (top_bull, bottom_bull, value_bull ,ts_event_bull) = detecter.get_bull_block(0);
        assert_abs_diff_eq!(top_bull ,10150.0);
        assert_abs_diff_eq!(bottom_bull ,10050.0);
        assert_abs_diff_eq!(value_bull ,10050.0);
        assert_eq!(ts_event_bull ,2u64);
        // check the details of the bear block
        let (top_bear ,bottom_bear ,value_bear ,ts_event_bear) = detecter.get_bear_block(0);
        assert_abs_diff_eq!(top_bear ,10400.0);
        assert_abs_diff_eq!(bottom_bear ,10325.0);
        assert_abs_diff_eq!(value_bear ,10400.0);
        assert_eq!(ts_event_bear ,5u64);
    
    }
}