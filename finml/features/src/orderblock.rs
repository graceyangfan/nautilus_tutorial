use pyo3::prelude::*;
use std::collections::VecDeque;

#[derive(Debug,Clone)]
pub struct FVG 
{
    pub is_bull: bool,
    pub mitigated_times: usize,
    pub top: f64,
    pub bottom: f64,
    pub value: f64,
    pub ts_event: u64,
}


#[derive(Debug,Clone)]
pub struct OrderBlock
{
    pub is_bull: bool,
    pub mitigated_times: usize,
    pub top: f64,
    pub bottom: f64,
    pub value: f64,
    pub ts_event: u64,
}

pub fn check_mitigated(block: &OrderBlock, last_target: f64, current_target: f64) -> bool 
{
    if block.is_bull 
    {
        if current_target < block.bottom && block.bottom < last_target
        {
            return true;
        }
    }
    else
    {
        if current_target > block.top && block.top > last_target
        {
            return true;
        }
    }
    false
}

pub fn check_mitigated_for_fvg(fvg:&FVG, last_target: f64, current_target: f64) -> bool 
{
    if fvg.is_bull 
    {
        if current_target < fvg.bottom && fvg.bottom < last_target
        {
            return true;
        }
    }
    else
    {
        if current_target > fvg.top && fvg.top > last_target
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
    ts_event_array: VecDeque<u64>,
    blocks: VecDeque<OrderBlock>,
    fvgs: VecDeque<FVG>,
    period: usize,
    order: usize,
    block_nums: usize,
    initialized: bool,
}

#[pymethods]
impl OrderBlockDetector {
    #[new]
    #[args(block_nums = "1")]
    pub fn new(period: usize, order: usize, block_nums: usize) -> Self {
        Self {
                high_array: VecDeque::with_capacity(period),
                low_array: VecDeque::with_capacity(period),
                close_array: VecDeque::with_capacity(period),
                ts_event_array: VecDeque::with_capacity(period),
                blocks: VecDeque::with_capacity(block_nums),
                fvgs:VecDeque::with_capacity(block_nums),
                period: period,
                order: order,
                block_nums: block_nums,
                initialized: false,
        }
    }
    
    pub fn update_raw(
        &mut self, 
        high: f64,
        low: f64,
        close: f64,
        ts_event: u64,
    )
    {   
        self.high_array.push_back(high);
        self.low_array.push_back(low);
        self.close_array.push_back(close);
        self.ts_event_array.push_back(ts_event);
        if self.high_array.len() >= self.period
        {
            self.initialized = true;
        }
        self.pop_front();

        if self.high_array.len() >= 2*self.order + 1
        {
            let high_values = self.high_array.range(self.high_array.len()-2*self.order-1..self.high_array.len()).copied().collect::<Vec<_>>();
            let low_values = self.low_array.range(self.low_array.len()-2*self.order-1..self.low_array.len()).copied().collect::<Vec<_>>();
            let max_high_idx =  high_values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            let min_low_idx = low_values.iter().enumerate().min_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();            
            let h = high_values[self.order];
            let l = low_values[self.order];
            if max_high_idx  == self.order
            {
                self.blocks.push_back(
                    OrderBlock{
                        is_bull: false,
                        mitigated_times: 0,
                        top: h,
                        bottom: l,
                        value: self.close_array[self.close_array.len()-1-self.order],
                        ts_event: self.ts_event_array[self.ts_event_array.len()-1-self.order]
                    }
                );
            }
            
            if min_low_idx == self.order
            {
                self.blocks.push_back(
                    OrderBlock{
                        is_bull: true,
                        mitigated_times: 0,
                        top: h,
                        bottom: l,
                        value: self.close_array[self.close_array.len()-1-self.order],
                        ts_event: self.ts_event_array[self.ts_event_array.len()-1-self.order]
                    }
                )
            }
            //push fvg 
            let h2 = self.high_array[self.high_array.len() -1-2];
            let h0 = self.high_array[self.high_array.len() -1];
            let l2 = self.low_array[self.low_array.len() -1-2];
            let l0 = self.low_array[self.low_array.len() -1];
            let c2 = self.close_array[self.close_array.len() -1-2];
            let c1 = self.close_array[self.close_array.len() -1-1];
            let bull_fvg_condition = l0 > h2 && c1 > c2 && c1 > h2;
            let bear_fvg_condition = h0 < l2 && c1 < c2 && c1 < l2;

            if bull_fvg_condition
            {
                self.fvgs.push_back(
                    FVG{
                        is_bull: true,
                        mitigated_times: 0,
                        top: l0,
                        bottom: h2,
                        value: (l0+h2)/2.0,
                        ts_event: ts_event,
                    }
                );
            }
            if bear_fvg_condition
            {
                self.fvgs.push_back(
                    FVG{
                        is_bull: false,
                        mitigated_times: 0,
                        top: l2,
                        bottom: h0,
                        value: (l2+h0)/2.0,
                        ts_event: ts_event,
                    }
                );
            }
        }

        // mitigated block 
        for i in (0..self.blocks.len()).rev()
        {
            if self.blocks[i].is_bull 
            {
                if  self.close_array.len()>=2 && check_mitigated(&self.blocks[i], self.close_array[self.close_array.len()-2], close)
                {
                    self.blocks[i].mitigated_times += 1;
                }
            }
            else
            {
                if  self.close_array.len()>=2 && check_mitigated(&self.blocks[i], self.close_array[self.close_array.len()-2], close)
                {
                    self.blocks[i].mitigated_times += 1;
                }
            }
        }

        // check_migated for fvg 
        for i in (0..self.fvgs.len()).rev()
        {
            if self.fvgs[i].is_bull 
            {
                if  self.close_array.len()>=2 && check_mitigated_for_fvg(&self.fvgs[i], self.close_array[self.close_array.len()-2], close)
                {
                    self.fvgs[i].mitigated_times += 1;
                }
            }
            else
            {
                if  self.close_array.len()>=2 && check_mitigated_for_fvg(&self.fvgs[i], self.close_array[self.close_array.len()-2], close)
                {
                    self.fvgs[i].mitigated_times += 1;
                }
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
            self.ts_event_array.pop_front();
        }
        if self.blocks.len() >= self.block_nums + 1
        {
            self.blocks.pop_front();
        } 
        if self.fvgs.len() >=self.block_nums + 1
        {
            self.fvgs.pop_front();
        }
    }


    pub fn block_nums(&self) -> usize
    {
        self.blocks.len()
    }

    pub fn fvg_nums(&self) ->usize
    {
        self.fvgs.len()
    }
    
    pub fn get_block(&self, idx: usize) -> (bool,usize,f64,f64,f64)
    {
        if idx > self.blocks.len() - 1 
        {
            return (false, 0, 0.0, 0.0, 0.0);
        }
        let block = &self.blocks[self.blocks.len()-1-idx];
        (block.is_bull, block.mitigated_times, block.top, block.bottom, block.value)
    }

    pub fn get_fvg(&self, idx: usize) -> (bool,usize,f64,f64,f64)
    {
        if idx > self.fvgs.len() - 1
        {
            return (false, 0, 0.0, 0.0, 0.0);
        }
        let fvg = &self.fvgs[self.fvgs.len()-1-idx];
        (fvg.is_bull, fvg.mitigated_times, fvg.top, fvg.bottom, fvg.value)
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
        // create an instance of OrderBlockDetector with period = 5, threshold = 0.01 and block_nums = 1
        let mut detecter =  OrderBlockDetector::new(5, 0.01, 1);
        // use some sample data for high, low, close, volume and ts_event
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
