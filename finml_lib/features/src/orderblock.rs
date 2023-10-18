use pyo3::prelude::*;
use std::collections::VecDeque;

#[derive(Debug,Clone)]
pub struct ZSBlock
{
    pub start: OrderBlock,
    pub end: OrderBlock,
    pub gg: f64,
    pub dd: f64,
    pub zg: f64,
    pub zd: f64
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



#[pyclass]
#[derive(Debug)]
pub struct OrderBlockDetector {
    high_array:VecDeque<f64>,
    low_array:VecDeque<f64>,
    close_array: VecDeque<f64>,
    ts_event_array: VecDeque<u64>,
    blocks: VecDeque<OrderBlock>,
    big_blocks: VecDeque<OrderBlock>,
    zsblocks: VecDeque<ZSBlock>,
    zs_threshold: f64,
    small_block_updated: bool,
    big_block_updated: bool,
    small_block_values: VecDeque<f64>,
    period: usize,
    order: usize,
    reduce_order: usize,
    block_nums: usize,
    zs_nums: usize,
    initialized: bool,
}

#[pymethods]
impl OrderBlockDetector {
    #[new]
    #[args(block_nums = "8",reduce_order = "3",zs_num = 3)] 
    pub fn new(
        period: usize, 
        order: usize, 
        reduce_order: usize, 
        block_nums: usize,
        zs_nums: usize,
        zs_threshold: f64,
    ) -> Self {
        Self {
                high_array: VecDeque::with_capacity(period),
                low_array: VecDeque::with_capacity(period),
                close_array: VecDeque::with_capacity(period),
                ts_event_array: VecDeque::with_capacity(period),
                blocks: VecDeque::with_capacity(block_nums),
                big_blocks: VecDeque::with_capacity(block_nums),
                zsblocks: VecDeque::with_capacity(block_nums),
                zs_threshold: zs_threshold,
                small_block_updated: false,
                big_block_updated: false,
                small_block_values: VecDeque::with_capacity(block_nums),
                period: period,
                order: order,
                reduce_order: reduce_order,
                block_nums: block_nums,
                zs_nums: zs_nums,
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
        self.small_block_updated = false;
        self.big_block_updated = false;
        self.high_array.push_back(high);
        self.low_array.push_back(low);
        self.close_array.push_back(close);
        self.ts_event_array.push_back(ts_event);
        if self.zsblocks.len() >= self.zs_nums
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
                        value: h,
                        ts_event: self.ts_event_array[self.ts_event_array.len()-1-self.order]
                    }
                );
                self.small_block_updated = true;
                self.small_block_values.push_back(h);
            }
            
            if min_low_idx == self.order 
            {
                self.blocks.push_back(
                    OrderBlock{
                        is_bull: true,
                        mitigated_times: 0,
                        top: h,
                        bottom: l,
                        value: l,
                        ts_event: self.ts_event_array[self.ts_event_array.len()-1-self.order]
                    }
                );
                self.small_block_updated = true;
                self.small_block_values.push_back(l);
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
        // small blocks into bigblock 
        if self.small_block_updated && self.small_block_values.len() >=2* self.reduce_order + 1
        {
            let block_values = self.small_block_values.range(self.small_block_values.len()-2*self.reduce_order-1..self.small_block_values.len()).copied().collect::<Vec<_>>();
            let max_idx =  block_values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            let min_idx = block_values.iter().enumerate().min_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();  
            if max_idx == self.reduce_order  
            {
                let block = &self.blocks[max_idx];
                self.big_blocks.push_back(
                    OrderBlock{
                        is_bull: false,
                        mitigated_times: block.mitigated_times,
                        top: block.top,
                        bottom: block.bottom,
                        value: block.value,
                        ts_event: block.ts_event
                    }
                );
                self.big_block_updated = true;
            }
            if  min_idx == self.reduce_order  
            {   
                let block = &self.blocks[min_idx];
                self.big_blocks.push_back(
                    OrderBlock{
                        is_bull: true,
                        mitigated_times: block.mitigated_times,
                        top: block.top,
                        bottom: block.bottom,
                        value: block.value,
                        ts_event: block.ts_event
                    }
                );
                self.big_block_updated = true;
            }
        }
        //update ZS structure 
        if self.big_block_updated && self.big_blocks.len() >=4
        {
            let cur_block = &self.big_blocks[self.big_blocks.len()-1];
            // no ZS blocks or ZSblock end is not the last bigblock 
            if self.zsblocks.len() == 0 || self.zsblocks[self.zsblocks.len()-1].end.ts_event != self.big_blocks[self.big_blocks.len()-2].ts_event 
            {
                //create a new zs 
                let back_block_1 = &self.big_blocks[self.big_blocks.len()-2];
                let back_block_2 = &self.big_blocks[self.big_blocks.len()-3];
                let back_block_3 = &self.big_blocks[self.big_blocks.len()-4];

                let max_one = f64::max(cur_block.value, back_block_1.value);
                let min_one = f64::min(cur_block.value, back_block_1.value);
                let max_two = f64::max(back_block_2.value, back_block_3.value);
                let min_two = f64::min(back_block_2.value, back_block_3.value);
                let zg = f64::min(max_two, max_one);
                let zd = f64::max(min_two, min_one);
                let gg = f64::max(max_two, max_one);
                let dd = f64::min(min_two, min_one);
                if zg > zd && (zg-zd)/(gg - dd) < self.zs_threshold 
                {
                    self.zsblocks.push_back(
                        ZSBlock{
                            start: back_block_3.clone(),
                            end: cur_block.clone(),
                            gg: gg,
                            dd: dd,
                            zg: zg,
                            zd: zd,
                        }
                    )
                };
            }    
            else 
            {
                let mut cur_zs = self.zsblocks.pop_back().unwrap();
                // check the old ZS should continue 
                if cur_block.is_bull 
                {
                    if cur_block.top > cur_zs.gg && (cur_zs.zg-cur_zs.zd)/(cur_block.top - cur_zs.dd) < self.zs_threshold
                    {
                        cur_zs.end = cur_block.clone();
                        cur_zs.gg = cur_block.top;
                    }
                    else if cur_block.top < cur_zs.gg && cur_block.top > cur_zs.zg 
                    {
                        cur_zs.end = cur_block.clone();
                        cur_zs.zg = cur_block.top;
                    }
                }
                else 
                {
                    if cur_block.bottom < cur_zs.dd && (cur_zs.zg-cur_zs.zd)/(cur_zs.gg - cur_block.bottom) < self.zs_threshold
                    {
                        cur_zs.end = cur_block.clone();
                        cur_zs.dd = cur_block.bottom;
                    }
                    else if cur_block.bottom > cur_zs.dd && cur_block.bottom < cur_zs.zd 
                    {
                        cur_zs.end = cur_block.clone();
                        cur_zs.zd = cur_block.bottom;
                    }
                }
                self.zsblocks.push_back(cur_zs);
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
            self.small_block_values.pop_front();
        } 
        if self.big_blocks.len() >= self.block_nums + 1
        {
            self.big_blocks.pop_front();
        }
        if self.zsblocks.len() >= self.zs_nums + 1
        {
            self.zsblocks.pop_front();
        }
    }


    pub fn zs_nums(&self) -> usize
    {
        self.zsblocks.len()
    }

    pub fn get_zs(&self, idx: usize) -> (u64,u64,f64, f64, f64, f64, f64, f64)
    {
        if idx > self.zsblocks.len() - 1 
        {
            return (0, 0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let zs = &self.zsblocks[self.zsblocks.len()-1-idx];
        (zs.start.ts_event, zs.end.ts_event,zs.start.value, zs.end.value, zs.gg, zs.dd, zs.zg, zs.zd)
    }

    pub fn big_block_nums(&self) -> usize
    {
        self.big_blocks.len()
    }

    
    pub fn get_big_block(&self, idx: usize) -> (u64, bool,usize,f64,f64,f64)
    {
        if idx > self.big_blocks.len() - 1 
        {
            return (0, false, 0, 0.0, 0.0, 0.0);
        }
        let block = &self.big_blocks[self.big_blocks.len()-1-idx];
        (block.ts_event, block.is_bull, block.mitigated_times, block.top, block.bottom, block.value)
    }

   
    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }
}

