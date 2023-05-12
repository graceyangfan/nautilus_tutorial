use pyo3::prelude::*;
use std::collections::VecDeque;
use std::cmp;

#[derive(Debug,Clone)]
pub struct OrderBlock
{
    pub is_bull: bool,
    pub top: f64,
    pub bottom: f64,
    pub value: f64,
    pub ts_event: u64,
}

#[derive(Debug,Clone)]
pub struct FVG 
{
    pub is_bull: bool,
    pub top: f64,
    pub bottom: f64,
    pub value: f64,
    pub ts_event: u64,
}

#[pyclass]
#[derive(Debug)]
pub struct SmartMoneyConcept {
    high_array:VecDeque<f64>,
    low_array:VecDeque<f64>,
    close_array: VecDeque<f64>,
    volume_array: VecDeque<f64>,
    trend: usize,
    itrend: usize,
    top_y: f64,
    top_x: usize,
    btm_y: f64,
    btm_x: usize,
    itop_y: f64,    
    itop_x: usize,
    ibtm_y: f64,
    ibtm_x: usize,
    trail_up: f64,
    trail_dn: f64,
    trail_up_x: usize,
    trail_dn_x: usize,
    top_cross: bool,
    btm_cross: bool,
    itop_cross: bool,
    ibtm_cross: bool,   
    eq_prev_top: f64,
    eq_prev_btm: f64,
    os: usize,
    last_os: usize,
    initialized: bool,
}

#[pymethods]
impl SmartMoneyConcept {
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
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        ts_event: u64,
    )
    {   

        


        self.top, self.btm = self.swings(length);
        self.itop, self.ibtm = self.swings(5);

        //Pivot High
        if self.top.abs() > 1e-10
        {
            self.top_cross = true;
            if self.top > self.top_y
            {
                self.top_type = "HH";
            }
            else
            {
                self.top_type = "LH";
            }
            self.top_y = self.top;
            self.trail_up = self.top;
        }

        if self.itop.abs() > 1e-10
        {
            self.itop_cross = true;
            self.itop_y = self.itop;
        }

        self.trail_up = cmp::max(high,self.trail_up);

        
        //Pivot Low
        if self.btm.abs() > 1e-10
        {
            self.btm_cross = true;
            if self.btm < self.btm_y
            {
                self.btm_type = "LL";
            }
            else
            {
                self.btm_type = "HL";
            }
            self.btm_y = self.btm;
            self.trail_dn = self.btm;
        }

        if self.ibtm.abs() > 1e-10
        {
            self.ibtm_cross = true;
            self.ibtm_y = self.ibtm;
        }

        self.trail_dn = cmp::min(low,self.trail_dn);

        //Pivot High BOS/CHoCH
        let bull_concordant =  high - cmp::max(close,open) > cmp::min(close,open) - low;

        //Detect internal bullish Structure
        if close > self.top_y && open < self.last_top_y && self.top_cross
        {
            let txt = "BOS";
            if self.trend < 0
            {
                self.choch = true; //change of character
                txt = "CHoCH";
                self.bull_choch_alert = true;
            }
            else
            {
                self.bull_bos_alert = true;
            }
            self.top_cross = false;
            self.trend = 1;
        }
        
        //Pivot Low BOS/CHoCH
        let bear_concordant =  high - cmp::max(close,open) < cmp::min(close,open) - low;

        if cloe < self.btm_y && open > self.last_btm_y && self.btm_cross
        {
            let txt = "BOS";
            if self.trend > 0
            {
                self.choch = true; //change of character
                txt = "CHoCH";
                self.bear_choch_alert = true;
            }
            else
            {
                self.bear_bos_alert = true;
            }
            self.btm_cross = false;
            self.trend = -1;
        }

        //Order Blocks detect 

        //EQH/EQL 
        self.eq_top = self.high_array[(self.high_array.len()-1)/2] == self.upper;
        self.eq_btm = self.low_array[(self.low_array.len()-1)/2] == self.lower;

        if self.eq_top 
        {
            let max = cmp::max(self.eq_top, self.eq_prev_top);
            let min = cmp::min(self.eq_top, self.eq_prev_top);
            if max < min + self.atr * self.eq_threshold
            {
                self.eqh_line = Some(Line::new(self.eq_top_x, self.eq_prev_top, self.n-self.eq_len, self.eq_top));
                self.eqh_lbl = Some(Label::new((self.n-self.eq_len+self.eq_top_x)/2, self.eq_top, "EQH"));
                self.eqh_alert = true;
            }
            self.eq_prev_top = self.eq_top;
        }
        if self.eq_btm
        {
            let max = cmp::max(self.eq_btm, self.eq_prev_btm);
            let min = cmp::min(self.eq_btm, self.eq_prev_btm);
            if max < min + self.atr * self.eq_threshold
            {
                self.eql_line = Some(Line::new(self.eq_btm_x, self.eq_prev_btm, self.n-self.eq_len, self.eq_btm));
                self.eql_lbl = Some(Label::new((self.n-self.eq_len+self.eq_btm_x)/2, self.eq_btm, "EQL"));
                self.eql_alert = true;
            }
            self.eq_prev_btm = self.eq_btm;
        }

        //Fair Value Gaps
        let src_c1 = self.close_array[self.close_array.len()-2];
        let scr_o1 = self.open_array[self.open_array.len()-2];
        let src_h = high;
        let src_l = low;
        let src_h2 = self.high_array[self.high_array.len()-3];
        let src_l2 = self.low_array[self.low_array.len()-3];
        let delta_per = (src_c1 - src_o1) / src_o1 * 100.0;
        let threshold = 0;
        if src_l > src_h2 && src_c1 > src_h2  && delta_per > threshold
        {
            self.bullish_fvg_cnd  = true;
            //insert FVG 

        }
        if src_h < src_l2 && src_c1 < src_l2 && delta_per < -threshold
        {
            self.bearish_fvg_cnd = true;
        }
        // delete remove_mitigated FVG 

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

    pub fn swings(&mut self, length: usize)-> (f64,f64)
    {
       let high_arry = self.high_array.range(self.high_array.len() - length..).copied().collect::<Vec<_>>();
       let low_array = self.low_array.range(self.low_array.len() - length..).copied().collect::<Vec<_>>();
       let upper = high_array.iter().max_by(|a, b| a.total_cmp(b)).cloned().unwrap(); 
       let lower = low_array.iter().min_by(|a, b| a.total_cmp(b)).cloned().unwrap(); 
       
       if high_array[0] > upper
       {
           self.os = 0;
       }
       else
       {
           if low_array[0] < lower
           {
               self.os = 1;
           }
       }

       let top = 0.0;
       let btm = 0.0;
       if self.os == 0 && self.last_os != 0
       {
           top = high_array[0];
       }
       
       if self.os == 1 && self.last_os != 1
       {
           btm = low_array[0];
       }
       (top, btm)
    }

    
    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

}
