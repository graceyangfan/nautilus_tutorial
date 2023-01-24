use pyo3::prelude::*;
use std::collections::{VecDeque,HashMap};
use std::ops::Neg;
use std::cmp;

#[pyclass]
#[derive(Debug)]
pub struct Entropy {
    period: usize,
    min_value: f64,
    quantile_size: f64,
    encoded_array: VecDeque<char>,
    full_alphabet: [char; 62],
    initialized: bool,
}

pub fn word_pmf(encoded_array: &VecDeque<char>, window: usize) -> HashMap<Vec<char>, f64> {
    let mut lib = HashMap::new();
    let period = encoded_array.len() as usize;
    for i in window..period + 1{
        let msg_sub: Vec<char> = encoded_array.range(i - window..i).copied().collect::<Vec<_>>();
        let count = lib.entry(msg_sub).or_insert(vec![]);
        count.push(i - window);
    }
    let n_words = (period - window + 1) as f64;
    let pmf = lib.iter()
        .map(|(word, positions)| (word.clone(), positions.len() as f64 / n_words))
        .collect::<HashMap<_,_>>();
    pmf 
}

pub fn longest_match(msg: &VecDeque<char>, i: usize, n :usize) ->(usize,Vec<char>)
{
    if i < n {
        panic!("start_index should larger than the window");
    }
    let mut match_ = vec![];
    for length in 0..n{
        let msg_right = msg.range(i..cmp::min(i + length + 1,msg.len())).copied().collect::<Vec<_>>();
        for j in i - n..i{
            let msg_left = msg.range(j..cmp::min(j + length + 1,msg.len())).copied().collect::<Vec<_>>();
            if msg_right == msg_left {
                match_ = msg_right;
                break;
            }
        }
    }
    (match_.len() + 1, match_)
}


#[pymethods]
impl Entropy {
    #[new]
    #[args(min_value = "0.0",max_value = "1.0")]
    pub fn new(
        period: usize, 
        n_quantiles: usize,
        min_value: f64, 
        max_value: f64) -> Self {
    
        let full_alphabet: [char; 62] = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ];
        let range = max_value - min_value;
        let quantile_size = range / n_quantiles as f64;
        Self {
            period: period,
            min_value: min_value,
            quantile_size: quantile_size,
            encoded_array: VecDeque::with_capacity(period),
            full_alphabet: full_alphabet,
            initialized: false,
        }
    }

    pub fn update_raw(&mut self, input: f64) {
        let quantile = (input - self.min_value) / self.quantile_size;
        let index = quantile.floor() as usize % self.full_alphabet.len();
        self.encoded_array.push_back(self.full_alphabet[index]);
        if self.encoded_array.len() >= self.period {
            self.initialized = true;
        }
        if self.encoded_array.len() >= self.period + 1 {
            self.encoded_array.pop_front();
        }
    }

    pub fn shannon_entropy(&self) -> f64 
    {
        let mut exr = HashMap::new();
        let mut entropy = 0.0;
        for each in self.encoded_array.iter() {
            let count = exr.entry(each).or_insert(0);
            *count += 1;
        }
        let textlen = self.period as f64;
        for value in exr.values() {
            let freq = *value as f64 / textlen;
            entropy += freq * freq.log2();
        }
        entropy *= -1.0;
        entropy
    }


    pub fn plugin_entropy(&self, window: usize) -> f64 {
        let pmf = word_pmf(&self.encoded_array,window);
        let res = pmf.iter()
                      .map(|(_, prob)| prob * prob.log2())
                      .sum::<f64>()
                      .neg() / window as f64;
        res 
    }

    #[args(window = "0")]
    pub fn konto_entropy(&self, window: usize) -> f64
    {
        let mut sum_entropy = 0.0;
        let mut n_points = 0;
        if window == 0 
        {
            for i in 1..(self.period / 2 + 1) {
                let (l, _) = longest_match(&self.encoded_array, i, i);
                sum_entropy += (i as f64 + 1.0).log2() / l as f64;
                n_points += 1;
            }
        }
        else
        {
            for  i in window..self.period - window + 1
            {
                let (l, _) = longest_match(&self.encoded_array, i ,window);
                sum_entropy += (window  as f64 + 1.0).log2() / l as f64;
                n_points += 1;
            }
        }
        sum_entropy / n_points as f64
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
    use std::collections::VecDeque;

    #[test]
    fn test_word_pmf() {
        let encoded_array = VecDeque::from(vec!['a','b','c','d','a','w','a','d','w','a','f','e']);
        let window = 3;
        let expected_pmf = vec![
            (vec!['a', 'b', 'c'], 0.1),
            (vec!['b', 'c', 'd'], 0.1),
            (vec!['c', 'd', 'a'], 0.1),
            (vec!['d', 'a', 'w'], 0.1),
            (vec!['a', 'w', 'a'], 0.1),
            (vec!['w', 'a', 'd'], 0.1),
            (vec!['a', 'd', 'w'], 0.1),
            (vec!['d', 'w', 'a'], 0.1),
            (vec!['w', 'a', 'f'], 0.1),
            (vec!['a', 'f', 'e'], 0.1),
        ].iter().cloned().collect::<HashMap<_, _>>();
        let result = word_pmf(&encoded_array, window);
        assert_eq!(result, expected_pmf);
    }

    #[test]
    fn test_longest_match() {
        let encoded_array = VecDeque::from(
            "1111101111121221111111"
            .chars()
            .collect::<Vec<char>>()
        );
        
        let (l, sub_msg) = longest_match(&encoded_array, 3, 2);
        assert_eq!(l, 3);
        assert_eq!(sub_msg, vec!['1', '1']);
    }

    #[test]
    fn test_shannon_entropy() {
        let mut entropy = Entropy::new(8, 2, 0.0, 1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(1.0);
        let result = entropy.shannon_entropy();
        assert_eq!(result,1.0);
    }

    #[test]
    fn test_plugin_entropy() {
        let mut entropy = Entropy::new(8, 2, 0.0, 1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(1.0);
        let result = entropy.plugin_entropy(1);
        assert_abs_diff_eq!(result,1.0,epsilon=1e-6);
        let result = entropy.plugin_entropy(2);
        assert_abs_diff_eq!(result,0.9211854965885542,epsilon=1e-6);
        let result = entropy.plugin_entropy(3);
        assert_abs_diff_eq!(result,0.750543055795941,epsilon=1e-6);
    }

    #[test]
    fn test_konto_entropy() {
        let mut entropy = Entropy::new(8, 2, 0.0, 1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(1.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(0.0);
        entropy.update_raw(1.0);
        let result = entropy.konto_entropy(0);
        assert_abs_diff_eq!(result,0.9682408185206046,epsilon=1e-6);
        let result = entropy.konto_entropy(1);
        assert_abs_diff_eq!(result,0.6428571428571429,epsilon=1e-6);
        let result = entropy.konto_entropy(2);
        assert_abs_diff_eq!(result,0.8453133337179499,epsilon=1e-6);
    }
}
