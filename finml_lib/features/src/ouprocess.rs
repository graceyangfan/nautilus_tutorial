use pyo3::prelude::*;
extern crate nalgebra;


#[pyclass]
#[derive(Debug)]
struct OUProcess {
    mu: f64,
    sigma: f64,
    theta: f64,
    initialized:bool,
}

#[pymethods]
impl OUProcess {

    fn new(mu: f64, sigma: f64, theta: f64) -> Self {
        OUProcess {
            mu,
            sigma,
            theta,
            initialized: false,
        }
    }
    
    fn update_raw(&mut self, input_x: f64, input_y: f64) {
        self.initialized = true;
        
    }

    fn reset(&mut self) {
        self.initialized = false;
    }

    fn initialized(&self) -> bool {
        self.initialized
    }
}