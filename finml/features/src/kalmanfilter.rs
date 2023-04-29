
use pyo3::prelude::*;
extern crate nalgebra;
use nalgebra::SMatrix;

#[pyclass]
#[derive(Debug)]
pub struct KalmanFilter {
    state_mean: SMatrix<f64,2,1>, // state_mean for kalman filter,[beta1,beta0]^T (2,1)
    q: f64,// moving noise 
    r: f64,//observe noise 
    k: SMatrix<f64,2,1>,//kalman gain 
    p: SMatrix<f64,2,2>, // Covariance matrix (2,2)
    initialized:bool,
}

#[pymethods]
impl KalmanFilter {
    #[new]
    #[args(q = "0.00001", r = "0.0001", p = "10000.0")]
    pub fn new(q:f64, r:f64, p:f64) -> Self {
        Self {
            state_mean: SMatrix::<f64, 2, 1>::zeros(),
            q:q,
            r:r,
            k: SMatrix::<f64, 2, 1>::zeros(),
            p: SMatrix::<f64, 2, 2>::from_vec(vec![p,0.,0.,p]), // Covariance matrix (2,2)
            initialized:false,
        }
    }
    
    pub fn update_raw(&mut self, input_x: f64, input_y: f64) {
        self.initialized = true;
        //forward
        let f = SMatrix::<f64, 2, 2>::identity();
        let x_pred = f * self.state_mean; //2*2*2*1 = (2,1)
        let p_pred = f * self.p * f.transpose() + self.q*SMatrix::<f64, 2, 2>::identity(); // Covariance prediction
        // Correct
        let h = SMatrix::<f64, 1, 2>::from_vec(vec![input_x,1.]);
        let y = (-h*x_pred).add_scalar(input_y);// Innovation
        let s = (h * p_pred * h.transpose()).add_scalar(self.r); // Innovation covariance
        self.k = p_pred * h.transpose() * s.try_inverse().unwrap(); // Kalman gain 2*2*2*1*1 = (2,1)
        self.state_mean = x_pred + self.k * y; // Updated state estimate (2,1)+(2,1) = (2,1)
        self.p = (SMatrix::<f64, 2, 2>::identity() - self.k * h) * p_pred; // Updated covariance estimate
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn intercept(&self) -> f64 {
        self.state_mean[(0,0)]
    }

    pub fn residual(&self) -> f64 {
        self.state_mean[(1,0)]
    }
}

#[cfg(test)]
mod tests {
    // Use the super keyword to bring the KalmanFilter class into scope
    use super::*;

    // Define a test function with the #[test] attribute
    #[test]
    fn test_kalman_filter() {
        // Initialize a KalmanFilter object with some parameters
        let q = 1e-5; // Process noise covariance
        let r = 1e-4; // Measurement noise covariance
        let p = 10000.0; // Initial state covariance
        let mut kf = KalmanFilter::new(q, r, p);
        // Define two arrays of input values
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        // Call the update_raw method with the input values and store the state_mean values
        let mut state_means = Vec::new();
        for (x, y) in x.iter().zip(y.iter()) {
            kf.update_raw(*x, *y);
            state_means.push(vec![kf.intercept(), kf.residual()]);
            println!("intercept: {}, residual: {}", kf.intercept(), kf.residual());
        }
    }
}

