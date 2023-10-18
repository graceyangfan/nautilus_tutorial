use nalgebra::DMatrix;
use std::collections::VecDeque;


pub fn get_betas(x: &VecDeque<f64>, y: &VecDeque<f64>) -> (f64, f64) {
    assert!(x.len() > 0 && y.len() > 0 && x.len() == y.len(),
     "Input vectors must have equal lengths and at least one element each.");
    let x_matrix = DMatrix::from_iterator(x.len(), 1, x.iter().cloned());
    let y_vector = DMatrix::from_iterator(y.len(), 1, y.iter().cloned());

    let xy = &x_matrix.transpose() * &y_vector;
    let xx = &x_matrix.transpose() * &x_matrix;

    let xx_inv = xx.try_inverse().unwrap();

    let b_mean = &xx_inv * xy;
    let err = y_vector - x_matrix * &b_mean;
    let b_var = err.transpose() * err / (x.len() as f64 - 1.0) * xx_inv;

    (b_mean[(0, 0)], b_var[(0, 0)])
}




pub fn ols_linear_regression(x: &VecDeque<f64>, y: &VecDeque<f64>) -> (f64, f64, f64) {
    let n = x.len();
    let x_sum = x.iter().sum::<f64>();
    let y_sum = y.iter().sum::<f64>();
    let xy_sum = x.iter().zip(y.iter()).map(|(&x_i, &y_i)| x_i * y_i).sum::<f64>();
    let x_sq_sum = x.iter().map(|&x_i| x_i * x_i).sum::<f64>();

    let slope = (n as f64 * xy_sum - x_sum * y_sum) / (n as f64 * x_sq_sum - x_sum * x_sum);
    let intercept = (y_sum - slope * x_sum) / n as f64;

    let y_hat = x.iter().map(|&x_i| slope * x_i + intercept);
    let residuals = y.iter().zip(y_hat).map(|(&y_i, y_hat_i)| y_i - y_hat_i);
    let last_residual = residuals.rev().next().unwrap();

    (slope, intercept, last_residual)
}






#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_get_betas() {
        let x = VecDeque::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = VecDeque::from(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let (betas, variances) = get_betas(&x, &y);

        assert_abs_diff_eq!(betas,2.0,epsilon=1e-6);
        assert_abs_diff_eq!(variances,0.0,epsilon=1e-6);

        let x = VecDeque::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = VecDeque::from(vec![1.0, 4.0, 6.0, 8.0, 10.0]);

        let (betas, variances) = get_betas(&x, &y);

        assert_abs_diff_eq!(betas,1.98181818,epsilon=1e-6);
        assert_abs_diff_eq!(variances,0.00446281,epsilon=1e-6);
    }
    #[test]
    fn test_ols_linear_regression() {
        let x: VecDeque<f64> = vec![1.0, 2.0, 3.0, 4.0].into_iter().collect();
        let y: VecDeque<f64> = vec![2.0, 4.0, 6.0, 8.0].into_iter().collect();
        let (slope, intercept, residual) = ols_linear_regression(&x, &y);
        assert_abs_diff_eq!(slope, 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(intercept, 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(residual, 0.0, epsilon = 1e-8);
    }
}
