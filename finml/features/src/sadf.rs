use nalgebra::DMatrix;
use std::collections::VecDeque;

pub fn get_betas(x: &VecDeque<f64>, y: &VecDeque<f64>) -> (f64, f64) {
    let x_matrix = DMatrix::from_row_slice(x.len(), 1, x.as_slices().0);
    let y_vector = DMatrix::from_row_slice(y.len(), 1, y.as_slices().0);

    let xy = &x_matrix.transpose() * &y_vector;
    let xx = &x_matrix.transpose() * &x_matrix;

    let xx_inv = xx.try_inverse().unwrap();

    let b_mean = &xx_inv * xy;
    let err = y_vector - x_matrix * &b_mean;
    let b_var = err.transpose() * err / (x.len() as f64 - 1.0) * xx_inv;

    (b_mean[(0, 0)], b_var[(0, 0)])
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
}
