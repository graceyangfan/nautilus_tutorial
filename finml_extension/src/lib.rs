#![feature(float_gamma)]
use pyo3::prelude::*;
mod hop;
#[pymodule]
fn _finml_extension(_py: Python, _m: &PyModule) -> PyResult<()> {
    Ok(())
}