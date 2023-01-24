use pyo3::prelude::*;
pub mod fracdiff;
pub mod entropy;
pub mod sadf;
pub mod microstructure;

use crate::entropy::Entropy;
use crate::fracdiff::FracDiff;
use crate::microstructure::MicroStructucture;

#[pymodule]
#[pyo3(name = "features")]
fn features(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroStructucture>()?;
    m.add_class::<FracDiff>()?;
    m.add_class::< Entropy>()?;
    Ok(())
}