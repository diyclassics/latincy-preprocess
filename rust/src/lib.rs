pub mod uv;
pub mod long_s;

#[cfg(feature = "pyo3-backend")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3-backend")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // U/V normalization functions
    m.add_function(wrap_pyfunction!(uv::normalize_uv, m)?)?;
    m.add_function(wrap_pyfunction!(uv::normalize_uv_char, m)?)?;
    m.add_function(wrap_pyfunction!(uv::normalize_uv_detailed, m)?)?;

    // Long-s normalization functions
    m.add_function(wrap_pyfunction!(long_s::normalize_long_s_word_pass1, m)?)?;
    m.add_function(wrap_pyfunction!(long_s::normalize_long_s_word_pass2, m)?)?;
    m.add_function(wrap_pyfunction!(long_s::normalize_long_s_word_full, m)?)?;
    m.add_function(wrap_pyfunction!(long_s::normalize_long_s_text_full, m)?)?;

    Ok(())
}
