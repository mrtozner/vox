//! Error conversion from VoxError to Python exceptions.

use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;
use vox::VoxError;

/// Convert a VoxError into a Python exception.
pub(crate) fn to_py_err(err: VoxError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
