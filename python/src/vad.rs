//! Python wrapper for the Silero VAD backend.

use pyo3::prelude::*;

use crate::util::default_model_dir;

/// Voice Activity Detection powered by the Silero VAD v5 ONNX model.
///
/// Stores the model path for lazy initialization. The actual ONNX session
/// is created when the pipeline starts listening, not at construction time.
///
/// Args:
///     model_path: Path to the `silero_vad.onnx` file.
///                 Defaults to `~/.vox/models/silero_vad.onnx`.
#[pyclass]
#[derive(Clone)]
pub struct SileroVad {
    pub(crate) model_path: String,
}

impl SileroVad {
    pub(crate) fn repr(&self) -> String {
        format!("SileroVad(model_path={:?})", self.model_path)
    }
}

#[pymethods]
impl SileroVad {
    #[new]
    #[pyo3(signature = (model_path=None))]
    fn new(model_path: Option<String>) -> PyResult<Self> {
        let path = model_path.unwrap_or_else(|| {
            default_model_dir()
                .join("silero_vad.onnx")
                .to_string_lossy()
                .into_owned()
        });
        Ok(Self { model_path: path })
    }

    fn __repr__(&self) -> String {
        self.repr()
    }
}
