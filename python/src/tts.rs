//! Python wrapper for the Kokoro TTS backend.

use std::sync::Arc;

use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::runtime::RUNTIME;
use crate::util::default_model_dir;

/// Synthesized audio output from a TTS engine.
///
/// Attributes:
///     sample_rate: Sample rate in Hz (typically 24000 for Kokoro).
///     duration_ms: Duration of the audio in milliseconds.
#[pyclass]
pub struct AudioOutput {
    samples: Vec<f32>,
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl AudioOutput {
    /// Save the audio to a WAV file.
    ///
    /// Args:
    ///     path: Destination file path (e.g. "output.wav").
    fn save(&self, path: &str) -> PyResult<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create WAV file: {e}")))?;

        for &sample in &self.samples {
            writer
                .write_sample(sample)
                .map_err(|e| PyRuntimeError::new_err(format!("failed to write sample: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to finalize WAV: {e}")))?;

        Ok(())
    }

    /// Return the raw f32 audio samples as a Python list.
    fn samples(&self) -> Vec<f32> {
        self.samples.clone()
    }

    /// Number of audio samples.
    fn __len__(&self) -> usize {
        self.samples.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioOutput(samples={}, sample_rate={}, duration_ms={})",
            self.samples.len(),
            self.sample_rate,
            self.duration_ms
        )
    }
}

/// Kokoro text-to-speech engine.
///
/// Uses the Kokoro-82M ONNX model for high-quality local speech synthesis.
/// Model loading is lazy -- the ONNX model is loaded on first call to
/// `synthesize()` and cached for subsequent calls.
///
/// Args:
///     model_path: Path to the Kokoro ONNX model file.
///                 Defaults to `<data_dir>/vox/models/kokoro-v1.0.onnx`.
///     voices_path: Path to the voices binary file.
///                  Defaults to `<data_dir>/vox/models/voices.bin`.
#[pyclass]
#[derive(Clone)]
pub struct KokoroTts {
    pub(crate) model_path: String,
    pub(crate) voices_path: String,
    /// Lazily-initialized backend, shared across clones.
    backend: Arc<OnceCell<vox::KokoroBackend>>,
}

#[pymethods]
impl KokoroTts {
    #[new]
    #[pyo3(signature = (model_path=None, voices_path=None))]
    fn new(model_path: Option<String>, voices_path: Option<String>) -> PyResult<Self> {
        let model_dir = default_model_dir();
        let mp = model_path.unwrap_or_else(|| {
            model_dir
                .join("kokoro-v1.0.onnx")
                .to_string_lossy()
                .into_owned()
        });
        let vp = voices_path.unwrap_or_else(|| {
            model_dir
                .join("voices.bin")
                .to_string_lossy()
                .into_owned()
        });
        Ok(Self {
            model_path: mp,
            voices_path: vp,
            backend: Arc::new(OnceCell::new()),
        })
    }

    /// Synthesize text to audio.
    ///
    /// Args:
    ///     text: The text to synthesize.
    ///     voice: Optional voice name (e.g. "af_heart", "am_adam").
    ///            Defaults to "af_heart".
    ///
    /// Returns:
    ///     AudioOutput containing the synthesized samples.
    #[pyo3(signature = (text, voice=None))]
    fn synthesize(
        &self,
        py: Python<'_>,
        text: &str,
        voice: Option<String>,
    ) -> PyResult<AudioOutput> {
        let model_path = self.model_path.clone();
        let voices_path = self.voices_path.clone();
        let text = text.to_string();
        let backend_cell = Arc::clone(&self.backend);

        py.allow_threads(|| {
            // Lazily initialise the backend on first call; reuse on subsequent calls.
            let backend = backend_cell.get_or_try_init(|| {
                RUNTIME
                    .block_on(vox::KokoroBackend::new(&model_path, &voices_path))
                    .map_err(to_py_err)
            })?;

            let request = vox::TtsRequest { text, voice };

            let output = RUNTIME
                .block_on(async {
                    use vox::TtsBackend;
                    backend.synthesize(&request).await
                })
                .map_err(to_py_err)?;

            Ok(AudioOutput {
                samples: output.audio.samples,
                sample_rate: output.audio.sample_rate,
                duration_ms: output.duration_ms,
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KokoroTts(model_path={:?}, voices_path={:?})",
            self.model_path, self.voices_path
        )
    }
}
