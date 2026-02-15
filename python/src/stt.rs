//! Python wrapper for the Whisper STT backend.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::runtime::RUNTIME;
use crate::util::default_model_dir;

/// Result of a speech-to-text transcription.
///
/// Attributes:
///     text: The transcribed text.
///     language: Detected language code (e.g. "en"), or None.
///     duration_ms: Duration of the source audio in milliseconds.
///     processing_time_ms: Time spent transcribing in milliseconds.
#[pyclass]
#[derive(Clone)]
pub struct TranscribeResult {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub language: Option<String>,
    #[pyo3(get)]
    pub duration_ms: u64,
    #[pyo3(get)]
    pub processing_time_ms: u64,
}

#[pymethods]
impl TranscribeResult {
    fn __repr__(&self) -> String {
        format!(
            "TranscribeResult(text={:?}, language={:?}, duration_ms={}, processing_time_ms={})",
            self.text, self.language, self.duration_ms, self.processing_time_ms
        )
    }

    fn __str__(&self) -> &str {
        &self.text
    }
}

/// Map a short model name like "tiny.en" to the GGML filename.
fn resolve_model_path(model: &str) -> String {
    let model_dir = default_model_dir();

    // If the user passed a full path, use it directly.
    if model.contains('/') || model.contains('\\') || model.ends_with(".bin") {
        return model.to_string();
    }

    // Map shorthand names to GGML filenames.
    let filename = match model {
        "tiny" => "ggml-tiny.bin",
        "tiny.en" => "ggml-tiny.en.bin",
        "base" => "ggml-base.bin",
        "base.en" => "ggml-base.en.bin",
        "small" => "ggml-small.bin",
        "small.en" => "ggml-small.en.bin",
        "medium" => "ggml-medium.bin",
        "medium.en" => "ggml-medium.en.bin",
        other => other,
    };

    model_dir.join(filename).to_string_lossy().into_owned()
}

/// Whisper-based speech-to-text engine.
///
/// Wraps the whisper.cpp bindings for local transcription.
/// Model loading is lazy -- the GGML model is loaded on first use.
///
/// Args:
///     model: Model name (e.g. "tiny.en", "base") or full path to a
///            GGML model file. Defaults to "tiny.en".
#[pyclass]
#[derive(Clone)]
pub struct WhisperStt {
    pub(crate) model_path: String,
}

impl WhisperStt {
    pub(crate) fn repr(&self) -> String {
        format!("WhisperStt(model_path={:?})", self.model_path)
    }
}

#[pymethods]
impl WhisperStt {
    #[new]
    #[pyo3(signature = (model=None))]
    fn new(model: Option<String>) -> PyResult<Self> {
        let model_name = model.unwrap_or_else(|| "tiny.en".to_string());
        let path = resolve_model_path(&model_name);
        Ok(Self { model_path: path })
    }

    /// Transcribe a WAV file on disk.
    ///
    /// Args:
    ///     path: Path to a WAV file (16-bit PCM or 32-bit float).
    ///
    /// Returns:
    ///     TranscribeResult with the transcribed text and metadata.
    fn transcribe_file(&self, py: Python<'_>, path: &str) -> PyResult<TranscribeResult> {
        let model_path = self.model_path.clone();
        let wav_path = path.to_string();

        py.allow_threads(|| {
            // Read WAV file.
            let reader = hound::WavReader::open(&wav_path)
                .map_err(|e| PyRuntimeError::new_err(format!("failed to open WAV: {e}")))?;

            let spec = reader.spec();
            let sample_rate = spec.sample_rate;

            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Int => {
                    let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    reader
                        .into_samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max_val)
                        .collect()
                }
                hound::SampleFormat::Float => reader
                    .into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect(),
            };

            let duration_ms = if sample_rate > 0 {
                (samples.len() as u64 * 1000) / sample_rate as u64
            } else {
                0
            };

            let utterance = vox::Utterance {
                audio: vox::AudioChunk {
                    samples,
                    sample_rate,
                    channels: spec.channels,
                },
                duration_ms,
            };

            // Load model and transcribe.
            let backend = vox::WhisperBackend::from_model(&model_path).map_err(to_py_err)?;

            let result = RUNTIME
                .block_on(async {
                    use vox::SttBackend;
                    backend.transcribe(&utterance).await
                })
                .map_err(to_py_err)?;

            Ok(TranscribeResult {
                text: result.text,
                language: result.language,
                duration_ms: result.duration_ms,
                processing_time_ms: result.processing_time_ms,
            })
        })
    }

    fn __repr__(&self) -> String {
        self.repr()
    }
}
