//! Sherpa-ONNX STT backend using direct FFI to the sherpa-onnx C API.
//!
//! Supports SenseVoice, Zipformer transducer, Paraformer, Whisper, and
//! Moonshine models through sherpa-onnx's unified offline recognizer.

use async_trait::async_trait;
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

use crate::error::VoxError;
use crate::traits::SttBackend;
use crate::types::{SttResult, Utterance};

/// Supported model architectures for the Sherpa-ONNX offline recognizer.
#[derive(Debug, Clone)]
pub enum SherpaModel {
    /// SenseVoice model (single ONNX file + tokens).
    SenseVoice {
        model: PathBuf,
        tokens: PathBuf,
        language: Option<String>,
    },
    /// Whisper model exported for sherpa-onnx (encoder + decoder + tokens).
    Whisper {
        encoder: PathBuf,
        decoder: PathBuf,
        tokens: PathBuf,
        language: Option<String>,
    },
    /// Paraformer model (single ONNX file + tokens).
    Paraformer { model: PathBuf, tokens: PathBuf },
    /// Zipformer transducer model (encoder + decoder + joiner + tokens).
    Transducer {
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
        tokens: PathBuf,
    },
}

/// Configuration for the Sherpa-ONNX STT backend.
#[derive(Debug, Clone)]
pub struct SherpaConfig {
    /// The model variant and its file paths.
    pub model: SherpaModel,
    /// Number of CPU threads for inference (default: 4).
    pub num_threads: i32,
    /// Execution provider, e.g. "cpu" (default: "cpu").
    pub provider: String,
    /// Decoding method, e.g. "greedy_search" (default: "greedy_search").
    pub decoding_method: String,
}

/// A wrapper around the raw recognizer pointer that is Send.
///
/// Safety: The sherpa-onnx offline recognizer is thread-safe — each call
/// creates an independent stream, and the recognizer can be shared across
/// threads without data races.
struct RecognizerPtr(*mut sherpa_sys::SherpaOnnxOfflineRecognizer);
unsafe impl Send for RecognizerPtr {}

/// Sherpa-ONNX STT backend using the offline recognizer C API.
pub struct SherpaBackend {
    recognizer: RecognizerPtr,
    _config: SherpaConfig,
}

// Safety: see RecognizerPtr comment above.
unsafe impl Send for SherpaBackend {}
unsafe impl Sync for SherpaBackend {}

/// Push a CString into the pool, returning a pointer valid while the pool lives.
fn push_cstring(pool: &mut Vec<CString>, s: &str) -> Result<*const std::os::raw::c_char, VoxError> {
    let cs = CString::new(s).map_err(|e| VoxError::Stt(format!("invalid C string: {e}")))?;
    let ptr = cs.as_ptr();
    pool.push(cs);
    Ok(ptr)
}

/// Push a path as CString into the pool.
fn push_path(pool: &mut Vec<CString>, p: &Path) -> Result<*const std::os::raw::c_char, VoxError> {
    let s = p
        .to_str()
        .ok_or_else(|| VoxError::Stt("non-UTF-8 model path".into()))?;
    push_cstring(pool, s)
}

impl SherpaBackend {
    /// Create a new backend with default settings (4 threads, cpu, greedy_search).
    pub fn new(model: SherpaModel) -> Result<Self, VoxError> {
        Self::with_config(SherpaConfig {
            model,
            num_threads: 4,
            provider: "cpu".into(),
            decoding_method: "greedy_search".into(),
        })
    }

    /// Create a new backend with the given configuration.
    pub fn with_config(config: SherpaConfig) -> Result<Self, VoxError> {
        validate_model_paths(&config.model)?;

        // CString pool — all strings must stay alive until after CreateOfflineRecognizer.
        let mut pool: Vec<CString> = Vec::new();

        let mut cfg = sherpa_sys::SherpaOnnxOfflineRecognizerConfig::default();

        // Feature extraction.
        cfg.feat_config.sample_rate = 16000;
        cfg.feat_config.feature_dim = 80;

        // Model-specific fields.
        let tokens_ptr = match &config.model {
            SherpaModel::SenseVoice {
                model,
                tokens,
                language,
            } => {
                cfg.model_config.sense_voice.model = push_path(&mut pool, model)?;
                if let Some(lang) = language {
                    cfg.model_config.sense_voice.language = push_cstring(&mut pool, lang)?;
                }
                cfg.model_config.sense_voice.use_itn = 1;
                push_path(&mut pool, tokens)?
            }
            SherpaModel::Whisper {
                encoder,
                decoder,
                tokens,
                language,
            } => {
                cfg.model_config.whisper.encoder = push_path(&mut pool, encoder)?;
                cfg.model_config.whisper.decoder = push_path(&mut pool, decoder)?;
                if let Some(lang) = language {
                    cfg.model_config.whisper.language = push_cstring(&mut pool, lang)?;
                }
                cfg.model_config.whisper.task = push_cstring(&mut pool, "transcribe")?;
                push_path(&mut pool, tokens)?
            }
            SherpaModel::Paraformer { model, tokens } => {
                cfg.model_config.paraformer.model = push_path(&mut pool, model)?;
                push_path(&mut pool, tokens)?
            }
            SherpaModel::Transducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => {
                cfg.model_config.transducer.encoder = push_path(&mut pool, encoder)?;
                cfg.model_config.transducer.decoder = push_path(&mut pool, decoder)?;
                cfg.model_config.transducer.joiner = push_path(&mut pool, joiner)?;
                push_path(&mut pool, tokens)?
            }
        };

        cfg.model_config.tokens = tokens_ptr;
        cfg.model_config.num_threads = config.num_threads;
        cfg.model_config.provider = push_cstring(&mut pool, &config.provider)?;
        cfg.model_config.debug = 0;
        cfg.decoding_method = push_cstring(&mut pool, &config.decoding_method)?;

        let recognizer = unsafe { sherpa_sys::SherpaOnnxCreateOfflineRecognizer(&cfg) };

        if recognizer.is_null() {
            return Err(VoxError::Stt(
                "failed to create sherpa-onnx offline recognizer".into(),
            ));
        }

        // pool drops here — that's fine, the recognizer has copied the strings.

        Ok(Self {
            recognizer: RecognizerPtr(recognizer),
            _config: config,
        })
    }

    /// Convenience constructor for SenseVoice models in a directory.
    ///
    /// Looks for `model.int8.onnx` (preferred) or `model.onnx`, plus
    /// `tokens.txt`. Language detection is automatic.
    pub fn from_sensevoice(model_dir: impl AsRef<Path>) -> Result<Self, VoxError> {
        let dir = model_dir.as_ref();
        let model = if dir.join("model.int8.onnx").exists() {
            dir.join("model.int8.onnx")
        } else {
            dir.join("model.onnx")
        };
        let tokens = dir.join("tokens.txt");
        Self::new(SherpaModel::SenseVoice {
            model,
            tokens,
            language: None,
        })
    }
}

impl Drop for SherpaBackend {
    fn drop(&mut self) {
        unsafe {
            sherpa_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer.0);
        }
    }
}

#[async_trait]
impl SttBackend for SherpaBackend {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        let samples = audio.audio.samples.clone();
        let duration_ms = audio.duration_ms;
        // Cast to usize to cross the Send boundary — raw pointers aren't Send.
        let rec_addr = self.recognizer.0 as usize;

        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let rec = rec_addr as *mut sherpa_sys::SherpaOnnxOfflineRecognizer;

            let stream = unsafe { sherpa_sys::SherpaOnnxCreateOfflineStream(rec) };
            if stream.is_null() {
                return Err(VoxError::Stt(
                    "failed to create sherpa-onnx offline stream".into(),
                ));
            }

            unsafe {
                sherpa_sys::SherpaOnnxAcceptWaveformOffline(
                    stream,
                    16000,
                    samples.as_ptr(),
                    samples.len() as c_int,
                );
                sherpa_sys::SherpaOnnxDecodeOfflineStream(rec, stream);
            }

            let result_ptr = unsafe { sherpa_sys::SherpaOnnxGetOfflineStreamResult(stream) };
            if result_ptr.is_null() {
                unsafe { sherpa_sys::SherpaOnnxDestroyOfflineStream(stream) };
                return Err(VoxError::Stt("sherpa-onnx returned null result".into()));
            }

            let (text, language) = unsafe {
                let result = &*result_ptr;

                let text = if result.text.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(result.text)
                        .to_string_lossy()
                        .trim()
                        .to_string()
                };

                let language = if result.lang.is_null() {
                    None
                } else {
                    let lang = CStr::from_ptr(result.lang)
                        .to_string_lossy()
                        .trim()
                        .to_string();
                    if lang.is_empty() { None } else { Some(lang) }
                };

                (text, language)
            };

            unsafe {
                sherpa_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
                sherpa_sys::SherpaOnnxDestroyOfflineStream(stream);
            }

            let processing_time_ms = start.elapsed().as_millis() as u64;

            Ok::<SttResult, VoxError>(SttResult {
                text,
                language,
                duration_ms,
                processing_time_ms,
            })
        })
        .await
        .map_err(|e| VoxError::Stt(format!("sherpa-onnx task panicked: {e}")))?
    }
}

/// Validate that all paths referenced by the model variant actually exist.
fn validate_model_paths(model: &SherpaModel) -> Result<(), VoxError> {
    let paths: Vec<&Path> = match model {
        SherpaModel::SenseVoice { model, tokens, .. } => {
            vec![model.as_path(), tokens.as_path()]
        }
        SherpaModel::Whisper {
            encoder,
            decoder,
            tokens,
            ..
        } => vec![encoder.as_path(), decoder.as_path(), tokens.as_path()],
        SherpaModel::Paraformer { model, tokens } => {
            vec![model.as_path(), tokens.as_path()]
        }
        SherpaModel::Transducer {
            encoder,
            decoder,
            joiner,
            tokens,
        } => vec![
            encoder.as_path(),
            decoder.as_path(),
            joiner.as_path(),
            tokens.as_path(),
        ],
    };

    for path in paths {
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }
    }

    Ok(())
}
