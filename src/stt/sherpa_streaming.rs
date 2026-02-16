//! Sherpa-ONNX streaming (online) STT backend using direct FFI.
//!
//! Uses the sherpa-onnx online recognizer for incremental speech-to-text.
//! Audio is processed chunk-by-chunk with partial results available
//! after each push, instead of re-transcribing the entire buffer.

use async_trait::async_trait;
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::error::VoxError;
use crate::traits::{StreamingSttBackend, SttBackend, SttSession};
use crate::types::{SttResult, Utterance};

/// Supported streaming model architectures.
#[derive(Debug, Clone)]
pub enum SherpaStreamingModel {
    /// Transducer model (encoder + decoder + joiner + tokens).
    Transducer {
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
        tokens: PathBuf,
    },
    /// Paraformer streaming model (encoder + decoder + tokens).
    Paraformer {
        encoder: PathBuf,
        decoder: PathBuf,
        tokens: PathBuf,
    },
}

/// Configuration for the streaming Sherpa-ONNX STT backend.
#[derive(Debug, Clone)]
pub struct SherpaStreamingConfig {
    /// The model variant and its file paths.
    pub model: SherpaStreamingModel,
    /// Number of CPU threads for inference (default: 4).
    pub num_threads: i32,
    /// Execution provider, e.g. "cpu" (default: "cpu").
    pub provider: String,
    /// Decoding method, e.g. "greedy_search" (default: "greedy_search").
    pub decoding_method: String,
    /// Whether to enable endpoint detection (default: true).
    pub enable_endpoint: bool,
    /// Rule1: min trailing silence to detect endpoint (default: 2.4).
    pub rule1_min_trailing_silence: f32,
    /// Rule2: min trailing silence (default: 1.2).
    pub rule2_min_trailing_silence: f32,
    /// Rule3: min utterance length (default: 20.0).
    pub rule3_min_utterance_length: f32,
}

impl Default for SherpaStreamingConfig {
    fn default() -> Self {
        Self {
            model: SherpaStreamingModel::Transducer {
                encoder: PathBuf::new(),
                decoder: PathBuf::new(),
                joiner: PathBuf::new(),
                tokens: PathBuf::new(),
            },
            num_threads: 4,
            provider: "cpu".into(),
            decoding_method: "greedy_search".into(),
            enable_endpoint: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
        }
    }
}

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

/// Wrapper for the online recognizer pointer.
///
/// Safety: sherpa-onnx online recognizer is thread-safe for creating streams
/// and decoding -- each stream holds independent state. The recognizer itself
/// is immutable after creation (model weights, config). Multiple threads can
/// call CreateOnlineStream / DecodeOnlineStream concurrently on the same
/// recognizer without data races (verified in sherpa-onnx source: recognizer
/// holds shared_ptr to model, each stream has its own decoder state).
///
/// Drop calls DestroyOnlineRecognizer. Because this is wrapped in Arc,
/// destruction only happens when ALL sessions and the backend are dropped.
struct OnlineRecognizerPtr(*const sherpa_sys::SherpaOnnxOnlineRecognizer);
unsafe impl Send for OnlineRecognizerPtr {}
unsafe impl Sync for OnlineRecognizerPtr {}

impl Drop for OnlineRecognizerPtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sherpa_sys::SherpaOnnxDestroyOnlineRecognizer(self.0) };
        }
    }
}

/// Sherpa-ONNX streaming STT backend using the online recognizer C API.
pub struct SherpaStreamingBackend {
    recognizer: Arc<OnlineRecognizerPtr>,
    _config: SherpaStreamingConfig,
}

// Send+Sync derived from Arc<OnlineRecognizerPtr> which is Send+Sync.
unsafe impl Send for SherpaStreamingBackend {}
unsafe impl Sync for SherpaStreamingBackend {}

impl SherpaStreamingBackend {
    /// Create a new streaming backend with the given configuration.
    pub fn with_config(config: SherpaStreamingConfig) -> Result<Self, VoxError> {
        let t = Instant::now();
        validate_model_paths(&config.model)?;

        // CString pool -- all strings must stay alive until after CreateOnlineRecognizer.
        let mut pool: Vec<CString> = Vec::new();

        let mut cfg = sherpa_sys::SherpaOnnxOnlineRecognizerConfig::default();

        cfg.feat_config.sample_rate = 16000;
        cfg.feat_config.feature_dim = 80;

        let tokens_ptr = match &config.model {
            SherpaStreamingModel::Transducer {
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
            SherpaStreamingModel::Paraformer {
                encoder,
                decoder,
                tokens,
            } => {
                cfg.model_config.paraformer.encoder = push_path(&mut pool, encoder)?;
                cfg.model_config.paraformer.decoder = push_path(&mut pool, decoder)?;
                push_path(&mut pool, tokens)?
            }
        };

        cfg.model_config.tokens = tokens_ptr;
        cfg.model_config.num_threads = config.num_threads;
        cfg.model_config.provider = push_cstring(&mut pool, &config.provider)?;
        cfg.model_config.debug = 0;

        cfg.decoding_method = push_cstring(&mut pool, &config.decoding_method)?;
        cfg.enable_endpoint = if config.enable_endpoint { 1 } else { 0 };
        cfg.rule1_min_trailing_silence = config.rule1_min_trailing_silence;
        cfg.rule2_min_trailing_silence = config.rule2_min_trailing_silence;
        cfg.rule3_min_utterance_length = config.rule3_min_utterance_length;

        let recognizer = unsafe { sherpa_sys::SherpaOnnxCreateOnlineRecognizer(&cfg) };

        if recognizer.is_null() {
            return Err(VoxError::Stt(
                "failed to create sherpa-onnx online recognizer".into(),
            ));
        }

        // pool drops here -- that's fine, the recognizer has copied the strings.

        tracing::debug!(
            elapsed_ms = t.elapsed().as_millis(),
            "sherpa streaming model loaded"
        );

        Ok(Self {
            recognizer: Arc::new(OnlineRecognizerPtr(recognizer)),
            _config: config,
        })
    }

    /// Convenience constructor for transducer models in a directory.
    ///
    /// Looks for canonical filenames:
    /// - `encoder.int8.onnx` (preferred) or `encoder.onnx`
    /// - `decoder.int8.onnx` (preferred) or `decoder.onnx`
    /// - `joiner.int8.onnx` (preferred) or `joiner.onnx`
    /// - `tokens.txt`
    pub fn from_transducer(model_dir: impl AsRef<Path>) -> Result<Self, VoxError> {
        let dir = model_dir.as_ref();
        let encoder = if dir.join("encoder.int8.onnx").exists() {
            dir.join("encoder.int8.onnx")
        } else {
            dir.join("encoder.onnx")
        };
        let decoder = if dir.join("decoder.int8.onnx").exists() {
            dir.join("decoder.int8.onnx")
        } else {
            dir.join("decoder.onnx")
        };
        let joiner = if dir.join("joiner.int8.onnx").exists() {
            dir.join("joiner.int8.onnx")
        } else {
            dir.join("joiner.onnx")
        };
        let tokens = dir.join("tokens.txt");

        Self::with_config(SherpaStreamingConfig {
            model: SherpaStreamingModel::Transducer {
                encoder,
                decoder,
                joiner,
                tokens,
            },
            ..SherpaStreamingConfig::default()
        })
    }
}

impl StreamingSttBackend for SherpaStreamingBackend {
    fn create_session(&self) -> Result<Box<dyn SttSession>, VoxError> {
        let stream = unsafe { sherpa_sys::SherpaOnnxCreateOnlineStream(self.recognizer.0) };
        if stream.is_null() {
            return Err(VoxError::Stt("failed to create online stream".into()));
        }
        Ok(Box::new(SherpaStreamingSession {
            recognizer: Arc::clone(&self.recognizer),
            stream,
            start_time: Instant::now(),
            total_samples: 0,
            last_text: String::new(),
            finished: false,
        }))
    }
}

#[async_trait]
impl SttBackend for SherpaStreamingBackend {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        // Batch mode: create a session, push all audio, finish.
        let mut session = self.create_session()?;
        session.push_audio(&audio.audio.samples, audio.audio.sample_rate)?;
        session.finish()
    }
}

/// A single streaming recognition session.
///
/// Holds a raw pointer to an online stream. The stream MUST be destroyed
/// when the session is dropped, even if `finish()` was never called
/// (e.g., on error or cancellation).
pub struct SherpaStreamingSession {
    recognizer: Arc<OnlineRecognizerPtr>,
    stream: *const sherpa_sys::SherpaOnnxOnlineStream,
    start_time: Instant,
    total_samples: usize,
    last_text: String,
    finished: bool,
}

// Safety: The stream pointer is only accessed through &mut self methods
// (push_audio, finish), so there's no concurrent access. The recognizer
// Arc is Send+Sync. The stream is created from a thread-safe recognizer
// and each stream is independent (own decoder state).
unsafe impl Send for SherpaStreamingSession {}

impl Drop for SherpaStreamingSession {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { sherpa_sys::SherpaOnnxDestroyOnlineStream(self.stream) };
            self.stream = std::ptr::null();
        }
    }
}

impl SttSession for SherpaStreamingSession {
    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<String>, VoxError> {
        if self.finished {
            return Err(VoxError::Stt("session already finished".into()));
        }

        let t = Instant::now();
        unsafe {
            sherpa_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as c_int,
                samples.as_ptr(),
                samples.len() as c_int,
            );

            while sherpa_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer.0, self.stream) != 0 {
                sherpa_sys::SherpaOnnxDecodeOnlineStream(self.recognizer.0, self.stream);
            }

            let result_ptr =
                sherpa_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer.0, self.stream);
            if result_ptr.is_null() {
                self.total_samples += samples.len();
                tracing::debug!(
                    elapsed_us = t.elapsed().as_micros(),
                    "sherpa push_audio decode"
                );
                return Ok(None);
            }

            let result = &*result_ptr;
            let text = if result.text.is_null() {
                String::new()
            } else {
                CStr::from_ptr(result.text)
                    .to_string_lossy()
                    .trim()
                    .to_string()
            };

            sherpa_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
            self.total_samples += samples.len();

            let ret = if !text.is_empty() && text != self.last_text {
                self.last_text = text.clone();
                Ok(Some(text))
            } else {
                Ok(None)
            };

            tracing::debug!(
                elapsed_us = t.elapsed().as_micros(),
                "sherpa push_audio decode"
            );

            ret
        }
    }

    fn finish(&mut self) -> Result<SttResult, VoxError> {
        if self.finished {
            return Err(VoxError::Stt("session already finished".into()));
        }
        self.finished = true;

        let t_finish = Instant::now();
        unsafe {
            sherpa_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);

            while sherpa_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer.0, self.stream) != 0 {
                sherpa_sys::SherpaOnnxDecodeOnlineStream(self.recognizer.0, self.stream);
            }

            let result_ptr =
                sherpa_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer.0, self.stream);

            let text = if result_ptr.is_null() {
                self.last_text.clone()
            } else {
                let result = &*result_ptr;
                let t = if result.text.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(result.text)
                        .to_string_lossy()
                        .trim()
                        .to_string()
                };
                sherpa_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
                if t.is_empty() {
                    self.last_text.clone()
                } else {
                    t
                }
            };

            let processing_time_ms = self.start_time.elapsed().as_millis() as u64;
            let duration_ms = (self.total_samples as u64 * 1000) / 16000;

            tracing::debug!(
                elapsed_us = t_finish.elapsed().as_micros(),
                "sherpa finish finalized"
            );

            Ok(SttResult {
                text,
                language: None,
                duration_ms,
                processing_time_ms,
            })
        }
    }
}

/// Validate that all paths referenced by the model variant actually exist.
fn validate_model_paths(model: &SherpaStreamingModel) -> Result<(), VoxError> {
    let paths: Vec<&Path> = match model {
        SherpaStreamingModel::Transducer {
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
        SherpaStreamingModel::Paraformer {
            encoder,
            decoder,
            tokens,
        } => vec![encoder.as_path(), decoder.as_path(), tokens.as_path()],
    };

    for path in paths {
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }
    }

    Ok(())
}
