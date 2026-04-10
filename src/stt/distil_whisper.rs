//! Distil-Whisper STT backend using whisper-rs bindings to whisper.cpp.
//!
//! Distil-Whisper is a distilled version of Whisper that is 6x faster while maintaining
//! comparable accuracy. It uses the same whisper-rs crate as the standard Whisper backend.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::VoxError;
use crate::traits::SttBackend;
use crate::types::{SttResult, Utterance};

/// Distil-Whisper model size.
#[derive(Debug, Clone, Copy)]
pub enum DistilWhisperModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
}

/// Configuration for the Distil-Whisper STT backend.
#[derive(Debug, Clone)]
pub struct DistilWhisperConfig {
    /// Path to the Distil-Whisper GGML model file.
    pub model_path: PathBuf,
    /// Language code (e.g. "en"). None = auto-detect.
    pub language: Option<String>,
    /// Whether to translate to English.
    pub translate: bool,
    /// Number of CPU threads (default: 4).
    pub n_threads: i32,
}

/// Distil-Whisper STT backend using whisper-rs bindings to whisper.cpp.
///
/// Distil-Whisper provides 6x faster inference compared to standard Whisper
/// while maintaining comparable accuracy. It uses the same whisper.cpp
/// bindings but with distilled model weights.
pub struct DistilWhisperBackend {
    ctx: Arc<WhisperContext>,
    config: DistilWhisperConfig,
}

impl DistilWhisperBackend {
    /// Create from a model file path.
    pub fn from_model(path: impl AsRef<Path>) -> Result<Self, VoxError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path));
        }
        let ctx = WhisperContext::new_with_params(
            path.to_str()
                .ok_or_else(|| VoxError::Stt("invalid model path".into()))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| VoxError::Stt(format!("failed to load distil-whisper model: {e}")))?;

        Ok(Self {
            ctx: Arc::new(ctx),
            config: DistilWhisperConfig {
                model_path: path,
                language: None,
                translate: false,
                n_threads: 4,
            },
        })
    }

    /// Create with custom config.
    pub fn with_config(config: DistilWhisperConfig) -> Result<Self, VoxError> {
        if !config.model_path.exists() {
            return Err(VoxError::ModelNotFound(config.model_path.clone()));
        }
        let ctx = WhisperContext::new_with_params(
            config
                .model_path
                .to_str()
                .ok_or_else(|| VoxError::Stt("invalid model path".into()))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| VoxError::Stt(format!("failed to load distil-whisper model: {e}")))?;

        Ok(Self {
            ctx: Arc::new(ctx),
            config,
        })
    }

    /// Convenience: load a model by size from a directory.
    /// Looks for files like "ggml-distil-tiny.en.bin" in the given directory.
    pub fn from_dir(dir: impl AsRef<Path>, model: DistilWhisperModel) -> Result<Self, VoxError> {
        let filename = match model {
            DistilWhisperModel::Tiny => "ggml-distil-tiny.bin",
            DistilWhisperModel::TinyEn => "ggml-distil-tiny.en.bin",
            DistilWhisperModel::Base => "ggml-distil-base.bin",
            DistilWhisperModel::BaseEn => "ggml-distil-base.en.bin",
            DistilWhisperModel::Small => "ggml-distil-small.bin",
            DistilWhisperModel::SmallEn => "ggml-distil-small.en.bin",
            DistilWhisperModel::Medium => "ggml-distil-medium.bin",
            DistilWhisperModel::MediumEn => "ggml-distil-medium.en.bin",
        };
        let path = dir.as_ref().join(filename);
        Self::from_model(path)
    }
}

#[async_trait]
impl SttBackend for DistilWhisperBackend {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        let ctx = self.ctx.clone();
        let samples = audio.audio.samples.clone();
        let duration_ms = audio.duration_ms;
        let language = self.config.language.clone();
        let translate = self.config.translate;
        let n_threads = self.config.n_threads;

        // whisper-rs is synchronous -- run in a blocking task
        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();

            // Create state from context
            let mut state = ctx.create_state().map_err(|e| {
                VoxError::Stt(format!("failed to create distil-whisper state: {e}"))
            })?;

            // Configure parameters
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(n_threads);
            params.set_translate(translate);
            params.set_no_timestamps(true);
            params.set_single_segment(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            if let Some(ref lang) = language {
                params.set_language(Some(lang));
            }

            // Run inference
            state
                .full(params, &samples)
                .map_err(|e| VoxError::Stt(format!("distil-whisper inference failed: {e}")))?;

            // Collect segments
            let num_segments = state.full_n_segments();

            let mut text = String::new();
            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    if let Ok(segment_text) = segment.to_str_lossy() {
                        text.push_str(&segment_text);
                    }
                }
            }

            let processing_time_ms = start.elapsed().as_millis() as u64;

            Ok::<SttResult, VoxError>(SttResult {
                text: text.trim().to_string(),
                language: language.clone(),
                duration_ms,
                processing_time_ms,
            })
        })
        .await
        .map_err(|e| VoxError::Stt(format!("distil-whisper task panicked: {e}")))?
    }
}
