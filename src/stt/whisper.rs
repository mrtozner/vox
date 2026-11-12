//! Whisper STT backend using whisper-rs bindings to whisper.cpp.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::VoxError;
use crate::traits::SttBackend;
use crate::types::{SttResult, Utterance};

/// Whisper model size.
#[derive(Debug, Clone, Copy)]
pub enum WhisperModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
}

/// Configuration for the Whisper STT backend.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Path to the Whisper GGML model file.
    pub model_path: PathBuf,
    /// Spoken language as a Whisper code (e.g. `"en"`, `"tr"`, `"de"`).
    ///
    /// `None` or `Some("auto")` lets a multilingual model detect the
    /// language per utterance. English-only models (`*.en`) always
    /// transcribe English, whatever is set here.
    pub language: Option<String>,
    /// Whether to translate to English.
    pub translate: bool,
    /// Number of CPU threads. See [`default_n_threads`].
    pub n_threads: i32,
}

impl WhisperConfig {
    /// Config for `model_path` with automatic language detection and a
    /// thread count derived from the machine ([`default_n_threads`]).
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            language: None,
            translate: false,
            n_threads: default_n_threads(),
        }
    }

    /// Set the spoken language (`"auto"` or `None` for detection).
    pub fn with_language(mut self, language: Option<impl Into<String>>) -> Self {
        self.language = language.map(Into::into);
        self
    }

    /// Set the number of inference threads (values below 1 are clamped to 1).
    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads.max(1);
        self
    }

    /// Check the configuration without loading the model. Currently this
    /// rejects language codes whisper.cpp does not know.
    pub fn validate(&self) -> Result<(), VoxError> {
        validate_language(self.language.as_deref())
    }
}

/// Upper bound for the default thread count. whisper.cpp stops scaling
/// (and often regresses) well before this on desktop CPUs.
const MAX_DEFAULT_THREADS: i32 = 8;

/// Default number of inference threads for this machine: all cores on tiny
/// systems, otherwise one core is left free for audio I/O, capped at 8.
pub fn default_n_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    threads_for_cores(cores)
}

pub(crate) fn threads_for_cores(cores: i32) -> i32 {
    if cores <= 2 {
        cores.max(1)
    } else {
        (cores - 1).min(MAX_DEFAULT_THREADS)
    }
}

/// Normalise a requested language: `None`, empty and `"auto"` mean
/// auto-detect (`None`); anything else is trimmed and lower-cased.
pub(crate) fn normalize_language(requested: Option<&str>) -> Option<String> {
    let lang = requested?.trim().to_ascii_lowercase();
    if lang.is_empty() || lang == "auto" {
        None
    } else {
        Some(lang)
    }
}

/// The language string handed to whisper.cpp for a given request.
///
/// English-only models always get `"en"`; multilingual models get the
/// requested code, or `"auto"` when none was requested.
pub(crate) fn language_param(requested: Option<&str>, multilingual: bool) -> String {
    if !multilingual {
        return "en".to_string();
    }
    normalize_language(requested).unwrap_or_else(|| "auto".to_string())
}

/// Reject language codes whisper.cpp does not know, so a typo fails at
/// startup instead of on the first utterance.
pub(crate) fn validate_language(requested: Option<&str>) -> Result<(), VoxError> {
    match normalize_language(requested) {
        Some(lang) if whisper_rs::get_lang_id(&lang).is_none() => Err(VoxError::Stt(format!(
            "unknown language '{lang}' (expected a Whisper language code such as en, tr, de, or 'auto')"
        ))),
        _ => Ok(()),
    }
}

/// Warn when a non-English language is requested from an English-only model.
pub(crate) fn warn_if_language_unsupported(
    requested: Option<&str>,
    multilingual: bool,
    model: &Path,
) {
    if multilingual {
        return;
    }
    if let Some(lang) = normalize_language(requested) {
        if lang != "en" {
            tracing::warn!(
                model = %model.display(),
                language = %lang,
                "model is English-only; the requested language is ignored. \
                 Use a multilingual model (e.g. base, small, turbo)"
            );
        }
    }
}

/// Language to report in [`SttResult`]: the configured code, or the one
/// whisper.cpp detected when running in auto mode.
pub(crate) fn reported_language(param: &str, detected_lang_id: i32) -> Option<String> {
    if param != "auto" {
        return Some(param.to_string());
    }
    whisper_rs::get_lang_str(detected_lang_id).map(str::to_string)
}

/// Whisper STT backend using whisper-rs bindings to whisper.cpp.
pub struct WhisperBackend {
    ctx: Arc<WhisperContext>,
    config: WhisperConfig,
}

impl WhisperBackend {
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
        .map_err(|e| VoxError::Stt(format!("failed to load whisper model: {e}")))?;

        Ok(Self {
            ctx: Arc::new(ctx),
            config: WhisperConfig::new(path),
        })
    }

    /// Create with custom config.
    pub fn with_config(config: WhisperConfig) -> Result<Self, VoxError> {
        if !config.model_path.exists() {
            return Err(VoxError::ModelNotFound(config.model_path.clone()));
        }
        validate_language(config.language.as_deref())?;
        let ctx = WhisperContext::new_with_params(
            config
                .model_path
                .to_str()
                .ok_or_else(|| VoxError::Stt("invalid model path".into()))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| VoxError::Stt(format!("failed to load whisper model: {e}")))?;
        warn_if_language_unsupported(
            config.language.as_deref(),
            ctx.is_multilingual(),
            &config.model_path,
        );

        Ok(Self {
            ctx: Arc::new(ctx),
            config,
        })
    }

    /// The configuration this backend runs with.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Whether the loaded model supports languages other than English.
    pub fn is_multilingual(&self) -> bool {
        self.ctx.is_multilingual()
    }

    /// Convenience: load a model by size from a directory.
    /// Looks for files like "ggml-tiny.en.bin" in the given directory.
    pub fn from_dir(dir: impl AsRef<Path>, model: WhisperModel) -> Result<Self, VoxError> {
        let filename = match model {
            WhisperModel::Tiny => "ggml-tiny.bin",
            WhisperModel::TinyEn => "ggml-tiny.en.bin",
            WhisperModel::Base => "ggml-base.bin",
            WhisperModel::BaseEn => "ggml-base.en.bin",
            WhisperModel::Small => "ggml-small.bin",
            WhisperModel::SmallEn => "ggml-small.en.bin",
            WhisperModel::Medium => "ggml-medium.bin",
            WhisperModel::MediumEn => "ggml-medium.en.bin",
        };
        let path = dir.as_ref().join(filename);
        Self::from_model(path)
    }
}

#[async_trait]
impl SttBackend for WhisperBackend {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        let ctx = self.ctx.clone();
        let samples = audio.audio.samples.clone();
        let duration_ms = audio.duration_ms;
        let language = language_param(self.config.language.as_deref(), ctx.is_multilingual());
        let translate = self.config.translate;
        let n_threads = self.config.n_threads;
        #[cfg(feature = "diarization")]
        let speaker_id = audio.speaker_id.clone();

        // whisper-rs is synchronous -- run in a blocking task
        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();

            // Create state from context
            let mut state = ctx
                .create_state()
                .map_err(|e| VoxError::Stt(format!("failed to create whisper state: {e}")))?;

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
            // Filter non-speech audio: segments with a no_speech probability
            // above this threshold are suppressed. This prevents Whisper from
            // hallucinating text for echo, noise, or silence captured during
            // Live Talk TTS playback.
            params.set_no_speech_thold(0.6);

            // Always set the language explicitly: whisper.cpp's own default
            // is "en", which silently breaks every other language.
            params.set_language(Some(&language));

            // Run inference
            state
                .full(params, &samples)
                .map_err(|e| VoxError::Stt(format!("whisper inference failed: {e}")))?;

            // Collect segments, skipping those with high no_speech probability
            let num_segments = state.full_n_segments();

            let mut text = String::new();
            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    // Skip segments that are likely non-speech (echo, noise).
                    if segment.no_speech_probability() > 0.6 {
                        tracing::debug!(
                            segment = i,
                            no_speech_prob = segment.no_speech_probability(),
                            "whisper: skipping high no_speech segment"
                        );
                        continue;
                    }
                    if let Ok(segment_text) = segment.to_str_lossy() {
                        text.push_str(&segment_text);
                    }
                }
            }

            let processing_time_ms = start.elapsed().as_millis() as u64;

            Ok::<SttResult, VoxError>(SttResult {
                text: text.trim().to_string(),
                language: reported_language(&language, state.full_lang_id_from_state()),
                duration_ms,
                processing_time_ms,
                #[cfg(feature = "diarization")]
                speaker_id,
            })
        })
        .await
        .map_err(|e| VoxError::Stt(format!("whisper task panicked: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_language_reaches_whisper_param() {
        assert_eq!(language_param(Some("tr"), true), "tr");
        assert_eq!(language_param(Some(" TR "), true), "tr");
        assert_eq!(language_param(Some("de"), true), "de");
    }

    #[test]
    fn missing_or_auto_language_means_detection_on_multilingual_models() {
        assert_eq!(language_param(None, true), "auto");
        assert_eq!(language_param(Some("auto"), true), "auto");
        assert_eq!(language_param(Some(""), true), "auto");
    }

    #[test]
    fn english_only_models_always_use_english() {
        assert_eq!(language_param(None, false), "en");
        assert_eq!(language_param(Some("tr"), false), "en");
        assert_eq!(language_param(Some("auto"), false), "en");
    }

    #[test]
    fn config_builder_carries_language_and_threads() {
        let config = WhisperConfig::new("ggml-base.bin")
            .with_language(Some("tr"))
            .with_threads(6);
        assert_eq!(config.language.as_deref(), Some("tr"));
        assert_eq!(config.n_threads, 6);
        assert_eq!(
            language_param(config.language.as_deref(), true),
            "tr",
            "configured language must be the one passed to whisper.cpp"
        );
        assert_eq!(WhisperConfig::new("m.bin").with_threads(0).n_threads, 1);
    }

    #[test]
    fn unknown_language_is_rejected_before_loading() {
        assert!(validate_language(Some("tr")).is_ok());
        assert!(validate_language(Some("auto")).is_ok());
        assert!(validate_language(None).is_ok());
        assert!(validate_language(Some("klingon")).is_err());
    }

    #[test]
    fn with_config_checks_language_before_touching_the_model() {
        // Cargo.toml is "a path that exists" but not a model: the language
        // check must fire before whisper.cpp tries to parse the file.
        let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let config = WhisperConfig::new(manifest).with_language(Some("klingon"));
        let err = WhisperBackend::with_config(config)
            .err()
            .expect("unknown language must fail");
        assert!(err.to_string().contains("unknown language"), "{err}");
    }

    #[test]
    fn default_threads_leave_headroom_and_are_capped() {
        assert_eq!(threads_for_cores(1), 1);
        assert_eq!(threads_for_cores(2), 2);
        assert_eq!(threads_for_cores(4), 3);
        assert_eq!(threads_for_cores(8), 7);
        assert_eq!(threads_for_cores(16), 8);
        assert!(default_n_threads() >= 1);
    }

    #[test]
    fn reported_language_prefers_configured_then_detected() {
        assert_eq!(reported_language("tr", 0).as_deref(), Some("tr"));
        let tr_id = whisper_rs::get_lang_id("tr").unwrap();
        assert_eq!(reported_language("auto", tr_id).as_deref(), Some("tr"));
        assert_eq!(reported_language("auto", -1), None);
    }
}
