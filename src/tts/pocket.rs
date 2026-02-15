//! Pocket TTS backend using the pocket-tts crate (Kyutai, 100M params).
//!
//! Pure Rust via Candle — no ONNX Runtime. Optimized for edge/embedded:
//! Jetson, Raspberry Pi, and lightweight CPU inference.

use async_trait::async_trait;
use std::sync::Arc;

use pocket_tts::TTSModel;

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest};

/// Pocket TTS output sample rate.
const POCKET_SAMPLE_RATE: u32 = 24_000;

/// Default model variant on HuggingFace.
const DEFAULT_VARIANT: &str = "b6369a24";

/// Built-in voice names shipped with the Pocket TTS model.
const BUILTIN_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

/// Configuration for the Pocket TTS backend.
#[derive(Debug, Clone)]
pub struct PocketTtsConfig {
    /// Model variant identifier (default: "b6369a24").
    pub variant: String,
    /// Directory containing voice embedding `.safetensors` files.
    pub voices_dir: std::path::PathBuf,
    /// Default voice name (default: "alba").
    pub default_voice: String,
    /// Temperature for generation (lower = more deterministic, default: 0.7).
    pub temperature: f32,
    /// Number of LSD decode steps (default: 1).
    pub lsd_decode_steps: usize,
    /// End-of-sequence threshold (default: -4.0).
    pub eos_threshold: f32,
}

impl Default for PocketTtsConfig {
    fn default() -> Self {
        Self {
            variant: DEFAULT_VARIANT.into(),
            voices_dir: "models/pocket-voices".into(),
            default_voice: "alba".into(),
            temperature: 0.7,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
        }
    }
}

/// Pocket TTS backend — lightweight, pure Rust, edge-optimized.
///
/// Uses Kyutai's Pocket TTS model (100M parameters) via the Candle
/// ML framework. Runs on CPU, Metal (macOS), or CUDA (Jetson).
///
/// Models auto-download from HuggingFace on first use.
/// Requires `HF_TOKEN` environment variable (model is gated).
pub struct PocketTtsBackend {
    model: Arc<TTSModel>,
    config: PocketTtsConfig,
}

impl PocketTtsBackend {
    /// Create a new Pocket TTS backend with default settings.
    ///
    /// Downloads the model from HuggingFace on first use (~236MB).
    /// Requires `HF_TOKEN` environment variable.
    pub fn new() -> Result<Self, VoxError> {
        Self::with_config(PocketTtsConfig::default())
    }

    /// Create with a specific voice and voices directory.
    ///
    /// Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma.
    pub fn with_voice(voice: &str, voices_dir: impl Into<std::path::PathBuf>) -> Result<Self, VoxError> {
        Self::with_config(PocketTtsConfig {
            default_voice: voice.into(),
            voices_dir: voices_dir.into(),
            ..Default::default()
        })
    }

    /// Create with custom configuration.
    pub fn with_config(config: PocketTtsConfig) -> Result<Self, VoxError> {
        let model = TTSModel::load_with_params(
            &config.variant,
            config.temperature,
            config.lsd_decode_steps,
            config.eos_threshold,
        )
        .map_err(|e| VoxError::Tts(format!("failed to load pocket-tts model: {e}")))?;

        Ok(Self {
            model: Arc::new(model),
            config,
        })
    }

    /// Load a voice state from a voice name or file path.
    ///
    /// Built-in voices (alba, marius, etc.) are resolved from the voices directory.
    /// File paths ending in `.safetensors` are loaded as pre-computed embeddings.
    /// Other file paths (e.g. `.wav`) are used for voice cloning.
    fn load_voice_state(
        &self,
        voice: &str,
    ) -> Result<pocket_tts::ModelState, VoxError> {
        if BUILTIN_VOICES.contains(&voice) || !voice.contains('.') {
            // Resolve built-in voice name to embedding file in voices_dir.
            let embedding_path = self.config.voices_dir.join(format!("{voice}.safetensors"));
            if !embedding_path.exists() {
                return Err(VoxError::Tts(format!(
                    "voice embedding not found: {} (download with: scripts/download_models.sh)",
                    embedding_path.display()
                )));
            }
            self.model
                .get_voice_state_from_prompt_file(&embedding_path)
                .map_err(|e| VoxError::Tts(format!("failed to load voice '{voice}': {e}")))
        } else if voice.ends_with(".safetensors") {
            // Pre-computed voice embedding file (absolute or relative path).
            self.model
                .get_voice_state_from_prompt_file(voice)
                .map_err(|e| VoxError::Tts(format!("failed to load voice embedding: {e}")))
        } else {
            // WAV file for voice cloning.
            self.model
                .get_voice_state(voice)
                .map_err(|e| VoxError::Tts(format!("failed to clone voice from '{voice}': {e}")))
        }
    }
}

#[async_trait]
impl TtsBackend for PocketTtsBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        let voice_name = request
            .voice
            .as_deref()
            .unwrap_or(&self.config.default_voice);

        let voice_state = self.load_voice_state(voice_name)?;
        let model = self.model.clone();
        let text = request.text.clone();

        // Pocket TTS is synchronous — run on a blocking thread pool.
        let audio_tensor = tokio::task::spawn_blocking(move || {
            model.generate(&text, &voice_state)
        })
        .await
        .map_err(|e| VoxError::Tts(format!("pocket-tts task panicked: {e}")))?
        .map_err(|e| VoxError::Tts(format!("pocket-tts synthesis failed: {e}")))?;

        // Convert Candle Tensor to Vec<f32>.
        let samples: Vec<f32> = audio_tensor
            .flatten_all()
            .map_err(|e| VoxError::Tts(format!("tensor flatten failed: {e}")))?
            .to_vec1()
            .map_err(|e| VoxError::Tts(format!("tensor to vec failed: {e}")))?;

        let duration_ms = (samples.len() as u64 * 1000) / u64::from(POCKET_SAMPLE_RATE);

        Ok(TtsOutput {
            audio: AudioChunk {
                samples,
                sample_rate: POCKET_SAMPLE_RATE,
                channels: 1,
            },
            duration_ms,
        })
    }
}
