//! Chatterbox Turbo TTS backend (Resemble AI, 350M params).
//!
//! High-quality voice cloning TTS via ONNX Runtime. Designed for
//! desktop/Mac targets. Models auto-download from HuggingFace.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chatterbox_rs::chatterbox::Chatterbox;
use chatterbox_rs::hf::{self, ModelVariant};
use chatterbox_rs::voice::VoiceProfile;

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest, VoiceInfo};

const CHATTERBOX_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_REPO_ID: &str = "ResembleAI/chatterbox-turbo-ONNX";
const DEFAULT_REVISION: &str = "main";

/// Configuration for the Chatterbox Turbo backend.
#[derive(Debug, Clone)]
pub struct ChatterboxConfig {
    /// Model quantization variant: "fp32", "fp16", "q4", "q4f16", "q8" (default: "q4").
    pub dtype: String,
    /// Path to reference WAV for voice cloning (required).
    pub reference_audio: PathBuf,
    /// Maximum tokens to generate (default: 256).
    pub max_new_tokens: usize,
    /// Repetition penalty (default: 1.2).
    pub repetition_penalty: f32,
    /// Number of intra-op threads for ONNX Runtime (default: 4).
    pub intra_threads: usize,
    /// Use CoreML execution provider for Apple Neural Engine / GPU acceleration.
    /// Requires the `chatterbox-coreml` feature (default: false).
    pub coreml: bool,
}

impl Default for ChatterboxConfig {
    fn default() -> Self {
        Self {
            dtype: "fp16".into(),
            reference_audio: "reference.wav".into(),
            max_new_tokens: 256,
            repetition_penalty: 1.2,
            intra_threads: 4,
            coreml: false,
        }
    }
}

/// Chatterbox Turbo TTS backend — high-quality voice cloning.
///
/// Uses Resemble AI's Chatterbox Turbo model (350M parameters)
/// via ONNX Runtime. Supports fp32, fp16, and quantized variants.
///
/// Models auto-download from HuggingFace on first use (~720MB for q4).
pub struct ChatterboxBackend {
    model: Arc<Mutex<Chatterbox>>,
    config: ChatterboxConfig,
    /// In-memory cache of encoded voice profiles keyed by canonical WAV path.
    voice_cache: Arc<Mutex<HashMap<PathBuf, VoiceProfile>>>,
}

impl ChatterboxBackend {
    /// Create a new Chatterbox backend with default q4 quantization.
    ///
    /// Downloads the model from HuggingFace on first use (~720MB).
    ///
    /// `reference_audio` is the path to a 5-20s WAV file of the
    /// target voice for cloning.
    pub fn new(reference_audio: impl AsRef<Path>) -> Result<Self, VoxError> {
        Self::with_config(ChatterboxConfig {
            reference_audio: reference_audio.as_ref().to_path_buf(),
            ..Default::default()
        })
    }

    /// Create from a local model directory (skips HuggingFace download).
    ///
    /// The directory must contain: `tokenizer.json` and the four ONNX model
    /// pairs (e.g. `conditional_decoder_q4.onnx` + `.onnx_data`).
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        reference_audio: impl AsRef<Path>,
    ) -> Result<Self, VoxError> {
        Self::from_model_dir_with_config(
            model_dir,
            ChatterboxConfig {
                reference_audio: reference_audio.as_ref().to_path_buf(),
                ..Default::default()
            },
        )
    }

    /// Create from a local model directory with custom config.
    pub fn from_model_dir_with_config(
        model_dir: impl AsRef<Path>,
        config: ChatterboxConfig,
    ) -> Result<Self, VoxError> {
        let dir = model_dir.as_ref();
        let suffix = match config.dtype.as_str() {
            "fp32" => "",
            "fp16" => "_fp16",
            "q4" => "_q4",
            "q4f16" => "_q4f16",
            "q8" => "_q8",
            "q8f16" => "_q8f16",
            "quantized" => "_quantized",
            other => {
                return Err(VoxError::Tts(format!("unknown dtype: {other}")));
            }
        };

        let paths = hf::ChatterboxPaths {
            tokenizer_json: dir.join("tokenizer.json"),
            conditional_decoder: dir.join(format!("conditional_decoder{suffix}.onnx")),
            speech_encoder: dir.join(format!("speech_encoder{suffix}.onnx")),
            embed_tokens: dir.join(format!("embed_tokens{suffix}.onnx")),
            language_model: dir.join(format!("language_model{suffix}.onnx")),
        };

        Self::load_from_paths(paths, config)
    }

    /// Create with custom configuration (downloads models from HuggingFace).
    pub fn with_config(config: ChatterboxConfig) -> Result<Self, VoxError> {
        let variant = parse_model_variant(&config.dtype)?;

        tracing::info!(
            dtype = %config.dtype,
            intra_threads = config.intra_threads,
            "downloading chatterbox turbo model assets"
        );

        let paths = hf::download_chatterbox_assets(DEFAULT_REPO_ID, DEFAULT_REVISION, variant)
            .map_err(|e| VoxError::Tts(format!("failed to download chatterbox models: {e}")))?;

        // ONNX Runtime resolves external data files (.onnx_data) relative to the
        // canonical path of the .onnx file. HuggingFace Hub stores files as symlinks
        // to content-addressed blobs, which breaks this resolution. Copy files to a
        // temporary directory with real paths.
        let paths = copy_to_real_paths(&paths)?;

        Self::load_from_paths(paths, config)
    }

    fn load_from_paths(
        paths: hf::ChatterboxPaths,
        config: ChatterboxConfig,
    ) -> Result<Self, VoxError> {
        let execution_provider = if config.coreml {
            tracing::info!("CoreML execution provider requested");
            chatterbox_rs::chatterbox::ExecutionProvider::CoreML
        } else {
            chatterbox_rs::chatterbox::ExecutionProvider::Auto
        };

        let session_config = chatterbox_rs::chatterbox::SessionConfig {
            intra_threads: Some(config.intra_threads),
            inter_threads: None,
            parallel_execution: false,
            execution_provider,
            coreml_cache_dir: None,
        };

        tracing::info!("loading chatterbox turbo model");

        let model = Chatterbox::load_with(&paths, &session_config)
            .map_err(|e| VoxError::Tts(format!("failed to load chatterbox model: {e}")))?;

        tracing::info!("chatterbox turbo model loaded");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
            voice_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl TtsBackend for ChatterboxBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        if request.seed.is_some() {
            tracing::debug!("chatterbox backend does not support seed; ignoring");
        }

        let reference_wav = request
            .voice
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.config.reference_audio.clone());

        if !reference_wav.exists() {
            return Err(VoxError::Tts(format!(
                "reference audio not found: {}",
                reference_wav.display()
            )));
        }

        let model = self.model.clone();
        let text = request.text.clone();
        let max_new_tokens = self.config.max_new_tokens;
        let repetition_penalty = self.config.repetition_penalty;
        let voice_cache = self.voice_cache.clone();
        let repo_id = DEFAULT_REPO_ID.to_string();
        let revision = DEFAULT_REVISION.to_string();
        let dtype = self.config.dtype.clone();

        // Chatterbox synthesis is synchronous — run on a blocking thread pool.
        let samples = tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|e| VoxError::Tts(format!("chatterbox mutex poisoned: {e}")))?;

            // Use canonical path as cache key to handle symlinks/relative paths.
            let cache_key = reference_wav.canonicalize().unwrap_or(reference_wav.clone());

            // Check voice profile cache; encode on miss.
            let mut cache = voice_cache
                .lock()
                .map_err(|e| VoxError::Tts(format!("voice cache mutex poisoned: {e}")))?;

            let profile = if let Some(cached) = cache.get(&cache_key) {
                tracing::debug!(path = %cache_key.display(), "voice profile cache hit");
                cached.clone()
            } else {
                tracing::debug!(path = %cache_key.display(), "voice profile cache miss, encoding");
                let p = model
                    .encode_voice_profile(&reference_wav, &repo_id, &revision, &dtype)
                    .map_err(|e| VoxError::Tts(format!("voice profile encoding failed: {e}")))?;
                cache.insert(cache_key, p.clone());
                p
            };
            drop(cache);

            model
                .synthesize_with_voice_profile(
                    &text,
                    &repo_id,
                    &revision,
                    &dtype,
                    &profile,
                    max_new_tokens,
                    repetition_penalty,
                )
                .map_err(|e| VoxError::Tts(format!("chatterbox synthesis failed: {e}")))
        })
        .await
        .map_err(|e| VoxError::Tts(format!("chatterbox task panicked: {e}")))??;

        let duration_ms = (samples.len() as u64 * 1000) / u64::from(CHATTERBOX_SAMPLE_RATE);

        Ok(TtsOutput {
            audio: AudioChunk {
                samples,
                sample_rate: CHATTERBOX_SAMPLE_RATE,
                channels: 1,
            },
            duration_ms,
        })
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        vec![VoiceInfo {
            id: "custom".to_string(),
            name: "Custom (Voice Cloning)".to_string(),
            gender: "any".to_string(),
            language: "en".to_string(),
            accent: "Cloned".to_string(),
        }]
    }

    fn backend_name(&self) -> &str {
        "chatterbox"
    }
}

// Copy HF Hub symlinked files to a directory with real paths so ONNX
// Runtime can resolve external data files.
fn copy_to_real_paths(paths: &hf::ChatterboxPaths) -> Result<hf::ChatterboxPaths, VoxError> {
    // Find the HF snapshot root (e.g. .../snapshots/<hash>/).
    let snapshot_root = hf::snapshot_root_from_cache_path(&paths.tokenizer_json)
        .ok_or_else(|| VoxError::Tts("cannot determine HF snapshot root".into()))?;

    let real_dir = snapshot_root.join("real");
    std::fs::create_dir_all(&real_dir)
        .map_err(|e| VoxError::Tts(format!("failed to create real model dir: {e}")))?;

    fn copy_if_needed(src: &Path, dest_dir: &Path) -> Result<PathBuf, VoxError> {
        let name = src
            .file_name()
            .ok_or_else(|| VoxError::Tts("model path has no filename".into()))?;
        let dest = dest_dir.join(name);
        if !dest.exists() {
            std::fs::copy(src, &dest)
                .map_err(|e| VoxError::Tts(format!("failed to copy {}: {e}", src.display())))?;
        }
        Ok(dest)
    }

    fn copy_onnx_pair(src: &Path, dest_dir: &Path) -> Result<PathBuf, VoxError> {
        let onnx_dest = copy_if_needed(src, dest_dir)?;
        // Also copy the companion .onnx_data file if it exists.
        let data_name = format!(
            "{}.onnx_data",
            src.file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default()
        );
        let data_src = src.parent().unwrap().join(&data_name);
        if data_src.exists() {
            copy_if_needed(&data_src, dest_dir)?;
        }
        Ok(onnx_dest)
    }

    Ok(hf::ChatterboxPaths {
        tokenizer_json: copy_if_needed(&paths.tokenizer_json, &real_dir)?,
        conditional_decoder: copy_onnx_pair(&paths.conditional_decoder, &real_dir)?,
        speech_encoder: copy_onnx_pair(&paths.speech_encoder, &real_dir)?,
        embed_tokens: copy_onnx_pair(&paths.embed_tokens, &real_dir)?,
        language_model: copy_onnx_pair(&paths.language_model, &real_dir)?,
    })
}

fn parse_model_variant(dtype: &str) -> Result<ModelVariant, VoxError> {
    match dtype.to_lowercase().as_str() {
        "fp32" => Ok(ModelVariant::Fp32),
        "fp16" => Ok(ModelVariant::Fp16),
        "q4" => Ok(ModelVariant::Q4),
        "q4f16" => Ok(ModelVariant::Q4f16),
        "q8" => Ok(ModelVariant::Q8),
        "q8f16" => Ok(ModelVariant::Q8f16),
        "quantized" => Ok(ModelVariant::Quantized),
        other => Err(VoxError::Tts(format!(
            "unknown chatterbox dtype '{other}'; expected: fp32, fp16, q4, q4f16, q8, q8f16"
        ))),
    }
}
