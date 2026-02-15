//! Kokoro TTS backend using the kokoro-tts crate.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kokoro_tts::{KokoroTts, Voice as KokoroVoice};

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest};

/// Kokoro output sample rate.
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Configuration for the Kokoro TTS backend.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    /// Path to the Kokoro ONNX model file.
    pub model_path: PathBuf,
    /// Path to the voices binary file.
    pub voices_path: PathBuf,
    /// Default voice to use when none is specified.
    pub default_voice: String,
    /// Speed multiplier (1.0 = normal).
    pub speed: f32,
}

/// Kokoro TTS backend powered by the kokoro-tts crate.
///
/// Uses the Kokoro-82M model (82 million parameters) for high-quality
/// text-to-speech synthesis. Runs entirely locally via ONNX Runtime.
pub struct KokoroBackend {
    engine: Arc<KokoroTts>,
    config: KokoroConfig,
}

impl KokoroBackend {
    /// Create a new Kokoro TTS backend.
    ///
    /// `model_path` should point to `kokoro-v1.0.onnx` (or int8 variant).
    /// `voices_path` should point to `voices-v1.0.bin`.
    pub async fn new(
        model_path: impl AsRef<Path>,
        voices_path: impl AsRef<Path>,
    ) -> Result<Self, VoxError> {
        let model_path = model_path.as_ref().to_path_buf();
        let voices_path = voices_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(VoxError::ModelNotFound(model_path));
        }
        if !voices_path.exists() {
            return Err(VoxError::ModelNotFound(voices_path));
        }

        let engine = KokoroTts::new(&model_path, &voices_path)
            .await
            .map_err(|e| VoxError::Tts(format!("failed to load kokoro model: {e}")))?;

        Ok(Self {
            engine: Arc::new(engine),
            config: KokoroConfig {
                model_path,
                voices_path,
                default_voice: "af_heart".into(),
                speed: 1.0,
            },
        })
    }

    /// Create with custom configuration.
    pub async fn with_config(config: KokoroConfig) -> Result<Self, VoxError> {
        if !config.model_path.exists() {
            return Err(VoxError::ModelNotFound(config.model_path.clone()));
        }
        if !config.voices_path.exists() {
            return Err(VoxError::ModelNotFound(config.voices_path.clone()));
        }

        let engine = KokoroTts::new(&config.model_path, &config.voices_path)
            .await
            .map_err(|e| VoxError::Tts(format!("failed to load kokoro model: {e}")))?;

        Ok(Self {
            engine: Arc::new(engine),
            config,
        })
    }

    /// Parse a voice name string into a Kokoro Voice enum variant.
    fn parse_voice(&self, name: &str, speed: f32) -> Result<KokoroVoice, VoxError> {
        // Map common voice name strings to the Voice enum.
        // v1.0 voices take an f32 speed parameter; v1.1 voices take i32.
        let speed_i32 = speed as i32;
        match name.to_lowercase().as_str() {
            // American female (v1.0)
            "af_heart" => Ok(KokoroVoice::AfHeart(speed)),
            "af_sky" => Ok(KokoroVoice::AfSky(speed)),
            "af_bella" => Ok(KokoroVoice::AfBella(speed)),
            "af_sarah" => Ok(KokoroVoice::AfSarah(speed)),
            "af_nicole" => Ok(KokoroVoice::AfNicole(speed)),
            "af_nova" => Ok(KokoroVoice::AfNova(speed)),
            "af_river" => Ok(KokoroVoice::AfRiver(speed)),
            "af_alloy" => Ok(KokoroVoice::AfAlloy(speed)),
            "af_aoede" => Ok(KokoroVoice::AfAoede(speed)),
            "af_jessica" => Ok(KokoroVoice::AfJessica(speed)),
            "af_kore" => Ok(KokoroVoice::AfKore(speed)),
            // American female (v1.1 -- i32 speed)
            "af_maple" => Ok(KokoroVoice::AfMaple(speed_i32)),
            "af_sol" => Ok(KokoroVoice::AfSol(speed_i32)),
            // American male (v1.0)
            "am_adam" => Ok(KokoroVoice::AmAdam(speed)),
            "am_echo" => Ok(KokoroVoice::AmEcho(speed)),
            "am_eric" => Ok(KokoroVoice::AmEric(speed)),
            "am_liam" => Ok(KokoroVoice::AmLiam(speed)),
            "am_michael" => Ok(KokoroVoice::AmMichael(speed)),
            "am_onyx" => Ok(KokoroVoice::AmOnyx(speed)),
            "am_puck" => Ok(KokoroVoice::AmPuck(speed)),
            // British female (v1.0)
            "bf_alice" => Ok(KokoroVoice::BfAlice(speed)),
            "bf_emma" => Ok(KokoroVoice::BfEmma(speed)),
            "bf_isabella" => Ok(KokoroVoice::BfIsabella(speed)),
            "bf_lily" => Ok(KokoroVoice::BfLily(speed)),
            // British male (v1.0)
            "bm_daniel" => Ok(KokoroVoice::BmDaniel(speed)),
            "bm_fable" => Ok(KokoroVoice::BmFable(speed)),
            "bm_george" => Ok(KokoroVoice::BmGeorge(speed)),
            "bm_lewis" => Ok(KokoroVoice::BmLewis(speed)),
            _ => Err(VoxError::Tts(format!("unknown voice: {name}"))),
        }
    }
}

#[async_trait]
impl TtsBackend for KokoroBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        let voice_name = request
            .voice
            .as_deref()
            .unwrap_or(&self.config.default_voice);
        let voice = self.parse_voice(voice_name, self.config.speed)?;

        let (samples, duration) = self
            .engine
            .synth(&request.text, voice)
            .await
            .map_err(|e| VoxError::Tts(format!("kokoro synthesis failed: {e}")))?;

        let duration_ms = duration.as_millis() as u64;

        Ok(TtsOutput {
            audio: AudioChunk {
                samples,
                sample_rate: KOKORO_SAMPLE_RATE,
                channels: 1,
            },
            duration_ms,
        })
    }
}
