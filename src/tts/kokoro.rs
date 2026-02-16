//! Kokoro TTS backend using the kokoro-tts crate.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kokoro_tts::{KokoroTts, Voice as KokoroVoice};

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest, VoiceInfo};

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
        // Map voice name strings to the Voice enum.
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
            "am_fenrir" => Ok(KokoroVoice::AmFenrir(speed)),
            "am_santa" => Ok(KokoroVoice::AmSanta(speed)),
            // British female (v1.0)
            "bf_alice" => Ok(KokoroVoice::BfAlice(speed)),
            "bf_emma" => Ok(KokoroVoice::BfEmma(speed)),
            "bf_isabella" => Ok(KokoroVoice::BfIsabella(speed)),
            "bf_lily" => Ok(KokoroVoice::BfLily(speed)),
            // British female (v1.1 -- i32 speed)
            "bf_vale" => Ok(KokoroVoice::BfVale(speed_i32)),
            // British male (v1.0)
            "bm_daniel" => Ok(KokoroVoice::BmDaniel(speed)),
            "bm_fable" => Ok(KokoroVoice::BmFable(speed)),
            "bm_george" => Ok(KokoroVoice::BmGeorge(speed)),
            "bm_lewis" => Ok(KokoroVoice::BmLewis(speed)),
            // Japanese (v1.0)
            "jf_alpha" => Ok(KokoroVoice::JfAlpha(speed)),
            "jf_gongitsune" => Ok(KokoroVoice::JfGongitsune(speed)),
            "jf_tebukuro" => Ok(KokoroVoice::JfTebukuro(speed)),
            "jf_nezumi" => Ok(KokoroVoice::JfNezumi(speed)),
            "jm_kumo" => Ok(KokoroVoice::JmKumo(speed)),
            // Chinese (v1.0)
            "zf_xiaobei" => Ok(KokoroVoice::ZfXiaobei(speed)),
            "zf_xiaoni" => Ok(KokoroVoice::ZfXiaoni(speed)),
            "zf_xiaoxiao" => Ok(KokoroVoice::ZfXiaoxiao(speed)),
            "zf_xiaoyi" => Ok(KokoroVoice::ZfXiaoyi(speed)),
            "zm_yunjian" => Ok(KokoroVoice::ZmYunjian(speed)),
            "zm_yunxi" => Ok(KokoroVoice::ZmYunxi(speed)),
            "zm_yunxia" => Ok(KokoroVoice::ZmYunxia(speed)),
            "zm_yunyang" => Ok(KokoroVoice::ZmYunyang(speed)),
            // Spanish (v1.0)
            "ef_dora" => Ok(KokoroVoice::EfDora(speed)),
            "em_alex" => Ok(KokoroVoice::EmAlex(speed)),
            "em_santa" => Ok(KokoroVoice::EmSanta(speed)),
            // French (v1.0)
            "ff_siwis" => Ok(KokoroVoice::FfSiwis(speed)),
            // Hindi (v1.0)
            "hf_alpha" => Ok(KokoroVoice::HfAlpha(speed)),
            "hf_beta" => Ok(KokoroVoice::HfBeta(speed)),
            "hm_omega" => Ok(KokoroVoice::HmOmega(speed)),
            "hm_psi" => Ok(KokoroVoice::HmPsi(speed)),
            // Italian (v1.0)
            "if_sara" => Ok(KokoroVoice::IfSara(speed)),
            "im_nicola" => Ok(KokoroVoice::ImNicola(speed)),
            // Portuguese (v1.0)
            "pf_dora" => Ok(KokoroVoice::PfDora(speed)),
            "pm_alex" => Ok(KokoroVoice::PmAlex(speed)),
            "pm_santa" => Ok(KokoroVoice::PmSanta(speed)),
            _ => Err(VoxError::Tts(format!("unknown voice: {name}"))),
        }
    }

    /// Build the full list of supported named voices with metadata.
    fn voice_list() -> Vec<VoiceInfo> {
        vec![
            // American English female
            VoiceInfo {
                id: "af_heart".into(),
                name: "Heart".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_alloy".into(),
                name: "Alloy".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_aoede".into(),
                name: "Aoede".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_bella".into(),
                name: "Bella".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_jessica".into(),
                name: "Jessica".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_kore".into(),
                name: "Kore".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_nicole".into(),
                name: "Nicole".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_nova".into(),
                name: "Nova".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_river".into(),
                name: "River".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_sarah".into(),
                name: "Sarah".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_sky".into(),
                name: "Sky".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_maple".into(),
                name: "Maple".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_sol".into(),
                name: "Sol".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // American English male
            VoiceInfo {
                id: "am_adam".into(),
                name: "Adam".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_echo".into(),
                name: "Echo".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_eric".into(),
                name: "Eric".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_fenrir".into(),
                name: "Fenrir".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_liam".into(),
                name: "Liam".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_michael".into(),
                name: "Michael".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_onyx".into(),
                name: "Onyx".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_puck".into(),
                name: "Puck".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // British English female
            VoiceInfo {
                id: "bf_alice".into(),
                name: "Alice".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_emma".into(),
                name: "Emma".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_isabella".into(),
                name: "Isabella".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_lily".into(),
                name: "Lily".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_vale".into(),
                name: "Vale".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // British English male
            VoiceInfo {
                id: "bm_daniel".into(),
                name: "Daniel".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_fable".into(),
                name: "Fable".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_george".into(),
                name: "George".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_lewis".into(),
                name: "Lewis".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // Japanese
            VoiceInfo {
                id: "jf_alpha".into(),
                name: "Alpha".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese".into(),
            },
            VoiceInfo {
                id: "jf_gongitsune".into(),
                name: "Gongitsune".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese".into(),
            },
            VoiceInfo {
                id: "jf_tebukuro".into(),
                name: "Tebukuro".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese".into(),
            },
            VoiceInfo {
                id: "jf_nezumi".into(),
                name: "Nezumi".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese".into(),
            },
            VoiceInfo {
                id: "jm_kumo".into(),
                name: "Kumo".into(),
                gender: "male".into(),
                language: "ja".into(),
                accent: "Japanese".into(),
            },
            // Chinese
            VoiceInfo {
                id: "zf_xiaobei".into(),
                name: "Xiaobei".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoni".into(),
                name: "Xiaoni".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoxiao".into(),
                name: "Xiaoxiao".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoyi".into(),
                name: "Xiaoyi".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunjian".into(),
                name: "Yunjian".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunxi".into(),
                name: "Yunxi".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunxia".into(),
                name: "Yunxia".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunyang".into(),
                name: "Yunyang".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            // Spanish
            VoiceInfo {
                id: "ef_dora".into(),
                name: "Dora".into(),
                gender: "female".into(),
                language: "es".into(),
                accent: "Spanish".into(),
            },
            VoiceInfo {
                id: "em_alex".into(),
                name: "Alex".into(),
                gender: "male".into(),
                language: "es".into(),
                accent: "Spanish".into(),
            },
            VoiceInfo {
                id: "em_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "es".into(),
                accent: "Spanish".into(),
            },
            // French
            VoiceInfo {
                id: "ff_siwis".into(),
                name: "Siwis".into(),
                gender: "female".into(),
                language: "fr".into(),
                accent: "French".into(),
            },
            // Hindi
            VoiceInfo {
                id: "hf_alpha".into(),
                name: "Alpha".into(),
                gender: "female".into(),
                language: "hi".into(),
                accent: "Hindi".into(),
            },
            VoiceInfo {
                id: "hf_beta".into(),
                name: "Beta".into(),
                gender: "female".into(),
                language: "hi".into(),
                accent: "Hindi".into(),
            },
            VoiceInfo {
                id: "hm_omega".into(),
                name: "Omega".into(),
                gender: "male".into(),
                language: "hi".into(),
                accent: "Hindi".into(),
            },
            VoiceInfo {
                id: "hm_psi".into(),
                name: "Psi".into(),
                gender: "male".into(),
                language: "hi".into(),
                accent: "Hindi".into(),
            },
            // Italian
            VoiceInfo {
                id: "if_sara".into(),
                name: "Sara".into(),
                gender: "female".into(),
                language: "it".into(),
                accent: "Italian".into(),
            },
            VoiceInfo {
                id: "im_nicola".into(),
                name: "Nicola".into(),
                gender: "male".into(),
                language: "it".into(),
                accent: "Italian".into(),
            },
            // Portuguese (Brazilian)
            VoiceInfo {
                id: "pf_dora".into(),
                name: "Dora".into(),
                gender: "female".into(),
                language: "pt-BR".into(),
                accent: "Brazilian".into(),
            },
            VoiceInfo {
                id: "pm_alex".into(),
                name: "Alex".into(),
                gender: "male".into(),
                language: "pt-BR".into(),
                accent: "Brazilian".into(),
            },
            VoiceInfo {
                id: "pm_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "pt-BR".into(),
                accent: "Brazilian".into(),
            },
        ]
    }
}

#[async_trait]
impl TtsBackend for KokoroBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        if request.seed.is_some() {
            tracing::debug!("kokoro backend does not support seed; ignoring");
        }

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

    fn list_voices(&self) -> Vec<VoiceInfo> {
        Self::voice_list()
    }

    fn backend_name(&self) -> &str {
        "kokoro"
    }
}
