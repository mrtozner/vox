//! Piper TTS backend for multilingual speech synthesis.
//!
//! Uses piper-rs (VITS-based ONNX models) to support 35+ languages
//! with hundreds of pre-trained voices. Models are lightweight
//! (15-100MB each) and run entirely locally via ONNX Runtime.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use piper_rs::synth::PiperSpeechSynthesizer;

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest, VoiceInfo};

/// Configuration for the Piper TTS backend.
#[derive(Debug, Clone)]
pub struct PiperConfig {
    /// Path to the `.onnx.json` config file.
    /// The `.onnx` model file must be in the same directory.
    pub config_path: PathBuf,
    /// Default speaker ID for multi-speaker models.
    pub default_speaker: Option<i64>,
    /// Speaking speed scale (1.0 = normal, lower = faster, higher = slower).
    pub length_scale: Option<f32>,
}

/// Piper TTS backend -- fast, lightweight, multilingual.
///
/// Uses VITS models via piper-rs and ONNX Runtime. Each voice is a
/// small ONNX model (15-100MB) that runs entirely on CPU.
///
/// Typical output sample rate is 22050 Hz mono.
pub struct PiperBackend {
    synth: Arc<Mutex<PiperSpeechSynthesizer>>,
    model: Arc<dyn piper_rs::PiperModel + Send + Sync>,
    #[allow(dead_code)]
    config: PiperConfig,
    sample_rate: u32,
    language: String,
}

impl PiperBackend {
    /// Create a new Piper TTS backend from a `.onnx.json` config file.
    ///
    /// The ONNX model file is expected to be alongside the config file
    /// (piper convention: `model.onnx.json` -> `model.onnx`).
    pub fn new(config_path: impl AsRef<Path>) -> Result<Self, VoxError> {
        Self::with_config(PiperConfig {
            config_path: config_path.as_ref().to_path_buf(),
            default_speaker: None,
            length_scale: None,
        })
    }

    /// Create with custom configuration.
    pub fn with_config(config: PiperConfig) -> Result<Self, VoxError> {
        if !config.config_path.exists() {
            return Err(VoxError::ModelNotFound(config.config_path.clone()));
        }

        let model = piper_rs::from_config_path(&config.config_path)
            .map_err(|e| VoxError::Tts(format!("failed to load piper model: {e}")))?;

        // Set default speaker if specified
        if let Some(sid) = config.default_speaker {
            if let Some(err) = model.set_speaker(sid) {
                return Err(VoxError::Tts(format!(
                    "failed to set default speaker {sid}: {err}"
                )));
            }
        }

        // Set length_scale if specified
        if let Some(length_scale) = config.length_scale {
            let synth_config = piper_rs::PiperSynthesisConfig {
                speaker: config.default_speaker,
                length_scale,
                ..Default::default()
            };
            model
                .set_fallback_synthesis_config(&synth_config)
                .map_err(|e| VoxError::Tts(format!("failed to set synthesis config: {e}")))?;
        }

        // Extract audio info from the model
        let audio_info = model
            .audio_output_info()
            .map_err(|e| VoxError::Tts(format!("failed to get audio output info: {e}")))?;
        let sample_rate = audio_info.sample_rate as u32;

        // Extract language
        let language = model
            .get_language()
            .ok()
            .flatten()
            .unwrap_or_else(|| "unknown".to_string());

        let synth = PiperSpeechSynthesizer::new(Arc::clone(&model))
            .map_err(|e| VoxError::Tts(format!("failed to create piper synthesizer: {e}")))?;

        tracing::info!(
            sample_rate = sample_rate,
            language = %language,
            "loaded piper TTS backend"
        );

        Ok(Self {
            synth: Arc::new(Mutex::new(synth)),
            model,
            config,
            sample_rate,
            language,
        })
    }
}

#[async_trait]
impl TtsBackend for PiperBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        if request.seed.is_some() {
            tracing::debug!("piper backend does not support seed; ignoring");
        }

        // Handle speaker selection from the voice field
        if let Some(ref voice) = request.voice {
            // Try parsing as a numeric speaker ID first
            if let Ok(sid) = voice.parse::<i64>() {
                if let Some(err) = self.model.set_speaker(sid) {
                    return Err(VoxError::Tts(format!(
                        "failed to set speaker ID {sid}: {err}"
                    )));
                }
            } else {
                // Try looking up by speaker name
                match self.model.speaker_name_to_id(voice) {
                    Ok(Some(sid)) => {
                        if let Some(err) = self.model.set_speaker(sid) {
                            return Err(VoxError::Tts(format!(
                                "failed to set speaker '{voice}': {err}"
                            )));
                        }
                    }
                    Ok(None) => {
                        // Not a known speaker name -- ignore for single-speaker models
                        tracing::debug!(voice = %voice, "voice not found as speaker name, using default");
                    }
                    Err(e) => {
                        return Err(VoxError::Tts(format!(
                            "failed to look up speaker '{voice}': {e}"
                        )));
                    }
                }
            }
        }

        let synth = self.synth.clone();
        let text = request.text.clone();
        let sample_rate = self.sample_rate;

        // Piper synthesis is synchronous and CPU-bound -- run on blocking thread pool.
        let samples = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, VoxError> {
            let synth = synth
                .lock()
                .map_err(|e| VoxError::Tts(format!("piper mutex poisoned: {e}")))?;

            let audio_stream = synth
                .synthesize_parallel(text, None)
                .map_err(|e| VoxError::Tts(format!("piper synthesis failed: {e}")))?;

            let mut samples: Vec<f32> = Vec::new();
            for result in audio_stream {
                let audio =
                    result.map_err(|e| VoxError::Tts(format!("piper audio chunk error: {e}")))?;
                samples.append(&mut audio.into_vec());
            }

            Ok(samples)
        })
        .await
        .map_err(|e| VoxError::Tts(format!("piper task panicked: {e}")))??;

        let duration_ms = (samples.len() as u64 * 1000) / u64::from(sample_rate);

        Ok(TtsOutput {
            audio: AudioChunk {
                samples,
                sample_rate,
                channels: 1,
            },
            duration_ms,
        })
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        match self.model.get_speakers() {
            Ok(Some(speakers)) if !speakers.is_empty() => {
                let mut voices: Vec<VoiceInfo> = speakers
                    .iter()
                    .map(|(id, name)| VoiceInfo {
                        id: id.to_string(),
                        name: name.clone(),
                        gender: "unknown".to_string(),
                        language: self.language.clone(),
                        accent: "Piper".to_string(),
                    })
                    .collect();
                voices.sort_by(|a, b| a.id.cmp(&b.id));
                voices
            }
            _ => {
                // Single-speaker model
                vec![VoiceInfo {
                    id: "0".to_string(),
                    name: "Default".to_string(),
                    gender: "unknown".to_string(),
                    language: self.language.clone(),
                    accent: "Piper".to_string(),
                }]
            }
        }
    }

    fn backend_name(&self) -> &str {
        "piper"
    }
}
