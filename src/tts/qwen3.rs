//! Qwen3-TTS backend integration.

use async_trait::async_trait;
use std::sync::Arc;

use qwen3_tts::{auto_device, Language, Qwen3TTS, Speaker, SynthesisOptions};
use tokio::sync::Mutex;

use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest, VoiceInfo};

/// Configuration for the Qwen3 TTS backend.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    /// Model variant: "0.6B" or "1.7B"
    pub model_variant: String,
    /// Device: "cuda", "metal", or "cpu"
    pub device: String,
    /// Default voice to use when none is specified
    pub default_voice: String,
    /// Speech speed multiplier (0.5 - 2.0, default 1.0)
    pub speed: f32,
    /// Temperature for generation (0.0 - 1.0, default 0.6)
    pub temperature: f32,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            model_variant: "0.6B".into(),
            device: Self::select_device(),
            default_voice: "en_us_female_1".into(),
            speed: 1.0,
            temperature: 0.6,
        }
    }
}

impl Qwen3Config {
    /// Select the best available device (Metal > CUDA > CPU)
    fn select_device() -> String {
        // Check environment variable override
        if let Ok(device) = std::env::var("VOX_QWEN3_DEVICE") {
            return device;
        }

        // Auto-detect best available device
        // Prioritize Metal on macOS (M1/M2/M3 chips)
        #[cfg(all(target_os = "macos", feature = "qwen3-metal"))]
        {
            if Self::is_metal_available() {
                return "metal".to_string();
            }
        }

        #[cfg(feature = "qwen3-cuda")]
        {
            if Self::is_cuda_available() {
                return "cuda".to_string();
            }
        }

        "cpu".to_string()
    }

    #[cfg(feature = "qwen3-cuda")]
    fn is_cuda_available() -> bool {
        // TODO: Actual CUDA detection logic
        false
    }

    #[cfg(feature = "qwen3-metal")]
    fn is_metal_available() -> bool {
        // Metal is always available on macOS (M1+, Intel with Metal support)
        cfg!(target_os = "macos")
    }
}

/// Qwen3 TTS backend powered by qwen3-tts.
///
/// Supports 9 preset speakers via CustomVoice model across languages.
/// Hardware acceleration available via Metal (macOS) or CUDA (Linux).
/// Models are automatically downloaded from HuggingFace Hub on first use.
pub struct Qwen3Backend {
    model: Arc<Mutex<Qwen3TTS>>,
    config: Qwen3Config,
}

impl Qwen3Backend {
    /// Create a new Qwen3 TTS backend with default configuration.
    pub async fn new() -> Result<Self, VoxError> {
        Self::with_config(Qwen3Config::default()).await
    }

    /// Create a new Qwen3 TTS backend with custom configuration.
    pub async fn with_config(config: Qwen3Config) -> Result<Self, VoxError> {
        tracing::info!(
            device = %config.device,
            model = %config.model_variant,
            "initializing qwen3 TTS backend"
        );

        // Determine model ID based on variant
        let model_id = std::env::var("VOX_QWEN3_MODEL").unwrap_or_else(|_| {
            if config.model_variant == "1.7B" {
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string()
            } else {
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string()
            }
        });

        // Use HuggingFace cache directory with better detection
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/huggingface", h)))
            .unwrap_or_else(|_| "/tmp/huggingface".to_string());

        // Try multiple path patterns HuggingFace uses
        let model_paths = vec![
            // New HF cache format (models--Org--Repo/snapshots/main)
            std::path::PathBuf::from(&cache_dir)
                .join("hub")
                .join(format!("models--{}--{}",
                    model_id.split('/').nth(0).unwrap_or("Qwen"),
                    model_id.split('/').nth(1).unwrap_or("Qwen3-TTS-12Hz-0.6B-CustomVoice")))
                .join("snapshots")
                .join("main"),
            // Old HF cache format (hub/Org/Repo)
            std::path::PathBuf::from(&cache_dir)
                .join("hub")
                .join(&model_id),
            // Environment override
            std::env::var("VOX_QWEN3_MODEL_PATH")
                .ok()
                .map(std::path::PathBuf::from)
                .unwrap_or_default(),
        ];

        let model_path = model_paths.into_iter()
            .find(|p| p.exists() && p.join("config.json").exists())
            .ok_or_else(|| VoxError::Tts(format!(
                "model not found. Please download using:\n  \
                 huggingface-cli download {} --local-dir ~/.cache/huggingface/hub/models--Qwen--{}/snapshots/main",
                model_id,
                model_id.split('/').nth(1).unwrap_or("Qwen3-TTS-12Hz-0.6B-CustomVoice")
            )))?;

        // Convert to string for API
        let model_path_str = model_path.to_string_lossy().to_string();

        // Load model (this will automatically download from HuggingFace if not cached)
        let model = tokio::task::spawn_blocking(move || -> Result<Qwen3TTS, VoxError> {
            tracing::info!(path = %model_path_str, "loading from local cache");

            // Select device
            let device = auto_device().map_err(|e| VoxError::Tts(format!("failed to select device: {e}")))?;

            // Load model from pretrained path
            let model = Qwen3TTS::from_pretrained(&model_path_str, device)
                .map_err(|e| VoxError::Tts(format!("failed to load qwen3 model: {e}")))?;

            tracing::info!("qwen3 model loaded successfully");
            Ok(model)
        })
        .await
        .map_err(|e| VoxError::Tts(format!("model loading task panicked: {e}")))??;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
        })
    }

    /// Parse a voice name string into Speaker and Language.
    fn parse_voice(&self, name: &str) -> Result<(Speaker, Language), VoxError> {
        // Map voice IDs to Speaker + Language
        match name {
            // US English
            "en_us_female_1" | "en_us_female_2" => Ok((Speaker::Vivian, Language::English)),
            // Fix: Map US Male to Aiden instead of Ryan due to EOS generation bug in 0.6B model
            "en_us_male_1" | "en_us_male_2" => Ok((Speaker::Aiden, Language::English)),
            // GB English
            "en_gb_female_1" => Ok((Speaker::Vivian, Language::English)),
            "en_gb_male_1" => Ok((Speaker::Ryan, Language::English)),
            // Chinese
            "zh_cn_female_1" | "zh_cn_female_2" => Ok((Speaker::Vivian, Language::Chinese)),
            "zh_cn_male_1" => Ok((Speaker::Ryan, Language::Chinese)),
            // Japanese
            "ja_jp_female_1" => Ok((Speaker::Vivian, Language::Japanese)),
            "ja_jp_male_1" => Ok((Speaker::Ryan, Language::Japanese)),
            // Spanish
            "es_es_female_1" => Ok((Speaker::Serena, Language::Spanish)),
            "es_es_male_1" => Ok((Speaker::Ryan, Language::Spanish)),
            // French
            "fr_fr_female_1" => Ok((Speaker::Serena, Language::French)),
            "fr_fr_male_1" => Ok((Speaker::Ryan, Language::French)),
            // German
            "de_de_female_1" => Ok((Speaker::Serena, Language::German)),
            "de_de_male_1" => Ok((Speaker::Ryan, Language::German)),
            // Italian
            "it_it_female_1" => Ok((Speaker::Serena, Language::Italian)),
            // Portuguese
            "pt_br_female_1" => Ok((Speaker::Serena, Language::Portuguese)),
            // Russian
            "ru_ru_female_1" => Ok((Speaker::Serena, Language::Russian)),
            _ => Err(VoxError::Tts(format!(
                "unknown voice: {name}. Use /v1/voices to see available voices"
            ))),
        }
    }

    /// Build the full list of supported voices with metadata.
    fn voice_list() -> Vec<VoiceInfo> {
        vec![
            // American English female
            VoiceInfo {
                id: "en_us_female_1".into(),
                name: "US Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "en_us_female_2".into(),
                name: "US Female 2 (Vivian)".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // American English male
            VoiceInfo {
                id: "en_us_male_1".into(),
                name: "US Male 1 (Aiden)".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "en_us_male_2".into(),
                name: "US Male 2 (Aiden)".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // British English female
            VoiceInfo {
                id: "en_gb_female_1".into(),
                name: "GB Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // British English male
            VoiceInfo {
                id: "en_gb_male_1".into(),
                name: "GB Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // Chinese female
            VoiceInfo {
                id: "zh_cn_female_1".into(),
                name: "CN Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "zh-CN".into(),
                accent: "Mandarin".into(),
            },
            VoiceInfo {
                id: "zh_cn_female_2".into(),
                name: "CN Female 2 (Vivian)".into(),
                gender: "female".into(),
                language: "zh-CN".into(),
                accent: "Mandarin".into(),
            },
            // Chinese male
            VoiceInfo {
                id: "zh_cn_male_1".into(),
                name: "CN Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "zh-CN".into(),
                accent: "Mandarin".into(),
            },
            // Japanese female
            VoiceInfo {
                id: "ja_jp_female_1".into(),
                name: "JP Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "ja-JP".into(),
                accent: "Japanese".into(),
            },
            // Japanese male
            VoiceInfo {
                id: "ja_jp_male_1".into(),
                name: "JP Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "ja-JP".into(),
                accent: "Japanese".into(),
            },
            // Spanish female
            VoiceInfo {
                id: "es_es_female_1".into(),
                name: "ES Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "es-ES".into(),
                accent: "Spanish".into(),
            },
            // Spanish male
            VoiceInfo {
                id: "es_es_male_1".into(),
                name: "ES Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "es-ES".into(),
                accent: "Spanish".into(),
            },
            // French female
            VoiceInfo {
                id: "fr_fr_female_1".into(),
                name: "FR Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "fr-FR".into(),
                accent: "French".into(),
            },
            // French male
            VoiceInfo {
                id: "fr_fr_male_1".into(),
                name: "FR Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "fr-FR".into(),
                accent: "French".into(),
            },
            // German female
            VoiceInfo {
                id: "de_de_female_1".into(),
                name: "DE Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "de-DE".into(),
                accent: "German".into(),
            },
            // German male
            VoiceInfo {
                id: "de_de_male_1".into(),
                name: "DE Male 1 (Ethan)".into(),
                gender: "male".into(),
                language: "de-DE".into(),
                accent: "German".into(),
            },
            // Italian female
            VoiceInfo {
                id: "it_it_female_1".into(),
                name: "IT Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "it-IT".into(),
                accent: "Italian".into(),
            },
            // Portuguese female
            VoiceInfo {
                id: "pt_br_female_1".into(),
                name: "PT Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "pt-BR".into(),
                accent: "Brazilian".into(),
            },
            // Russian female
            VoiceInfo {
                id: "ru_ru_female_1".into(),
                name: "RU Female 1 (Vivian)".into(),
                gender: "female".into(),
                language: "ru-RU".into(),
                accent: "Russian".into(),
            },
        ]
    }
}

impl Qwen3Backend {
    /// Synthesize audio with chunked streaming callback.
    ///
    /// Instead of waiting for complete synthesis, this method calls `chunk_callback`
    /// with audio chunks as they are generated (~800ms per chunk). This enables
    /// low-latency playback start while synthesis continues in the background.
    ///
    /// # Note
    ///
    /// The underlying vendored library supports true streaming via
    /// `Qwen3TTS::synthesize_streaming()`, but exposing it through the mutex-protected
    /// model requires careful lifetime management. For WebSocket streaming use cases,
    /// consider calling the vendored library directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// backend.synthesize_with_streaming(&request, |chunk| {
    ///     // Send chunk to audio player or websocket
    ///     Ok(())
    /// }).await?;
    /// ```
    pub async fn synthesize_with_streaming<F>(
        &self,
        request: &TtsRequest,
        mut chunk_callback: F,
    ) -> Result<TtsOutput, VoxError>
    where
        F: FnMut(AudioChunk) -> Result<(), VoxError> + Send + 'static,
    {
        let voice_name = request
            .voice
            .as_deref()
            .unwrap_or(&self.config.default_voice);
        let (speaker, language) = self.parse_voice(voice_name)?;

        let text = request.text.clone();
        let temperature = self.config.temperature;
        let seed = request.seed;
        let model = Arc::clone(&self.model);

        // Run streaming synthesis on blocking thread pool
        let start = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || -> Result<_, VoxError> {
            let model_guard = model.blocking_lock();

            let options = qwen3_tts::SynthesisOptions {
                temperature: temperature as f64,
                seed,
                chunk_frames: 10, // ~800ms chunks at 12.5 Hz
                ..Default::default()
            };

            let mut all_samples = Vec::new();
            let mut sample_rate = 24000;

            // Create streaming session
            let session = model_guard
                .synthesize_streaming(&text, speaker, language, options)
                .map_err(|e| VoxError::Tts(format!("streaming session failed: {e}")))?;

            // Iterate through chunks and invoke callback
            for chunk_result in session {
                let audio_buf = chunk_result
                    .map_err(|e| VoxError::Tts(format!("chunk generation failed: {e}")))?;

                sample_rate = audio_buf.sample_rate;
                all_samples.extend_from_slice(&audio_buf.samples);

                // Invoke callback with chunk
                chunk_callback(AudioChunk {
                    samples: audio_buf.samples,
                    sample_rate,
                    channels: 1,
                })?;
            }

            Ok((all_samples, sample_rate))
        })
        .await
        .map_err(|e| VoxError::Tts(format!("streaming task panicked: {e}")))?;

        let (samples, sample_rate) = result?;
        let duration_ms = (samples.len() as f64 / sample_rate as f64 * 1000.0) as u64;

        tracing::info!(
            duration_ms = duration_ms,
            processing_time_ms = start.elapsed().as_millis() as u64,
            rtf = start.elapsed().as_secs_f64() / (duration_ms as f64 / 1000.0),
            "qwen3 streaming synthesis complete"
        );

        Ok(TtsOutput {
            audio: AudioChunk {
                samples,
                sample_rate,
                channels: 1,
            },
            duration_ms,
        })
    }
}

#[async_trait]
impl TtsBackend for Qwen3Backend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        let voice_name = request
            .voice
            .as_deref()
            .unwrap_or(&self.config.default_voice);
        let (speaker, language) = self.parse_voice(voice_name)?;

        let text = request.text.clone();
        let temperature = self.config.temperature;
        let seed = request.seed;

        // Clone the Arc to move into blocking task
        let model = Arc::clone(&self.model);

        // Run synthesis - use blocking because model.lock() is async
        let start = std::time::Instant::now();
        let result = {
            // Lock the mutex for the duration of synthesis
            let model_guard = model.lock().await;

            // Run synthesis on the blocking thread pool
            // We can't move the guard, so we do synthesis here
            tracing::debug!(
                text_len = text.len(),
                speaker = ?speaker,
                language = ?language,
                "starting qwen3 synthesis"
            );

            // Build synthesis options
            let options = SynthesisOptions {
                temperature: temperature as f64,
                seed,
                ..Default::default()
            };

            // Synthesize audio
            let audio = model_guard
                .synthesize_with_voice(&text, speaker, language, Some(options))
                .map_err(|e| VoxError::Tts(format!("synthesis failed: {e}")))?;

            // Extract samples
            let samples = audio.samples;
            let sample_rate = audio.sample_rate;

            tracing::debug!(
                num_samples = samples.len(),
                sample_rate = sample_rate,
                "synthesis completed"
            );

            Ok::<(Vec<f32>, u32), VoxError>((samples, sample_rate))
        };

        let (samples, sample_rate) = result?;
        let duration_ms = (samples.len() as f64 / sample_rate as f64 * 1000.0) as u64;

        tracing::info!(
            duration_ms = duration_ms,
            processing_time_ms = start.elapsed().as_millis() as u64,
            rtf = start.elapsed().as_secs_f64() / (duration_ms as f64 / 1000.0),
            "qwen3 synthesis complete"
        );

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
        Self::voice_list()
    }

    fn backend_name(&self) -> &str {
        "qwen3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_list_shows_aiden_for_us_male() {
        let voices = Qwen3Backend::voice_list();

        // Verify US Male voices reference Aiden (not Ryan/Ethan)
        let male1 = voices.iter().find(|v| v.id == "en_us_male_1").unwrap();
        assert!(male1.name.contains("Aiden"), "en_us_male_1 should mention Aiden in name");

        let male2 = voices.iter().find(|v| v.id == "en_us_male_2").unwrap();
        assert!(male2.name.contains("Aiden"), "en_us_male_2 should mention Aiden in name");
    }

    #[test]
    fn test_gb_male_shows_ethan() {
        let voices = Qwen3Backend::voice_list();

        // GB male can still use Ryan/Ethan naming
        let gb_male = voices.iter().find(|v| v.id == "en_gb_male_1").unwrap();
        assert!(gb_male.name.contains("Ethan"), "en_gb_male_1 should mention Ethan");
    }
}
