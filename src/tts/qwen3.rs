//! Qwen3-TTS backend integration.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use qwen3_tts::{Language, Qwen3TTS, Speaker, SynthesisOptions, auto_device};
use tokio::sync::Mutex;

use super::normalize;
use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{AudioChunk, TtsOutput, TtsRequest, VoiceInfo};

/// The HuggingFace hub cache: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
fn hub_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".cache").join("huggingface"))
        .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
        .join("hub")
}

fn is_model_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file()
}

/// Find the directory holding the files of `model_id` (`Org/Repo`).
///
/// Tried in order:
///
/// 1. `override_dir`, when given. A wrong override is an error, not a reason
///    to load some other copy.
/// 2. The hub cache entry `models--Org--Repo`: the snapshot `refs/main`
///    names, else the only snapshot present. The hub stores snapshots under
///    their commit hash, so there is no `snapshots/main` to look for.
/// 3. `hub/Org/Repo`, a plain directory made with `--local-dir`.
fn resolve_model_dir(
    hub: &Path,
    model_id: &str,
    override_dir: Option<&Path>,
) -> Result<PathBuf, VoxError> {
    if let Some(dir) = override_dir {
        return if is_model_dir(dir) {
            Ok(dir.to_path_buf())
        } else {
            Err(VoxError::Tts(format!(
                "VOX_QWEN3_MODEL_PATH is set to {}, which has no config.json",
                dir.display()
            )))
        };
    }

    let repo = hub.join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots = repo.join("snapshots");

    if let Ok(commit) = std::fs::read_to_string(repo.join("refs").join("main")) {
        let commit = commit.trim();
        // A ref is a bare commit hash; anything else is not a path to follow.
        if !commit.is_empty() && commit.chars().all(|c| c.is_ascii_alphanumeric()) {
            let dir = snapshots.join(commit);
            if is_model_dir(&dir) {
                return Ok(dir);
            }
        }
    }

    let mut found: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|path| is_model_dir(path))
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    match found.len() {
        0 => {}
        1 => return Ok(found.remove(0)),
        n => {
            return Err(VoxError::Tts(format!(
                "{n} snapshots of {model_id} in {} and no usable refs/main to choose one; \
                 set VOX_QWEN3_MODEL_PATH to the snapshot to load",
                snapshots.display()
            )));
        }
    }

    let flat = hub.join(model_id);
    if is_model_dir(&flat) {
        return Ok(flat);
    }

    Err(VoxError::Tts(format!(
        "qwen3 model {model_id} not found under {}. Download it with:\n  \
         huggingface-cli download {model_id}\n\
         or point VOX_QWEN3_MODEL_PATH at a directory that holds its config.json",
        hub.display()
    )))
}

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

        let override_dir = std::env::var_os("VOX_QWEN3_MODEL_PATH")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let model_path = resolve_model_dir(&hub_cache_dir(), &model_id, override_dir.as_deref())?;

        // Convert to string for API
        let model_path_str = model_path.to_string_lossy().to_string();

        // Load the model from the resolved local directory.
        let model = tokio::task::spawn_blocking(move || -> Result<Qwen3TTS, VoxError> {
            tracing::info!(path = %model_path_str, "loading from local cache");

            // Select device
            let device = auto_device()
                .map_err(|e| VoxError::Tts(format!("failed to select device: {e}")))?;

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

        let text = normalize::clean(&request.text);
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
        let output = TtsOutput::from_mono(samples, sample_rate);

        tracing::info!(
            duration_ms = output.duration_ms,
            processing_time_ms = start.elapsed().as_millis() as u64,
            rtf = real_time_factor(start.elapsed(), output.duration_ms),
            "qwen3 streaming synthesis complete"
        );

        Ok(output)
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

        let text = normalize::clean(&request.text);
        let temperature = self.config.temperature;
        let seed = request.seed;
        let model = Arc::clone(&self.model);

        // Inference is synchronous and takes seconds. Run it on the blocking
        // pool, and take the model lock there too, so neither the work nor
        // the wait for the lock ties up an async worker.
        let start = std::time::Instant::now();
        let (samples, sample_rate) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32), VoxError> {
                let model_guard = model.blocking_lock();

                tracing::debug!(
                    text_len = text.len(),
                    speaker = ?speaker,
                    language = ?language,
                    "starting qwen3 synthesis"
                );

                let options = SynthesisOptions {
                    temperature: temperature as f64,
                    seed,
                    ..Default::default()
                };

                let audio = model_guard
                    .synthesize_with_voice(&text, speaker, language, Some(options))
                    .map_err(|e| VoxError::Tts(format!("synthesis failed: {e}")))?;

                Ok((audio.samples, audio.sample_rate))
            })
            .await
            .map_err(|e| VoxError::Tts(format!("qwen3 synthesis task panicked: {e}")))??;

        let output = TtsOutput::from_mono(samples, sample_rate);

        tracing::info!(
            duration_ms = output.duration_ms,
            processing_time_ms = start.elapsed().as_millis() as u64,
            rtf = real_time_factor(start.elapsed(), output.duration_ms),
            "qwen3 synthesis complete"
        );

        Ok(output)
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        Self::voice_list()
    }

    fn backend_name(&self) -> &str {
        "qwen3"
    }
}

/// Processing time over audio time; 0 when there is no audio.
fn real_time_factor(elapsed: std::time::Duration, duration_ms: u64) -> f64 {
    if duration_ms == 0 {
        return 0.0;
    }
    elapsed.as_secs_f64() / (duration_ms as f64 / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    const MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
    const REPO_DIR: &str = "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice";

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempHub(PathBuf);

    impl TempHub {
        fn new() -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "vox-qwen3-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        /// Create `relative` with a `config.json` in it.
        fn model_dir(&self, relative: &str) -> PathBuf {
            let dir = self.0.join(relative);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), "{}").unwrap();
            dir
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
    }

    impl Drop for TempHub {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolves_the_snapshot_refs_main_points_to() {
        let hub = TempHub::new();
        let old = hub.model_dir(&format!("{REPO_DIR}/snapshots/aaa111"));
        let current = hub.model_dir(&format!("{REPO_DIR}/snapshots/bbb222"));
        // Trailing newline, as `git`-style ref files have.
        hub.write(&format!("{REPO_DIR}/refs/main"), "bbb222\n");

        let resolved = resolve_model_dir(&hub.0, MODEL_ID, None).unwrap();
        assert_eq!(resolved, current);
        assert_ne!(resolved, old);
    }

    #[test]
    fn resolves_a_lone_snapshot_without_refs() {
        let hub = TempHub::new();
        let only = hub.model_dir(&format!("{REPO_DIR}/snapshots/0123abcd"));
        assert_eq!(resolve_model_dir(&hub.0, MODEL_ID, None).unwrap(), only);
    }

    #[test]
    fn a_stale_ref_falls_back_to_the_lone_snapshot() {
        let hub = TempHub::new();
        let only = hub.model_dir(&format!("{REPO_DIR}/snapshots/0123abcd"));
        hub.write(&format!("{REPO_DIR}/refs/main"), "deadbeef");
        assert_eq!(resolve_model_dir(&hub.0, MODEL_ID, None).unwrap(), only);
    }

    #[test]
    fn incomplete_snapshots_are_ignored() {
        let hub = TempHub::new();
        let complete = hub.model_dir(&format!("{REPO_DIR}/snapshots/complete"));
        // An interrupted download: the directory exists, config.json does not.
        std::fs::create_dir_all(hub.0.join(format!("{REPO_DIR}/snapshots/partial"))).unwrap();
        assert_eq!(resolve_model_dir(&hub.0, MODEL_ID, None).unwrap(), complete);
    }

    #[test]
    fn several_snapshots_without_a_ref_are_ambiguous() {
        let hub = TempHub::new();
        hub.model_dir(&format!("{REPO_DIR}/snapshots/aaa111"));
        hub.model_dir(&format!("{REPO_DIR}/snapshots/bbb222"));
        let err = resolve_model_dir(&hub.0, MODEL_ID, None).unwrap_err();
        assert!(err.to_string().contains("2 snapshots"), "{err}");
    }

    #[test]
    fn a_ref_cannot_point_outside_the_snapshots_directory() {
        let hub = TempHub::new();
        hub.model_dir("elsewhere");
        hub.write(&format!("{REPO_DIR}/refs/main"), "../../elsewhere");
        assert!(resolve_model_dir(&hub.0, MODEL_ID, None).is_err());
    }

    #[test]
    fn resolves_a_flat_local_dir() {
        let hub = TempHub::new();
        let flat = hub.model_dir(MODEL_ID);
        assert_eq!(resolve_model_dir(&hub.0, MODEL_ID, None).unwrap(), flat);
    }

    #[test]
    fn the_override_wins_and_must_be_valid() {
        let hub = TempHub::new();
        hub.model_dir(&format!("{REPO_DIR}/snapshots/0123abcd"));
        let custom = hub.model_dir("custom");
        assert_eq!(
            resolve_model_dir(&hub.0, MODEL_ID, Some(&custom)).unwrap(),
            custom
        );

        let missing = hub.0.join("missing");
        let err = resolve_model_dir(&hub.0, MODEL_ID, Some(&missing)).unwrap_err();
        assert!(err.to_string().contains("VOX_QWEN3_MODEL_PATH"), "{err}");
    }

    #[test]
    fn a_missing_model_says_how_to_get_it() {
        let hub = TempHub::new();
        let err = resolve_model_dir(&hub.0, MODEL_ID, None).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&format!("huggingface-cli download {MODEL_ID}")),
            "{message}"
        );
    }

    #[test]
    fn real_time_factor_handles_empty_audio() {
        let second = std::time::Duration::from_secs(1);
        assert_eq!(real_time_factor(second, 0), 0.0);
        assert_eq!(real_time_factor(second, 2000), 0.5);
    }

    #[test]
    fn test_voice_list_shows_aiden_for_us_male() {
        let voices = Qwen3Backend::voice_list();

        // Verify US Male voices reference Aiden (not Ryan/Ethan)
        let male1 = voices.iter().find(|v| v.id == "en_us_male_1").unwrap();
        assert!(
            male1.name.contains("Aiden"),
            "en_us_male_1 should mention Aiden in name"
        );

        let male2 = voices.iter().find(|v| v.id == "en_us_male_2").unwrap();
        assert!(
            male2.name.contains("Aiden"),
            "en_us_male_2 should mention Aiden in name"
        );
    }

    #[test]
    fn test_gb_male_shows_ethan() {
        let voices = Qwen3Backend::voice_list();

        // GB male can still use Ryan/Ethan naming
        let gb_male = voices.iter().find(|v| v.id == "en_gb_male_1").unwrap();
        assert!(
            gb_male.name.contains("Ethan"),
            "en_gb_male_1 should mention Ethan"
        );
    }
}
