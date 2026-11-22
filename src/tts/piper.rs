//! Piper TTS backend for multilingual speech synthesis.
//!
//! Uses piper-rs (VITS-based ONNX models) to support 35+ languages
//! with hundreds of pre-trained voices. Models are lightweight
//! (15-100MB each) and run entirely locally via ONNX Runtime.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use piper_rs::synth::PiperSpeechSynthesizer;

use super::normalize;
use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{TtsOutput, TtsRequest, VoiceInfo};

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
    /// Guards the model's speaker setting together with synthesis: piper-rs
    /// keeps the speaker on the model, so selecting one and synthesizing
    /// with it has to be a single critical section.
    synth: Arc<Mutex<PiperSpeechSynthesizer>>,
    model: Arc<dyn piper_rs::PiperModel + Send + Sync>,
    #[allow(dead_code)]
    config: PiperConfig,
    sample_rate: u32,
    language: String,
    /// Speaker id to name; empty for single-speaker models.
    speakers: HashMap<i64, String>,
    /// Speaker used when a request names none.
    default_speaker: Option<i64>,
}

/// Decide which speaker a request selects.
///
/// Returns `None` for single-speaker models, which take no speaker id.
/// A request without a voice gets `default`, never whatever an earlier
/// request chose.
fn resolve_speaker(
    voice: Option<&str>,
    speakers: &HashMap<i64, String>,
    default: Option<i64>,
) -> Result<Option<i64>, VoxError> {
    if speakers.is_empty() {
        return Ok(None);
    }
    let fallback = default
        .filter(|sid| speakers.contains_key(sid))
        .or_else(|| speakers.keys().copied().min());

    let Some(voice) = voice.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(fallback);
    };
    if voice.eq_ignore_ascii_case("default") {
        return Ok(fallback);
    }
    if let Ok(sid) = voice.parse::<i64>() {
        return if speakers.contains_key(&sid) {
            Ok(Some(sid))
        } else {
            Err(VoxError::Tts(format!(
                "piper model has no speaker {sid} ({} speakers)",
                speakers.len()
            )))
        };
    }
    let by_name = speakers
        .iter()
        .filter(|(_, name)| name.eq_ignore_ascii_case(voice))
        .map(|(&sid, _)| sid)
        .min();
    if by_name.is_none() {
        // Callers pass voice ids of other backends here (the UI shares one
        // voice picker), so an unknown name is not an error.
        tracing::debug!(voice = %voice, "voice is not a piper speaker; using the default");
    }
    Ok(by_name.or(fallback))
}

impl PiperBackend {
    /// Check the local eSpeak NG installation using an English Piper voice.
    ///
    /// This does not require an ONNX model. Loading a model also checks the
    /// particular voice configured for it.
    pub fn check_phonemizer() -> Result<(), VoxError> {
        piper_rs::check_phonemizer("en-us").map_err(|e| VoxError::Tts(e.to_string()))
    }

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
        if let Some(sid) = config.default_speaker
            && let Some(err) = model.set_speaker(sid)
        {
            return Err(VoxError::Tts(format!(
                "failed to set default speaker {sid}: {err}"
            )));
        }

        // Set length_scale if specified. Start from the model's own settings:
        // a default-constructed config would zero both noise scales.
        if let Some(length_scale) = config.length_scale {
            let current = model
                .get_fallback_synthesis_config()
                .map_err(|e| VoxError::Tts(format!("failed to read synthesis config: {e}")))?;
            let mut synth_config = current
                .downcast_ref::<piper_rs::PiperSynthesisConfig>()
                .cloned()
                .ok_or_else(|| VoxError::Tts("unexpected piper synthesis config type".into()))?;
            synth_config.length_scale = length_scale;
            synth_config.speaker = config.default_speaker;
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

        let speakers = model
            .get_speakers()
            .ok()
            .flatten()
            .cloned()
            .unwrap_or_default();

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
            default_speaker: config.default_speaker,
            config,
            sample_rate,
            language,
            speakers,
        })
    }
}

#[async_trait]
impl TtsBackend for PiperBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        if request.seed.is_some() {
            tracing::debug!("piper backend does not support seed; ignoring");
        }

        // Piper's espeak-ng front end reads numbers and abbreviations itself;
        // it only needs markup and unspeakable characters removed.
        let text = normalize::clean(&request.text);
        if text.is_empty() {
            return Ok(TtsOutput::from_mono(Vec::new(), self.sample_rate));
        }

        let speaker = resolve_speaker(
            request.voice.as_deref(),
            &self.speakers,
            self.default_speaker,
        )?;

        let synth = self.synth.clone();
        let model = Arc::clone(&self.model);

        // Piper synthesis is synchronous and CPU-bound -- run on blocking thread pool.
        let samples = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, VoxError> {
            // Nothing behind the lock is left half-updated by a panic: the
            // speaker is set again below on every request.
            let synth = synth.lock().unwrap_or_else(|e| e.into_inner());

            // The speaker lives on the shared model. Setting it under the
            // lock, on every request, keeps one request's choice from
            // leaking into another.
            if let Some(sid) = speaker
                && let Some(err) = model.set_speaker(sid)
            {
                return Err(VoxError::Tts(format!(
                    "failed to select piper speaker {sid}: {err}"
                )));
            }

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

        Ok(TtsOutput::from_mono(samples, self.sample_rate))
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        if self.speakers.is_empty() {
            // Single-speaker model
            return vec![VoiceInfo {
                id: "default".to_string(),
                name: "Default".to_string(),
                gender: "neutral".to_string(),
                language: self.language.clone(),
                accent: "Piper".to_string(),
            }];
        }
        let mut ids: Vec<i64> = self.speakers.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| VoiceInfo {
                id: id.to_string(),
                name: self.speakers[&id].clone(),
                gender: "unknown".to_string(),
                language: self.language.clone(),
                accent: "Piper".to_string(),
            })
            .collect()
    }

    fn backend_name(&self) -> &str {
        "piper"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speakers(entries: &[(i64, &str)]) -> HashMap<i64, String> {
        entries
            .iter()
            .map(|&(id, name)| (id, name.to_string()))
            .collect()
    }

    #[test]
    fn single_speaker_models_take_no_speaker() {
        let none = HashMap::new();
        assert_eq!(resolve_speaker(None, &none, None).unwrap(), None);
        assert_eq!(resolve_speaker(Some("0"), &none, None).unwrap(), None);
        assert_eq!(
            resolve_speaker(Some("af_heart"), &none, Some(3)).unwrap(),
            None
        );
    }

    #[test]
    fn a_request_without_a_voice_gets_the_default_every_time() {
        let map = speakers(&[(0, "amy"), (1, "joe"), (2, "kim")]);
        // One request picks speaker 2 ...
        assert_eq!(resolve_speaker(Some("2"), &map, Some(1)).unwrap(), Some(2));
        // ... and the next, naming none, is back on the default.
        assert_eq!(resolve_speaker(None, &map, Some(1)).unwrap(), Some(1));
        assert_eq!(resolve_speaker(Some(""), &map, Some(1)).unwrap(), Some(1));
        assert_eq!(
            resolve_speaker(Some("default"), &map, Some(1)).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn default_falls_back_to_the_lowest_speaker_id() {
        let map = speakers(&[(4, "amy"), (7, "joe")]);
        assert_eq!(resolve_speaker(None, &map, None).unwrap(), Some(4));
        // A configured default the model does not have is ignored.
        assert_eq!(resolve_speaker(None, &map, Some(99)).unwrap(), Some(4));
    }

    #[test]
    fn speakers_resolve_by_id_and_by_name() {
        let map = speakers(&[(0, "amy"), (1, "joe")]);
        assert_eq!(resolve_speaker(Some("1"), &map, None).unwrap(), Some(1));
        assert_eq!(resolve_speaker(Some("joe"), &map, None).unwrap(), Some(1));
        assert_eq!(resolve_speaker(Some("JOE"), &map, None).unwrap(), Some(1));
        assert_eq!(
            resolve_speaker(Some(" amy "), &map, Some(1)).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn unknown_ids_fail_and_unknown_names_use_the_default() {
        let map = speakers(&[(0, "amy"), (1, "joe")]);
        let err = resolve_speaker(Some("5"), &map, None).unwrap_err();
        assert!(err.to_string().contains("no speaker 5"), "{err}");
        // A Kokoro voice id from the shared voice picker.
        assert_eq!(
            resolve_speaker(Some("af_heart"), &map, Some(1)).unwrap(),
            Some(1)
        );
    }
}
