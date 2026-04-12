//! Model inventory for the capability registry.
//!
//! This module captures which STT/TTS/VAD backends are currently loaded
//! and serializes them into a uniform [`LoadedModel`] shape. The server
//! already tracks the essential fields (backend name, model name, size
//! on disk); this module just wraps them into a single inventory so the
//! registry can expose them consistently.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct LoadedModel {
    pub backend: String,
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ModelInventory {
    pub stt: Option<LoadedModel>,
    pub streaming_stt: Option<LoadedModel>,
    pub tts: Option<LoadedModel>,
    pub vad: Option<LoadedModel>,
}

impl ModelInventory {
    /// Build from the tuple-form arguments the server already tracks.
    pub fn from_parts(
        stt: Option<(&str, &str, Option<u64>)>,
        streaming_stt: Option<(&str, &str, Option<u64>)>,
        tts: Option<(&str, &str, Option<u64>)>,
        vad: Option<&std::path::Path>,
    ) -> Self {
        Self {
            stt: stt.map(|(b, m, s)| LoadedModel {
                backend: b.to_string(),
                model_name: m.to_string(),
                size_mb: s,
                sample_rate: Some(16000),
                language: None,
            }),
            streaming_stt: streaming_stt.map(|(b, m, s)| LoadedModel {
                backend: b.to_string(),
                model_name: m.to_string(),
                size_mb: s,
                sample_rate: Some(16000),
                language: None,
            }),
            tts: tts.map(|(b, m, s)| LoadedModel {
                backend: b.to_string(),
                model_name: m.to_string(),
                size_mb: s,
                sample_rate: None,
                language: None,
            }),
            vad: vad.map(|path| {
                let size_mb = std::fs::metadata(path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                LoadedModel {
                    backend: "silero".to_string(),
                    model_name: path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("silero_vad")
                        .to_string(),
                    size_mb,
                    sample_rate: Some(16000),
                    language: None,
                }
            }),
        }
    }
}
