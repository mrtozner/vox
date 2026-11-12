//! Shared server state and request counters.

use std::sync::Arc;

/// Shared server state accessible from all handlers.
///
/// `stt` is the primary batch STT backend (Whisper or offline Sherpa) used for
/// `/v1/transcribe` and as the fallback in WebSocket streaming. `streaming_stt`
/// is the optional incremental backend (online Sherpa) used *only* for the
/// WebSocket `/v1/listen` session path. They must be separate instances —
/// do not load the same backend as both.
///
/// `diarization` and `speaker_db` are loaded together from `speaker_encoder.onnx`.
/// Both are `None` if the model is missing or fails to load; `/v1/listen` then
/// falls back to speaker-less transcription with no behavioral change. This is
/// intentional — diarization is an opt-in biometric feature.
pub struct ServerState {
    pub stt: Option<Arc<dyn vox::traits::SttBackend>>,
    pub tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub conversation_tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>>,
    pub vad_model_path: Option<std::path::PathBuf>,
    pub stats: Arc<std::sync::Mutex<ServerStats>>,
    pub start_time: std::time::Instant,
    pub ollama_host: String,
    pub http_client: reqwest::Client,
    pub stt_model_name: Option<String>,
    pub stt_model_size: Option<u64>,
    pub tts_model_name: Option<String>,
    pub tts_model_size: Option<u64>,
    pub capabilities: Arc<vox::CapabilityRegistry>,
    /// Live Talk conversation memory by session id, kept across reconnects.
    pub sessions: Arc<vox::conversation::SessionStore>,
    #[cfg(feature = "diarization")]
    pub diarization: Option<Arc<vox::DiarizationPipeline>>,
    #[cfg(feature = "diarization")]
    pub speaker_db: Option<Arc<vox::SpeakerDatabase>>,
}

/// Cumulative request counters.
pub struct ServerStats {
    pub requests: u64,
    pub transcriptions: u64,
    pub syntheses: u64,
}
