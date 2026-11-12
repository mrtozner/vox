//! Request/response DTOs for the HTTP API.

use serde::{Deserialize, Serialize};

/// Response from the transcription endpoint.
#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration_ms: u64,
    pub processing_time_ms: u64,
}

/// Request body for the synthesis endpoint.
#[derive(Debug, Deserialize)]
pub struct SynthesizeRequest {
    pub text: String,
    pub voice: Option<String>,
    pub seed: Option<u64>,
    /// Audio container/encoding. `wav` (default): 16-bit PCM WAV, playable
    /// in every browser. `wav_f32`: 32-bit float WAV (the pre-0.7 default).
    /// `pcm`: headerless 16-bit little-endian samples.
    pub format: Option<String>,
}

/// Response listing available backends (`GET /v1/vox/models`; also merged
/// into `GET /v1/models` for one release).
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub stt: Option<BackendInfo>,
    pub tts: Option<BackendInfo>,
    pub streaming_stt: Option<BackendInfo>,
    pub ollama: Option<OllamaStatus>,
}

/// Ollama connectivity status.
#[derive(Debug, Serialize)]
pub struct OllamaStatus {
    pub connected: bool,
    pub host: String,
    pub model_count: usize,
}

/// Info about a loaded backend.
#[derive(Debug, Serialize)]
pub struct BackendInfo {
    pub name: String,
    pub loaded: bool,
    pub model: Option<String>,
    pub size_mb: Option<u64>,
}

/// Server statistics response.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    /// HTTP requests served, not counting `/v1/stats` itself.
    pub requests: u64,
    pub transcriptions: u64,
    pub syntheses: u64,
    pub uptime_secs: u64,
    /// Successful `/v1/chat` completions.
    pub chats: u64,
    /// Responses with a 4xx status.
    pub client_errors: u64,
    /// Responses with a 5xx status.
    pub server_errors: u64,
    /// Latency counters (cumulative totals and derived averages).
    pub latency: LatencyStats,
    /// WebSocket sessions currently open / opened since start.
    pub ws_active: u64,
    pub ws_total: u64,
    /// Inference slots: configured maximum and how many are free right now.
    pub inference_slots: usize,
    pub inference_slots_free: usize,
}

/// Cumulative latency counters reported by `/v1/stats`.
#[derive(Debug, Serialize)]
pub struct LatencyStats {
    /// Total audio transcribed / time spent transcribing it.
    pub transcribe_audio_ms: u64,
    pub transcribe_ms: u64,
    pub transcribe_avg_ms: Option<u64>,
    /// Processing time divided by audio duration (below 1.0 is faster than real time).
    pub transcribe_rtf: Option<f64>,
    /// Total audio synthesized / time spent synthesizing it.
    pub synthesize_audio_ms: u64,
    pub synthesize_ms: u64,
    pub synthesize_avg_ms: Option<u64>,
    pub synthesize_rtf: Option<f64>,
    /// Time spent waiting on the LLM for `/v1/chat`.
    pub chat_ms: u64,
    pub chat_avg_ms: Option<u64>,
}

/// Health check response (`GET /health`): the process is up.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    /// vox version (`CARGO_PKG_VERSION`).
    pub version: String,
}

/// Readiness response (`GET /health/ready`): 200 when the speech backends
/// are loaded, 503 otherwise.
#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    /// `ready` or `not_ready`.
    pub status: String,
    pub version: String,
    pub stt: bool,
    pub tts: bool,
    pub vad: bool,
    /// Names of the required backends that are not loaded.
    pub missing: Vec<String>,
}

/// Request body for the chat endpoint (Ollama proxy).
///
/// The Ollama address is fixed by the operator (`vox serve --ollama-host`).
/// A `host` field sent by older clients is ignored: letting a request pick
/// the upstream turned the server into an open proxy.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub text: String,
    pub model: Option<String>,
}

/// Response from the chat endpoint.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub model: String,
}

/// Response listing available TTS voices.
#[derive(Debug, Serialize)]
pub struct VoicesResponse {
    /// Voices of the main TTS engine (`/v1/synthesize`, `/v1/speak`).
    pub voices: Vec<VoiceInfo>,
    /// Backend name of the main TTS engine, `null` when none is loaded.
    pub engine: Option<String>,
    /// Backend that speaks in Live Talk / Converse. It can differ from
    /// `engine` (Piper for conversation, Kokoro for the Speak tab), in which
    /// case `voices` does not apply to conversations.
    pub conversation_engine: Option<String>,
    /// Voices of the conversation engine.
    pub conversation_voices: Vec<VoiceInfo>,
}

/// Info about a TTS voice.
#[derive(Debug, Serialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: String,
    pub language: String,
    pub accent: String,
    pub backend: String,
}

/// Response listing available Ollama models.
#[derive(Debug, Serialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModelInfo>,
}

/// Info about an available Ollama model.
#[derive(Debug, Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub size: Option<u64>,
}

/// Request from client on the `/v1/speak` WebSocket.
#[derive(Debug, Deserialize)]
pub struct SpeakWsRequest {
    pub text: String,
    pub voice: Option<String>,
}

/// JSON events sent from server to client on the `/v1/speak` WebSocket.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum TtsWsEvent {
    #[serde(rename = "tts_start")]
    Start {},
    #[serde(rename = "tts_progress")]
    Progress { chunk: usize, progress: f32 },
    #[serde(rename = "tts_done")]
    Done { chunks: usize },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Optional configuration sent by the client on the first JSON text message
/// of the `/v1/converse` WebSocket. All fields are optional; server defaults
/// are used when absent.
#[derive(Debug, Default, Deserialize)]
pub struct ConverseConfig {
    /// Ollama model name. Defaults to `llama3.2`.
    pub model: Option<String>,
    /// Ollama host (`host:port`). Defaults to the server's configured host.
    pub host: Option<String>,
    /// TTS voice id. Defaults to backend's default voice.
    pub voice: Option<String>,
}

/// JSON events sent from server to client on the `/v1/converse` WebSocket.
///
/// Wire protocol (server → client):
/// - `ready`          — handshake complete, server is listening for audio
/// - `speech_start`   — VAD detected speech begin
/// - `speech_end`     — VAD detected speech end, STT starting
/// - `transcript`     — Whisper STT result for the last utterance
/// - `thinking`       — LLM generation began (no text yet)
/// - `sentence`       — one complete sentence from the LLM plus its audio (base64 WAV)
/// - `done`           — LLM + TTS finished for this turn; client may speak again
/// - `error`          — fatal error for this turn; connection remains open
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ConverseWsEvent {
    #[serde(rename = "ready")]
    Ready {
        model: String,
        voice: Option<String>,
    },
    #[serde(rename = "speech_start")]
    SpeechStart,
    #[serde(rename = "speech_end")]
    SpeechEnd,
    #[serde(rename = "transcript")]
    Transcript {
        text: String,
        duration_ms: u64,
        processing_time_ms: u64,
    },
    #[serde(rename = "thinking")]
    Thinking,
    #[serde(rename = "sentence")]
    Sentence {
        index: usize,
        text: String,
        /// Base64-encoded WAV bytes (float32 mono).
        audio_b64: String,
        sample_rate: u32,
    },
    #[serde(rename = "done")]
    Done { sentences: usize },
    #[serde(rename = "error")]
    Error { message: String },
}

// ─── Live Talk (/v1/live-talk) ───────────────────────────────

/// JSON events sent from server to client on `/v1/live-talk`.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum LiveTalkWsEvent {
    #[serde(rename = "ready")]
    Ready {
        model: String,
        voice: Option<String>,
        mode: LiveTalkMode,
    },
    #[serde(rename = "speech_start")]
    SpeechStart,
    #[serde(rename = "partial")]
    Partial { text: String, stability: f32 },
    #[serde(rename = "transcript")]
    Transcript {
        text: String,
        duration_ms: u64,
        processing_time_ms: u64,
    },
    #[serde(rename = "thinking")]
    Thinking,
    #[serde(rename = "token")]
    Token { text: String },
    #[serde(rename = "sentence")]
    Sentence { index: usize, text: String },
    #[serde(rename = "audio_chunk")]
    AudioChunk {
        sentence_index: usize,
        sample_rate: u32,
    },
    #[serde(rename = "turn_done")]
    TurnDone { sentences: usize },
    #[serde(rename = "cancelled")]
    Cancelled { reason: CancelReason },
    #[serde(rename = "error")]
    Error { message: String, fatal: bool },
}

/// VAD or push-to-talk segmentation mode.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum LiveTalkMode {
    Vad,
    PushToTalk,
}

/// Why a turn was cancelled.
#[derive(Debug, serde::Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum CancelReason {
    UserBargeIn,
    ClientRequest,
    InternalError,
}

/// Client config for `/v1/live-talk` session.
#[derive(Debug, serde::Deserialize)]
pub struct LiveTalkConfig {
    pub model: Option<String>,
    pub host: Option<String>,
    pub voice: Option<String>,
    #[serde(default)]
    pub mode: LiveTalkModeConfig,
    #[serde(default = "default_true")]
    pub barge_in_enabled: bool,
    #[serde(default)]
    pub system_prompt_override: Option<String>,
}

/// Deserialization wrapper for `LiveTalkMode` with default.
#[derive(Debug, serde::Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum LiveTalkModeConfig {
    #[default]
    Vad,
    PushToTalk,
}

impl From<LiveTalkModeConfig> for LiveTalkMode {
    fn from(m: LiveTalkModeConfig) -> Self {
        match m {
            LiveTalkModeConfig::Vad => LiveTalkMode::Vad,
            LiveTalkModeConfig::PushToTalk => LiveTalkMode::PushToTalk,
        }
    }
}

/// Client commands for `/v1/live-talk`.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveTalkClientMsg {
    Config(LiveTalkConfig),
    Cancel,
    PttStart,
    PttEnd,
}

fn default_true() -> bool {
    true
}
