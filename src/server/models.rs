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

/// Configuration sent by the client as a JSON text message on the
/// `/v1/converse` WebSocket, as the first message or at any later time. All
/// fields are optional. A field that is present replaces the current value
/// and applies from the next turn on; a turn that is already running keeps
/// the values it started with.
///
/// The Ollama host is server configuration (`VOX_OLLAMA_HOST`) and cannot be
/// chosen by the client; a `host` field is ignored.
#[derive(Debug, Default, Deserialize)]
pub struct ConverseConfig {
    /// Ollama model name. Defaults to `llama3.2`.
    pub model: Option<String>,
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
///
/// The connection keeps a conversation memory for its lifetime, so later
/// turns see earlier ones.
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
        /// Base64-encoded WAV bytes (16-bit PCM mono).
        audio_b64: String,
        sample_rate: u32,
    },
    #[serde(rename = "done")]
    Done { sentences: usize },
    #[serde(rename = "error")]
    Error { message: String },
}

// ─── Live Talk (/v1/live-talk) ───────────────────────────────

/// Version of the Live Talk wire protocol, sent in `ready`.
pub const LIVE_TALK_PROTOCOL_VERSION: u32 = 2;

/// JSON events sent from server to client on `/v1/live-talk` (protocol v2).
///
/// Every event that belongs to a turn carries `turn_id`. Ids start at 1 and
/// grow by one per turn within a connection. A client must drop any event
/// (and the binary frame after an `audio_chunk`) whose `turn_id` is older
/// than the turn it is showing, or that belongs to a turn it cancelled.
///
/// - `ready`        — handshake done. `session_id` identifies the memory;
///   send it back in `config` after a reconnect to keep the conversation.
/// - `speech_start` / `speech_end` — the user started / stopped speaking.
///   After `speech_end` the server is transcribing.
/// - `transcript`   — what the user said. The speaker fields are present only
///   when the server identified the speaker.
/// - `thinking`     — the LLM request was sent.
/// - `token`        — raw LLM text as it arrives, for display only (may
///   contain markdown or reasoning text that is never spoken).
/// - `sentence`     — a speakable sentence; its audio follows.
/// - `audio_chunk`  — header for exactly one binary WAV frame, which is the
///   next binary message on the socket.
/// - `turn_metrics` — latencies of the turn; a stage that was not reached is
///   `null`.
/// - `turn_done`    — the turn is complete; all its audio has been sent.
/// - `cancelled`    — the turn was cut short. `turn_id` is 0 when a `cancel`
///   arrived before any turn existed.
/// - `memory`       — the whole conversation memory, sent after `ready`,
///   after every change to it and after `forget_memory`.
/// - `error`        — `fatal: false` leaves the session usable (the turn, if
///   any, is over and the microphone is accepted again).
#[derive(Debug, serde::Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum LiveTalkWsEvent {
    #[serde(rename = "ready")]
    Ready {
        model: String,
        voice: Option<String>,
        mode: LiveTalkMode,
        protocol_version: u32,
        session_id: String,
    },
    #[serde(rename = "speech_start")]
    SpeechStart,
    #[serde(rename = "speech_end")]
    SpeechEnd,
    #[serde(rename = "partial")]
    Partial { text: String, stability: f32 },
    #[serde(rename = "transcript")]
    Transcript {
        turn_id: u64,
        text: String,
        duration_ms: u64,
        processing_time_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_color: Option<String>,
    },
    #[serde(rename = "thinking")]
    Thinking { turn_id: u64 },
    #[serde(rename = "token")]
    Token { turn_id: u64, text: String },
    #[serde(rename = "sentence")]
    Sentence {
        turn_id: u64,
        index: usize,
        text: String,
    },
    #[serde(rename = "audio_chunk")]
    AudioChunk {
        turn_id: u64,
        sentence_index: usize,
        sample_rate: u32,
    },
    /// `stt_ms`: time spent transcribing. `llm_first_token_ms`: from sending
    /// the LLM request to its first token. `tts_first_audio_ms`: from the
    /// first sentence being ready to its audio being ready.
    #[serde(rename = "turn_metrics")]
    TurnMetrics {
        turn_id: u64,
        stt_ms: u64,
        llm_first_token_ms: Option<u64>,
        tts_first_audio_ms: Option<u64>,
    },
    #[serde(rename = "turn_done")]
    TurnDone { turn_id: u64, sentences: usize },
    #[serde(rename = "cancelled")]
    Cancelled { turn_id: u64, reason: CancelReason },
    #[serde(rename = "memory")]
    Memory(vox::conversation::ConversationSnapshot),
    #[serde(rename = "error")]
    Error {
        message: String,
        fatal: bool,
        /// The turn the error ended, if it happened inside one.
        #[serde(skip_serializing_if = "Option::is_none")]
        turn_id: Option<u64>,
    },
}

/// VAD or push-to-talk segmentation mode.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LiveTalkMode {
    Vad,
    PushToTalk,
}

/// Why a turn was cancelled.
#[derive(Debug, serde::Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum CancelReason {
    /// The user started speaking (server-side voice detection, or
    /// `ptt_start`) while the turn was running.
    UserBargeIn,
    /// The client sent `cancel`.
    ClientRequest,
    InternalError,
}

/// Client config for `/v1/live-talk` session.
///
/// The Ollama host is server configuration (`VOX_OLLAMA_HOST`) and cannot be
/// chosen by the client; a `host` field is ignored.
#[derive(Debug, serde::Deserialize)]
pub struct LiveTalkConfig {
    pub model: Option<String>,
    pub voice: Option<String>,
    #[serde(default)]
    pub mode: LiveTalkModeConfig,
    #[serde(default = "default_true")]
    pub barge_in_enabled: bool,
    #[serde(default)]
    pub system_prompt_override: Option<String>,
    /// Session to continue (from an earlier `ready`). Only read from the
    /// first message of a connection.
    #[serde(default)]
    pub session_id: Option<String>,
}

impl Default for LiveTalkConfig {
    fn default() -> Self {
        Self {
            model: None,
            voice: None,
            mode: LiveTalkModeConfig::default(),
            barge_in_enabled: true,
            system_prompt_override: None,
            session_id: None,
        }
    }
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

/// Client commands for `/v1/live-talk` (protocol v2).
///
/// - `config`        — see [`LiveTalkConfig`].
/// - `cancel`        — stop the turn. `turn_id` defaults to the current (or
///   last) turn. `heard_sentences` is the number of sentences whose playback
///   started in the browser; memory keeps only those. Works after
///   `turn_done` too, while the browser is still playing. The bare
///   `{"type":"cancel"}` of protocol v1 is still accepted.
/// - `playback_done` — the browser finished playing the turn; the server
///   listens to the microphone again.
/// - `forget_memory` — clear the conversation memory.
/// - `ptt_start` / `ptt_end` — push-to-talk key down / up.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveTalkClientMsg {
    Config(LiveTalkConfig),
    Cancel {
        #[serde(default)]
        turn_id: Option<u64>,
        #[serde(default)]
        heard_sentences: Option<usize>,
    },
    PlaybackDone {
        turn_id: u64,
        #[serde(default)]
        heard_sentences: Option<usize>,
    },
    ForgetMemory,
    PttStart,
    PttEnd,
}

fn default_true() -> bool {
    true
}
