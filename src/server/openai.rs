//! OpenAI-compatible endpoints, so existing OpenAI SDKs and tools can point
//! their `base_url` at a local vox server.
//!
//! | Endpoint | OpenAI equivalent |
//! |---|---|
//! | `POST /v1/audio/transcriptions` | `audio.transcriptions.create` |
//! | `POST /v1/audio/speech` | `audio.speech.create` |
//! | `GET /v1/models`, `GET /v1/models/{id}` | `models.list`, `models.retrieve` |
//!
//! Known differences from OpenAI, all reported as `400` with a clear message
//! rather than silently approximated:
//!
//! - Transcription uploads must be WAV (any rate / channels / bit depth).
//! - `response_format` for transcriptions: `json`, `text`, `verbose_json`
//!   (no `srt` / `vtt`; `verbose_json` has no segments).
//! - `response_format` for speech: `wav` (the default here; OpenAI defaults to
//!   `mp3`) and `pcm` (24 kHz 16-bit mono, as OpenAI). No `mp3`, `opus`,
//!   `aac`, `flac`.
//! - `speed` other than `1.0` is rejected.
//! - `language` and `prompt` are accepted and ignored: the STT language is
//!   chosen when the server starts (`vox serve --language`).
//! - No authentication: an `Authorization` header is accepted and ignored.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{Multipart, Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};

use crate::llm::LlmEndpoint;
use crate::types::{AudioChunk, TtsRequest};

use super::ServerState;
use super::error::ServerError;
use super::handlers::{
    LlmStatus, audio_response, encode_pcm16, encode_wav_pcm16, run_synthesis, run_transcription,
    vox_models_for_llm, vox_models_response,
};
use super::models::ModelsResponse;

type AppState = Arc<ServerState>;

/// Model id OpenAI clients send for transcription; always available as an alias.
pub const STT_ALIAS: &str = "whisper-1";
/// Model id OpenAI clients send for speech; always available as an alias.
pub const TTS_ALIAS: &str = "tts-1";
/// Sample rate of OpenAI's raw `pcm` speech format.
pub const OPENAI_PCM_SAMPLE_RATE: u32 = 24_000;

/// Voice names of the OpenAI speech API. Stock SDK snippets send one of
/// these; none exists locally, so they select the backend's default voice.
const OPENAI_VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse",
];

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// One entry of the OpenAI model list.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    /// Always `model`.
    pub object: &'static str,
    /// Unix time the server started.
    pub created: u64,
    /// Always `vox`.
    pub owned_by: &'static str,
}

/// `GET /v1/models` response: the OpenAI list, plus vox's pre-0.7 keys
/// (`stt`, `tts`, `streaming_stt`, `ollama`) for one release.
#[derive(Debug, Serialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: &'static str,
    pub data: Vec<ModelObject>,
    /// Deprecated: moved to `GET /v1/vox/models`.
    #[serde(flatten)]
    pub legacy: ModelsResponse,
}

fn model_objects(state: &ServerState) -> Vec<ModelObject> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_sub(state.start_time.elapsed().as_secs());

    let mut ids: Vec<String> = Vec::new();
    let mut push = |id: Option<String>| {
        if let Some(id) = id.filter(|id| !id.is_empty())
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    };

    if state.stt.is_some() {
        push(Some(STT_ALIAS.to_string()));
        push(state.stt_model_name.clone());
    }
    if let Some(tts) = state.tts.as_ref().or(state.conversation_tts.as_ref()) {
        push(Some(TTS_ALIAS.to_string()));
        push(Some(tts.backend_name().to_string()));
    }
    if state.tts.is_some() {
        push(state.tts_model_name.clone());
    }
    if let Some(conv) = state.conversation_tts.as_ref() {
        push(Some(conv.backend_name().to_string()));
    }

    ids.into_iter()
        .map(|id| ModelObject {
            id,
            object: "model",
            created,
            owned_by: "vox",
        })
        .collect()
}

/// GET /v1/models — OpenAI-shaped model list.
///
/// The response also carries vox's old top-level keys so pre-0.7 clients keep
/// working for one release; new code should read `GET /v1/vox/models`.
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list",
        data: model_objects(&state),
        legacy: vox_models_response(&state).await,
    })
}

pub(super) async fn list_models_with_llm(
    State(state): State<AppState>,
    Extension(endpoint): Extension<Arc<LlmEndpoint>>,
) -> impl IntoResponse {
    #[derive(Serialize)]
    struct Response {
        #[serde(flatten)]
        models: ModelList,
        llm: LlmStatus,
    }
    let (legacy, llm) = vox_models_for_llm(&state, &endpoint).await;
    Json(Response {
        models: ModelList {
            object: "list",
            data: model_objects(&state),
            legacy,
        },
        llm,
    })
}

/// GET /v1/models/{id} — OpenAI-shaped single model.
pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ServerError> {
    model_objects(&state)
        .into_iter()
        .find(|m| m.id == id)
        .map(Json)
        .ok_or_else(|| {
            ServerError::not_found(format!("The model '{id}' does not exist"))
                .with_code("model_not_found")
                .with_param("model")
        })
}

// ---------------------------------------------------------------------------
// Transcriptions
// ---------------------------------------------------------------------------

/// `response_format: json` body.
#[derive(Debug, Serialize)]
pub struct TranscriptionJson {
    pub text: String,
}

/// `response_format: verbose_json` body.
#[derive(Debug, Serialize)]
pub struct TranscriptionVerboseJson {
    /// Always `transcribe`.
    pub task: &'static str,
    /// Detected or configured language, `unknown` when the backend does not
    /// report one.
    pub language: String,
    /// Audio duration in seconds.
    pub duration: f64,
    pub text: String,
    /// Always empty: the STT backends do not expose segment timestamps yet.
    pub segments: Vec<serde_json::Value>,
}

fn multipart_error(e: axum::extract::multipart::MultipartError) -> ServerError {
    ServerError::new(
        e.status(),
        format!("invalid multipart body: {}", e.body_text()),
    )
    .with_code("invalid_multipart")
}

/// POST /v1/audio/transcriptions — multipart upload, OpenAI request/response shape.
pub async fn transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ServerError> {
    let mut file: Option<Bytes> = None;
    let mut response_format = "json".to_string();
    let mut stream = false;

    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => file = Some(field.bytes().await.map_err(multipart_error)?),
            "response_format" => {
                response_format = field
                    .text()
                    .await
                    .map_err(multipart_error)?
                    .trim()
                    .to_ascii_lowercase();
            }
            "stream" => {
                let value = field.text().await.map_err(multipart_error)?;
                stream = matches!(value.trim().to_ascii_lowercase().as_str(), "true" | "1");
            }
            // model, language, prompt, temperature, timestamp_granularities[],
            // ...: accepted for compatibility, not used. Drain and move on.
            _ => {
                field.bytes().await.map_err(multipart_error)?;
            }
        }
    }

    if !matches!(response_format.as_str(), "json" | "text" | "verbose_json") {
        return Err(ServerError::bad_request(format!(
            "response_format '{response_format}' is not supported (expected json, text or verbose_json)"
        ))
        .with_code("unsupported_format")
        .with_param("response_format"));
    }
    if stream {
        return Err(
            ServerError::bad_request("streaming transcriptions are not supported")
                .with_code("unsupported_parameter")
                .with_param("stream"),
        );
    }
    let file = file.ok_or_else(|| {
        ServerError::bad_request("missing 'file' field: upload the audio as multipart/form-data")
            .with_code("missing_file")
            .with_param("file")
    })?;
    if !file.starts_with(b"RIFF") {
        return Err(ServerError::bad_request(
            "unsupported audio format: vox accepts WAV uploads only. Convert first, e.g. \
             `ffmpeg -i input.mp3 output.wav`.",
        )
        .with_code("unsupported_audio_format")
        .with_param("file"));
    }

    let result = run_transcription(&state, file)
        .await
        .map_err(|e| e.with_param_if_unset("file"))?;

    Ok(match response_format.as_str() {
        "text" => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            format!("{}\n", result.text),
        )
            .into_response(),
        "verbose_json" => Json(TranscriptionVerboseJson {
            task: "transcribe",
            language: result.language.unwrap_or_else(|| "unknown".to_string()),
            duration: result.duration_ms as f64 / 1000.0,
            text: result.text,
            segments: Vec::new(),
        })
        .into_response(),
        _ => Json(TranscriptionJson { text: result.text }).into_response(),
    })
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// `POST /v1/audio/speech` request body.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Model id. `tts-1` (or anything unknown) selects the main TTS engine;
    /// a backend name such as `piper` selects that engine when it is loaded.
    #[serde(default)]
    pub model: Option<String>,
    /// Text to speak.
    pub input: String,
    /// A vox voice id (see `GET /v1/voices`). OpenAI voice names (`alloy`,
    /// ...) select the default voice.
    #[serde(default)]
    pub voice: Option<String>,
    /// `wav` (default) or `pcm`.
    #[serde(default)]
    pub response_format: Option<String>,
    /// Must be absent or `1.0`.
    #[serde(default)]
    pub speed: Option<f64>,
    /// `audio` (default). `sse` is not supported.
    #[serde(default)]
    pub stream_format: Option<String>,
}

/// Pick the TTS engine for an OpenAI `model` id.
fn select_tts<'a>(
    state: &'a ServerState,
    model: Option<&str>,
) -> Option<&'a Arc<dyn crate::traits::TtsBackend>> {
    if let (Some(model), Some(conv)) = (model, state.conversation_tts.as_ref())
        && conv.backend_name().eq_ignore_ascii_case(model)
    {
        return Some(conv);
    }
    state.tts.as_ref().or(state.conversation_tts.as_ref())
}

/// Map the requested voice onto one the backend has.
fn resolve_voice(
    tts: &Arc<dyn crate::traits::TtsBackend>,
    voice: Option<String>,
) -> Option<String> {
    let voice = voice
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())?;
    if tts.list_voices().iter().any(|v| v.id == voice) {
        return Some(voice);
    }
    if OPENAI_VOICES.contains(&voice.to_ascii_lowercase().as_str()) {
        // An OpenAI voice name from a stock snippet: use the default voice.
        return None;
    }
    // Not listed, but backends accept more than they list (speaker numbers,
    // language aliases); let the backend decide.
    Some(voice)
}

/// POST /v1/audio/speech — JSON in, audio out, OpenAI request shape.
pub async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response, ServerError> {
    let format = req
        .response_format
        .as_deref()
        .map(|f| f.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "wav".to_string());
    if !matches!(format.as_str(), "wav" | "pcm") {
        return Err(ServerError::bad_request(format!(
            "response_format '{format}' is not supported: vox produces 'wav' (default) and 'pcm'"
        ))
        .with_code("unsupported_format")
        .with_param("response_format"));
    }
    if let Some(speed) = req.speed
        && (speed - 1.0).abs() > 1e-6
    {
        return Err(
            ServerError::bad_request("speed is not supported yet: omit it or send 1.0")
                .with_code("unsupported_parameter")
                .with_param("speed"),
        );
    }
    if req
        .stream_format
        .as_deref()
        .is_some_and(|f| !f.eq_ignore_ascii_case("audio"))
    {
        return Err(
            ServerError::bad_request("stream_format other than 'audio' is not supported")
                .with_code("unsupported_parameter")
                .with_param("stream_format"),
        );
    }

    let tts = select_tts(&state, req.model.as_deref()).ok_or_else(|| {
        ServerError::service_unavailable("TTS backend not loaded").with_code("tts_unavailable")
    })?;

    let request = TtsRequest {
        text: req.input,
        voice: resolve_voice(tts, req.voice),
        seed: None,
    };
    let output = run_synthesis(&state, tts, request, "input").await?;

    if format == "pcm" {
        // OpenAI's pcm is 24 kHz 16-bit mono; clients hard-code that.
        let audio = to_openai_pcm(output.audio).await?;
        let bytes = encode_pcm16(&audio.samples);
        return Ok(audio_response("audio/pcm", &audio, bytes));
    }
    let bytes = encode_wav_pcm16(&output.audio)?;
    Ok(audio_response("audio/wav", &output.audio, bytes))
}

/// Convert to 24 kHz mono.
async fn to_openai_pcm(audio: AudioChunk) -> Result<AudioChunk, ServerError> {
    if audio.sample_rate == OPENAI_PCM_SAMPLE_RATE && audio.channels <= 1 {
        return Ok(audio);
    }
    tokio::task::spawn_blocking(move || crate::audio::resample_once(&audio, OPENAI_PCM_SAMPLE_RATE))
        .await
        .map_err(|e| ServerError::internal(format!("resampling task failed: {e}")))?
        .map_err(|e| ServerError::internal(format!("resampling failed: {e}")))
}
