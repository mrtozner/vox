//! Endpoint handlers for the Vox HTTP API.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};

use crate::types::{AudioChunk, SttResult, TtsOutput, TtsRequest, Utterance};

use super::ServerState;
use super::access::content_security_policy;
use super::error::ServerError;
use super::models::*;

type AppState = Arc<ServerState>;

/// Longest audio accepted by the transcription endpoints. Bounds the memory
/// a single request can pin: 30 minutes of 16 kHz mono f32 is ~115 MB.
pub const MAX_AUDIO_SECS: u64 = 30 * 60;

/// Sample rates accepted in uploaded WAV files. The lower bound keeps the
/// upsampling ratio (and with it the output size) small.
const WAV_SAMPLE_RATES: std::ops::RangeInclusive<u32> = 4_000..=384_000;

const OLLAMA_TAGS_TIMEOUT: Duration = Duration::from_secs(3);
const OLLAMA_SHOW_TIMEOUT: Duration = Duration::from_secs(2);
const OLLAMA_CHAT_TIMEOUT: Duration = Duration::from_secs(90);

// ---------------------------------------------------------------------------
// Audio decoding / encoding
// ---------------------------------------------------------------------------

/// Audio decoded from an upload, ready for the speech models.
#[derive(Debug)]
pub struct DecodedAudio {
    /// 16 kHz mono samples.
    pub samples: Vec<f32>,
    /// Duration of the uploaded audio in milliseconds.
    pub duration_ms: u64,
}

/// Decode a WAV file and convert it to the 16 kHz mono the speech models
/// expect, whatever its sample rate, channel count or sample format.
///
/// CPU-bound; call it from `spawn_blocking`.
pub fn decode_wav_to_mono_16k(bytes: &[u8]) -> Result<DecodedAudio, ServerError> {
    let invalid = |msg: String| ServerError::bad_request(msg).with_code("invalid_wav");

    let reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| invalid(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();

    if spec.channels == 0 {
        return Err(invalid("invalid WAV: zero channels".into()));
    }
    if !WAV_SAMPLE_RATES.contains(&spec.sample_rate) {
        return Err(invalid(format!(
            "invalid WAV: unsupported sample rate {} Hz (expected {}..={} Hz)",
            spec.sample_rate,
            WAV_SAMPLE_RATES.start(),
            WAV_SAMPLE_RATES.end()
        )));
    }
    let bits = spec.bits_per_sample;
    let bits_ok = match spec.sample_format {
        hound::SampleFormat::Int => (8..=32).contains(&bits),
        hound::SampleFormat::Float => bits == 32,
    };
    if !bits_ok {
        return Err(invalid(format!(
            "invalid WAV: unsupported {bits}-bit {} samples (expected 8-32 bit integer or 32-bit float)",
            match spec.sample_format {
                hound::SampleFormat::Int => "integer",
                hound::SampleFormat::Float => "float",
            }
        )));
    }

    let channels = usize::from(spec.channels);
    let max_samples = (MAX_AUDIO_SECS as usize)
        .saturating_mul(spec.sample_rate as usize)
        .saturating_mul(channels);
    let too_long = || {
        ServerError::payload_too_large(format!(
            "audio is longer than the {} minute limit",
            MAX_AUDIO_SECS / 60
        ))
        .with_code("audio_too_long")
    };
    let decode_err = |e: hound::Error| invalid(format!("WAV decode error: {e}"));

    // Collect by hand rather than through `collect()`: the header's claimed
    // length is attacker-controlled, the bytes actually present are not.
    let mut samples: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.into_samples::<f32>() {
                if samples.len() >= max_samples {
                    return Err(too_long());
                }
                samples.push(s.map_err(decode_err)?);
            }
        }
        hound::SampleFormat::Int => {
            // `bits` is in 8..=32 here, so the shift cannot underflow.
            let scale = (1u64 << (bits - 1)) as f32;
            for s in reader.into_samples::<i32>() {
                if samples.len() >= max_samples {
                    return Err(too_long());
                }
                samples.push(s.map_err(decode_err)? as f32 / scale);
            }
        }
    }

    // Drop a trailing partial frame so the channels stay aligned.
    samples.truncate(samples.len() - samples.len() % channels);
    let frames = (samples.len() / channels) as u64;
    let duration_ms = frames * 1000 / u64::from(spec.sample_rate);

    let samples = crate::audio::to_mono_16k(&samples, spec.sample_rate, spec.channels)
        .map_err(|e| ServerError::internal(format!("audio conversion failed: {e}")))?;

    Ok(DecodedAudio {
        samples,
        duration_ms,
    })
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16
}

fn wav_error(e: hound::Error) -> ServerError {
    // Encoding our own audio failing is our bug, not the client's.
    ServerError::internal(format!("WAV encode error: {e}")).with_code("encode_failed")
}

/// Encode audio as 16-bit PCM WAV, the format every browser and player decodes.
pub fn encode_wav_pcm16(audio: &AudioChunk) -> Result<Vec<u8>, ServerError> {
    let spec = hound::WavSpec {
        channels: audio.channels.max(1),
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::with_capacity(44 + audio.samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).map_err(wav_error)?;
        for &sample in &audio.samples {
            writer.write_sample(to_i16(sample)).map_err(wav_error)?;
        }
        writer.finalize().map_err(wav_error)?;
    }
    Ok(buf.into_inner())
}

/// Encode audio as 32-bit float WAV (lossless; not playable in Chrome's
/// `<audio>` element).
pub fn encode_wav_f32(audio: &AudioChunk) -> Result<Vec<u8>, ServerError> {
    let spec = hound::WavSpec {
        channels: audio.channels.max(1),
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = Cursor::new(Vec::with_capacity(58 + audio.samples.len() * 4));
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).map_err(wav_error)?;
        for &sample in &audio.samples {
            writer.write_sample(sample).map_err(wav_error)?;
        }
        writer.finalize().map_err(wav_error)?;
    }
    Ok(buf.into_inner())
}

/// Headerless 16-bit little-endian PCM.
pub fn encode_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        out.extend_from_slice(&to_i16(sample).to_le_bytes());
    }
    out
}

/// Build an audio response with the sample rate and channel count in headers.
pub fn audio_response(content_type: &'static str, audio: &AudioChunk, body: Vec<u8>) -> Response {
    let mut response = ([(header::CONTENT_TYPE, content_type)], body).into_response();
    let headers = response.headers_mut();
    headers.insert("x-sample-rate", HeaderValue::from(audio.sample_rate));
    headers.insert("x-channels", HeaderValue::from(audio.channels.max(1)));
    response
}

// ---------------------------------------------------------------------------
// Shared STT / TTS execution (also used by the OpenAI-compatible endpoints)
// ---------------------------------------------------------------------------

/// Decode an uploaded WAV and transcribe it under an inference permit.
pub async fn run_transcription(state: &ServerState, body: Bytes) -> Result<SttResult, ServerError> {
    let stt = state.stt.as_ref().ok_or_else(|| {
        ServerError::service_unavailable("STT backend not loaded").with_code("stt_unavailable")
    })?;

    let audio = tokio::task::spawn_blocking(move || decode_wav_to_mono_16k(&body))
        .await
        .map_err(|e| ServerError::internal(format!("audio decoding task failed: {e}")))??;

    let audio_ms = audio.duration_ms;
    let utterance = Utterance {
        audio: AudioChunk {
            samples: audio.samples,
            sample_rate: crate::audio::SPEECH_SAMPLE_RATE,
            channels: 1,
        },
        duration_ms: audio_ms,
        #[cfg(feature = "diarization")]
        speaker_id: None,
    };

    let _permit = state.inference_permit().await?;
    let started = Instant::now();
    let mut result = stt.transcribe(&utterance).await?;
    state
        .stats
        .record_transcription(audio_ms, started.elapsed());

    if result.duration_ms == 0 {
        result.duration_ms = audio_ms;
    }
    Ok(result)
}

/// Validate the text and synthesize it under an inference permit.
pub async fn run_synthesis(
    state: &ServerState,
    tts: &Arc<dyn crate::traits::TtsBackend>,
    request: TtsRequest,
    text_param: &'static str,
) -> Result<TtsOutput, ServerError> {
    if request.text.trim().is_empty() {
        return Err(ServerError::bad_request("text is empty")
            .with_code("empty_text")
            .with_param(text_param));
    }
    let chars = request.text.chars().count();
    let limit = state.limits.max_text_chars;
    if chars > limit {
        return Err(ServerError::payload_too_large(format!(
            "text is {chars} characters, the limit is {limit}. Split it into several requests \
             or use the /v1/speak WebSocket."
        ))
        .with_code("text_too_long")
        .with_param(text_param));
    }

    let _permit = state.inference_permit().await?;
    let started = Instant::now();
    let output = tts.synthesize(&request).await?;
    state
        .stats
        .record_synthesis(output.audio.duration_ms(), started.elapsed());
    Ok(output)
}

// ---------------------------------------------------------------------------
// Speech endpoints
// ---------------------------------------------------------------------------

/// POST /v1/transcribe — accepts WAV body, returns transcription JSON.
///
/// Any sample rate (4-384 kHz), channel count and 8-32 bit integer or 32-bit
/// float format is accepted; the audio is downmixed and resampled to 16 kHz
/// mono before it reaches the STT backend.
pub async fn transcribe(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<impl IntoResponse, ServerError> {
    let result = run_transcription(&state, body).await?;

    Ok(Json(TranscribeResponse {
        text: result.text,
        language: result.language,
        duration_ms: result.duration_ms,
        processing_time_ms: result.processing_time_ms,
    }))
}

/// POST /v1/synthesize — accepts JSON request, returns audio.
///
/// Returns 16-bit PCM WAV unless `format` asks for `wav_f32` or raw `pcm`.
pub async fn synthesize(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<Response, ServerError> {
    let tts = state.tts.as_ref().ok_or_else(|| {
        ServerError::service_unavailable("TTS backend not loaded").with_code("tts_unavailable")
    })?;

    #[derive(Clone, Copy)]
    enum Format {
        Wav16,
        WavF32,
        Pcm16,
    }
    let format = match req
        .format
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("wav" | "wav16" | "wav_pcm16") => Format::Wav16,
        Some("wav_f32" | "f32" | "float") => Format::WavF32,
        Some("pcm" | "pcm16") => Format::Pcm16,
        Some(other) => {
            return Err(ServerError::bad_request(format!(
                "unsupported format '{other}' (expected wav, wav_f32 or pcm)"
            ))
            .with_code("unsupported_format")
            .with_param("format"));
        }
    };

    let tts_request = TtsRequest {
        text: req.text,
        voice: req.voice,
        seed: req.seed,
    };
    let output = run_synthesis(&state, tts, tts_request, "text").await?;
    let audio = &output.audio;

    Ok(match format {
        Format::Wav16 => audio_response("audio/wav", audio, encode_wav_pcm16(audio)?),
        Format::WavF32 => audio_response("audio/wav", audio, encode_wav_f32(audio)?),
        Format::Pcm16 => audio_response("audio/pcm", audio, encode_pcm16(&audio.samples)),
    })
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// Build the vox-shaped backend listing (shared by `/v1/vox/models` and the
/// legacy keys of `/v1/models`).
pub async fn vox_models_response(state: &ServerState) -> ModelsResponse {
    // Check Ollama connectivity, bounded so a hung Ollama cannot stall the UI.
    let ollama = {
        let request =
            crate::privacy::ollama_request(&state.ollama_host, "/api/tags", "llm status check", 0);
        let response = match &request {
            Ok((url, _)) => {
                let request = state.http_client.get(url).timeout(OLLAMA_TAGS_TIMEOUT);
                request.send().await.ok()
            }
            // Refused by offline mode: reported as not connected.
            Err(_) => None,
        };
        match response {
            Some(resp) if resp.status().is_success() => {
                if let (Ok((_, egress)), Some(len)) = (&request, resp.content_length()) {
                    egress.add_received(len);
                }
                let body: serde_json::Value = resp.json().await.unwrap_or_default();
                let count = body["models"].as_array().map(|a| a.len()).unwrap_or(0);
                Some(OllamaStatus {
                    connected: true,
                    host: state.ollama_host.clone(),
                    model_count: count,
                })
            }
            _ => Some(OllamaStatus {
                connected: false,
                host: state.ollama_host.clone(),
                model_count: 0,
            }),
        }
    };

    ModelsResponse {
        stt: state.stt.as_ref().map(|_| BackendInfo {
            name: "whisper".to_string(),
            loaded: true,
            model: state.stt_model_name.clone(),
            size_mb: state.stt_model_size,
        }),
        tts: state.tts.as_ref().map(|tts| BackendInfo {
            name: tts.backend_name().to_string(),
            loaded: true,
            model: state.tts_model_name.clone(),
            size_mb: state.tts_model_size,
        }),
        streaming_stt: state.streaming_stt.as_ref().map(|_| BackendInfo {
            name: "sherpa-streaming".to_string(),
            loaded: true,
            model: Some("zipformer".to_string()),
            size_mb: Some(27),
        }),
        ollama,
    }
}

/// GET /v1/vox/models — list loaded backends (vox's own shape).
///
/// `GET /v1/models` is the OpenAI-shaped listing; see `openai::list_models`.
pub async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(vox_models_response(&state).await)
}

fn avg(total: u64, count: u64) -> Option<u64> {
    (count > 0).then(|| total / count)
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| (numerator as f64 / denominator as f64 * 1000.0).round() / 1000.0)
}

/// GET /v1/stats — server statistics.
///
/// Requests are counted by middleware; reading the statistics does not
/// change them.
pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let s = &state.stats;
    let get = |c: &std::sync::atomic::AtomicU64| c.load(Ordering::Relaxed);

    let transcriptions = get(&s.transcriptions);
    let syntheses = get(&s.syntheses);
    let chats = get(&s.chats);
    let transcribe_ms = get(&s.transcribe_ms);
    let transcribe_audio_ms = get(&s.transcribe_audio_ms);
    let synthesize_ms = get(&s.synthesize_ms);
    let synthesize_audio_ms = get(&s.synthesize_audio_ms);
    let chat_ms = get(&s.chat_ms);

    Json(StatsResponse {
        requests: get(&s.requests),
        transcriptions,
        syntheses,
        uptime_secs: state.start_time.elapsed().as_secs(),
        chats,
        client_errors: get(&s.client_errors),
        server_errors: get(&s.server_errors),
        latency: LatencyStats {
            transcribe_audio_ms,
            transcribe_ms,
            transcribe_avg_ms: avg(transcribe_ms, transcriptions),
            transcribe_rtf: ratio(transcribe_ms, transcribe_audio_ms),
            synthesize_audio_ms,
            synthesize_ms,
            synthesize_avg_ms: avg(synthesize_ms, syntheses),
            synthesize_rtf: ratio(synthesize_ms, synthesize_audio_ms),
            chat_ms,
            chat_avg_ms: avg(chat_ms, chats),
        },
        ws_active: get(&s.ws_active),
        ws_total: get(&s.ws_total),
        inference_slots: state.inference_slots,
        inference_slots_free: state.inference.available_permits(),
    })
}

/// GET /health — liveness: the process is up and serving.
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// GET /health/ready — readiness: 200 once STT, TTS and VAD are loaded,
/// 503 (with the list of what is missing) otherwise.
pub async fn health_ready(State(state): State<AppState>) -> impl IntoResponse {
    let stt = state.stt.is_some();
    let tts = state.tts.is_some() || state.conversation_tts.is_some();
    let vad = state.vad_model_path.is_some();
    let missing: Vec<String> = [("stt", stt), ("tts", tts), ("vad", vad)]
        .into_iter()
        .filter(|(_, loaded)| !loaded)
        .map(|(name, _)| name.to_string())
        .collect();
    let ready = missing.is_empty();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(ReadyResponse {
            status: if ready { "ready" } else { "not_ready" }.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            stt,
            tts,
            vad,
            missing,
        }),
    )
}

/// GET / — serve the WebUI with a Content-Security-Policy.
pub async fn index(headers: HeaderMap) -> impl IntoResponse {
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    let csp = HeaderValue::from_str(&content_security_policy(host))
        .unwrap_or_else(|_| HeaderValue::from_static("default-src 'self'"));
    (
        [
            (header::CONTENT_SECURITY_POLICY, csp),
            (header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY")),
            (
                header::REFERRER_POLICY,
                HeaderValue::from_static("no-referrer"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        Html(include_str!("ui.html")),
    )
}

fn voice_infos(tts: &Arc<dyn crate::traits::TtsBackend>) -> Vec<VoiceInfo> {
    let backend = tts.backend_name().to_string();
    tts.list_voices()
        .into_iter()
        .map(|v| VoiceInfo {
            id: v.id,
            name: v.name,
            gender: v.gender,
            language: v.language,
            accent: v.accent,
            backend: backend.clone(),
        })
        .collect()
}

/// GET /v1/voices — list available TTS voices.
///
/// `voices` belong to the main engine. Live Talk and Converse speak through
/// `conversation_engine`, which may be a different backend with its own
/// `conversation_voices`.
pub async fn voices(State(state): State<AppState>) -> impl IntoResponse {
    let conversation = state.conversation_tts.as_ref().or(state.tts.as_ref());
    Json(VoicesResponse {
        voices: state.tts.as_ref().map(voice_infos).unwrap_or_default(),
        engine: state.tts.as_ref().map(|t| t.backend_name().to_string()),
        conversation_engine: conversation.map(|t| t.backend_name().to_string()),
        conversation_voices: conversation.map(voice_infos).unwrap_or_default(),
    })
}

// ---------------------------------------------------------------------------
// Ollama proxy
// ---------------------------------------------------------------------------

/// Ollama generate request (internal).
#[derive(serde::Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
    keep_alive: &'static str,
    options: serde_json::Value,
}

/// Ollama generate response (internal).
#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

/// Map a failed request to Ollama onto 503 (unreachable), 504 (timed out) or
/// 502 (anything else). The reqwest error goes to the log, not the client.
fn ollama_send_error(host: &str, err: &reqwest::Error) -> ServerError {
    tracing::warn!(ollama_host = host, "Ollama request failed: {err}");
    if err.is_timeout() {
        ServerError::gateway_timeout("Ollama did not answer in time").with_code("upstream_timeout")
    } else if err.is_connect() {
        ServerError::service_unavailable(format!(
            "Ollama is not reachable at {host}. Is it running? Start it with: ollama serve"
        ))
        .with_code("ollama_unreachable")
    } else {
        ServerError::bad_gateway("request to Ollama failed")
    }
}

/// An Ollama request refused by offline mode: 403 `offline_mode`.
fn offline_denied(err: crate::privacy::EgressDenied) -> ServerError {
    tracing::warn!("{err}");
    ServerError::forbidden(err.to_string()).with_code("offline_mode")
}

/// Log (truncated) what an upstream said without handing it to the client:
/// upstream bodies can carry internal paths, and with a mis-pointed
/// `--ollama-host` they are whatever that other service returns.
fn log_upstream_body(context: &str, body: &str) {
    let snippet: String = body.chars().take(500).collect();
    tracing::warn!(body = %snippet, "{context}");
}

/// POST /v1/chat — proxy a chat message through Ollama.
///
/// The upstream is always the operator-configured Ollama; a `host` field in
/// the request is ignored.
pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, ServerError> {
    if req.text.trim().is_empty() {
        return Err(ServerError::bad_request("text is empty")
            .with_code("empty_text")
            .with_param("text"));
    }

    let host = &state.ollama_host;
    let model = req.model.unwrap_or_else(|| "llama3.2".to_string());

    // Build environment-aware system prompt
    let system_prompt = crate::prompts::build_system_prompt_with_registry(
        crate::prompts::VoicePromptMode::Standard,
        &state.capabilities,
    );

    let body = OllamaRequest {
        model: model.clone(),
        prompt: req.text,
        system: Some(system_prompt),
        stream: false,
        keep_alive: crate::conversation::OLLAMA_KEEP_ALIVE,
        options: crate::conversation::ollama_options(),
    };
    let (url, egress) = crate::privacy::ollama_request(
        host,
        "/api/generate",
        "llm chat",
        crate::privacy::json_len(&body),
    )
    .map_err(offline_denied)?;

    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .timeout(OLLAMA_CHAT_TIMEOUT)
        .build()
        .map_err(|e| ServerError::internal(format!("failed to initialize Ollama client: {e}")))?;

    let started = Instant::now();
    let resp = client
        .post(&url)
        .header(reqwest::header::CONNECTION, "close")
        .json(&body)
        .send()
        .await
        .map_err(|e| ollama_send_error(host, &e))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        egress.add_received(text.len() as u64);
        log_upstream_body(&format!("Ollama returned HTTP {status}"), &text);
        return Err(if status == reqwest::StatusCode::NOT_FOUND {
            ServerError::not_found(format!(
                "model '{model}' is not available in Ollama. Pull it with: ollama pull {model}"
            ))
            .with_code("model_not_found")
            .with_param("model")
        } else {
            ServerError::bad_gateway(format!("Ollama returned HTTP {}", status.as_u16()))
        });
    }

    let body = resp.text().await.map_err(|e| {
        tracing::warn!("failed to read Ollama response: {e}");
        if e.is_timeout() {
            ServerError::gateway_timeout("Ollama did not answer in time")
                .with_code("upstream_timeout")
        } else {
            ServerError::bad_gateway("failed to read the Ollama response")
        }
    })?;

    egress.add_received(body.len() as u64);

    let ollama_resp: OllamaResponse = serde_json::from_str(&body).map_err(|e| {
        log_upstream_body(&format!("invalid Ollama response: {e}"), &body);
        ServerError::bad_gateway("Ollama returned an invalid response")
    })?;
    state.stats.record_chat(started.elapsed());

    Ok(Json(ChatResponse {
        response: ollama_resp.response,
        model,
    }))
}

/// GET /v1/ollama-models — list locally available Ollama models.
pub async fn ollama_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ServerError> {
    let (url, egress) =
        crate::privacy::ollama_request(&state.ollama_host, "/api/tags", "llm model list", 0)
            .map_err(offline_denied)?;

    let resp = state
        .http_client
        .get(&url)
        .timeout(OLLAMA_TAGS_TIMEOUT)
        .send()
        .await
        .map_err(|e| ollama_send_error(&state.ollama_host, &e))?;
    egress.add_received(resp.content_length().unwrap_or(0));

    if !resp.status().is_success() {
        return Err(ServerError::bad_gateway(format!(
            "Ollama returned HTTP {}",
            resp.status().as_u16()
        )));
    }

    // Ollama /api/tags returns { "models": [{ "name": "llama3.2:latest", "size": 123456, ... }] }
    let body: serde_json::Value = resp.json().await.map_err(|e| {
        tracing::warn!("invalid Ollama /api/tags response: {e}");
        ServerError::bad_gateway("Ollama returned an invalid response")
    })?;

    // Filter to only generative (chat) models. `/api/tags` already returns a
    // `details.family` field for every installed model — we can use that to
    // skip obvious embedding families without a second round-trip. For anything
    // not conclusively classified, we probe `/api/show` in parallel with a
    // total budget so a slow Ollama can never hang the UI.
    let embedding_families = [
        "bert",
        "nomic-bert",
        "nomic-embed-text",
        "mxbai-embed-large",
        "jina-bert",
        "snowflake-arctic-embed",
        "stella",
    ];

    let mut candidates: Vec<(String, Option<u64>, bool)> = Vec::new();
    if let Some(arr) = body["models"].as_array() {
        for m in arr {
            let name = m["name"].as_str().unwrap_or("unknown").to_string();
            let size = m["size"].as_u64();
            let family = m["details"]["family"].as_str().unwrap_or("").to_lowercase();

            // Definitely an embedding model: drop it without a probe.
            if embedding_families.iter().any(|e| family == *e) {
                continue;
            }

            // Known chat families: trust the tag and skip the probe.
            let chat_families = ["llama", "gemma", "mistral", "qwen", "phi", "gpt"];
            let trusted = chat_families.iter().any(|c| family.contains(c));
            candidates.push((name, size, trusted));
        }
    }

    let client = state.http_client.clone();

    // Probe untrusted candidates in parallel, with a global 3s budget.
    let probes = candidates.into_iter().map(|(name, size, trusted)| {
        let client = client.clone();
        let host = state.ollama_host.clone();
        async move {
            if trusted {
                return Some(OllamaModelInfo { name, size });
            }
            let show_body = serde_json::json!({ "name": name.clone() });
            let (show_url, egress) = crate::privacy::ollama_request(
                &host,
                "/api/show",
                "llm model details",
                crate::privacy::json_len(&show_body),
            )
            .ok()?;
            let req = client
                .post(&show_url)
                .json(&show_body)
                .timeout(OLLAMA_SHOW_TIMEOUT)
                .send();
            // Per-probe cap (2s) on top of client timeout — keeps the overall
            // budget tight even if reqwest is blocked on DNS or TCP.
            let result = tokio::time::timeout(Duration::from_secs(2), req).await;
            let has_template = match result {
                Ok(Ok(r)) => {
                    egress.add_received(r.content_length().unwrap_or(0));
                    let info_fut = r.json::<serde_json::Value>();
                    tokio::time::timeout(Duration::from_secs(1), info_fut)
                        .await
                        .ok()
                        .and_then(|r| r.ok())
                        .map(|info| info["template"].as_str().unwrap_or("").len() > 50)
                        .unwrap_or(true)
                }
                _ => true, // include on timeout or error — better to show than hide
            };
            if has_template {
                Some(OllamaModelInfo { name, size })
            } else {
                None
            }
        }
    });

    // Global cap on the whole fan-out. If Ollama is really stuck, we return
    // whatever we have (possibly empty) rather than blocking the client.
    let all = futures_util::future::join_all(probes);
    let models: Vec<OllamaModelInfo> = match tokio::time::timeout(Duration::from_secs(3), all).await
    {
        Ok(results) => results.into_iter().flatten().collect(),
        Err(_) => {
            // Budget exceeded — fall back to a best-effort list from /api/tags
            // without template filtering. User still sees their chat models.
            if let Some(arr) = body["models"].as_array() {
                arr.iter()
                    .map(|m| OllamaModelInfo {
                        name: m["name"].as_str().unwrap_or("unknown").to_string(),
                        size: m["size"].as_u64(),
                    })
                    .collect()
            } else {
                Vec::new()
            }
        }
    };

    Ok(Json(OllamaModelsResponse { models }))
}

/// GET /v1/capabilities — return the full environment capability registry.
pub async fn capabilities(State(state): State<AppState>) -> impl IntoResponse {
    Json((*state.capabilities).clone())
}
