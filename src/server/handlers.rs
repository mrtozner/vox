//! Endpoint handlers for the Vox HTTP API.

use std::io::Cursor;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::{Html, IntoResponse};

use vox::types::{AudioChunk, TtsRequest, Utterance};

use super::ServerState;
use super::error::ServerError;
use super::models::*;

type AppState = Arc<ServerState>;

/// POST /v1/transcribe — accepts WAV body, returns transcription JSON.
pub async fn transcribe(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<impl IntoResponse, ServerError> {
    let stt = state
        .stt
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("STT backend not loaded"))?;

    // Bump request counters
    {
        let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.requests += 1;
        stats.transcriptions += 1;
    }

    // Decode WAV
    let cursor = Cursor::new(body.as_ref());
    let reader = hound::WavReader::new(cursor)
        .map_err(|e| ServerError::bad_request(format!("invalid WAV: {e}")))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ServerError::bad_request(format!("WAV decode error: {e}")))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| ServerError::bad_request(format!("WAV decode error: {e}")))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    let duration_ms = if sample_rate > 0 {
        (samples.len() as u64 * 1000) / (sample_rate as u64 * channels as u64)
    } else {
        0
    };

    let utterance = Utterance {
        audio: AudioChunk {
            samples,
            sample_rate,
            channels,
        },
        duration_ms,
    };

    let result = stt.transcribe(&utterance).await?;

    Ok(Json(TranscribeResponse {
        text: result.text,
        language: result.language,
        duration_ms: result.duration_ms,
        processing_time_ms: result.processing_time_ms,
    }))
}

/// POST /v1/synthesize — accepts JSON request, returns WAV audio.
pub async fn synthesize(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let tts = state
        .tts
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("TTS backend not loaded"))?;

    // Bump request counters
    {
        let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.requests += 1;
        stats.syntheses += 1;
    }

    let tts_request = TtsRequest {
        text: req.text,
        voice: req.voice,
        seed: req.seed,
    };

    let output = tts.synthesize(&tts_request).await?;

    // Encode audio as WAV into a byte buffer
    let spec = hound::WavSpec {
        channels: output.audio.channels,
        sample_rate: output.audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec)
            .map_err(|e| ServerError::bad_request(format!("WAV encode error: {e}")))?;
        for &sample in &output.audio.samples {
            writer
                .write_sample(sample)
                .map_err(|e| ServerError::bad_request(format!("WAV write error: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| ServerError::bad_request(format!("WAV finalize error: {e}")))?;
    }

    let wav_bytes = buf.into_inner();

    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav_bytes))
}

/// GET /v1/models — list loaded backends.
pub async fn models(State(state): State<AppState>) -> impl IntoResponse {
    {
        let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.requests += 1;
    }

    // Check Ollama connectivity
    let ollama = {
        let url = format!("http://{}/api/tags", state.ollama_host);
        match state.http_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
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

    Json(ModelsResponse {
        stt: state.stt.as_ref().map(|_| BackendInfo {
            name: "whisper".to_string(),
            loaded: true,
            model: state.stt_model_name.clone(),
            size_mb: state.stt_model_size,
        }),
        tts: state.tts.as_ref().map(|_| BackendInfo {
            name: "kokoro".to_string(),
            loaded: true,
            model: state.tts_model_name.clone(),
            size_mb: state.tts_model_size,
        }),
        ollama,
    })
}

/// GET /v1/stats — server statistics.
pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let mut s = state.stats.lock().unwrap_or_else(|e| e.into_inner());
    s.requests += 1;
    let uptime_secs = state.start_time.elapsed().as_secs();

    Json(StatsResponse {
        requests: s.requests,
        transcriptions: s.transcriptions,
        syntheses: s.syntheses,
        uptime_secs,
    })
}

/// GET /health — simple health check.
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// GET / — serve the WebUI.
pub async fn index() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

/// GET /v1/voices — list available TTS voices.
pub async fn voices(State(state): State<AppState>) -> impl IntoResponse {
    let voices = if let Some(tts) = state.tts.as_ref() {
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
    } else {
        vec![]
    };
    Json(VoicesResponse { voices })
}

/// Ollama generate request (internal).
#[derive(serde::Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

/// Ollama generate response (internal).
#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

/// POST /v1/chat — proxy a chat message through Ollama.
pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, ServerError> {
    {
        let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.requests += 1;
    }

    let host = req.host.unwrap_or_else(|| state.ollama_host.clone());
    let model = req.model.unwrap_or_else(|| "llama3.2".to_string());

    let url = format!("http://{host}/api/generate");
    let body = OllamaRequest {
        model: model.clone(),
        prompt: req.text,
        stream: false,
    };

    let resp = state
        .http_client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ServerError::service_unavailable(format!(
                "Ollama request failed: {e}\n\nIs Ollama running? Start it with: ollama serve"
            ))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(ServerError::service_unavailable(format!(
            "Ollama returned HTTP {status}: {text}"
        )));
    }

    let ollama_resp: OllamaResponse = resp
        .json()
        .await
        .map_err(|e| ServerError::bad_request(format!("invalid Ollama response: {e}")))?;

    Ok(Json(ChatResponse {
        response: ollama_resp.response,
        model,
    }))
}

/// GET /v1/ollama-models — list locally available Ollama models.
pub async fn ollama_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ServerError> {
    let url = format!("http://{}/api/tags", state.ollama_host);

    let resp = state.http_client.get(&url).send().await.map_err(|e| {
        ServerError::service_unavailable(format!(
            "Ollama not reachable: {e}\n\nStart it with: ollama serve"
        ))
    })?;

    if !resp.status().is_success() {
        return Err(ServerError::service_unavailable("Ollama returned an error"));
    }

    // Ollama /api/tags returns { "models": [{ "name": "llama3.2:latest", "size": 123456, ... }] }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ServerError::bad_request(format!("invalid Ollama response: {e}")))?;

    // Filter to only generative (chat) models by checking if the model has a
    // prompt template via /api/show.  Embedding models never have one.
    let show_url = format!("http://{}/api/show", state.ollama_host);
    let mut models = Vec::new();
    if let Some(arr) = body["models"].as_array() {
        for m in arr {
            let name = m["name"].as_str().unwrap_or("unknown");
            let size = m["size"].as_u64();

            let has_template = match state
                .http_client
                .post(&show_url)
                .json(&serde_json::json!({ "name": name }))
                .send()
                .await
            {
                Ok(r) => r
                    .json::<serde_json::Value>()
                    .await
                    .ok()
                    .map(|info| {
                        // Chat models have rich templates with role markers.
                        // Embedding models have only "{{ .Prompt }}" or empty.
                        let tmpl = info["template"].as_str().unwrap_or("");
                        tmpl.len() > 50
                    })
                    .unwrap_or(true),
                Err(_) => true, // include if we can't determine
            };

            if has_template {
                models.push(OllamaModelInfo {
                    name: name.to_string(),
                    size,
                });
            }
        }
    }

    Ok(Json(OllamaModelsResponse { models }))
}
