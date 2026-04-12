//! Endpoint handlers for the Vox HTTP API.

use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

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
        #[cfg(feature = "diarization")]
        speaker_id: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
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

    // Build environment-aware system prompt
    let system_prompt = vox::prompts::build_system_prompt_with_registry(
        vox::prompts::VoicePromptMode::Standard,
        &state.capabilities,
    );

    let url = format!("http://{host}/api/generate");
    let body = OllamaRequest {
        model: model.clone(),
        prompt: req.text,
        system: Some(system_prompt),
        stream: false,
    };

    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|e| {
            ServerError::service_unavailable(format!("failed to initialize Ollama client: {e}"))
        })?;

    let resp = client
        .post(&url)
        .header(reqwest::header::CONNECTION, "close")
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

    let body = resp
        .text()
        .await
        .map_err(|e| ServerError::bad_request(format!("failed to read Ollama response: {e}")))?;

    let ollama_resp: OllamaResponse = serde_json::from_str(&body)
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

    let show_url = format!("http://{}/api/show", state.ollama_host);
    let client = state.http_client.clone();

    // Probe untrusted candidates in parallel, with a global 3s budget.
    let probes = candidates.into_iter().map(|(name, size, trusted)| {
        let client = client.clone();
        let show_url = show_url.clone();
        async move {
            if trusted {
                return Some(OllamaModelInfo { name, size });
            }
            let req = client
                .post(&show_url)
                .json(&serde_json::json!({ "name": name.clone() }))
                .send();
            // Per-probe cap (2s) on top of client timeout — keeps the overall
            // budget tight even if reqwest is blocked on DNS or TCP.
            let result = tokio::time::timeout(Duration::from_secs(2), req).await;
            let has_template = match result {
                Ok(Ok(r)) => {
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
    {
        let mut stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.requests += 1;
    }
    Json((*state.capabilities).clone())
}

/// GET /v1/cache/stats — model cache statistics.
pub async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    if let Some(cache) = &state.model_cache {
        let stats = cache.stats();
        Json(CacheStatsResponse {
            enabled: true,
            entries: stats.entries,
            max_entries: stats.max_entries,
            hits: stats.hits,
            misses: stats.misses,
            hit_rate: stats.hit_rate,
        })
    } else {
        Json(CacheStatsResponse {
            enabled: false,
            entries: 0,
            max_entries: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        })
    }
}
