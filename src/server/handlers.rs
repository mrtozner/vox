//! Endpoint handlers for the Vox HTTP API.

use std::io::Cursor;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;

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

    Json(ModelsResponse {
        stt: state.stt.as_ref().map(|_| BackendInfo {
            name: "whisper".to_string(),
            loaded: true,
        }),
        tts: state.tts.as_ref().map(|_| BackendInfo {
            name: "kokoro".to_string(),
            loaded: true,
        }),
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
