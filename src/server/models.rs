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
}

/// Response listing available backends.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub stt: Option<BackendInfo>,
    pub tts: Option<BackendInfo>,
}

/// Info about a loaded backend.
#[derive(Debug, Serialize)]
pub struct BackendInfo {
    pub name: String,
    pub loaded: bool,
}

/// Server statistics response.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub requests: u64,
    pub transcriptions: u64,
    pub syntheses: u64,
    pub uptime_secs: u64,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}
