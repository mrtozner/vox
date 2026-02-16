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
}

/// Response listing available backends.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub stt: Option<BackendInfo>,
    pub tts: Option<BackendInfo>,
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

/// Request body for the chat endpoint (Ollama proxy).
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub text: String,
    pub model: Option<String>,
    pub host: Option<String>,
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
    pub voices: Vec<VoiceInfo>,
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
