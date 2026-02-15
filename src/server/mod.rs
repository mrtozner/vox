//! HTTP server for the Vox voice AI framework.
//!
//! Provides a REST API for speech-to-text and text-to-speech operations.
//! Start with [`run`] from the `vox serve` CLI command.

mod error;
mod handlers;
mod models;
mod ws;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Shared server state accessible from all handlers.
pub struct ServerState {
    pub stt: Option<Arc<dyn vox::traits::SttBackend>>,
    pub tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub vad_model_path: Option<std::path::PathBuf>,
    pub stats: Arc<std::sync::Mutex<ServerStats>>,
    pub start_time: std::time::Instant,
    pub ollama_host: String,
    pub http_client: reqwest::Client,
    pub stt_model_name: Option<String>,
    pub stt_model_size: Option<u64>,
    pub tts_model_name: Option<String>,
    pub tts_model_size: Option<u64>,
}

/// Cumulative request counters.
pub struct ServerStats {
    pub requests: u64,
    pub transcriptions: u64,
    pub syntheses: u64,
}

/// Start the HTTP server on `host:port`.
///
/// Attempts to load STT (Whisper) and TTS (Kokoro) backends from
/// `~/.vox/models/`. Missing backends are tolerated -- the corresponding
/// endpoints return 503 until models are available.
pub async fn run(host: &str, port: u16) -> anyhow::Result<()> {
    let models_dir = dirs::data_dir()
        .map(|d| d.join("vox").join("models"))
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".vox")
                .join("models")
        });

    // --- Load STT backend (optional) -------------------------------------------
    #[cfg(feature = "whisper")]
    let (stt, stt_model_name, stt_model_size): (Option<Arc<dyn vox::traits::SttBackend>>, Option<String>, Option<u64>) = {
        // Try common whisper model names in preference order
        let candidates = [
            "ggml-base.en.bin",
            "ggml-tiny.en.bin",
            "ggml-base.bin",
            "ggml-tiny.bin",
            "ggml-small.en.bin",
            "ggml-small.bin",
        ];
        let mut loaded = None;
        let mut model_name = None;
        let mut model_size = None;
        for name in &candidates {
            let path = models_dir.join(name);
            if path.exists() {
                match vox::stt::WhisperBackend::from_model(&path) {
                    Ok(backend) => {
                        info!(model = %path.display(), "loaded whisper STT backend");
                        // Extract model variant from filename: "ggml-base.en.bin" -> "base.en"
                        let variant = name
                            .strip_prefix("ggml-")
                            .unwrap_or(name)
                            .strip_suffix(".bin")
                            .unwrap_or(name)
                            .to_string();
                        model_name = Some(variant);
                        model_size = std::fs::metadata(&path).ok().map(|m| m.len() / (1024 * 1024));
                        loaded = Some(Arc::new(backend) as Arc<dyn vox::traits::SttBackend>);
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(model = %path.display(), error = %e, "failed to load whisper model");
                    }
                }
            }
        }
        if loaded.is_none() {
            tracing::warn!(
                "no whisper model found in {}, STT disabled",
                models_dir.display()
            );
        }
        (loaded, model_name, model_size)
    };
    #[cfg(not(feature = "whisper"))]
    let (stt, stt_model_name, stt_model_size): (Option<Arc<dyn vox::traits::SttBackend>>, Option<String>, Option<u64>) = {
        tracing::warn!("STT disabled (compiled without 'whisper' feature)");
        (None, None, None)
    };

    // --- Load TTS backend (optional) -------------------------------------------
    let (tts, tts_model_name, tts_model_size) = load_tts(&models_dir).await;

    // --- Detect VAD model for WebSocket endpoint --------------------------------
    let vad_model_path = {
        let path = models_dir.join("silero_vad.onnx");
        if path.exists() {
            info!(model = %path.display(), "VAD model found for WebSocket endpoint");
            Some(path)
        } else {
            tracing::warn!("VAD model not found, WebSocket /v1/listen will be unavailable");
            None
        }
    };

    // --- Build state and router ------------------------------------------------
    let ollama_host =
        std::env::var("VOX_OLLAMA_HOST").unwrap_or_else(|_| "localhost:11434".to_string());

    let state = Arc::new(ServerState {
        stt,
        tts,
        vad_model_path,
        stats: Arc::new(std::sync::Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: std::time::Instant::now(),
        ollama_host,
        http_client: reqwest::Client::new(),
        stt_model_name,
        stt_model_size,
        tts_model_name,
        tts_model_size,
    });

    let app = Router::new()
        .route("/", get(handlers::index))
        .route("/v1/chat", post(handlers::chat))
        .route("/v1/voices", get(handlers::voices))
        .route("/v1/ollama-models", get(handlers::ollama_models))
        .route("/v1/transcribe", post(handlers::transcribe))
        .route("/v1/synthesize", post(handlers::synthesize))
        .route("/v1/models", get(handlers::models))
        .route("/v1/stats", get(handlers::stats))
        .route("/health", get(handlers::health))
        .route("/v1/listen", get(ws::listen_ws))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB max request body
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // --- Bind and serve --------------------------------------------------------
    let addr = format!("{host}:{port}");
    println!();
    println!("  vox server v{}", env!("CARGO_PKG_VERSION"));
    println!("  listening on http://{addr}");
    println!("  open http://{addr}/ in your browser for the web interface");
    println!();
    println!("  endpoints:");
    println!("    GET  /            — web interface");
    println!("    POST /v1/chat        — LLM chat via Ollama");
    println!("    GET  /v1/voices      — list TTS voices");
    println!("    GET  /v1/ollama-models — list Ollama models");
    println!("    POST /v1/transcribe  — speech-to-text (WAV body)");
    println!("    POST /v1/synthesize  — text-to-speech (JSON body)");
    println!("    GET  /v1/models      — list loaded backends");
    println!("    GET  /v1/stats       — server statistics");
    println!("    WS   /v1/listen       — real-time voice transcription");
    println!("    GET  /health         — health check");
    println!();

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(feature = "kokoro")]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let model_path = models_dir.join("kokoro-v1.0.onnx");
    let voices_path = models_dir.join("voices.bin");
    if model_path.exists() && voices_path.exists() {
        match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
            Ok(backend) => {
                info!("loaded kokoro TTS backend");
                // Extract model name from filename: "kokoro-v1.0.onnx" -> "kokoro v1.0"
                let model_name = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.replace('-', " "))
                    .unwrap_or_else(|| "kokoro".to_string());
                let model_size =
                    std::fs::metadata(&model_path).ok().map(|m| m.len() / (1024 * 1024));
                (
                    Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                    Some(model_name),
                    model_size,
                )
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load kokoro TTS backend");
                (None, None, None)
            }
        }
    } else {
        tracing::warn!(
            "kokoro model files not found in {}, TTS disabled",
            models_dir.display()
        );
        (None, None, None)
    }
}

#[cfg(not(feature = "kokoro"))]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro' feature); models dir: {}",
        models_dir.display()
    );
    (None, None, None)
}
