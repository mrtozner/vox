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
    let stt: Option<Arc<dyn vox::traits::SttBackend>> = {
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
        for name in &candidates {
            let path = models_dir.join(name);
            if path.exists() {
                match vox::stt::WhisperBackend::from_model(&path) {
                    Ok(backend) => {
                        info!(model = %path.display(), "loaded whisper STT backend");
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
        loaded
    };
    #[cfg(not(feature = "whisper"))]
    let stt: Option<Arc<dyn vox::traits::SttBackend>> = {
        tracing::warn!("STT disabled (compiled without 'whisper' feature)");
        None
    };

    // --- Load TTS backend (optional) -------------------------------------------
    let tts: Option<Arc<dyn vox::traits::TtsBackend>> = load_tts(&models_dir).await;

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
    });

    let app = Router::new()
        .route("/v1/transcribe", post(handlers::transcribe))
        .route("/v1/synthesize", post(handlers::synthesize))
        .route("/v1/models", get(handlers::models))
        .route("/v1/stats", get(handlers::stats))
        .route("/health", get(handlers::health))
        .route("/v1/listen", get(ws::listen_ws))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // --- Bind and serve --------------------------------------------------------
    let addr = format!("{host}:{port}");
    println!();
    println!("  vox server v{}", env!("CARGO_PKG_VERSION"));
    println!("  listening on http://{addr}");
    println!();
    println!("  endpoints:");
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
async fn load_tts(models_dir: &std::path::Path) -> Option<Arc<dyn vox::traits::TtsBackend>> {
    let model_path = models_dir.join("kokoro-v1.0.onnx");
    let voices_path = models_dir.join("voices.bin");
    if model_path.exists() && voices_path.exists() {
        match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
            Ok(backend) => {
                info!("loaded kokoro TTS backend");
                Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>)
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load kokoro TTS backend");
                None
            }
        }
    } else {
        tracing::warn!(
            "kokoro model files not found in {}, TTS disabled",
            models_dir.display()
        );
        None
    }
}

#[cfg(not(feature = "kokoro"))]
async fn load_tts(models_dir: &std::path::Path) -> Option<Arc<dyn vox::traits::TtsBackend>> {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro' feature); models dir: {}",
        models_dir.display()
    );
    None
}
