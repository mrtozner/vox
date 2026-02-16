//! HTTP server for the Vox voice AI framework.
//!
//! Provides a REST API for speech-to-text and text-to-speech operations.
//! Start with [`run`] from the `vox serve` CLI command.

pub mod error;
pub mod handlers;
pub mod models;
pub mod ws;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Shared server state accessible from all handlers.
///
/// `stt` is the primary batch STT backend (Whisper or offline Sherpa) used for
/// `/v1/transcribe` and as the fallback in WebSocket streaming. `streaming_stt`
/// is the optional incremental backend (online Sherpa) used *only* for the
/// WebSocket `/v1/listen` session path. They must be separate instances —
/// do not load the same backend as both.
pub struct ServerState {
    pub stt: Option<Arc<dyn vox::traits::SttBackend>>,
    pub tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>>,
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

    #[cfg(feature = "whisper")]
    let (stt, stt_model_name, stt_model_size): (
        Option<Arc<dyn vox::traits::SttBackend>>,
        Option<String>,
        Option<u64>,
    ) = {
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
                        let variant = name
                            .strip_prefix("ggml-")
                            .unwrap_or(name)
                            .strip_suffix(".bin")
                            .unwrap_or(name)
                            .to_string();
                        model_name = Some(variant);
                        model_size = std::fs::metadata(&path)
                            .ok()
                            .map(|m| m.len() / (1024 * 1024));
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
    let (stt, stt_model_name, stt_model_size): (
        Option<Arc<dyn vox::traits::SttBackend>>,
        Option<String>,
        Option<u64>,
    ) = {
        tracing::warn!("STT disabled (compiled without 'whisper' feature)");
        (None, None, None)
    };

    let (tts, tts_model_name, tts_model_size) = load_tts(&models_dir).await;

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

    #[cfg(feature = "sherpa")]
    let streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>> = {
        let streaming_dir = models_dir.join("sherpa-streaming");
        if streaming_dir.exists() {
            match vox::SherpaStreamingBackend::from_transducer(&streaming_dir) {
                Ok(backend) => {
                    info!(
                        model = %streaming_dir.display(),
                        "loaded sherpa streaming STT backend"
                    );
                    Some(Arc::new(backend))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load streaming STT");
                    None
                }
            }
        } else {
            None
        }
    };
    #[cfg(not(feature = "sherpa"))]
    let streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>> = None;

    let ollama_host =
        std::env::var("VOX_OLLAMA_HOST").unwrap_or_else(|_| "localhost:11434".to_string());

    let state = Arc::new(ServerState {
        stt,
        tts,
        streaming_stt,
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
                let model_name = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.replace('-', " "))
                    .unwrap_or_else(|| "kokoro".to_string());
                let model_size = std::fs::metadata(&model_path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                return (
                    Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                    Some(model_name),
                    model_size,
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load kokoro TTS backend");
            }
        }
    } else {
        tracing::warn!(
            "kokoro model files not found in {}, trying piper fallback",
            models_dir.display()
        );
    }

    // Fallback: try Piper if Kokoro is unavailable
    #[cfg(feature = "piper")]
    {
        if let Some(result) = try_load_piper(models_dir) {
            return result;
        }
    }

    (None, None, None)
}

#[cfg(all(feature = "piper", not(feature = "kokoro")))]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    if let Some(result) = try_load_piper(models_dir) {
        return result;
    }
    tracing::warn!(
        "no piper model found in {}, TTS disabled",
        models_dir.display()
    );
    (None, None, None)
}

#[cfg(not(any(feature = "kokoro", feature = "piper")))]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro' or 'piper' feature); models dir: {}",
        models_dir.display()
    );
    (None, None, None)
}

/// Scan models_dir/piper/ for any *.onnx.json config files and try to load the first valid one.
#[cfg(feature = "piper")]
fn try_load_piper(
    models_dir: &std::path::Path,
) -> Option<(
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
)> {
    let piper_dir = models_dir.join("piper");
    if !piper_dir.exists() {
        return None;
    }

    let entries = std::fs::read_dir(&piper_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json")
            && path.to_str().is_some_and(|s| s.ends_with(".onnx.json"))
        {
            match vox::tts::PiperBackend::new(&path) {
                Ok(backend) => {
                    info!(model = %path.display(), "loaded piper TTS backend");
                    let model_name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| {
                            // "en_US-lessac-medium.onnx" -> "piper en_US-lessac-medium"
                            format!("piper {}", s.strip_suffix(".onnx").unwrap_or(s))
                        })
                        .unwrap_or_else(|| "piper".to_string());
                    // Size of the .onnx model file (not the json config)
                    let onnx_path = path.with_extension(""); // strips .json -> .onnx
                    let model_size = std::fs::metadata(&onnx_path)
                        .ok()
                        .map(|m| m.len() / (1024 * 1024));
                    return Some((
                        Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                        Some(model_name),
                        model_size,
                    ));
                }
                Err(e) => {
                    tracing::warn!(model = %path.display(), error = %e, "failed to load piper model");
                }
            }
        }
    }
    None
}
