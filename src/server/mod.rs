//! HTTP server for the Vox voice AI framework.
//!
//! Provides a REST API for speech-to-text and text-to-speech operations.
//! Start with [`run`] from the `vox serve` CLI command.

#[cfg(feature = "diarization")]
mod diarization_init;
pub mod error;
pub mod handlers;
pub mod live_talk;
mod loaders;
pub mod models;
mod speakers;
mod state;
pub mod ws;
pub mod ws_converse;
pub mod ws_listen;
pub mod ws_speak;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

#[cfg(feature = "diarization")]
use diarization_init::load_diarization;
#[cfg(feature = "piper")]
use loaders::try_load_piper;
use loaders::{load_tts, resolve_models_dir};
pub use state::{ServerState, ServerStats};

/// Start the HTTP server on `host:port`.
///
/// Attempts to load STT (Whisper) and TTS (Kokoro/Piper) backends from
/// the repo-relative `models/` directory (see
/// [`crate::cli::models::models_dir`]). Missing backends are tolerated --
/// the corresponding endpoints return 503 until models are available.
///
/// # Arguments
/// - `cache_models`: deprecated, ignored (kept so `vox serve --cache-models` still parses)
pub async fn run(host: &str, port: u16, cache_models: bool) -> anyhow::Result<()> {
    // Use the single source of truth for model resolution -- repo-relative
    // by default, overridable via `VOX_MODELS_DIR`. Mirrors the logic in
    // `crate::cli::models::models_dir` so the server module is self-contained
    // when pulled into test crates via `#[path = ...]`.
    let models_dir = resolve_models_dir();
    info!(dir = %models_dir.display(), "server using models directory");

    // `--cache-models` used to build an LRU model cache that nothing ever
    // read from or wrote to. The cache is gone; the flag is accepted and
    // ignored so existing invocations keep working.
    if cache_models {
        tracing::warn!("--cache-models is deprecated and has no effect");
    }

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

    // Load Piper as conversation TTS (better pronunciation for Live Talk/Converse)
    #[cfg(feature = "piper")]
    let conversation_tts: Option<Arc<dyn vox::traits::TtsBackend>> = {
        // Only load separately if main TTS isn't already Piper
        if tts_model_name
            .as_deref()
            .is_some_and(|n| n.starts_with("piper"))
        {
            tts.clone() // Main TTS is already Piper, reuse it
        } else {
            match try_load_piper(&models_dir) {
                Some((Some(piper), _, _)) => {
                    info!("loaded piper as conversation TTS (Live Talk/Converse)");
                    Some(piper)
                }
                _ => None,
            }
        }
    };
    #[cfg(not(feature = "piper"))]
    let conversation_tts: Option<Arc<dyn vox::traits::TtsBackend>> = None;

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

    #[cfg(feature = "diarization")]
    let (diarization, speaker_db) = load_diarization(&models_dir).await;

    // Build capability registry from already-loaded state.
    let profile = vox::system_profile::SystemProfile::detect();
    let http_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let stt_tuple: Option<(&str, &str, Option<u64>)> = stt_model_name
        .as_deref()
        .map(|n| ("whisper", n, stt_model_size));

    let streaming_stt_tuple: Option<(&str, &str, Option<u64>)> = if streaming_stt.is_some() {
        Some(("sherpa-streaming", "zipformer", Some(27)))
    } else {
        None
    };

    // Infer TTS backend name from model name prefix if possible.
    let tts_tuple: Option<(&str, &str, Option<u64>)> = tts_model_name.as_deref().map(|name| {
        let backend = if name.starts_with("piper") {
            "piper"
        } else if name == "qwen3" {
            "qwen3"
        } else if name.starts_with("kokoro") {
            "kokoro"
        } else {
            "tts"
        };
        (backend, name, tts_model_size)
    });

    let capabilities = Arc::new(
        vox::CapabilityRegistry::build(
            &profile,
            stt_tuple,
            streaming_stt_tuple,
            tts_tuple,
            vad_model_path.as_deref(),
            &ollama_host,
            &http_client,
        )
        .await,
    );
    info!(
        ollama_models = capabilities.ollama_models.len(),
        "capability registry built"
    );

    let state = Arc::new(ServerState {
        stt,
        tts,
        conversation_tts,
        streaming_stt,
        vad_model_path,
        stats: Arc::new(std::sync::Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: std::time::Instant::now(),
        ollama_host,
        http_client: http_client.clone(),
        stt_model_name,
        stt_model_size,
        tts_model_name,
        tts_model_size,
        capabilities,
        #[cfg(feature = "diarization")]
        diarization,
        #[cfg(feature = "diarization")]
        speaker_db,
    });

    let app = Router::new()
        .route("/", get(handlers::index))
        .route("/v1/chat", post(handlers::chat))
        .route("/v1/voices", get(handlers::voices))
        .route("/v1/ollama-models", get(handlers::ollama_models))
        .route("/v1/capabilities", get(handlers::capabilities))
        .route("/v1/transcribe", post(handlers::transcribe))
        .route("/v1/synthesize", post(handlers::synthesize))
        .route("/v1/models", get(handlers::models))
        .route("/v1/stats", get(handlers::stats))
        .route("/health", get(handlers::health))
        .route("/v1/listen", get(ws_listen::listen_ws))
        .route("/v1/speak", get(ws_speak::speak_ws))
        .route("/v1/converse", get(ws_converse::converse_ws))
        .route("/v1/live-talk", get(live_talk::live_talk_ws))
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
    println!("    GET  /v1/capabilities — environment capability registry");
    println!("    POST /v1/transcribe  — speech-to-text (WAV body)");
    println!("    POST /v1/synthesize  — text-to-speech (JSON body)");
    println!("    GET  /v1/models      — list loaded backends");
    println!("    GET  /v1/stats       — server statistics");
    println!("    WS   /v1/listen       — real-time voice transcription");
    println!("    WS   /v1/speak        — streaming text-to-speech");
    println!("    WS   /v1/converse     — continuous voice chat (VAD+STT+LLM+TTS)");
    println!("    WS   /v1/live-talk    — barge-in voice chat (experimental)");
    println!("    GET  /health         — health check");
    println!();

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
