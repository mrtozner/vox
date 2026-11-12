//! HTTP server for the Vox voice AI framework.
//!
//! Provides a REST API for speech-to-text and text-to-speech operations.
//! Start with [`run`] from the `vox serve` CLI command.

pub mod access;
#[cfg(feature = "diarization")]
mod diarization_init;
pub mod error;
pub mod handlers;
pub mod live_talk;
mod loaders;
pub mod middleware;
pub mod models;
pub mod openai;
mod speakers;
mod state;
pub mod ws;
pub mod ws_converse;
pub mod ws_listen;
pub mod ws_speak;

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

#[cfg(feature = "diarization")]
use diarization_init::load_diarization;
pub use loaders::LoadOptions;
#[cfg(feature = "piper")]
use loaders::try_load_piper;
use loaders::{load_stt, load_tts, resolve_models_dir};
#[allow(unused_imports)]
pub use state::{
    DEFAULT_OLLAMA_HOST, Limits, ServerOptions, ServerState, ServerStats, WsSessionGuard,
    normalize_ollama_host, shutdown_close_message,
};

/// Largest request body accepted (audio uploads).
pub const MAX_BODY_BYTES: usize = 50 * 1024 * 1024;

/// Build the application router with every route and middleware layer.
///
/// Layers, outermost first: request tracing, statistics + uniform error
/// bodies, `Host`/`Origin` enforcement, CORS for allowed origins, request
/// timeout, body size limit.
pub fn build_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/", get(handlers::index))
        .route("/v1/chat", post(handlers::chat))
        .route("/v1/voices", get(handlers::voices))
        .route("/v1/ollama-models", get(handlers::ollama_models))
        .route("/v1/capabilities", get(handlers::capabilities))
        .route("/v1/transcribe", post(handlers::transcribe))
        .route("/v1/synthesize", post(handlers::synthesize))
        .route("/v1/vox/models", get(handlers::models))
        .route("/v1/stats", get(handlers::stats))
        .route("/health", get(handlers::health))
        .route("/health/ready", get(handlers::health_ready))
        // OpenAI-compatible surface
        .route("/v1/models", get(openai::list_models))
        .route("/v1/models/{id}", get(openai::retrieve_model))
        .route("/v1/audio/transcriptions", post(openai::transcriptions))
        .route("/v1/audio/speech", post(openai::speech))
        // WebSockets
        .route("/v1/listen", get(ws_listen::listen_ws))
        .route("/v1/speak", get(ws_speak::speak_ws))
        .route("/v1/converse", get(ws_converse::converse_ws))
        .route("/v1/live-talk", get(live_talk::live_talk_ws))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            state.limits.request_timeout,
        ))
        .layer(access::cors_layer(&state.access))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            access::enforce,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            middleware::track,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Serve `state` on `listener` until `shutdown` resolves, then drain.
///
/// On shutdown the server stops accepting connections, cancels
/// `state.shutdown` (WebSocket handlers answer with a `1001` close frame) and
/// waits up to `grace` for in-flight requests and open WebSocket sessions to
/// finish. Whatever is still running after that is dropped.
pub async fn serve(
    listener: TcpListener,
    state: Arc<ServerState>,
    grace: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let app = build_router(Arc::clone(&state));
    let token = state.shutdown.clone();

    let server = {
        let token = token.clone();
        async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move { token.cancelled().await })
                .await
        }
    };
    tokio::pin!(server);

    tokio::select! {
        result = &mut server => return result,
        () = shutdown => {}
        // Something else (a test, an embedding application) cancelled the token.
        () = token.cancelled() => {}
    }

    info!(
        grace_secs = grace.as_secs_f32(),
        "shutting down, draining connections"
    );
    token.cancel();

    let stats = Arc::clone(&state.stats);
    let drain = async {
        let result = server.await;
        while stats.ws_active.load(Ordering::Relaxed) > 0 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        result
    };
    match tokio::time::timeout(grace, drain).await {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!(
                open_websockets = state.stats.ws_active.load(Ordering::Relaxed),
                "grace period elapsed, dropping remaining connections"
            );
            Ok(())
        }
    }
}

/// Resolves on Ctrl-C or (on Unix) SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            // No signal handler available: never resolve, rely on SIGKILL.
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

/// Refuse a non-loopback bind unless the operator opted in.
///
/// vox has no authentication: anyone who can reach the port can use the
/// microphone pipeline, the LLM proxy and the speaker database.
pub fn check_bind_host(host: &str, allow_remote: bool) -> anyhow::Result<()> {
    if access::is_loopback_host(host) {
        return Ok(());
    }
    if !allow_remote {
        anyhow::bail!(
            "refusing to bind to non-loopback address '{host}': vox has no authentication, so \
             every device that can reach this port could transcribe, synthesize and chat through \
             it.\nIf that is what you want (trusted network, container, reverse proxy with its own \
             auth), set VOX_ALLOW_REMOTE=1 and start again."
        );
    }
    eprintln!();
    eprintln!("  WARNING: vox is listening on '{host}', reachable from other machines.");
    eprintln!("  WARNING: there is NO authentication. Only do this on a network you trust.");
    tracing::warn!(
        host,
        "serving on a non-loopback address without authentication"
    );
    Ok(())
}

/// Start the HTTP server on `host:port`.
///
/// Attempts to load STT (Whisper) and TTS (Kokoro/Piper) backends from
/// the models directory (see [`vox::paths::models_dir`]). Missing backends
/// are tolerated -- the corresponding endpoints return 503 until models are
/// available.
///
/// Language, STT model and Piper voice come from the `VOX_LANGUAGE`,
/// `VOX_STT_MODEL` and `VOX_PIPER_VOICE` environment variables; use
/// [`run_with_options`] to pass them explicitly.
///
/// # Arguments
/// - `cache_models`: deprecated, ignored (kept so `vox serve --cache-models` still parses)
#[allow(dead_code)]
pub async fn run(host: &str, port: u16, cache_models: bool) -> anyhow::Result<()> {
    run_with_options(host, port, cache_models, LoadOptions::from_env()).await
}

/// Like [`run`], with explicit model-loading options. Server options come
/// from the `VOX_*` environment variables (see [`ServerOptions::from_env`]).
#[allow(dead_code)]
pub async fn run_with_options(
    host: &str,
    port: u16,
    cache_models: bool,
    load_opts: LoadOptions,
) -> anyhow::Result<()> {
    run_with_config(
        host,
        port,
        cache_models,
        load_opts,
        ServerOptions::from_env(),
    )
    .await
}

/// Like [`run`], with explicit model-loading and server options.
pub async fn run_with_config(
    host: &str,
    port: u16,
    cache_models: bool,
    load_opts: LoadOptions,
    server_opts: ServerOptions,
) -> anyhow::Result<()> {
    // Fail before loading any model if the bind address is not allowed.
    check_bind_host(host, server_opts.allow_remote)?;

    // Single source of truth for model resolution, shared with the CLI:
    // `VOX_MODELS_DIR` → source checkout → per-user data directory.
    let models_dir = resolve_models_dir();
    info!(dir = %models_dir.display(), "server using models directory");

    // Hardware profile drives the STT candidate order and thread count.
    let profile = vox::system_profile::SystemProfile::detect();
    info!(profile = %profile.summary(), "detected hardware profile");

    // `--cache-models` used to build an LRU model cache that nothing ever
    // read from or wrote to. The cache is gone; the flag is accepted and
    // ignored so existing invocations keep working.
    if cache_models {
        tracing::warn!("--cache-models is deprecated and has no effect");
    }

    let (stt, stt_model_name, stt_model_size) = load_stt(&models_dir, &profile, &load_opts);

    let (tts, tts_model_name, tts_model_size) = load_tts(&models_dir, &load_opts).await;

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
            match try_load_piper(&models_dir, &load_opts) {
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

    let ollama_host = server_opts.resolved_ollama_host();

    #[cfg(feature = "diarization")]
    let (diarization, speaker_db) = load_diarization(&models_dir).await;

    // Build capability registry from already-loaded state.
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
        ..ServerState::with_options(host, port, &server_opts)
    });

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
    println!("    GET  /v1/vox/models  — list loaded backends");
    println!("    GET  /v1/stats       — server statistics");
    println!("    GET  /v1/models               — OpenAI-compatible model list");
    println!("    POST /v1/audio/transcriptions — OpenAI-compatible speech-to-text");
    println!("    POST /v1/audio/speech         — OpenAI-compatible text-to-speech");
    println!("    WS   /v1/listen       — real-time voice transcription");
    println!("    WS   /v1/speak        — streaming text-to-speech");
    println!("    WS   /v1/converse     — continuous voice chat (VAD+STT+LLM+TTS)");
    println!("    WS   /v1/live-talk    — barge-in voice chat (experimental)");
    println!("    GET  /health         — liveness (GET /health/ready — readiness)");
    println!();
    println!("  LLM: Ollama at {}", state.ollama_host);
    println!();

    let listener = TcpListener::bind(&addr).await?;
    serve(
        listener,
        state,
        server_opts.shutdown_grace,
        shutdown_signal(),
    )
    .await?;
    info!("server stopped");

    Ok(())
}
