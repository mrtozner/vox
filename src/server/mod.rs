//! HTTP server for the Vox voice AI framework.
//!
//! Provides a REST API for speech-to-text and text-to-speech operations.
//! Start with [`run`] from the `vox serve` CLI command.

pub mod error;
pub mod handlers;
pub mod live_talk;
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
///
/// `diarization` and `speaker_db` are loaded together from `speaker_encoder.onnx`.
/// Both are `None` if the model is missing or fails to load; `/v1/listen` then
/// falls back to speaker-less transcription with no behavioral change. This is
/// intentional — diarization is an opt-in biometric feature.
pub struct ServerState {
    pub stt: Option<Arc<dyn vox::traits::SttBackend>>,
    pub tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub conversation_tts: Option<Arc<dyn vox::traits::TtsBackend>>,
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
    pub model_cache: Option<vox::ModelCache>,
    pub capabilities: Arc<vox::CapabilityRegistry>,
    #[cfg(feature = "diarization")]
    pub diarization: Option<Arc<vox::DiarizationPipeline>>,
    #[cfg(feature = "diarization")]
    pub speaker_db: Option<Arc<vox::SpeakerDatabase>>,
}

/// Cumulative request counters.
pub struct ServerStats {
    pub requests: u64,
    pub transcriptions: u64,
    pub syntheses: u64,
}

/// Start the HTTP server on `host:port`.
///
/// Attempts to load STT (Whisper) and TTS (Kokoro/Piper) backends from
/// the repo-relative `models/` directory (see
/// [`crate::cli::models::models_dir`]). Missing backends are tolerated --
/// the corresponding endpoints return 503 until models are available.
///
/// # Arguments
/// - `cache_models`: Enable model caching (disabled by default for backwards compatibility)
pub async fn run(host: &str, port: u16, cache_models: bool) -> anyhow::Result<()> {
    // Use the single source of truth for model resolution -- repo-relative
    // by default, overridable via `VOX_MODELS_DIR`. Mirrors the logic in
    // `crate::cli::models::models_dir` so the server module is self-contained
    // when pulled into test crates via `#[path = ...]`.
    let models_dir = resolve_models_dir();
    info!(dir = %models_dir.display(), "server using models directory");

    // Initialize model cache if enabled
    let model_cache = if cache_models {
        info!("model caching enabled (--cache-models)");
        Some(vox::ModelCache::new(3)) // LRU cache with max 3 models
    } else {
        None
    };

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
        model_cache,
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
        .route("/v1/cache/stats", get(handlers::cache_stats))
        .route("/health", get(handlers::health))
        .route("/v1/listen", get(ws::listen_ws))
        .route("/v1/speak", get(ws::speak_ws))
        .route("/v1/converse", get(ws::converse_ws))
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
    println!("    GET  /v1/cache/stats — model cache statistics");
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

/// Resolve the models directory for the server.
///
/// Mirrors `crate::cli::models::models_dir` so this module can be pulled
/// into test crates via `#[path = ...]` without dragging in the CLI. The
/// resolution order is: `VOX_MODELS_DIR` env → `$CARGO_MANIFEST_DIR/models`
/// → walk upward from the running binary for a sibling `models/` dir.
fn resolve_models_dir() -> std::path::PathBuf {
    if let Ok(explicit) = std::env::var("VOX_MODELS_DIR") {
        let p = std::path::PathBuf::from(explicit);
        if !p.exists() {
            let _ = std::fs::create_dir_all(&p);
        }
        return p;
    }

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_models = manifest_dir.join("models");
    if repo_models.exists() {
        return repo_models;
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut cursor = exe.parent().map(|p| p.to_path_buf());
        while let Some(dir) = cursor {
            let candidate = dir.join("models");
            if candidate.exists() {
                return candidate;
            }
            cursor = dir.parent().map(|p| p.to_path_buf());
        }
    }

    let _ = std::fs::create_dir_all(&repo_models);
    repo_models
}

#[cfg(feature = "qwen3")]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let _ = models_dir;

    // Try Qwen3 first (if enabled)
    match vox::tts::Qwen3Backend::new().await {
        Ok(backend) => {
            info!("loaded qwen3 TTS backend");
            return (
                Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                Some("qwen3".to_string()),
                None, // Model size not applicable (auto-downloaded to HF cache)
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load qwen3 TTS backend, trying fallback");
        }
    }

    // Fallback: try Kokoro if Qwen3 is unavailable
    #[cfg(feature = "kokoro")]
    {
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
        }
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

#[cfg(all(feature = "kokoro", not(feature = "qwen3")))]
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

#[cfg(all(feature = "piper", not(any(feature = "kokoro", feature = "qwen3"))))]
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

#[cfg(not(any(feature = "kokoro", feature = "piper", feature = "qwen3")))]
async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro', 'piper', or 'qwen3' feature); models dir: {}",
        models_dir.display()
    );
    (None, None, None)
}

/// Scan models_dir/piper/ for any *.onnx.json config files and try to load the first valid one.
#[cfg(feature = "piper")]
type PiperLoadResult = Option<(
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
)>;

/// Resolve the path for the persistent speaker database.
///
/// Priority: `VOX_SPEAKER_DB` env var → OS data dir (`dirs::data_local_dir()/vox/speakers.db`)
/// → repo-relative fallback. The speaker DB is user data, not a model
/// artifact, so it lives outside `models/` by design.
#[cfg(feature = "diarization")]
fn speaker_db_path() -> std::path::PathBuf {
    if let Ok(custom) = std::env::var("VOX_SPEAKER_DB") {
        return std::path::PathBuf::from(custom);
    }
    if let Some(data_dir) = dirs::data_local_dir() {
        let vox_dir = data_dir.join("vox");
        let _ = std::fs::create_dir_all(&vox_dir);
        return vox_dir.join("speakers.db");
    }
    std::path::PathBuf::from("speakers.db")
}

/// Attempt to load the ECAPA-TDNN speaker encoder and hydrate the registry
/// from SQLite. Returns `(None, None)` if the model is missing so the server
/// keeps working with diarization disabled.
#[cfg(feature = "diarization")]
async fn load_diarization(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<vox::DiarizationPipeline>>,
    Option<Arc<vox::SpeakerDatabase>>,
) {
    let encoder_path = models_dir.join("speaker_encoder.onnx");
    if !encoder_path.exists() {
        tracing::warn!(
            "speaker_encoder.onnx not found in {}, diarization disabled",
            models_dir.display()
        );
        return (None, None);
    }

    let embedding = match vox::SpeakerEmbedding::new(&encoder_path) {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load speaker encoder, diarization disabled");
            return (None, None);
        }
    };

    let db_path = speaker_db_path();
    let db_str = db_path.to_string_lossy().into_owned();
    // Ensure the database file exists before sqlx tries to open it.
    if !db_path.exists() {
        if let Err(err) = std::fs::File::create(&db_path) {
            tracing::warn!(
                path = %db_path.display(),
                error = %err,
                "failed to create speaker database file, diarization disabled"
            );
            return (None, None);
        }
    }
    let db = match vox::SpeakerDatabase::open(db_str).await {
        Ok(db) => db,
        Err(err) => {
            tracing::warn!(
                path = %db_path.display(),
                error = %err,
                "failed to open speaker database, diarization disabled"
            );
            return (None, None);
        }
    };

    // Clear stale speakers if embedding pipeline version changed.
    // Different preprocessing (CMVN, thresholds) produces incompatible embeddings.
    const EMBEDDING_VERSION: &str = "v2-cmvn";
    let stored_version = db
        .get_system_preference("embedding_version")
        .await
        .unwrap_or_default()
        .unwrap_or_default();
    if stored_version != EMBEDDING_VERSION {
        tracing::info!(
            old = %stored_version,
            new = EMBEDDING_VERSION,
            "embedding version changed, clearing stale speakers"
        );
        if let Err(e) = db.clear_speakers().await {
            tracing::warn!(error = %e, "failed to clear stale speakers");
        }
        let _ = db
            .set_system_preference("embedding_version", EMBEDDING_VERSION)
            .await;
    }

    // Hydrate in-memory registry from persisted speakers.
    let mut registry = vox::SpeakerRegistry::new();
    match db.list_speakers().await {
        Ok(speakers) => {
            for speaker in speakers {
                let id = speaker.id.clone();
                let name = speaker.name.clone();
                if let Err(err) = registry.enroll(id.clone(), name, speaker.embedding) {
                    tracing::warn!(
                        speaker_id = %id,
                        error = %err,
                        "failed to hydrate speaker into registry"
                    );
                }
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "failed to list speakers from database");
        }
    }

    let pipeline = match vox::DiarizationPipelineBuilder::new()
        .embedding(embedding)
        .registry(registry)
        .auto_enroll(true)
        .build()
    {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!(error = %err, "failed to build diarization pipeline");
            return (None, None);
        }
    };

    info!(
        model = %encoder_path.display(),
        db = %db_path.display(),
        "loaded diarization pipeline"
    );
    (Some(Arc::new(pipeline)), Some(Arc::new(db)))
}

#[cfg(feature = "piper")]
fn try_load_piper(models_dir: &std::path::Path) -> PiperLoadResult {
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
