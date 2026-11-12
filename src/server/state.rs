//! Shared server state, runtime options and request counters.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use super::access::AccessPolicy;
use super::error::ServerError;

/// Default Ollama address when neither `--ollama-host` nor `VOX_OLLAMA_HOST` is set.
pub const DEFAULT_OLLAMA_HOST: &str = "localhost:11434";

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
///
/// Build one with [`ServerState::empty`] and struct-update syntax so a new
/// field only has to be given a default in one place:
///
/// ```ignore
/// let state = ServerState { stt: Some(stt), ..ServerState::empty() };
/// ```
pub struct ServerState {
    pub stt: Option<Arc<dyn vox::traits::SttBackend>>,
    pub tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub conversation_tts: Option<Arc<dyn vox::traits::TtsBackend>>,
    pub streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>>,
    pub vad_model_path: Option<std::path::PathBuf>,
    /// Lock-free counters; see [`ServerStats`].
    pub stats: Arc<ServerStats>,
    pub start_time: std::time::Instant,
    /// Ollama address as `host:port`. Set by the operator only
    /// (`--ollama-host` / `VOX_OLLAMA_HOST`), never by a request.
    pub ollama_host: String,
    pub http_client: reqwest::Client,
    pub stt_model_name: Option<String>,
    pub stt_model_size: Option<u64>,
    pub tts_model_name: Option<String>,
    pub tts_model_size: Option<u64>,
    pub capabilities: Arc<vox::CapabilityRegistry>,
    /// Which `Host` / `Origin` values may talk to this server.
    pub access: AccessPolicy,
    /// Request limits (text length, timeouts).
    pub limits: Limits,
    /// Bounds concurrent STT/TTS inference across REST and WebSocket paths.
    /// Use [`ServerState::inference_permit`] (REST, gives up with 503) or
    /// [`ServerState::inference_permit_wait`] (WebSocket, waits).
    pub inference: Arc<Semaphore>,
    /// Configured number of inference slots (the semaphore's capacity).
    pub inference_slots: usize,
    /// Cancelled when the server starts shutting down. WebSocket handlers
    /// select on it and send [`shutdown_close_message`] before returning.
    pub shutdown: CancellationToken,
    #[cfg(feature = "diarization")]
    pub diarization: Option<Arc<vox::DiarizationPipeline>>,
    #[cfg(feature = "diarization")]
    pub speaker_db: Option<Arc<vox::SpeakerDatabase>>,
}

impl ServerState {
    /// A state with no backends loaded and default options, bound to
    /// `127.0.0.1:3000` for access-control purposes.
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self::with_options("127.0.0.1", 3000, &ServerOptions::default())
    }

    /// A state with no backends loaded, configured from `opts` for a server
    /// bound to `host:port`.
    pub fn with_options(host: &str, port: u16, opts: &ServerOptions) -> Self {
        Self {
            stt: None,
            tts: None,
            conversation_tts: None,
            streaming_stt: None,
            vad_model_path: None,
            stats: Arc::new(ServerStats::default()),
            start_time: std::time::Instant::now(),
            ollama_host: opts.resolved_ollama_host(),
            http_client: reqwest::Client::new(),
            stt_model_name: None,
            stt_model_size: None,
            tts_model_name: None,
            tts_model_size: None,
            capabilities: Arc::new(vox::CapabilityRegistry::default()),
            access: AccessPolicy::new(host, port)
                .with_extra_hosts(opts.allowed_hosts.iter().cloned())
                .with_extra_origins(opts.allowed_origins.iter().cloned()),
            limits: Limits {
                max_text_chars: opts.max_text_chars,
                request_timeout: opts.request_timeout,
                inference_wait: opts.inference_wait,
            },
            inference: Arc::new(Semaphore::new(opts.max_concurrent_inference.max(1))),
            inference_slots: opts.max_concurrent_inference.max(1),
            shutdown: CancellationToken::new(),
            #[cfg(feature = "diarization")]
            diarization: None,
            #[cfg(feature = "diarization")]
            speaker_db: None,
        }
    }

    /// Acquire an inference slot for a REST request.
    ///
    /// Waits up to [`Limits::inference_wait`]; after that the request fails
    /// with `503 server_busy` so a saturated server sheds load instead of
    /// piling up requests.
    pub async fn inference_permit(&self) -> Result<OwnedSemaphorePermit, ServerError> {
        let acquire = Arc::clone(&self.inference).acquire_owned();
        match tokio::time::timeout(self.limits.inference_wait, acquire).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(ServerError::service_unavailable("server is shutting down")),
            Err(_) => Err(ServerError::busy(
                "all inference slots are busy, retry shortly",
            )),
        }
    }

    /// Acquire an inference slot for a WebSocket session, waiting as long as
    /// it takes. Returns `None` only when the server is shutting down.
    pub async fn inference_permit_wait(&self) -> Option<OwnedSemaphorePermit> {
        tokio::select! {
            permit = Arc::clone(&self.inference).acquire_owned() => permit.ok(),
            () = self.shutdown.cancelled() => None,
        }
    }

    /// Register an open WebSocket session. Hold the guard for the lifetime of
    /// the connection: shutdown waits (up to the grace period) until every
    /// guard is dropped.
    pub fn ws_session(&self) -> WsSessionGuard {
        self.stats.ws_total.fetch_add(1, Ordering::Relaxed);
        self.stats.ws_active.fetch_add(1, Ordering::Relaxed);
        WsSessionGuard {
            stats: Arc::clone(&self.stats),
        }
    }
}

/// RAII marker for an open WebSocket session; see [`ServerState::ws_session`].
pub struct WsSessionGuard {
    stats: Arc<ServerStats>,
}

impl Drop for WsSessionGuard {
    fn drop(&mut self) {
        self.stats.ws_active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The close frame WebSocket handlers send when the server shuts down
/// (`1001 Going Away`).
pub fn shutdown_close_message() -> axum::extract::ws::Message {
    axum::extract::ws::Message::Close(Some(axum::extract::ws::CloseFrame {
        code: axum::extract::ws::close_code::AWAY,
        reason: "server shutting down".into(),
    }))
}

/// Per-request limits.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Longest text (in characters) accepted by the synthesis endpoints.
    pub max_text_chars: usize,
    /// Upper bound on how long a REST request may run.
    pub request_timeout: Duration,
    /// How long a REST request waits for an inference slot before `503`.
    pub inference_wait: Duration,
}

/// Operator-controlled server options (`vox serve` flags and `VOX_*` variables).
#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// Ollama address (`host:port`, a scheme prefix is tolerated).
    /// `None` falls back to [`DEFAULT_OLLAMA_HOST`].
    pub ollama_host: Option<String>,
    /// Allow binding to a non-loopback address. The server has no
    /// authentication, so this is refused unless explicitly enabled.
    pub allow_remote: bool,
    /// Extra `Host` values to accept (`name` or `name:port`), e.g. the public
    /// name of a reverse proxy.
    pub allowed_hosts: Vec<String>,
    /// Extra `Origin` values to accept verbatim (`https://app.example`).
    pub allowed_origins: Vec<String>,
    /// Concurrent STT/TTS inferences across all clients.
    pub max_concurrent_inference: usize,
    /// See [`Limits::request_timeout`].
    pub request_timeout: Duration,
    /// See [`Limits::inference_wait`].
    pub inference_wait: Duration,
    /// How long shutdown waits for in-flight requests and open WebSockets.
    pub shutdown_grace: Duration,
    /// See [`Limits::max_text_chars`].
    pub max_text_chars: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            ollama_host: None,
            allow_remote: false,
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
            max_concurrent_inference: 4,
            request_timeout: Duration::from_secs(300),
            inference_wait: Duration::from_secs(30),
            shutdown_grace: Duration::from_secs(5),
            max_text_chars: 10_000,
        }
    }
}

impl ServerOptions {
    /// Read the `VOX_*` environment variables:
    ///
    /// | Variable | Meaning |
    /// |---|---|
    /// | `VOX_OLLAMA_HOST` | Ollama address |
    /// | `VOX_ALLOW_REMOTE` | `1`/`true` permits a non-loopback bind |
    /// | `VOX_ALLOWED_HOSTS` | comma-separated extra `Host` values |
    /// | `VOX_ALLOWED_ORIGINS` | comma-separated extra `Origin` values |
    /// | `VOX_MAX_CONCURRENT_INFERENCE` | inference slots (default 4) |
    /// | `VOX_REQUEST_TIMEOUT_SECS` | REST request timeout (default 300) |
    /// | `VOX_SHUTDOWN_GRACE_SECS` | shutdown drain period (default 5) |
    /// | `VOX_MAX_TEXT_CHARS` | synthesis text limit (default 10000) |
    pub fn from_env() -> Self {
        fn var(key: &str) -> Option<String> {
            std::env::var(key)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        }
        fn list(key: &str) -> Vec<String> {
            var(key)
                .map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default()
        }
        fn num<T: std::str::FromStr>(key: &str) -> Option<T> {
            let raw = var(key)?;
            match raw.parse() {
                Ok(v) => Some(v),
                Err(_) => {
                    tracing::warn!("ignoring {key}={raw}: not a valid number");
                    None
                }
            }
        }

        let d = Self::default();
        Self {
            ollama_host: var("VOX_OLLAMA_HOST"),
            allow_remote: var("VOX_ALLOW_REMOTE")
                .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes")),
            allowed_hosts: list("VOX_ALLOWED_HOSTS"),
            allowed_origins: list("VOX_ALLOWED_ORIGINS"),
            max_concurrent_inference: num::<usize>("VOX_MAX_CONCURRENT_INFERENCE")
                .filter(|n| *n > 0)
                .unwrap_or(d.max_concurrent_inference),
            request_timeout: num::<u64>("VOX_REQUEST_TIMEOUT_SECS")
                .filter(|n| *n > 0)
                .map(Duration::from_secs)
                .unwrap_or(d.request_timeout),
            inference_wait: d.inference_wait,
            shutdown_grace: num::<u64>("VOX_SHUTDOWN_GRACE_SECS")
                .map(Duration::from_secs)
                .unwrap_or(d.shutdown_grace),
            max_text_chars: num::<usize>("VOX_MAX_TEXT_CHARS")
                .filter(|n| *n > 0)
                .unwrap_or(d.max_text_chars),
        }
    }

    /// The Ollama address as bare `host:port`.
    pub fn resolved_ollama_host(&self) -> String {
        normalize_ollama_host(self.ollama_host.as_deref().unwrap_or(DEFAULT_OLLAMA_HOST))
    }
}

/// Reduce an Ollama address to `host:port`: the rest of the server builds
/// `http://{host}/api/...` URLs, so a scheme, path or trailing slash typed by
/// the operator would otherwise produce a broken URL.
pub fn normalize_ollama_host(raw: &str) -> String {
    let s = raw.trim();
    let s = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or(s);
    let s = s.split('/').next().unwrap_or(s).trim();
    if s.is_empty() {
        DEFAULT_OLLAMA_HOST.to_string()
    } else {
        s.to_string()
    }
}

/// Cumulative request counters.
///
/// All fields are atomics: there is no lock to poison, and a panicking
/// handler can never take the statistics down with it.
#[derive(Debug, Default)]
pub struct ServerStats {
    /// HTTP requests served (excluding `/v1/stats` itself).
    pub requests: AtomicU64,
    /// Responses with a 4xx status.
    pub client_errors: AtomicU64,
    /// Responses with a 5xx status.
    pub server_errors: AtomicU64,
    /// Successful transcriptions.
    pub transcriptions: AtomicU64,
    /// Successful syntheses.
    pub syntheses: AtomicU64,
    /// Successful `/v1/chat` completions.
    pub chats: AtomicU64,
    /// Total audio transcribed, in milliseconds.
    pub transcribe_audio_ms: AtomicU64,
    /// Total time spent in STT, in milliseconds.
    pub transcribe_ms: AtomicU64,
    /// Total audio synthesized, in milliseconds.
    pub synthesize_audio_ms: AtomicU64,
    /// Total time spent in TTS, in milliseconds.
    pub synthesize_ms: AtomicU64,
    /// Total time spent waiting on the LLM, in milliseconds.
    pub chat_ms: AtomicU64,
    /// WebSocket sessions currently open.
    pub ws_active: AtomicU64,
    /// WebSocket sessions opened since start.
    pub ws_total: AtomicU64,
}

impl ServerStats {
    /// Record one successful transcription.
    pub fn record_transcription(&self, audio_ms: u64, elapsed: Duration) {
        self.transcriptions.fetch_add(1, Ordering::Relaxed);
        self.transcribe_audio_ms
            .fetch_add(audio_ms, Ordering::Relaxed);
        self.transcribe_ms
            .fetch_add(elapsed.as_millis() as u64, Ordering::Relaxed);
    }

    /// Record one successful synthesis.
    pub fn record_synthesis(&self, audio_ms: u64, elapsed: Duration) {
        self.syntheses.fetch_add(1, Ordering::Relaxed);
        self.synthesize_audio_ms
            .fetch_add(audio_ms, Ordering::Relaxed);
        self.synthesize_ms
            .fetch_add(elapsed.as_millis() as u64, Ordering::Relaxed);
    }

    /// Record one successful chat completion.
    pub fn record_chat(&self, elapsed: Duration) {
        self.chats.fetch_add(1, Ordering::Relaxed);
        self.chat_ms
            .fetch_add(elapsed.as_millis() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_host_is_reduced_to_host_port() {
        assert_eq!(normalize_ollama_host("localhost:11434"), "localhost:11434");
        assert_eq!(
            normalize_ollama_host("http://10.0.0.5:11434/"),
            "10.0.0.5:11434"
        );
        assert_eq!(
            normalize_ollama_host(" https://gpu.lan:443/api "),
            "gpu.lan:443"
        );
        assert_eq!(normalize_ollama_host(""), DEFAULT_OLLAMA_HOST);
    }

    #[test]
    fn explicit_ollama_host_wins_over_default() {
        let opts = ServerOptions {
            ollama_host: Some("http://box:1234".into()),
            ..ServerOptions::default()
        };
        assert_eq!(opts.resolved_ollama_host(), "box:1234");
        assert_eq!(
            ServerOptions::default().resolved_ollama_host(),
            DEFAULT_OLLAMA_HOST
        );
    }

    #[tokio::test]
    async fn rest_permit_gives_up_when_saturated() {
        let state = ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                max_concurrent_inference: 1,
                inference_wait: Duration::from_millis(20),
                ..ServerOptions::default()
            },
        );
        let held = state.inference_permit().await.expect("first permit");
        assert!(state.inference_permit().await.is_err());
        drop(held);
        assert!(state.inference_permit().await.is_ok());
    }

    #[tokio::test]
    async fn ws_permit_waits_and_yields_to_shutdown() {
        let state = Arc::new(ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                max_concurrent_inference: 1,
                ..ServerOptions::default()
            },
        ));
        let held = state.inference_permit_wait().await.expect("first permit");

        let waiter = {
            let state = Arc::clone(&state);
            tokio::spawn(async move { state.inference_permit_wait().await.is_some() })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "second session must wait");
        drop(held);
        assert!(waiter.await.unwrap(), "waiter gets the freed slot");

        let held = state.inference_permit_wait().await.expect("permit");
        state.shutdown.cancel();
        assert!(state.inference_permit_wait().await.is_none());
        drop(held);
    }

    #[test]
    fn ws_guard_tracks_active_sessions() {
        let state = ServerState::empty();
        let a = state.ws_session();
        let b = state.ws_session();
        assert_eq!(state.stats.ws_active.load(Ordering::Relaxed), 2);
        drop(a);
        assert_eq!(state.stats.ws_active.load(Ordering::Relaxed), 1);
        drop(b);
        assert_eq!(state.stats.ws_active.load(Ordering::Relaxed), 0);
        assert_eq!(state.stats.ws_total.load(Ordering::Relaxed), 2);
    }
}
