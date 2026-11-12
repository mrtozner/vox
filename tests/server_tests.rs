//! Integration tests for the Vox HTTP API server.
#![allow(dead_code)]
//!
//! The tests drive the production router ([`server::build_router`], every
//! middleware layer included) with mock STT/TTS backends. Most use
//! `tower::ServiceExt::oneshot`; the WebSocket and shutdown tests bind a real
//! loopback listener and speak HTTP by hand.
//!
//! Run with: `cargo test --test server_tests --features server`

use vox::server;

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tower::ServiceExt;

use server::{ServerOptions, ServerState};

// ---------------------------------------------------------------------------
// Mock backends
// ---------------------------------------------------------------------------

/// What the STT backend was handed: (samples, sample_rate, channels).
type SeenAudio = Arc<Mutex<Vec<(usize, u32, u16)>>>;

#[derive(Default)]
struct MockStt {
    seen: SeenAudio,
    delay: Duration,
}

#[async_trait]
impl vox::SttBackend for MockStt {
    async fn transcribe(&self, audio: &vox::Utterance) -> Result<vox::SttResult, vox::VoxError> {
        self.seen.lock().unwrap().push((
            audio.audio.samples.len(),
            audio.audio.sample_rate,
            audio.audio.channels,
        ));
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        Ok(vox::SttResult {
            text: "hello world".into(),
            language: Some("en".into()),
            duration_ms: audio.duration_ms,
            processing_time_ms: 1,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        })
    }
}

struct MockTts {
    name: &'static str,
    sample_rate: u32,
    /// Samples produced per request.
    samples: usize,
    delay: Duration,
    fail: bool,
    seen_voices: Arc<Mutex<Vec<Option<String>>>>,
}

impl Default for MockTts {
    fn default() -> Self {
        Self {
            name: "mocktts",
            sample_rate: 16_000,
            samples: 16_000,
            delay: Duration::ZERO,
            fail: false,
            seen_voices: Arc::default(),
        }
    }
}

#[async_trait]
impl vox::TtsBackend for MockTts {
    async fn synthesize(&self, request: &vox::TtsRequest) -> Result<vox::TtsOutput, vox::VoxError> {
        self.seen_voices.lock().unwrap().push(request.voice.clone());
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        if self.fail {
            return Err(vox::VoxError::Tts("synthesis exploded".into()));
        }
        Ok(vox::TtsOutput::from_mono(
            vec![0.25; self.samples],
            self.sample_rate,
        ))
    }

    fn list_voices(&self) -> Vec<vox::types::VoiceInfo> {
        vec![vox::types::VoiceInfo {
            id: "af_heart".into(),
            name: "Heart".into(),
            gender: "female".into(),
            language: "en-US".into(),
            accent: "American".into(),
        }]
    }

    fn backend_name(&self) -> &str {
        self.name
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HOST: &str = "127.0.0.1:3000";

fn no_proxy_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn empty_state() -> ServerState {
    ServerState {
        http_client: no_proxy_client(),
        ..ServerState::empty()
    }
}

fn app(state: ServerState) -> Router {
    server::build_router(Arc::new(state))
}

fn get_req(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header(header::HOST, HOST)
        .body(Body::empty())
        .unwrap()
}

fn post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::HOST, HOST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn post_bytes(uri: &str, body: Vec<u8>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::HOST, HOST)
        .body(Body::from(body))
        .unwrap()
}

async fn body_bytes(response: axum::http::Response<Body>) -> Vec<u8> {
    response
        .into_body()
        .collect()
        .await
        .expect("failed to collect response body")
        .to_bytes()
        .to_vec()
}

async fn body_json(response: axum::http::Response<Body>) -> serde_json::Value {
    let bytes = body_bytes(response).await;
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "response body is not valid JSON ({e}): {:?}",
            String::from_utf8_lossy(&bytes)
        )
    })
}

/// Assert the standard error body and return it.
async fn assert_error(
    response: axum::http::Response<Body>,
    status: StatusCode,
    kind: &str,
    code: &str,
) -> serde_json::Value {
    assert_eq!(response.status(), status);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/json",
        "errors are always JSON"
    );
    let json = body_json(response).await;
    let message = json["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("error.message missing: {json}"));
    assert!(!message.is_empty(), "error.message is empty: {json}");
    assert_eq!(json["error"]["type"], kind, "{json}");
    assert_eq!(json["error"]["code"], code, "{json}");
    assert!(json["error"].get("param").is_some(), "{json}");
    assert_eq!(
        json["message"], json["error"]["message"],
        "top-level message mirrors error.message"
    );
    json
}

/// Interleaved 16-bit WAV with `frames` frames of a quiet tone.
fn wav_i16(sample_rate: u32, channels: u16, frames: usize) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for i in 0..frames {
            let v = ((i as f32 * 0.05).sin() * 8000.0) as i16;
            for _ in 0..channels {
                writer.write_sample(v).unwrap();
            }
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

/// A WAV header claiming an arbitrary format, followed by `data`.
fn raw_wav(format_tag: u16, channels: u16, sample_rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
    let block_align = channels.wrapping_mul(bits / 8);
    let byte_rate = sample_rate.wrapping_mul(u32::from(block_align));
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

const BOUNDARY: &str = "vox-test-boundary";

/// Build a multipart/form-data request. A field is `(name, filename, bytes)`.
fn multipart(uri: &str, fields: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
    let mut body = Vec::new();
    for (name, filename, value) in fields {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        match filename {
            Some(f) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::HOST, HOST)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

/// A stand-in for Ollama on a loopback port. Returns its `host:port` and a
/// counter of `/api/generate` calls.
async fn fake_ollama(
    generate_status: StatusCode,
    generate_body: &'static str,
    tags_delay: Duration,
) -> (String, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let router = Router::new()
        .route(
            "/api/generate",
            post(move || {
                let counter = Arc::clone(&counter);
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    (generate_status, generate_body)
                }
            }),
        )
        .route(
            "/api/tags",
            get(move || async move {
                tokio::time::sleep(tags_delay).await;
                axum::Json(serde_json::json!({
                    "models": [{ "name": "llama3.2:latest", "size": 1, "details": { "family": "llama" } }]
                }))
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (addr.to_string(), hits)
}

/// A loopback port nothing listens on.
async fn dead_port() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr.to_string()
}

// ---------------------------------------------------------------------------
// Health, readiness, stats
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_reports_status_and_version() {
    let response = app(empty_state())
        .oneshot(get_req("/health"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-content-type-options"], "nosniff");
    let json = body_json(response).await;
    assert_eq!(json["status"], "ok");
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn ready_is_503_without_backends() {
    let response = app(empty_state())
        .oneshot(get_req("/health/ready"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(response).await;
    assert_eq!(json["status"], "not_ready");
    assert_eq!(json["stt"], false);
    assert_eq!(json["tts"], false);
    assert_eq!(json["vad"], false);
    assert_eq!(json["missing"], serde_json::json!(["stt", "tts", "vad"]));
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn ready_is_200_with_backends_and_names_what_is_missing() {
    let state = ServerState {
        stt: Some(Arc::new(MockStt::default())),
        tts: Some(Arc::new(MockTts::default())),
        ..empty_state()
    };
    let response = app(state).oneshot(get_req("/health/ready")).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body_json(response).await["missing"],
        serde_json::json!(["vad"])
    );

    let state = ServerState {
        stt: Some(Arc::new(MockStt::default())),
        tts: Some(Arc::new(MockTts::default())),
        vad_model_path: Some("silero_vad.onnx".into()),
        ..empty_state()
    };
    let response = app(state).oneshot(get_req("/health/ready")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["status"], "ready");
    assert_eq!(json["missing"], serde_json::json!([]));
}

#[tokio::test]
async fn stats_does_not_count_itself() {
    let app = app(empty_state());

    for _ in 0..3 {
        let response = app.clone().oneshot(get_req("/v1/stats")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(
            json["requests"], 0,
            "polling stats must not move the counter"
        );
    }

    app.clone().oneshot(get_req("/health")).await.unwrap();
    app.clone().oneshot(get_req("/health")).await.unwrap();
    app.clone().oneshot(get_req("/v1/nope")).await.unwrap();

    let json = body_json(app.clone().oneshot(get_req("/v1/stats")).await.unwrap()).await;
    assert_eq!(json["requests"], 3);
    assert_eq!(json["client_errors"], 1);
    assert_eq!(json["server_errors"], 0);
    assert_eq!(json["transcriptions"], 0);
    assert_eq!(json["syntheses"], 0);
    assert!(json["uptime_secs"].is_number());
    assert_eq!(json["inference_slots"], 4);
    assert_eq!(json["inference_slots_free"], 4);
    assert_eq!(json["ws_active"], 0);
}

#[tokio::test]
async fn stats_report_latency_counters() {
    let state = ServerState {
        stt: Some(Arc::new(MockStt {
            delay: Duration::from_millis(20),
            ..MockStt::default()
        })),
        tts: Some(Arc::new(MockTts {
            delay: Duration::from_millis(20),
            ..MockTts::default()
        })),
        ..empty_state()
    };
    let app = app(state);

    let response = app
        .clone()
        .oneshot(post_bytes("/v1/transcribe", wav_i16(16_000, 1, 32_000)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // A failed request counts as a request, not as a transcription.
    let response = app
        .clone()
        .oneshot(post_bytes("/v1/transcribe", b"garbage".to_vec()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let json = body_json(app.oneshot(get_req("/v1/stats")).await.unwrap()).await;
    assert_eq!(json["requests"], 3);
    assert_eq!(json["transcriptions"], 1);
    assert_eq!(json["syntheses"], 1);
    assert_eq!(json["client_errors"], 1);
    let latency = &json["latency"];
    assert_eq!(latency["transcribe_audio_ms"], 2000);
    assert!(latency["transcribe_ms"].as_u64().unwrap() >= 20);
    assert!(latency["transcribe_avg_ms"].as_u64().unwrap() >= 20);
    assert!(latency["transcribe_rtf"].as_f64().unwrap() > 0.0);
    assert_eq!(latency["synthesize_audio_ms"], 1000);
    assert!(latency["synthesize_ms"].as_u64().unwrap() >= 20);
    assert!(latency["chat_avg_ms"].is_null());
}

// ---------------------------------------------------------------------------
// Web UI
// ---------------------------------------------------------------------------

#[tokio::test]
async fn index_serves_html_with_a_content_security_policy() {
    let response = app(empty_state()).oneshot(get_req("/")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let headers = response.headers().clone();
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("text/html")
    );
    let csp = headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap();
    assert!(csp.contains("default-src 'self'"), "{csp}");
    assert!(csp.contains("frame-ancestors 'none'"), "{csp}");
    assert!(
        csp.contains(&format!("connect-src 'self' ws://{HOST} wss://{HOST}")),
        "{csp}"
    );
    assert!(!csp.contains("https:"), "no remote sources: {csp}");
    assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");

    let html = String::from_utf8(body_bytes(response).await).unwrap();
    assert!(html.contains("<html") || html.contains("<!DOCTYPE"));
}

// ---------------------------------------------------------------------------
// Origin / Host enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cross_origin_requests_are_rejected_on_every_route() {
    let app = app(empty_state());
    for (method, uri) in [
        ("GET", "/health"),
        ("GET", "/"),
        ("GET", "/v1/stats"),
        ("POST", "/v1/chat"),
        ("POST", "/v1/synthesize"),
        ("POST", "/v1/audio/speech"),
        ("GET", "/v1/live-talk"),
        ("OPTIONS", "/v1/chat"),
    ] {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::HOST, HOST)
            .header(header::ORIGIN, "https://evil.example")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"text":"hi","input":"hi"}"#))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none(),
            "{method} {uri}"
        );
        assert_error(
            response,
            StatusCode::FORBIDDEN,
            "permission_error",
            "origin_not_allowed",
        )
        .await;
    }
}

#[tokio::test]
async fn dns_rebinding_host_is_rejected() {
    let request = Request::builder()
        .uri("/v1/stats")
        .header(header::HOST, "evil.example:3000")
        .body(Body::empty())
        .unwrap();
    let response = app(empty_state()).oneshot(request).await.unwrap();
    assert_error(
        response,
        StatusCode::FORBIDDEN,
        "permission_error",
        "host_not_allowed",
    )
    .await;

    // Right name, wrong port.
    let request = Request::builder()
        .uri("/v1/stats")
        .header(header::HOST, "localhost:9999")
        .body(Body::empty())
        .unwrap();
    let response = app(empty_state()).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn same_origin_and_origin_less_requests_are_accepted() {
    let app = app(empty_state());

    // No Origin (curl, SDKs, same-origin GET).
    let response = app.clone().oneshot(get_req("/health")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none()
    );

    // No Host and no Origin: a bare non-browser client.
    let request = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    for (host, origin) in [
        ("127.0.0.1:3000", "http://127.0.0.1:3000"),
        ("localhost:3000", "http://localhost:3000"),
        ("[::1]:3000", "http://[::1]:3000"),
    ] {
        let request = Request::builder()
            .uri("/health")
            .header(header::HOST, host)
            .header(header::ORIGIN, origin)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{origin}");
        assert_eq!(
            response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            origin
        );
    }
}

#[tokio::test]
async fn operator_allowed_origin_passes_preflight() {
    let state = ServerState {
        http_client: no_proxy_client(),
        ..ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                allowed_origins: vec!["https://app.example".into()],
                ..ServerOptions::default()
            },
        )
    };
    let request = Request::builder()
        .method("OPTIONS")
        .uri("/v1/synthesize")
        .header(header::HOST, HOST)
        .header(header::ORIGIN, "https://app.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type")
        .body(Body::empty())
        .unwrap();
    let response = app(state).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        "https://app.example"
    );
}

#[test]
fn non_loopback_bind_needs_an_explicit_opt_in() {
    for host in ["127.0.0.1", "localhost", "::1"] {
        assert!(server::check_bind_host(host, false).is_ok(), "{host}");
    }
    for host in ["0.0.0.0", "::", "192.168.1.20"] {
        let err = server::check_bind_host(host, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VOX_ALLOW_REMOTE"), "{err}");
        assert!(server::check_bind_host(host, true).is_ok(), "{host}");
    }
}

// ---------------------------------------------------------------------------
// WebSocket upgrade and shutdown (real sockets)
// ---------------------------------------------------------------------------

struct TestServer {
    addr: std::net::SocketAddr,
    state: Arc<ServerState>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

/// Start the real server (TTS mock loaded, so `/v1/speak` upgrades) on a
/// loopback port.
async fn start_server(grace: Duration) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(ServerState {
        tts: Some(Arc::new(MockTts::default())),
        http_client: no_proxy_client(),
        ..ServerState::with_options("127.0.0.1", addr.port(), &ServerOptions::default())
    });
    let task = tokio::spawn(server::serve(
        listener,
        Arc::clone(&state),
        grace,
        std::future::pending(),
    ));
    TestServer { addr, state, task }
}

/// Send a WebSocket upgrade request; return the stream, the HTTP status and
/// any bytes that arrived after the response head.
async fn ws_upgrade(
    addr: std::net::SocketAddr,
    path: &str,
    origin: Option<&str>,
) -> (TcpStream, u16, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         Sec-WebSocket-Version: 13\r\n"
    );
    if let Some(origin) = origin {
        request.push_str(&format!("Origin: {origin}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();

    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let mut chunk = [0u8; 1024];
        let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
            .await
            .expect("timed out waiting for the upgrade response")
            .unwrap();
        assert!(n > 0, "connection closed before the response head");
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("bad response head: {head}"));
    (stream, status, buf[head_end..].to_vec())
}

/// Read one (short, unmasked) server frame: returns (opcode, payload).
async fn read_ws_frame(stream: &mut TcpStream, mut pending: Vec<u8>) -> (u8, Vec<u8>) {
    async fn fill(stream: &mut TcpStream, pending: &mut Vec<u8>, want: usize) {
        while pending.len() < want {
            let mut chunk = [0u8; 256];
            let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
                .await
                .expect("timed out waiting for a WebSocket frame")
                .unwrap();
            assert!(n > 0, "connection closed without a WebSocket frame");
            pending.extend_from_slice(&chunk[..n]);
        }
    }
    fill(stream, &mut pending, 2).await;
    let opcode = pending[0] & 0x0f;
    let len = usize::from(pending[1] & 0x7f);
    assert!(len < 126, "test helper only reads short frames");
    fill(stream, &mut pending, 2 + len).await;
    (opcode, pending[2..2 + len].to_vec())
}

#[tokio::test]
async fn ws_upgrade_from_foreign_origin_is_403() {
    let srv = start_server(Duration::from_secs(1)).await;
    for path in ["/v1/speak", "/v1/live-talk", "/v1/listen", "/v1/converse"] {
        let (_stream, status, _) = ws_upgrade(srv.addr, path, Some("https://evil.example")).await;
        assert_eq!(status, 403, "{path}");
    }
    assert_eq!(srv.state.stats.ws_total.load(Ordering::Relaxed), 0);
    srv.task.abort();
}

#[tokio::test]
async fn ws_upgrade_without_origin_and_from_own_origin_is_accepted() {
    let srv = start_server(Duration::from_secs(1)).await;

    let (_a, status, _) = ws_upgrade(srv.addr, "/v1/speak", None).await;
    assert_eq!(status, 101, "origin-less client on loopback");

    let own = format!("http://127.0.0.1:{}", srv.addr.port());
    let (_b, status, _) = ws_upgrade(srv.addr, "/v1/speak", Some(&own)).await;
    assert_eq!(status, 101, "{own}");

    let localhost = format!("http://localhost:{}", srv.addr.port());
    let (_c, status, _) = ws_upgrade(srv.addr, "/v1/speak", Some(&localhost)).await;
    assert_eq!(status, 101, "{localhost}");

    // Same host, different port is a different origin.
    let other = format!("http://127.0.0.1:{}", srv.addr.port().wrapping_add(1));
    let (_d, status, _) = ws_upgrade(srv.addr, "/v1/speak", Some(&other)).await;
    assert_eq!(status, 403, "{other}");

    srv.task.abort();
}

#[tokio::test]
async fn shutdown_sends_close_frame_to_open_websockets() {
    let srv = start_server(Duration::from_secs(5)).await;

    let (mut stream, status, pending) = ws_upgrade(srv.addr, "/v1/speak", None).await;
    assert_eq!(status, 101);

    // The session is registered once the handler runs.
    for _ in 0..100 {
        if srv.state.stats.ws_active.load(Ordering::Relaxed) == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(srv.state.stats.ws_active.load(Ordering::Relaxed), 1);

    // Shutdown signal.
    srv.state.shutdown.cancel();

    let (opcode, payload) = read_ws_frame(&mut stream, pending).await;
    assert_eq!(opcode, 0x8, "expected a close frame");
    assert_eq!(
        u16::from_be_bytes([payload[0], payload[1]]),
        1001,
        "close code is Going Away"
    );

    // The server drains and stops well inside the grace period.
    let result = tokio::time::timeout(Duration::from_secs(3), srv.task)
        .await
        .expect("server did not stop after shutdown")
        .unwrap();
    assert!(result.is_ok());
    assert_eq!(srv.state.stats.ws_active.load(Ordering::Relaxed), 0);
    assert!(
        TcpStream::connect(srv.addr).await.is_err(),
        "listener is closed"
    );
}

#[tokio::test]
async fn shutdown_gives_up_after_the_grace_period() {
    // A session that never ends: hold the guard ourselves.
    let srv = start_server(Duration::from_millis(150)).await;
    let _stuck = srv.state.ws_session();

    srv.state.shutdown.cancel();
    let started = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(3), srv.task)
        .await
        .expect("grace period was not enforced")
        .unwrap()
        .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(140));
}

// ---------------------------------------------------------------------------
// /v1/transcribe
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transcribe_resamples_44k_stereo_to_16k_mono() {
    let seen = SeenAudio::default();
    let state = ServerState {
        stt: Some(Arc::new(MockStt {
            seen: Arc::clone(&seen),
            ..MockStt::default()
        })),
        ..empty_state()
    };

    // 1.5 s of 44.1 kHz stereo: 66150 frames, 132300 interleaved samples.
    let response = app(state)
        .oneshot(post_bytes("/v1/transcribe", wav_i16(44_100, 2, 66_150)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["text"], "hello world");
    assert_eq!(json["duration_ms"], 1500);

    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        vec![(24_000, 16_000, 1)],
        "STT must receive round(66150 * 16000 / 44100) = 24000 mono samples at 16 kHz"
    );
}

#[tokio::test]
async fn transcribe_passes_16k_mono_through_and_handles_other_layouts() {
    for (rate, channels, frames, expected) in [
        (16_000u32, 1u16, 16_000usize, 16_000usize),
        (16_000, 2, 16_000, 16_000),
        (8_000, 1, 8_000, 16_000),
        (48_000, 6, 48_000, 16_000),
    ] {
        let seen = SeenAudio::default();
        let state = ServerState {
            stt: Some(Arc::new(MockStt {
                seen: Arc::clone(&seen),
                ..MockStt::default()
            })),
            ..empty_state()
        };
        let response = app(state)
            .oneshot(post_bytes(
                "/v1/transcribe",
                wav_i16(rate, channels, frames),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{rate} Hz x{channels}");
        assert_eq!(body_json(response).await["duration_ms"], 1000);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![(expected, 16_000, 1)],
            "{rate} Hz x{channels}"
        );
    }
}

#[tokio::test]
async fn transcribe_rejects_unsupported_bit_depths_and_rates() {
    let state = || ServerState {
        stt: Some(Arc::new(MockStt::default())),
        ..empty_state()
    };
    let data = [0u8; 64];
    for (label, wav) in [
        ("0-bit int", raw_wav(1, 1, 16_000, 0, &data)),
        ("4-bit int", raw_wav(1, 1, 16_000, 4, &data)),
        ("64-bit int", raw_wav(1, 1, 16_000, 64, &data)),
        ("64-bit float", raw_wav(3, 1, 16_000, 64, &data)),
        ("16-bit float", raw_wav(3, 1, 16_000, 16, &data)),
        ("0 Hz", raw_wav(1, 1, 0, 16, &data)),
        ("100 Hz", raw_wav(1, 1, 100, 16, &data)),
        ("zero channels", raw_wav(1, 0, 16_000, 16, &data)),
        ("not a wav", b"definitely not audio".to_vec()),
    ] {
        let response = app(state())
            .oneshot(post_bytes("/v1/transcribe", wav))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
        let json = body_json(response).await;
        assert_eq!(json["error"]["code"], "invalid_wav", "{label}: {json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{label}");
    }

    // The supported depths all decode.
    for bits in [8u16, 16, 24, 32] {
        let wav = raw_wav(1, 1, 16_000, bits, &vec![0u8; usize::from(bits / 8) * 160]);
        let response = app(state())
            .oneshot(post_bytes("/v1/transcribe", wav))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{bits}-bit");
    }
}

// ---------------------------------------------------------------------------
// /v1/synthesize
// ---------------------------------------------------------------------------

fn tts_state(tts: MockTts) -> ServerState {
    ServerState {
        tts: Some(Arc::new(tts)),
        ..empty_state()
    }
}

#[tokio::test]
async fn synthesize_returns_16_bit_wav_by_default() {
    let response = app(tts_state(MockTts::default()))
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
    assert_eq!(response.headers()["x-sample-rate"], "16000");
    let bytes = body_bytes(response).await;
    let reader = hound::WavReader::new(Cursor::new(bytes)).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.bits_per_sample, 16);
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);
    assert_eq!(reader.len(), 16_000);
}

#[tokio::test]
async fn synthesize_supports_float_wav_and_raw_pcm() {
    let response = app(tts_state(MockTts::default()))
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi", "format": "wav_f32"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let reader = hound::WavReader::new(Cursor::new(body_bytes(response).await)).unwrap();
    assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
    assert_eq!(reader.spec().bits_per_sample, 32);

    let response = app(tts_state(MockTts::default()))
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi", "format": "pcm"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/pcm");
    let bytes = body_bytes(response).await;
    assert_eq!(bytes.len(), 32_000);
    assert_eq!(
        i16::from_le_bytes([bytes[0], bytes[1]]),
        (0.25 * f32::from(i16::MAX)) as i16
    );

    let response = app(tts_state(MockTts::default()))
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi", "format": "mp3"}),
        ))
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "unsupported_format",
    )
    .await;
    assert_eq!(json["error"]["param"], "format");
}

#[tokio::test]
async fn synthesize_enforces_the_text_limit() {
    let state = || ServerState {
        tts: Some(Arc::new(MockTts::default())),
        http_client: no_proxy_client(),
        ..ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                max_text_chars: 10,
                ..ServerOptions::default()
            },
        )
    };

    // Characters, not bytes: ten two-byte characters fit.
    let response = app(state())
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "ğğğğğğğğğğ"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app(state())
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "eleven char"}),
        ))
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
        "text_too_long",
    )
    .await;
    assert_eq!(json["error"]["param"], "text");

    let response = app(state())
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "  "}),
        ))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "empty_text",
    )
    .await;

    // The same limit guards the OpenAI endpoint.
    let response = app(state())
        .oneshot(post_json(
            "/v1/audio/speech",
            serde_json::json!({"model": "tts-1", "input": "eleven char", "voice": "alloy"}),
        ))
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
        "text_too_long",
    )
    .await;
    assert_eq!(json["error"]["param"], "input");
}

#[tokio::test]
async fn inference_slots_shed_load_with_503_and_retry_after() {
    let state = ServerState {
        tts: Some(Arc::new(MockTts {
            delay: Duration::from_millis(300),
            ..MockTts::default()
        })),
        http_client: no_proxy_client(),
        ..ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                max_concurrent_inference: 1,
                inference_wait: Duration::from_millis(40),
                ..ServerOptions::default()
            },
        )
    };
    let app = app(state);
    let request = || post_json("/v1/synthesize", serde_json::json!({"text": "hi"}));

    let (a, b) = tokio::join!(
        app.clone().oneshot(request()),
        app.clone().oneshot(request())
    );
    let (a, b) = (a.unwrap(), b.unwrap());
    let (ok, busy) = if a.status() == StatusCode::OK {
        (a, b)
    } else {
        (b, a)
    };
    assert_eq!(ok.status(), StatusCode::OK);
    assert_eq!(busy.headers()[header::RETRY_AFTER], "5");
    assert_error(
        busy,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "server_busy",
    )
    .await;

    // The slot is free again afterwards.
    let response = app.oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Error classes: status + body shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn every_error_class_has_the_right_status_and_body() {
    // 400: handler-level validation.
    let state = ServerState {
        stt: Some(Arc::new(MockStt::default())),
        ..empty_state()
    };
    let response = app(state)
        .oneshot(post_bytes("/v1/transcribe", b"nope".to_vec()))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_wav",
    )
    .await;

    // 400: syntactically invalid JSON, rejected by the extractor.
    let request = Request::builder()
        .method("POST")
        .uri("/v1/synthesize")
        .header(header::HOST, HOST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    let response = app(tts_state(MockTts::default()))
        .oneshot(request)
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "bad_request",
    )
    .await;

    // 422: valid JSON, missing field.
    let response = app(tts_state(MockTts::default()))
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"voice": "x"}),
        ))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request_error",
        "unprocessable_entity",
    )
    .await;

    // 415: JSON endpoint without a JSON content type.
    let response = app(tts_state(MockTts::default()))
        .oneshot(post_bytes("/v1/synthesize", br#"{"text":"hi"}"#.to_vec()))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "invalid_request_error",
        "unsupported_media_type",
    )
    .await;

    // 404 and 405 from the router.
    let response = app(empty_state())
        .oneshot(get_req("/v1/nonexistent"))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::NOT_FOUND,
        "not_found_error",
        "not_found",
    )
    .await;
    let response = app(empty_state())
        .oneshot(get_req("/v1/synthesize"))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "method_not_allowed",
    )
    .await;

    // 500: the backend failed.
    let response = app(tts_state(MockTts {
        fail: true,
        ..MockTts::default()
    }))
    .oneshot(post_json(
        "/v1/synthesize",
        serde_json::json!({"text": "hi"}),
    ))
    .await
    .unwrap();
    let json = assert_error(
        response,
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "tts_failed",
    )
    .await;
    assert!(
        json["message"]
            .as_str()
            .unwrap()
            .contains("synthesis exploded")
    );

    // 503: backend not loaded.
    let response = app(empty_state())
        .oneshot(post_json(
            "/v1/synthesize",
            serde_json::json!({"text": "hi"}),
        ))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "tts_unavailable",
    )
    .await;
    let response = app(empty_state())
        .oneshot(post_bytes("/v1/transcribe", wav_i16(16_000, 1, 160)))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "stt_unavailable",
    )
    .await;
}

#[tokio::test]
async fn slow_requests_time_out_with_504() {
    let state = ServerState {
        stt: Some(Arc::new(MockStt {
            delay: Duration::from_secs(5),
            ..MockStt::default()
        })),
        http_client: no_proxy_client(),
        ..ServerState::with_options(
            "127.0.0.1",
            3000,
            &ServerOptions {
                request_timeout: Duration::from_millis(60),
                ..ServerOptions::default()
            },
        )
    };
    let app = app(state);
    let started = std::time::Instant::now();
    let response = app
        .clone()
        .oneshot(post_bytes("/v1/transcribe", wav_i16(16_000, 1, 1_600)))
        .await
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_error(
        response,
        StatusCode::GATEWAY_TIMEOUT,
        "timeout_error",
        "timeout",
    )
    .await;

    // The cancelled request released its inference slot.
    let json = body_json(app.oneshot(get_req("/v1/stats")).await.unwrap()).await;
    assert_eq!(json["inference_slots_free"], json["inference_slots"]);
    assert_eq!(json["server_errors"], 1);
}

// ---------------------------------------------------------------------------
// /v1/chat and Ollama error mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_ignores_a_client_supplied_host() {
    let (ollama, hits) = fake_ollama(
        StatusCode::OK,
        r#"{"response":"hi from the configured ollama"}"#,
        Duration::ZERO,
    )
    .await;
    // Where an attacker would like the server to connect instead.
    let (decoy, decoy_hits) = fake_ollama(
        StatusCode::OK,
        r#"{"response":"hi from the decoy"}"#,
        Duration::ZERO,
    )
    .await;

    let state = ServerState {
        ollama_host: ollama,
        ..empty_state()
    };
    let response = app(state)
        .oneshot(post_json(
            "/v1/chat",
            serde_json::json!({"text": "hello", "model": "m", "host": decoy}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["response"], "hi from the configured ollama");
    assert_eq!(json["model"], "m");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        decoy_hits.load(Ordering::SeqCst),
        0,
        "the request's host field must not be contacted"
    );
}

#[tokio::test]
async fn chat_maps_upstream_failures_without_echoing_the_body() {
    const SECRET: &str = "SECRET-UPSTREAM-DETAIL /internal/path";
    let chat = |ollama: String| async move {
        let state = ServerState {
            ollama_host: ollama,
            ..empty_state()
        };
        app(state)
            .oneshot(post_json(
                "/v1/chat",
                serde_json::json!({"text": "hello", "model": "ghost"}),
            ))
            .await
            .unwrap()
    };

    // Upstream error -> 502, body not echoed.
    let (ollama, _) = fake_ollama(StatusCode::INTERNAL_SERVER_ERROR, SECRET, Duration::ZERO).await;
    let json = assert_error(
        chat(ollama).await,
        StatusCode::BAD_GATEWAY,
        "upstream_error",
        "upstream_error",
    )
    .await;
    assert!(!json.to_string().contains("SECRET"), "{json}");
    assert!(json["message"].as_str().unwrap().contains("500"));

    // Upstream 404 -> the model does not exist.
    let (ollama, _) = fake_ollama(StatusCode::NOT_FOUND, SECRET, Duration::ZERO).await;
    let json = assert_error(
        chat(ollama).await,
        StatusCode::NOT_FOUND,
        "not_found_error",
        "model_not_found",
    )
    .await;
    assert!(!json.to_string().contains("SECRET"), "{json}");
    assert!(
        json["message"]
            .as_str()
            .unwrap()
            .contains("ollama pull ghost")
    );

    // Upstream answered 200 with garbage -> 502.
    let (ollama, _) = fake_ollama(StatusCode::OK, SECRET, Duration::ZERO).await;
    let json = assert_error(
        chat(ollama).await,
        StatusCode::BAD_GATEWAY,
        "upstream_error",
        "upstream_error",
    )
    .await;
    assert!(!json.to_string().contains("SECRET"), "{json}");

    // Nothing listening -> 503.
    assert_error(
        chat(dead_port().await).await,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "ollama_unreachable",
    )
    .await;

    // Empty text never reaches Ollama.
    let (ollama, hits) = fake_ollama(StatusCode::OK, r#"{"response":"x"}"#, Duration::ZERO).await;
    let state = ServerState {
        ollama_host: ollama,
        ..empty_state()
    };
    let response = app(state)
        .oneshot(post_json("/v1/chat", serde_json::json!({"text": " "})))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "empty_text",
    )
    .await;
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn ollama_models_lists_models_and_times_out_per_request() {
    let (ollama, _) = fake_ollama(StatusCode::OK, "{}", Duration::ZERO).await;
    let state = ServerState {
        ollama_host: ollama,
        ..empty_state()
    };
    let response = app(state)
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["models"][0]["name"], "llama3.2:latest");

    // Unreachable -> 503.
    let state = ServerState {
        ollama_host: dead_port().await,
        ..empty_state()
    };
    let response = app(state)
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "ollama_unreachable",
    )
    .await;

    // A hung Ollama -> 504 after the per-request timeout (3 s), not forever.
    let (ollama, _) = fake_ollama(StatusCode::OK, "{}", Duration::from_secs(30)).await;
    let state = ServerState {
        ollama_host: ollama,
        ..empty_state()
    };
    let started = std::time::Instant::now();
    let response = app(state)
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(8));
    assert_error(
        response,
        StatusCode::GATEWAY_TIMEOUT,
        "timeout_error",
        "upstream_timeout",
    )
    .await;
}

// ---------------------------------------------------------------------------
// Voices and models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn voices_reports_the_conversation_engine() {
    // Nothing loaded.
    let json = body_json(
        app(empty_state())
            .oneshot(get_req("/v1/voices"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(json["voices"], serde_json::json!([]));
    assert!(json["engine"].is_null());
    assert!(json["conversation_engine"].is_null());
    assert_eq!(json["conversation_voices"], serde_json::json!([]));

    // Kokoro-like main engine, Piper-like conversation engine.
    let state = ServerState {
        tts: Some(Arc::new(MockTts {
            name: "kokoro",
            ..MockTts::default()
        })),
        conversation_tts: Some(Arc::new(MockTts {
            name: "piper",
            ..MockTts::default()
        })),
        ..empty_state()
    };
    let json = body_json(app(state).oneshot(get_req("/v1/voices")).await.unwrap()).await;
    assert_eq!(json["engine"], "kokoro");
    assert_eq!(json["voices"][0]["id"], "af_heart");
    assert_eq!(json["voices"][0]["backend"], "kokoro");
    assert_eq!(json["conversation_engine"], "piper");
    assert_eq!(json["conversation_voices"][0]["backend"], "piper");

    // Without a separate conversation engine, conversations use the main one.
    let json = body_json(
        app(tts_state(MockTts {
            name: "kokoro",
            ..MockTts::default()
        }))
        .oneshot(get_req("/v1/voices"))
        .await
        .unwrap(),
    )
    .await;
    assert_eq!(json["conversation_engine"], "kokoro");
    assert_eq!(json["conversation_voices"][0]["id"], "af_heart");
}

fn loaded_state() -> ServerState {
    ServerState {
        stt: Some(Arc::new(MockStt::default())),
        stt_model_name: Some("ggml-base.en".into()),
        tts: Some(Arc::new(MockTts {
            name: "kokoro",
            ..MockTts::default()
        })),
        tts_model_name: Some("kokoro-v1.0".into()),
        conversation_tts: Some(Arc::new(MockTts {
            name: "piper",
            ..MockTts::default()
        })),
        ollama_host: "127.0.0.1:1".into(),
        ..empty_state()
    }
}

#[tokio::test]
async fn vox_models_keeps_the_vox_shape() {
    let json = body_json(
        app(ServerState {
            ollama_host: "127.0.0.1:1".into(),
            ..empty_state()
        })
        .oneshot(get_req("/v1/vox/models"))
        .await
        .unwrap(),
    )
    .await;
    assert!(json["stt"].is_null());
    assert!(json["tts"].is_null());
    assert!(json["streaming_stt"].is_null());
    assert_eq!(json["ollama"]["connected"], false);
    assert!(json.get("object").is_none(), "no OpenAI keys here");

    let json = body_json(
        app(loaded_state())
            .oneshot(get_req("/v1/vox/models"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(json["stt"]["loaded"], true);
    assert_eq!(json["stt"]["model"], "ggml-base.en");
    assert_eq!(json["tts"]["name"], "kokoro");
}

#[tokio::test]
async fn models_is_openai_shaped_and_keeps_legacy_keys() {
    let response = app(loaded_state())
        .oneshot(get_req("/v1/models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;

    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "whisper-1",
            "ggml-base.en",
            "tts-1",
            "kokoro",
            "kokoro-v1.0",
            "piper"
        ]
    );
    for model in data {
        assert_eq!(model["object"], "model");
        assert_eq!(model["owned_by"], "vox");
        assert!(model["created"].as_u64().unwrap() > 1_600_000_000);
    }

    // Legacy keys stay for one release.
    assert_eq!(json["stt"]["model"], "ggml-base.en");
    assert_eq!(json["tts"]["name"], "kokoro");
    assert_eq!(json["ollama"]["connected"], false);

    // Nothing loaded: an empty list, still the OpenAI shape.
    let json = body_json(
        app(ServerState {
            ollama_host: "127.0.0.1:1".into(),
            ..empty_state()
        })
        .oneshot(get_req("/v1/models"))
        .await
        .unwrap(),
    )
    .await;
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"], serde_json::json!([]));
    assert!(json["stt"].is_null());
}

#[tokio::test]
async fn retrieve_model_follows_openai() {
    let response = app(loaded_state())
        .oneshot(get_req("/v1/models/whisper-1"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["id"], "whisper-1");
    assert_eq!(json["object"], "model");

    let response = app(loaded_state())
        .oneshot(get_req("/v1/models/gpt-4o"))
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::NOT_FOUND,
        "not_found_error",
        "model_not_found",
    )
    .await;
    assert_eq!(json["error"]["param"], "model");
}

// ---------------------------------------------------------------------------
// OpenAI: POST /v1/audio/transcriptions
// ---------------------------------------------------------------------------

fn stt_state(seen: &SeenAudio) -> ServerState {
    ServerState {
        stt: Some(Arc::new(MockStt {
            seen: Arc::clone(seen),
            ..MockStt::default()
        })),
        ..empty_state()
    }
}

#[tokio::test]
async fn openai_transcription_returns_json_text() {
    let seen = SeenAudio::default();
    let wav = wav_i16(44_100, 2, 44_100);
    let request = multipart(
        "/v1/audio/transcriptions",
        &[
            ("model", None, b"whisper-1"),
            ("file", Some("speech.wav"), &wav),
            ("language", None, b"en"),
            ("temperature", None, b"0"),
        ],
    );
    let response = app(stt_state(&seen)).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json, serde_json::json!({"text": "hello world"}));
    assert_eq!(*seen.lock().unwrap(), vec![(16_000, 16_000, 1)]);
}

#[tokio::test]
async fn openai_transcription_response_formats() {
    let wav = wav_i16(16_000, 1, 24_000);

    let request = multipart(
        "/v1/audio/transcriptions",
        &[
            ("file", Some("a.wav"), &wav),
            ("model", None, b"whisper-1"),
            ("response_format", None, b"text"),
        ],
    );
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/plain")
    );
    assert_eq!(body_bytes(response).await, b"hello world\n");

    let request = multipart(
        "/v1/audio/transcriptions",
        &[
            ("file", Some("a.wav"), &wav),
            ("model", None, b"whisper-1"),
            ("response_format", None, b"verbose_json"),
        ],
    );
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["task"], "transcribe");
    assert_eq!(json["language"], "en");
    assert_eq!(json["duration"], 1.5);
    assert_eq!(json["text"], "hello world");
    assert_eq!(json["segments"], serde_json::json!([]));

    let request = multipart(
        "/v1/audio/transcriptions",
        &[
            ("file", Some("a.wav"), &wav),
            ("response_format", None, b"srt"),
        ],
    );
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "unsupported_format",
    )
    .await;
    assert_eq!(json["error"]["param"], "response_format");
}

#[tokio::test]
async fn openai_transcription_rejects_bad_uploads() {
    // No file field.
    let request = multipart("/v1/audio/transcriptions", &[("model", None, b"whisper-1")]);
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "missing_file",
    )
    .await;
    assert_eq!(json["error"]["param"], "file");

    // Not a WAV (an MP3 header).
    let request = multipart(
        "/v1/audio/transcriptions",
        &[("file", Some("a.mp3"), b"ID3\x04\x00\x00\x00\x00\x00\x00")],
    );
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "unsupported_audio_format",
    )
    .await;
    assert!(json["message"].as_str().unwrap().contains("WAV"));

    // RIFF magic but broken.
    let request = multipart(
        "/v1/audio/transcriptions",
        &[("file", Some("a.wav"), b"RIFFbroken")],
    );
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(request)
        .await
        .unwrap();
    let json = assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_wav",
    )
    .await;
    assert_eq!(json["error"]["param"], "file");

    // Not multipart at all.
    let response = app(stt_state(&SeenAudio::default()))
        .oneshot(post_bytes(
            "/v1/audio/transcriptions",
            wav_i16(16_000, 1, 160),
        ))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "bad_request",
    )
    .await;

    // No STT backend.
    let wav = wav_i16(16_000, 1, 160);
    let request = multipart("/v1/audio/transcriptions", &[("file", Some("a.wav"), &wav)]);
    let response = app(empty_state()).oneshot(request).await.unwrap();
    assert_error(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "stt_unavailable",
    )
    .await;
}

// ---------------------------------------------------------------------------
// OpenAI: POST /v1/audio/speech
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_speech_returns_16_bit_wav() {
    let voices = Arc::new(Mutex::new(Vec::new()));
    let state = tts_state(MockTts {
        sample_rate: 22_050,
        samples: 22_050,
        seen_voices: Arc::clone(&voices),
        ..MockTts::default()
    });
    let response = app(state)
        .oneshot(post_json(
            "/v1/audio/speech",
            serde_json::json!({"model": "tts-1", "input": "Hello there", "voice": "alloy"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
    let reader = hound::WavReader::new(Cursor::new(body_bytes(response).await)).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 16);
    assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
    assert_eq!(reader.spec().sample_rate, 22_050);
    assert_eq!(reader.len(), 22_050);
    // "alloy" is an OpenAI voice: the backend's default voice is used.
    assert_eq!(*voices.lock().unwrap(), vec![None]);
}

#[tokio::test]
async fn openai_speech_pcm_is_24k_mono_s16le() {
    let state = tts_state(MockTts {
        sample_rate: 16_000,
        samples: 16_000,
        ..MockTts::default()
    });
    let response = app(state)
        .oneshot(post_json(
            "/v1/audio/speech",
            serde_json::json!({"model": "tts-1", "input": "Hello", "voice": "af_heart", "response_format": "pcm", "speed": 1.0}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/pcm");
    assert_eq!(response.headers()["x-sample-rate"], "24000");
    let bytes = body_bytes(response).await;
    assert_eq!(
        bytes.len(),
        24_000 * 2,
        "1 s resampled to 24 kHz, 2 bytes/sample"
    );
}

#[tokio::test]
async fn openai_speech_voice_and_model_selection() {
    let main_voices = Arc::new(Mutex::new(Vec::new()));
    let conv_voices = Arc::new(Mutex::new(Vec::new()));
    let state = || ServerState {
        tts: Some(Arc::new(MockTts {
            name: "kokoro",
            seen_voices: Arc::clone(&main_voices),
            ..MockTts::default()
        })),
        conversation_tts: Some(Arc::new(MockTts {
            name: "piper",
            seen_voices: Arc::clone(&conv_voices),
            ..MockTts::default()
        })),
        ..empty_state()
    };

    for (model, voice) in [
        ("tts-1", "af_heart"),
        ("piper", "NOVA"),
        ("kokoro", "speaker-7"),
    ] {
        let response = app(state())
            .oneshot(post_json(
                "/v1/audio/speech",
                serde_json::json!({"model": model, "input": "Hello", "voice": voice}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{model}/{voice}");
    }
    // A listed voice passes through; an unlisted non-OpenAI one is left to the backend.
    assert_eq!(
        *main_voices.lock().unwrap(),
        vec![Some("af_heart".to_string()), Some("speaker-7".to_string())]
    );
    // model "piper" selected the conversation engine; "NOVA" is an OpenAI voice.
    assert_eq!(*conv_voices.lock().unwrap(), vec![None]);
}

#[tokio::test]
async fn openai_speech_rejects_what_it_cannot_do() {
    let speech = |body: serde_json::Value| async move {
        app(tts_state(MockTts::default()))
            .oneshot(post_json("/v1/audio/speech", body))
            .await
            .unwrap()
    };

    let json = assert_error(
        speech(
            serde_json::json!({"model": "tts-1", "input": "Hi", "voice": "alloy", "speed": 1.5}),
        )
        .await,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "unsupported_parameter",
    )
    .await;
    assert_eq!(json["error"]["param"], "speed");

    for format in ["mp3", "opus", "aac", "flac"] {
        let json = assert_error(
            speech(serde_json::json!({"model": "tts-1", "input": "Hi", "voice": "alloy", "response_format": format}))
                .await,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unsupported_format",
        )
        .await;
        assert_eq!(json["error"]["param"], "response_format", "{format}");
    }

    let json = assert_error(
        speech(serde_json::json!({"model": "tts-1", "input": "", "voice": "alloy"})).await,
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "empty_text",
    )
    .await;
    assert_eq!(json["error"]["param"], "input");

    // `input` missing entirely.
    assert_error(
        speech(serde_json::json!({"model": "tts-1", "voice": "alloy"})).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request_error",
        "unprocessable_entity",
    )
    .await;

    // No TTS backend.
    let response = app(empty_state())
        .oneshot(post_json(
            "/v1/audio/speech",
            serde_json::json!({"model": "tts-1", "input": "Hi", "voice": "alloy"}),
        ))
        .await
        .unwrap();
    assert_error(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        "tts_unavailable",
    )
    .await;
}

// ---------------------------------------------------------------------------
// /v1/speakers through the production router
// ---------------------------------------------------------------------------

#[tokio::test]
async fn speaker_routes_are_mounted_behind_the_access_policy() {
    let app = app(empty_state());

    // Diarization is not loaded: the list answers, empty and unavailable.
    let response = app.clone().oneshot(get_req("/v1/speakers")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["available"], false);
    assert_eq!(json["speakers"], serde_json::json!([]));

    // Mutations fail with the server-wide error body.
    let request = Request::builder()
        .method("DELETE")
        .uri("/v1/speakers?confirm=all")
        .header(header::HOST, HOST)
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(response).await;
    assert_eq!(json["error"]["code"], "diarization_unavailable");
    assert_eq!(json["error"]["message"], json["message"]);
    assert!(json["message"].is_string());

    // A foreign origin cannot wipe or export the speaker store.
    for (method, uri) in [
        ("DELETE", "/v1/speakers?confirm=all"),
        (
            "GET",
            "/v1/speakers/speaker_1/export?include_embedding=true",
        ),
        ("PUT", "/v1/speakers/consent"),
    ] {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::HOST, HOST)
            .header(header::ORIGIN, "https://evil.example")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"granted":true}"#))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_error(
            response,
            StatusCode::FORBIDDEN,
            "permission_error",
            "origin_not_allowed",
        )
        .await;
    }
}
