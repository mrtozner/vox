//! Integration tests for the Vox HTTP API server.
#![allow(dead_code)]
//!
//! These tests build the Axum router with `None` backends (no models loaded)
//! and verify endpoint behavior using `tower::ServiceExt::oneshot`.
//!
//! Run with: `cargo test --test server_tests --features server`

#[path = "../src/server/mod.rs"]
mod server;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use tower::ServiceExt;

use server::{ServerState, ServerStats};

// ---------------------------------------------------------------------------
// Helper: build the router with no backends loaded
// ---------------------------------------------------------------------------

fn build_test_app() -> Router {
    let state = Arc::new(ServerState {
        stt: None,
        tts: None,
        streaming_stt: None,
        model_cache: None,
        vad_model_path: None,
        stats: Arc::new(Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: Instant::now(),
        ollama_host: "localhost:11434".into(),
        http_client: reqwest::Client::new(),
        stt_model_name: None,
        stt_model_size: None,
        tts_model_name: None,
        tts_model_size: None,
        capabilities: Arc::new(vox::CapabilityRegistry::default()),
        #[cfg(feature = "diarization")]
        diarization: None,
        #[cfg(feature = "diarization")]
        speaker_db: None,
    });

    Router::new()
        .route("/", get(server::handlers::index))
        .route("/v1/voices", get(server::handlers::voices))
        .route("/v1/transcribe", post(server::handlers::transcribe))
        .route("/v1/synthesize", post(server::handlers::synthesize))
        .route("/v1/models", get(server::handlers::models))
        .route("/v1/stats", get(server::handlers::stats))
        .route("/health", get(server::handlers::health))
        .with_state(state)
}

/// Convenience: collect the full response body as bytes.
async fn body_bytes(response: axum::http::Response<Body>) -> Vec<u8> {
    response
        .into_body()
        .collect()
        .await
        .expect("failed to collect response body")
        .to_bytes()
        .to_vec()
}

/// Convenience: collect the full response body as a JSON value.
async fn body_json(response: axum::http::Response<Body>) -> serde_json::Value {
    let bytes = body_bytes(response).await;
    serde_json::from_slice(&bytes).expect("response body is not valid JSON")
}

// ---------------------------------------------------------------------------
// 1. Health endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_returns_200_with_status_ok() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["status"], "ok");
}

// ---------------------------------------------------------------------------
// 2. Stats endpoint -- returns 200 with counters
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stats_returns_200_with_counters() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    // The stats endpoint increments `requests` before responding, so the
    // first call should show requests == 1.
    assert_eq!(json["requests"], 1);
    assert_eq!(json["transcriptions"], 0);
    assert_eq!(json["syntheses"], 0);
    assert!(json["uptime_secs"].is_number());
}

// ---------------------------------------------------------------------------
// 3. Models endpoint -- stt/tts are null when no backends loaded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_returns_200_with_null_backends() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    assert!(json["stt"].is_null(), "stt should be null when not loaded");
    assert!(json["tts"].is_null(), "tts should be null when not loaded");
    // ollama field may be present with connected: false (network attempt)
    assert!(
        json.get("ollama").is_some(),
        "ollama status should be present"
    );
}

// ---------------------------------------------------------------------------
// 4. Voices endpoint -- empty list when no TTS loaded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn voices_returns_200_with_empty_list() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/voices")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response).await;
    let voices = json["voices"]
        .as_array()
        .expect("voices should be an array");
    assert!(
        voices.is_empty(),
        "voices should be empty when no TTS loaded"
    );
}

// ---------------------------------------------------------------------------
// 5. Index endpoint -- returns HTML
// ---------------------------------------------------------------------------

#[tokio::test]
async fn index_returns_200_with_html() {
    let app = build_test_app();

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/html"),
        "expected text/html content-type, got: {content_type}"
    );

    let bytes = body_bytes(response).await;
    let body_str = String::from_utf8_lossy(&bytes);
    assert!(
        body_str.contains("<html") || body_str.contains("<!DOCTYPE") || body_str.contains("<HTML"),
        "response body should contain HTML markup"
    );
}

// ---------------------------------------------------------------------------
// 6. Transcribe without STT -- returns 503
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transcribe_without_stt_returns_503() {
    let app = build_test_app();

    // Send a minimal valid-looking WAV body (content does not matter since
    // the backend-not-loaded check happens first).
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcribe")
                .body(Body::from(vec![0u8; 64]))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let json = body_json(response).await;
    let error_msg = json["error"].as_str().unwrap_or("");
    assert!(
        error_msg.contains("STT"),
        "error should mention STT, got: {error_msg}"
    );
}

// ---------------------------------------------------------------------------
// 7. Synthesize without TTS -- returns 503
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synthesize_without_tts_returns_503() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/synthesize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let json = body_json(response).await;
    let error_msg = json["error"].as_str().unwrap_or("");
    assert!(
        error_msg.contains("TTS"),
        "error should mention TTS, got: {error_msg}"
    );
}

// ---------------------------------------------------------------------------
// 8. Transcribe with invalid WAV -- returns 400 (requires STT loaded, so
//    with None backend this returns 503 first; we test the 503 path here
//    and document the expected 400 behavior)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transcribe_with_garbage_bytes_returns_503_without_backend() {
    // Without an STT backend the 503 check fires before WAV parsing.
    // This test confirms the guard clause works with arbitrary input.
    let app = build_test_app();

    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF, 0x42, 0x13];
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcribe")
                .body(Body::from(garbage))
                .unwrap(),
        )
        .await
        .unwrap();

    // Backend check fires first: 503
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ---------------------------------------------------------------------------
// 9. Synthesize with valid JSON structure -- validates request parsing,
//    returns 503 because TTS is not loaded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synthesize_with_valid_json_returns_503_not_422() {
    let app = build_test_app();

    // A well-formed request body: the handler should parse it successfully
    // but then fail on the missing TTS backend (503), not on deserialization.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/synthesize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"hello","voice":"af_heart"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "valid JSON should not produce a 422 or 400"
    );
}

// ---------------------------------------------------------------------------
// 10. Stats counter increments across requests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stats_counter_increments_across_requests() {
    // We share state across multiple requests by building the state once
    // and cloning the router (oneshot consumes the router).
    let state = Arc::new(ServerState {
        stt: None,
        tts: None,
        streaming_stt: None,
        model_cache: None,
        vad_model_path: None,
        stats: Arc::new(Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: Instant::now(),
        ollama_host: "localhost:11434".into(),
        http_client: reqwest::Client::new(),
        stt_model_name: None,
        stt_model_size: None,
        tts_model_name: None,
        tts_model_size: None,
        capabilities: Arc::new(vox::CapabilityRegistry::default()),
        #[cfg(feature = "diarization")]
        diarization: None,
        #[cfg(feature = "diarization")]
        speaker_db: None,
    });

    let app = Router::new()
        .route("/v1/stats", get(server::handlers::stats))
        .route("/health", get(server::handlers::health))
        .with_state(state.clone());

    // First stats request
    let resp1 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json1 = body_json(resp1).await;
    assert_eq!(json1["requests"], 1);

    // Second stats request
    let resp2 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json2 = body_json(resp2).await;
    assert_eq!(json2["requests"], 2);

    // Third stats request
    let resp3 = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json3 = body_json(resp3).await;
    assert_eq!(json3["requests"], 3);
}

// ---------------------------------------------------------------------------
// 11. CORS headers are present (CorsLayer::permissive)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cors_headers_are_present_on_response() {
    // Build a router WITH the CorsLayer to verify headers.
    let state = Arc::new(ServerState {
        stt: None,
        tts: None,
        streaming_stt: None,
        model_cache: None,
        vad_model_path: None,
        stats: Arc::new(Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: Instant::now(),
        ollama_host: "localhost:11434".into(),
        http_client: reqwest::Client::new(),
        stt_model_name: None,
        stt_model_size: None,
        tts_model_name: None,
        tts_model_size: None,
        capabilities: Arc::new(vox::CapabilityRegistry::default()),
        #[cfg(feature = "diarization")]
        diarization: None,
        #[cfg(feature = "diarization")]
        speaker_db: None,
    });

    let app = Router::new()
        .route("/health", get(server::handlers::health))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    // Send a request with an Origin header to trigger CORS response headers.
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("origin", "http://example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let acl = response
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        acl, "*",
        "CorsLayer::permissive should set access-control-allow-origin to *"
    );
}

// ---------------------------------------------------------------------------
// 12. Synthesize with invalid JSON body -- returns 422
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synthesize_with_invalid_json_returns_422() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/synthesize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"not_text": true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum returns 422 Unprocessable Entity when JSON deserialization fails
    // because the required `text` field is missing.
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ---------------------------------------------------------------------------
// 13. Non-existent route returns 404
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_route_returns_404() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
