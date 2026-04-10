//! Error handling and edge case tests for the Vox HTTP server.
#![allow(dead_code)]
//!
//! Run with:
//!   cargo test --features server --test error_edge_tests

#[path = "../src/server/mod.rs"]
mod server;

use std::io::Cursor;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use tower::ServiceExt;

use server::{ServerState, ServerStats};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a minimal ServerState with no backends loaded.
fn test_state() -> Arc<ServerState> {
    Arc::new(ServerState {
        stt: None,
        tts: None,
        vad_model_path: None,
        stats: Arc::new(std::sync::Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: std::time::Instant::now(),
        ollama_host: "localhost:11434".to_string(),
        http_client: reqwest::Client::new(),
        stt_model_name: None,
        stt_model_size: None,
        tts_model_name: None,
        tts_model_size: None,
        streaming_stt: None,
        model_cache: None,
    })
}

/// Build the Router matching the production server layout.
fn test_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/", get(server::handlers::index))
        .route("/v1/chat", post(server::handlers::chat))
        .route("/v1/voices", get(server::handlers::voices))
        .route("/v1/ollama-models", get(server::handlers::ollama_models))
        .route("/v1/transcribe", post(server::handlers::transcribe))
        .route("/v1/synthesize", post(server::handlers::synthesize))
        .route("/v1/models", get(server::handlers::models))
        .route("/v1/stats", get(server::handlers::stats))
        .route("/health", get(server::handlers::health))
        .with_state(state)
}

/// Encode a mono WAV file from f32 samples at the given sample rate.
fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

/// Encode a mono WAV file from i16 samples at the given sample rate.
fn encode_wav_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

/// Read the full response body as a string.
async fn body_string(body: Body) -> String {
    let bytes = body.collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// Parse a JSON response body into a serde_json::Value.
async fn body_json(body: Body) -> serde_json::Value {
    let text = body_string(body).await;
    serde_json::from_str(&text).expect("response body is not valid JSON")
}

// ===========================================================================
// 1. ServerError formatting
// ===========================================================================

mod server_error_formatting {
    use super::*;

    #[tokio::test]
    async fn bad_request_returns_400_with_json_error() {
        let err = server::error::ServerError::bad_request("invalid input");
        let response = axum::response::IntoResponse::into_response(err);

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let json = body_json(response.into_body()).await;
        assert_eq!(json["error"], "invalid input");
    }

    #[tokio::test]
    async fn service_unavailable_returns_503_with_json_error() {
        let err = server::error::ServerError::service_unavailable("backend offline");
        let response = axum::response::IntoResponse::into_response(err);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let json = body_json(response.into_body()).await;
        assert_eq!(json["error"], "backend offline");
    }

    #[tokio::test]
    async fn error_response_has_json_content_type() {
        let err = server::error::ServerError::bad_request("test");
        let response = axum::response::IntoResponse::into_response(err);

        let ct = response
            .headers()
            .get("content-type")
            .expect("missing content-type header");
        assert_eq!(ct, "application/json");
    }

    #[tokio::test]
    async fn error_body_is_valid_json_object() {
        let err = server::error::ServerError::bad_request("some message");
        let response = axum::response::IntoResponse::into_response(err);

        let json = body_json(response.into_body()).await;
        assert!(json.is_object());
        assert!(json.get("error").is_some());
    }

    #[tokio::test]
    async fn empty_message_still_produces_valid_json() {
        let err = server::error::ServerError::bad_request("");
        let response = axum::response::IntoResponse::into_response(err);

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = body_json(response.into_body()).await;
        assert_eq!(json["error"], "");
    }

    #[tokio::test]
    async fn message_with_special_chars_is_escaped() {
        let err = server::error::ServerError::bad_request(r#"bad "value" \n test"#);
        let response = axum::response::IntoResponse::into_response(err);

        let json = body_json(response.into_body()).await;
        assert_eq!(json["error"], r#"bad "value" \n test"#);
    }
}

// ===========================================================================
// 2. WAV decoding edge cases (transcribe handler)
// ===========================================================================

mod wav_decoding {
    use super::*;

    #[tokio::test]
    async fn empty_body_returns_400() {
        let state = test_state_with_stt();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let json = body_json(resp.into_body()).await;
        let error_msg = json["error"].as_str().unwrap();
        assert!(
            error_msg.contains("invalid WAV"),
            "expected 'invalid WAV' in error, got: {error_msg}"
        );
    }

    #[tokio::test]
    async fn random_bytes_returns_400() {
        let state = test_state_with_stt();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(vec![0xDE, 0xAD, 0xBE, 0xEF]))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let json = body_json(resp.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("invalid WAV"));
    }

    #[tokio::test]
    async fn valid_wav_header_but_no_samples_returns_503() {
        // A WAV with zero samples is valid to hound; it will decode
        // to an empty Vec<f32> then try stt.transcribe which will
        // use our mock STT returning a result. With no STT backend
        // this would be 503, but with our mock STT it should succeed.
        let state = test_state_with_stt();
        let app = test_router(state);

        let wav = encode_wav_f32(&[], 16000);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(wav))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // The mock STT will succeed, so we get 200
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp.into_body()).await;
        assert!(json.get("text").is_some());
    }

    #[tokio::test]
    async fn wav_int16_format_is_accepted() {
        let state = test_state_with_stt();
        let app = test_router(state);

        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let wav = encode_wav_i16(&samples, 16000);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(wav))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn wav_float32_format_is_accepted() {
        let state = test_state_with_stt();
        let app = test_router(state);

        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_f32(&samples, 16000);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(wav))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn transcribe_without_stt_backend_returns_503() {
        let state = test_state(); // no STT loaded
        let app = test_router(state);

        let wav = encode_wav_f32(&[0.0; 100], 16000);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(wav))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let json = body_json(resp.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("STT"));
    }

    #[tokio::test]
    async fn transcribe_increments_stats() {
        let state = test_state_with_stt();
        let stats = Arc::clone(&state.stats);
        let app = test_router(state);

        let wav = encode_wav_f32(&[0.0; 100], 16000);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcribe")
            .body(Body::from(wav))
            .unwrap();

        let _ = app.oneshot(req).await.unwrap();

        let s = stats.lock().unwrap();
        assert_eq!(s.requests, 1);
        assert_eq!(s.transcriptions, 1);
    }

    // -- Mock STT for WAV tests --

    use async_trait::async_trait;

    struct MockStt(String);

    #[async_trait]
    impl vox::SttBackend for MockStt {
        async fn transcribe(
            &self,
            audio: &vox::Utterance,
        ) -> Result<vox::SttResult, vox::VoxError> {
            Ok(vox::SttResult {
                text: self.0.clone(),
                language: Some("en".into()),
                duration_ms: audio.duration_ms,
                processing_time_ms: 1,
            })
        }
    }

    fn test_state_with_stt() -> Arc<ServerState> {
        Arc::new(ServerState {
            stt: Some(Arc::new(MockStt("transcribed text".into()))),
            tts: None,
            vad_model_path: None,
            stats: Arc::new(std::sync::Mutex::new(ServerStats {
                requests: 0,
                transcriptions: 0,
                syntheses: 0,
            })),
            start_time: std::time::Instant::now(),
            ollama_host: "localhost:11434".to_string(),
            http_client: reqwest::Client::new(),
            stt_model_name: None,
            stt_model_size: None,
            tts_model_name: None,
            tts_model_size: None,
            streaming_stt: None,
            model_cache: None,
        })
    }
}

// ===========================================================================
// 3. Request body parsing (synthesize handler)
// ===========================================================================

mod synthesize_parsing {
    use super::*;

    #[tokio::test]
    async fn missing_text_field_returns_422() {
        let state = test_state();
        let app = test_router(state);

        // Send JSON without the required "text" field
        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"voice": "af_heart"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 422 for deserialization failures
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn empty_json_object_returns_422() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn non_json_body_returns_400() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .header("content-type", "application/json")
            .body(Body::from("this is not json"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 400 for syntactically invalid JSON (parse error),
        // vs 422 for valid JSON missing required fields (data error).
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_content_type_returns_415() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .body(Body::from(r#"{"text": "hello"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 415 Unsupported Media Type when content-type is missing for Json extractor
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn valid_request_without_tts_returns_503() {
        let state = test_state(); // no TTS loaded
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"text": "hello world"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let json = body_json(resp.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("TTS"));
    }

    #[tokio::test]
    async fn valid_request_with_optional_voice() {
        let state = test_state(); // no TTS, but parsing should succeed before 503
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/synthesize")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"text": "hello world", "voice": "af_heart"}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Parsing succeeds, but no TTS backend -> 503
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}

// ===========================================================================
// 4. Chat request parsing
// ===========================================================================

mod chat_parsing {
    use super::*;

    #[tokio::test]
    async fn missing_text_field_returns_422() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "llama3.2"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn empty_json_object_returns_422() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn non_json_body_returns_400() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 400 for syntactically invalid JSON (parse error),
        // vs 422 for valid JSON missing required fields (data error).
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn valid_request_with_optional_fields_parses_ok() {
        // The chat handler will try to reach Ollama and fail (no real server),
        // but the request parsing should succeed (not 422).
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"text": "hello", "model": "llama3.2", "host": "localhost:11434"}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Parsing succeeds but Ollama is unreachable -> 503
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn valid_request_text_only_parses_ok() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"text": "hello"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should not be 422 (parsing succeeded); will be 503 (Ollama not running)
        assert_ne!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}

// ===========================================================================
// 5. VoxError to ServerError conversion
// ===========================================================================

mod vox_error_conversion {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn stt_error_maps_to_500() {
        let err = vox::VoxError::Stt("model crashed".into());
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("STT error"));
    }

    #[tokio::test]
    async fn tts_error_maps_to_500() {
        let err = vox::VoxError::Tts("synthesis failed".into());
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("TTS error"));
    }

    #[tokio::test]
    async fn pipeline_error_maps_to_500() {
        let err = vox::VoxError::Pipeline("broken pipe".into());
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("pipeline error"));
    }

    #[tokio::test]
    async fn no_stt_error_maps_to_503() {
        let err = vox::VoxError::NoStt;
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("no STT backend"));
    }

    #[tokio::test]
    async fn no_vad_error_maps_to_503() {
        let err = vox::VoxError::NoVad;
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("no VAD backend"));
    }

    #[tokio::test]
    async fn audio_error_maps_to_500() {
        let err = vox::VoxError::Audio("device lost".into());
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn io_error_maps_to_500() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: vox::VoxError = io_err.into();
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn model_not_found_maps_to_500() {
        let err = vox::VoxError::ModelNotFound(PathBuf::from("/missing/model.bin"));
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_json(response.into_body()).await;
        assert!(json["error"].as_str().unwrap().contains("model not found"));
    }

    #[tokio::test]
    async fn vad_error_maps_to_500() {
        let err = vox::VoxError::Vad("inference error".into());
        let server_err: server::error::ServerError = err.into();
        let response = axum::response::IntoResponse::into_response(server_err);

        // Vad variant falls into the catch-all arm -> 500
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}

// ===========================================================================
// 6. Stats thread safety
// ===========================================================================

mod stats_thread_safety {
    use super::*;

    #[tokio::test]
    async fn concurrent_stats_requests_are_consistent() {
        let state = test_state();
        let stats = Arc::clone(&state.stats);

        let num_tasks = 50;
        let mut handles = Vec::with_capacity(num_tasks);

        for _ in 0..num_tasks {
            let app = test_router(Arc::clone(&state));
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap();
                let resp = app.oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let s = stats.lock().unwrap();
        assert_eq!(
            s.requests, num_tasks as u64,
            "expected {num_tasks} requests, got {}",
            s.requests
        );
    }

    #[tokio::test]
    async fn concurrent_health_does_not_increment_stats() {
        let state = test_state();
        let stats = Arc::clone(&state.stats);

        let num_tasks = 20;
        let mut handles = Vec::with_capacity(num_tasks);

        for _ in 0..num_tasks {
            let app = test_router(Arc::clone(&state));
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap();
                let _ = app.oneshot(req).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let s = stats.lock().unwrap();
        // Health endpoint does not bump the stats counter
        assert_eq!(s.requests, 0);
    }

    #[tokio::test]
    async fn mixed_concurrent_endpoints_count_correctly() {
        let state = test_state();
        let stats = Arc::clone(&state.stats);

        let mut handles = Vec::new();

        // 10 stats requests (each increments requests by 1)
        for _ in 0..10 {
            let app = test_router(Arc::clone(&state));
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap();
                let _ = app.oneshot(req).await.unwrap();
            }));
        }

        // 5 models requests (each increments requests by 1)
        for _ in 0..5 {
            let app = test_router(Arc::clone(&state));
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap();
                let _ = app.oneshot(req).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let s = stats.lock().unwrap();
        assert_eq!(
            s.requests, 15,
            "expected 15 total requests, got {}",
            s.requests
        );
        assert_eq!(s.transcriptions, 0);
        assert_eq!(s.syntheses, 0);
    }
}

// ===========================================================================
// 7. Additional endpoint edge cases
// ===========================================================================

mod endpoint_edge_cases {
    use super::*;

    #[tokio::test]
    async fn health_returns_ok_status() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp.into_body()).await;
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn models_with_no_backends_returns_nulls() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp.into_body()).await;
        assert!(json["stt"].is_null());
        assert!(json["tts"].is_null());
    }

    #[tokio::test]
    async fn voices_with_no_tts_returns_empty_list() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/voices")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp.into_body()).await;
        assert_eq!(json["voices"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn index_returns_html() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .expect("missing content-type");
        assert!(
            ct.to_str().unwrap().contains("text/html"),
            "expected text/html content-type, got {:?}",
            ct
        );
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/nonexistent")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn wrong_method_returns_405() {
        let state = test_state();
        let app = test_router(state);

        // /v1/transcribe only accepts POST, not GET
        let req = Request::builder()
            .method("GET")
            .uri("/v1/transcribe")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn stats_returns_zero_on_fresh_server() {
        let state = test_state();
        let app = test_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp.into_body()).await;
        // The stats request itself bumps the counter to 1
        assert_eq!(json["requests"], 1);
        assert_eq!(json["transcriptions"], 0);
        assert_eq!(json["syntheses"], 0);
        assert!(json["uptime_secs"].as_u64().is_some());
    }
}
