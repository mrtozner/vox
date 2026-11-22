//! `GET /v1/privacy/ledger`: the server reports what it connected to.
#![allow(dead_code)]
//!
//! Drives the production router against a stand-in Ollama on a loopback
//! port. Offline mode is off in this binary; see `offline_server_tests.rs`
//! for the other half.
//!
//! Run with: `cargo test --test privacy_ledger_tests --features server`

use vox::server;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use tokio::net::TcpListener;
use tower::ServiceExt;

use server::ServerState;

const HOST: &str = "127.0.0.1:3000";
const TAGS: &str =
    r#"{"models":[{"name":"llama3.2:latest","size":1,"details":{"family":"llama"}}]}"#;
const GENERATED: &str = r#"{"response":"hello from the stand-in"}"#;

/// A stand-in for Ollama. Returns its `host:port`.
async fn fake_ollama() -> String {
    let router = Router::new()
        .route("/api/tags", get(|| async { TAGS }))
        .route("/api/generate", post(|| async { GENERATED }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    addr.to_string()
}

fn app(ollama_host: &str) -> Router {
    server::build_router(Arc::new(ServerState {
        ollama_host: ollama_host.to_string(),
        http_client: reqwest::Client::builder().no_proxy().build().unwrap(),
        ..ServerState::empty()
    }))
}

fn get_req(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header(header::HOST, HOST)
        .body(Body::empty())
        .unwrap()
}

async fn json(response: axum::http::Response<Body>) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).expect("response is JSON")
}

async fn ledger(app: &Router) -> serde_json::Value {
    let response = app
        .clone()
        .oneshot(get_req("/v1/privacy/ledger"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    json(response).await
}

/// Rows of `destinations` for one destination, keyed by purpose.
fn rows<'a>(ledger: &'a serde_json::Value, destination: &str) -> Vec<&'a serde_json::Value> {
    ledger["destinations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["destination"] == destination)
        .collect()
}

#[tokio::test]
async fn ledger_has_the_documented_shape_before_any_call() {
    let ollama = fake_ollama().await;
    let app = app(&ollama);
    let body = ledger(&app).await;

    assert_eq!(body["offline"], false);
    assert_eq!(body["ollama_host"], ollama.as_str());
    assert_eq!(body["ollama_host_loopback"], true);
    assert!(body["fully_local"].is_boolean());
    assert!(body["totals"]["connections"].is_u64());
    assert!(body["totals"]["bytes_sent"].is_u64());
    assert!(body["totals"]["bytes_received"].is_u64());
    assert!(body["totals"]["non_loopback_connections"].is_u64());
    assert!(body["totals"]["blocked"].is_u64());
    assert!(body["destinations"].is_array());
    assert!(body["recent"].is_array());
    assert!(body["dropped_entries"].is_u64());
    // This server has not talked to its Ollama yet.
    assert!(rows(&body, &ollama).is_empty());
}

#[tokio::test]
async fn ledger_records_a_model_list_call() {
    let ollama = fake_ollama().await;
    let app = app(&ollama);

    let response = app
        .clone()
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = ledger(&app).await;
    let rows = rows(&body, &ollama);
    assert_eq!(rows.len(), 1, "{body}");
    let row = rows[0];
    assert_eq!(row["purpose"], "llm model list");
    assert_eq!(row["loopback"], true);
    assert_eq!(row["connections"], 1);
    assert_eq!(row["blocked"], 0);
    assert_eq!(row["bytes_sent"], 0, "a GET has no body");
    assert_eq!(row["bytes_received"], TAGS.len() as u64);
    assert!(row["last_unix_ms"].as_u64().unwrap() > 0);

    let entry = body["recent"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["destination"] == ollama.as_str())
        .expect("the call is among the recent entries");
    assert_eq!(entry["purpose"], "llm model list");
    assert_eq!(entry["blocked"], false);
    assert_eq!(entry["bytes_received"], TAGS.len() as u64);
    assert!(entry["seq"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn ledger_counts_chat_bytes_in_both_directions() {
    let ollama = fake_ollama().await;
    let app = app(&ollama);

    for _ in 0..2 {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat")
            .header(header::HOST, HOST)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"text":"hello there","model":"llama3.2"}"#))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let body = ledger(&app).await;
    let rows = rows(&body, &ollama);
    let chat = rows
        .iter()
        .find(|row| row["purpose"] == "llm chat")
        .unwrap_or_else(|| panic!("no chat row in {body}"));
    assert_eq!(chat["connections"], 2);
    assert_eq!(chat["bytes_received"], 2 * GENERATED.len() as u64);
    // The request carries the prompt, the system prompt and the options.
    let sent = chat["bytes_sent"].as_u64().unwrap();
    assert!(sent > 2 * "hello there".len() as u64, "bytes_sent = {sent}");
}

#[tokio::test]
async fn status_probe_and_streamed_chat_are_recorded_too() {
    let ollama = fake_ollama().await;
    let app = app(&ollama);

    // `/v1/vox/models` checks whether Ollama is up.
    let response = app
        .clone()
        .oneshot(get_req("/v1/vox/models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The streaming path used by Live Talk, Converse and `vox chat`. The
    // stand-in has no /api/chat, so the request fails after it was sent.
    let result = vox::streaming_chat::stream_chat(
        &reqwest::Client::builder().no_proxy().build().unwrap(),
        &ollama,
        "llama3.2",
        &[vox::conversation::ChatMessage::user("hi")],
        &vox::streaming_chat::StreamOptions::default(),
        tokio_util::sync::CancellationToken::new(),
        |_| {},
        |_| async { Ok(()) },
    )
    .await;
    assert!(result.is_err(), "the stand-in has no /api/chat");

    let body = ledger(&app).await;
    let purposes: Vec<&str> = rows(&body, &ollama)
        .iter()
        .map(|row| row["purpose"].as_str().unwrap())
        .collect();
    assert!(purposes.contains(&"llm status check"), "{purposes:?}");
    assert!(purposes.contains(&"llm chat"), "{purposes:?}");
}

#[tokio::test]
async fn nothing_in_this_run_left_the_machine() {
    let ollama = fake_ollama().await;
    let app = app(&ollama);
    app.clone()
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();

    let body = ledger(&app).await;
    assert_eq!(body["fully_local"], true, "{body}");
    assert_eq!(body["totals"]["non_loopback_connections"], 0);
    for row in body["destinations"].as_array().unwrap() {
        assert_eq!(row["loopback"], true, "{row}");
    }
}

#[tokio::test]
async fn a_remote_ollama_host_is_reported_as_not_loopback() {
    // Nothing is sent: only the configured address is described.
    let app = app("gpu-box.invalid:11434");
    let body = ledger(&app).await;
    assert_eq!(body["ollama_host"], "gpu-box.invalid:11434");
    assert_eq!(body["ollama_host_loopback"], false);
}
