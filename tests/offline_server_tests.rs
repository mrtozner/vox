//! Offline mode on the server: a remote Ollama is refused everywhere.
#![allow(dead_code)]
//!
//! Offline mode is process-wide, so these tests live in their own binary
//! and every one of them switches it on.
//!
//! Run with: `cargo test --test offline_server_tests --features server`

use vox::server;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::get;
use http_body_util::BodyExt;
use tokio::net::TcpListener;
use tower::ServiceExt;

use server::{LoadOptions, ServerOptions, ServerState};

const HOST: &str = "127.0.0.1:3000";
/// `.invalid` never resolves, so a regression cannot reach a real machine.
const REMOTE: &str = "gpu-box.invalid:11434";

fn offline() {
    vox::privacy::set_offline(true);
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

fn chat_req() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat")
        .header(header::HOST, HOST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"text":"hello"}"#))
        .unwrap()
}

async fn json(response: axum::http::Response<Body>) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).expect("response is JSON")
}

#[tokio::test]
async fn chat_through_a_remote_ollama_is_refused() {
    offline();
    let response = app(REMOTE).oneshot(chat_req()).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let body = json(response).await;
    assert_eq!(body["error"]["code"], "offline_mode", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("offline mode"), "{message}");
    assert!(message.contains(REMOTE), "{message}");
}

#[tokio::test]
async fn model_list_from_a_remote_ollama_is_refused() {
    offline();
    let response = app(REMOTE)
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(json(response).await["error"]["code"], "offline_mode");
}

#[tokio::test]
async fn status_probe_reports_a_remote_ollama_as_disconnected() {
    offline();
    let response = app(REMOTE)
        .oneshot(get_req("/v1/vox/models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json(response).await;
    assert_eq!(body["ollama"]["connected"], false, "{body}");
}

#[tokio::test]
async fn streamed_chat_to_a_remote_ollama_is_refused() {
    offline();
    let err = vox::streaming_chat::stream_chat(
        &reqwest::Client::new(),
        REMOTE,
        "llama3.2",
        &[vox::conversation::ChatMessage::user("hi")],
        &vox::streaming_chat::StreamOptions::default(),
        tokio_util::sync::CancellationToken::new(),
        |_| {},
        |_| async { Ok(()) },
    )
    .await
    .expect_err("Live Talk, Converse and `vox chat` all go through here");
    let message = err.to_string();
    assert!(message.contains("offline mode"), "{message}");
    assert!(message.contains(REMOTE), "{message}");
}

#[tokio::test]
async fn capability_probe_of_a_remote_ollama_finds_nothing() {
    offline();
    let models = vox::capabilities::ollama::probe(&reqwest::Client::new(), REMOTE)
        .await
        .unwrap();
    assert!(models.is_empty());
}

#[tokio::test]
async fn server_refuses_to_start_with_a_remote_ollama() {
    offline();
    let err = server::run_with_config(
        "127.0.0.1",
        0,
        false,
        LoadOptions::default(),
        ServerOptions {
            ollama_host: Some(format!("http://{REMOTE}/")),
            ..ServerOptions::default()
        },
    )
    .await
    .expect_err("must fail before loading models or binding");
    let message = err.to_string();
    assert!(message.contains("offline mode"), "{message}");
    assert!(message.contains(REMOTE), "{message}");
}

#[tokio::test]
async fn refused_attempts_are_in_the_ledger_and_nothing_was_sent() {
    offline();
    let app = app(REMOTE);
    app.clone().oneshot(chat_req()).await.unwrap();
    app.clone()
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();

    let body = json(app.oneshot(get_req("/v1/privacy/ledger")).await.unwrap()).await;
    assert_eq!(body["offline"], true);
    assert_eq!(body["ollama_host_loopback"], false);
    assert_eq!(
        body["fully_local"], true,
        "nothing left the machine: {body}"
    );
    assert_eq!(body["totals"]["non_loopback_connections"], 0);
    assert_eq!(body["totals"]["non_loopback_bytes_sent"], 0);

    let remote: Vec<_> = body["destinations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["destination"] == REMOTE)
        .collect();
    assert!(!remote.is_empty(), "{body}");
    for row in &remote {
        assert_eq!(row["connections"], 0, "{row}");
        assert_eq!(row["bytes_sent"], 0, "{row}");
        assert!(row["blocked"].as_u64().unwrap() >= 1, "{row}");
    }
    assert!(remote.iter().any(|row| row["purpose"] == "llm chat"));
    assert!(remote.iter().any(|row| row["purpose"] == "llm model list"));
}

#[tokio::test]
async fn a_loopback_ollama_keeps_working_offline() {
    offline();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let router = Router::new().route(
        "/api/tags",
        get(move || {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                r#"{"models":[{"name":"llama3.2:latest","size":1,"details":{"family":"llama"}}]}"#
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ollama = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let response = app(&ollama)
        .oneshot(get_req("/v1/ollama-models"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json(response).await;
    assert_eq!(body["models"][0]["name"], "llama3.2:latest", "{body}");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}
