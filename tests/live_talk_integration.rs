//! Integration tests for Live Talk WebSocket wire protocol and cancellable streaming.

#![cfg(feature = "server")]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

#[path = "../src/server/mod.rs"]
mod server;

use server::models::{
    CancelReason, LiveTalkClientMsg, LiveTalkMode, LiveTalkModeConfig, LiveTalkWsEvent,
};

#[test]
fn ready_event_serializes_with_mode() {
    let ev = LiveTalkWsEvent::Ready {
        model: "llama3.2".into(),
        voice: Some("en_US-amy-medium".into()),
        mode: LiveTalkMode::Vad,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "ready");
    assert_eq!(json["model"], "llama3.2");
    assert_eq!(json["voice"], "en_US-amy-medium");
    assert_eq!(json["mode"], "vad");
}

#[test]
fn ready_event_push_to_talk_mode() {
    let ev = LiveTalkWsEvent::Ready {
        model: "llama3.2".into(),
        voice: None,
        mode: LiveTalkMode::PushToTalk,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "ready");
    assert_eq!(json["mode"], "push_to_talk");
    assert!(json["voice"].is_null());
}

#[test]
fn speech_start_event_is_bare_tagged_unit() {
    let ev = LiveTalkWsEvent::SpeechStart;
    let s = serde_json::to_string(&ev).unwrap();
    assert_eq!(s, r#"{"type":"speech_start"}"#);
}

#[test]
fn transcript_event_includes_timing() {
    let ev = LiveTalkWsEvent::Transcript {
        text: "hello world".into(),
        duration_ms: 1234,
        processing_time_ms: 56,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "transcript");
    assert_eq!(json["text"], "hello world");
    assert_eq!(json["duration_ms"], 1234);
    assert_eq!(json["processing_time_ms"], 56);
}

#[test]
fn sentence_event_text_only_no_audio_field() {
    let ev = LiveTalkWsEvent::Sentence {
        index: 0,
        text: "Hi there.".into(),
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "sentence");
    assert_eq!(json["index"], 0);
    assert_eq!(json["text"], "Hi there.");
    assert!(json.get("audio_b64").is_none());
    assert!(json.get("sample_rate").is_none());
}

#[test]
fn audio_chunk_event_is_header_only() {
    let ev = LiveTalkWsEvent::AudioChunk {
        sentence_index: 2,
        sample_rate: 22050,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "audio_chunk");
    assert_eq!(json["sentence_index"], 2);
    assert_eq!(json["sample_rate"], 22050);
    assert!(json.get("audio_b64").is_none());
    assert!(json.get("data").is_none());
}

#[test]
fn turn_done_event_reports_sentence_count() {
    let ev = LiveTalkWsEvent::TurnDone { sentences: 3 };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "turn_done");
    assert_eq!(json["sentences"], 3);
}

#[test]
fn cancelled_event_serializes_reason() {
    let ev = LiveTalkWsEvent::Cancelled {
        reason: CancelReason::UserBargeIn,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "cancelled");
    assert_eq!(json["reason"], "user_barge_in");

    let ev2 = LiveTalkWsEvent::Cancelled {
        reason: CancelReason::ClientRequest,
    };
    let json2: serde_json::Value = serde_json::to_value(&ev2).unwrap();
    assert_eq!(json2["reason"], "client_request");
}

#[test]
fn error_event_has_fatal_flag() {
    let ev = LiveTalkWsEvent::Error {
        message: "ollama unreachable".into(),
        fatal: true,
    };
    let json: serde_json::Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["message"], "ollama unreachable");
    assert_eq!(json["fatal"], true);
}

#[test]
fn client_config_deserializes_with_defaults() {
    let raw = r#"{
        "type": "config",
        "model": "llama3.2",
        "voice": "en_US-amy-medium"
    }"#;
    let msg: LiveTalkClientMsg = serde_json::from_str(raw).unwrap();
    match msg {
        LiveTalkClientMsg::Config(cfg) => {
            assert_eq!(cfg.model.as_deref(), Some("llama3.2"));
            assert_eq!(cfg.voice.as_deref(), Some("en_US-amy-medium"));
            assert!(matches!(cfg.mode, LiveTalkModeConfig::Vad));
            assert!(cfg.barge_in_enabled);
            assert!(cfg.host.is_none());
            assert!(cfg.system_prompt_override.is_none());
        }
        other => panic!("expected Config, got {:?}", other),
    }
}

#[test]
fn client_config_deserializes_push_to_talk() {
    let raw = r#"{
        "type": "config",
        "mode": "push_to_talk",
        "barge_in_enabled": false
    }"#;
    let msg: LiveTalkClientMsg = serde_json::from_str(raw).unwrap();
    match msg {
        LiveTalkClientMsg::Config(cfg) => {
            assert!(matches!(cfg.mode, LiveTalkModeConfig::PushToTalk));
            assert!(!cfg.barge_in_enabled);
        }
        other => panic!("expected Config, got {:?}", other),
    }
}

#[test]
fn client_cancel_deserializes() {
    let msg: LiveTalkClientMsg = serde_json::from_str(r#"{"type":"cancel"}"#).unwrap();
    assert!(matches!(msg, LiveTalkClientMsg::Cancel));
}

#[test]
fn client_ptt_start_and_end_deserialize() {
    let start: LiveTalkClientMsg = serde_json::from_str(r#"{"type":"ptt_start"}"#).unwrap();
    assert!(matches!(start, LiveTalkClientMsg::PttStart));
    let end: LiveTalkClientMsg = serde_json::from_str(r#"{"type":"ptt_end"}"#).unwrap();
    assert!(matches!(end, LiveTalkClientMsg::PttEnd));
}

#[test]
fn client_unknown_variant_errors_cleanly() {
    let err = serde_json::from_str::<LiveTalkClientMsg>(r#"{"type":"bogus"}"#);
    assert!(err.is_err());
}

#[test]
fn mode_config_to_runtime_mode_conversion() {
    let vad: LiveTalkMode = LiveTalkModeConfig::Vad.into();
    let ptt: LiveTalkMode = LiveTalkModeConfig::PushToTalk.into();
    assert!(matches!(vad, LiveTalkMode::Vad));
    assert!(matches!(ptt, LiveTalkMode::PushToTalk));
}

#[allow(dead_code)]
struct NoopTts;

#[async_trait]
impl vox::traits::TtsBackend for NoopTts {
    async fn synthesize(
        &self,
        _request: &vox::types::TtsRequest,
    ) -> Result<vox::types::TtsOutput, vox::VoxError> {
        Ok(vox::types::TtsOutput {
            audio: vox::types::AudioChunk {
                samples: vec![],
                sample_rate: 22050,
                channels: 1,
            },
            duration_ms: 0,
        })
    }

    fn backend_name(&self) -> &str {
        "noop"
    }
}

async fn ollama_available(host: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .unwrap();
    let url = format!("http://{}/api/tags", host);
    matches!(client.get(&url).send().await, Ok(r) if r.status().is_success())
}

async fn ollama_resolve_model(host: &str, model: &str) -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let url = format!("http://{}/api/tags", host);
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    let models = body.get("models")?.as_array()?;
    let mut prefix_match: Option<String> = None;
    for m in models {
        if let Some(name) = m.get("name").and_then(|n| n.as_str()) {
            if name == model {
                return Some(name.to_string());
            }
            if prefix_match.is_none() && name.starts_with(&format!("{model}:")) {
                prefix_match = Some(name.to_string());
            }
        }
    }
    prefix_match
}

#[tokio::test]
async fn cancel_token_aborts_streaming_turn() {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "localhost:11434".into());

    if !ollama_available(&host).await {
        eprintln!("skipping: Ollama not reachable at {host}");
        return;
    }

    let requested_model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into());

    let Some(model) = ollama_resolve_model(&host, &requested_model).await else {
        eprintln!(
            "skipping: Ollama reachable at {host} but no model matching '{requested_model}' is installed (set OLLAMA_MODEL to an installed model to run this test)"
        );
        return;
    };
    eprintln!("using Ollama model: {model}");
    let client = reqwest::Client::new();
    let tts: Arc<dyn vox::traits::TtsBackend> = Arc::new(NoopTts);
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    let result = vox::streaming_chat::stream_chat_with_tts_cancellable(
        &client,
        &host,
        &model,
        "Please describe, at length and in great detail, the history of the Roman Empire from the founding of Rome through the fall of the Western Empire. Use at least five long sentences.",
        tts,
        Some("You are a helpful verbose assistant.".into()),
        None,
        cancel,
        |_sentence| async move { Ok(()) },
    )
    .await;

    match result {
        Ok(reason) => {
            assert_eq!(reason, vox::streaming_chat::StopReason::Cancelled);
        }
        Err(e) => panic!("streaming call errored instead of cancelling: {e:?}"),
    }
}

#[tokio::test]
async fn precancelled_token_returns_immediately() {
    let client = reqwest::Client::new();
    let tts: Arc<dyn vox::traits::TtsBackend> = Arc::new(NoopTts);
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();

    let start = std::time::Instant::now();
    let result = vox::streaming_chat::stream_chat_with_tts_cancellable(
        &client,
        "127.0.0.1:1",
        "llama3.2",
        "hi",
        tts,
        None,
        None,
        cancel,
        |_sentence| async move { Ok(()) },
    )
    .await;
    let elapsed = start.elapsed();

    assert!(elapsed < Duration::from_secs(2));
    match result {
        Ok(reason) => assert_eq!(reason, vox::streaming_chat::StopReason::Cancelled),
        Err(e) => panic!("expected Cancelled, got error: {e:?}"),
    }
}
