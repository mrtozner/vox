//! Integration tests for the Live Talk protocol v2 and `/v1/converse`.
//!
//! The session tests drive the real session loops
//! ([`server::live_talk::run_live_talk`], [`server::ws_converse::run_converse`])
//! over in-memory channels, with a scripted VAD, STT and TTS and a fake
//! Ollama on a local TCP port. No model files and no network are needed.

#![cfg(feature = "server")]

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::extract::ws::Message;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

#[path = "../src/server/mod.rs"]
mod server;

use server::live_talk::{LiveTalkDeps, run_live_talk};
use server::models::{
    CancelReason, ConverseConfig, LIVE_TALK_PROTOCOL_VERSION, LiveTalkClientMsg, LiveTalkConfig,
    LiveTalkMode, LiveTalkModeConfig, LiveTalkWsEvent,
};
use server::ws_converse::{ConverseDeps, run_converse};
use vox::conversation::{Conversation, SessionStore};
use vox::streaming_chat::StreamOptions;

// ─── Wire types ──────────────────────────────────────────────

#[test]
fn ready_event_carries_protocol_version_and_session_id() {
    let ev = LiveTalkWsEvent::Ready {
        model: "llama3.2".into(),
        voice: Some("en_US-amy-medium".into()),
        mode: LiveTalkMode::Vad,
        protocol_version: LIVE_TALK_PROTOCOL_VERSION,
        session_id: "abc12345".into(),
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "ready");
    assert_eq!(json["model"], "llama3.2");
    assert_eq!(json["voice"], "en_US-amy-medium");
    assert_eq!(json["mode"], "vad");
    assert_eq!(json["protocol_version"], 2);
    assert_eq!(json["session_id"], "abc12345");
}

#[test]
fn ready_event_push_to_talk_mode() {
    let ev = LiveTalkWsEvent::Ready {
        model: "llama3.2".into(),
        voice: None,
        mode: LiveTalkMode::PushToTalk,
        protocol_version: LIVE_TALK_PROTOCOL_VERSION,
        session_id: "abc12345".into(),
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["mode"], "push_to_talk");
    assert!(json["voice"].is_null());
}

#[test]
fn speech_start_and_end_are_bare_tagged_units() {
    let start = serde_json::to_string(&LiveTalkWsEvent::SpeechStart).unwrap();
    assert_eq!(start, r#"{"type":"speech_start"}"#);
    let end = serde_json::to_string(&LiveTalkWsEvent::SpeechEnd).unwrap();
    assert_eq!(end, r#"{"type":"speech_end"}"#);
}

#[test]
fn transcript_event_includes_timing_and_omits_unknown_speaker() {
    let ev = LiveTalkWsEvent::Transcript {
        turn_id: 4,
        text: "hello world".into(),
        duration_ms: 1234,
        processing_time_ms: 56,
        speaker_id: None,
        speaker_name: None,
        speaker_color: None,
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "transcript");
    assert_eq!(json["turn_id"], 4);
    assert_eq!(json["text"], "hello world");
    assert_eq!(json["duration_ms"], 1234);
    assert_eq!(json["processing_time_ms"], 56);
    assert!(json.get("speaker_id").is_none());
    assert!(json.get("speaker_name").is_none());
    assert!(json.get("speaker_color").is_none());

    let ev = LiveTalkWsEvent::Transcript {
        turn_id: 4,
        text: "hi".into(),
        duration_ms: 1,
        processing_time_ms: 1,
        speaker_id: Some("spk_1".into()),
        speaker_name: Some("Ada".into()),
        speaker_color: Some("#aa3366".into()),
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["speaker_id"], "spk_1");
    assert_eq!(json["speaker_name"], "Ada");
    assert_eq!(json["speaker_color"], "#aa3366");
}

#[test]
fn turn_scoped_events_serialize_turn_id() {
    let events = [
        LiveTalkWsEvent::Thinking { turn_id: 7 },
        LiveTalkWsEvent::Token {
            turn_id: 7,
            text: "Hi".into(),
        },
        LiveTalkWsEvent::Sentence {
            turn_id: 7,
            index: 0,
            text: "Hi there.".into(),
        },
        LiveTalkWsEvent::AudioChunk {
            turn_id: 7,
            sentence_index: 0,
            sample_rate: 22050,
        },
        LiveTalkWsEvent::TurnMetrics {
            turn_id: 7,
            stt_ms: 10,
            llm_first_token_ms: Some(20),
            tts_first_audio_ms: None,
        },
        LiveTalkWsEvent::TurnDone {
            turn_id: 7,
            sentences: 1,
        },
        LiveTalkWsEvent::Cancelled {
            turn_id: 7,
            reason: CancelReason::UserBargeIn,
        },
    ];
    let types: Vec<String> = events
        .iter()
        .map(|ev| {
            let json: Value = serde_json::to_value(ev).unwrap();
            assert_eq!(json["turn_id"], 7, "{json}");
            json["type"].as_str().unwrap().to_string()
        })
        .collect();
    assert_eq!(
        types,
        [
            "thinking",
            "token",
            "sentence",
            "audio_chunk",
            "turn_metrics",
            "turn_done",
            "cancelled"
        ]
    );
}

#[test]
fn sentence_and_audio_chunk_carry_no_audio() {
    let ev = LiveTalkWsEvent::Sentence {
        turn_id: 1,
        index: 0,
        text: "Hi there.".into(),
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["index"], 0);
    assert_eq!(json["text"], "Hi there.");
    assert!(json.get("audio_b64").is_none());

    let ev = LiveTalkWsEvent::AudioChunk {
        turn_id: 1,
        sentence_index: 2,
        sample_rate: 22050,
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["sentence_index"], 2);
    assert_eq!(json["sample_rate"], 22050);
    assert!(json.get("audio_b64").is_none());
    assert!(json.get("data").is_none());
}

#[test]
fn turn_metrics_uses_null_for_a_stage_that_was_not_reached() {
    let ev = LiveTalkWsEvent::TurnMetrics {
        turn_id: 1,
        stt_ms: 12,
        llm_first_token_ms: None,
        tts_first_audio_ms: None,
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "turn_metrics");
    assert_eq!(json["stt_ms"], 12);
    assert!(json["llm_first_token_ms"].is_null());
    assert!(json["tts_first_audio_ms"].is_null());
}

#[test]
fn cancelled_event_serializes_reason() {
    let reasons = [
        (CancelReason::UserBargeIn, "user_barge_in"),
        (CancelReason::ClientRequest, "client_request"),
    ];
    for (reason, wire) in reasons {
        let ev = LiveTalkWsEvent::Cancelled { turn_id: 3, reason };
        let json: Value = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["type"], "cancelled");
        assert_eq!(json["reason"], wire);
    }
}

#[test]
fn memory_event_is_flat() {
    let mut conversation = Conversation::with_budget(1000);
    conversation.commit("hi", &["Hello.".to_string()], true);
    let ev = LiveTalkWsEvent::Memory(conversation.snapshot());
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "memory");
    assert_eq!(json["turns"][0]["user"], "hi");
    assert_eq!(json["turns"][0]["assistant"], "Hello.");
    assert_eq!(json["turns"][0]["interrupted"], true);
    assert!(json["turns"][0]["tokens"].as_u64().unwrap() > 0);
    assert_eq!(json["total_tokens"], json["turns"][0]["tokens"]);
    assert_eq!(json["budget_tokens"], 1000);
    assert_eq!(json["low_watermark_tokens"], 600);
    assert_eq!(json["dropped_turns"], 0);
}

#[test]
fn error_event_has_fatal_flag_and_optional_turn_id() {
    let ev = LiveTalkWsEvent::Error {
        message: "ollama unreachable".into(),
        fatal: true,
        turn_id: None,
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["message"], "ollama unreachable");
    assert_eq!(json["fatal"], true);
    assert!(json.get("turn_id").is_none());

    let ev = LiveTalkWsEvent::Error {
        message: "stalled".into(),
        fatal: false,
        turn_id: Some(9),
    };
    let json: Value = serde_json::to_value(&ev).unwrap();
    assert_eq!(json["fatal"], false);
    assert_eq!(json["turn_id"], 9);
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
            assert!(cfg.system_prompt_override.is_none());
            assert!(cfg.session_id.is_none());
        }
        other => panic!("expected Config, got {:?}", other),
    }
}

#[test]
fn client_config_deserializes_push_to_talk_and_session_id() {
    let raw = r#"{
        "type": "config",
        "mode": "push_to_talk",
        "barge_in_enabled": false,
        "session_id": "0123456789abcdef"
    }"#;
    let msg: LiveTalkClientMsg = serde_json::from_str(raw).unwrap();
    match msg {
        LiveTalkClientMsg::Config(cfg) => {
            assert!(matches!(cfg.mode, LiveTalkModeConfig::PushToTalk));
            assert!(!cfg.barge_in_enabled);
            assert_eq!(cfg.session_id.as_deref(), Some("0123456789abcdef"));
        }
        other => panic!("expected Config, got {:?}", other),
    }
}

/// The Ollama host is server configuration. A client that still sends
/// `host` (protocol v1) is accepted and the field has no effect: neither
/// config type has such a field any more.
#[test]
fn client_supplied_host_is_ignored() {
    let raw = r#"{"type":"config","model":"m","host":"evil.example:80"}"#;
    let msg: LiveTalkClientMsg = serde_json::from_str(raw).unwrap();
    assert!(matches!(msg, LiveTalkClientMsg::Config(_)));

    let bare: LiveTalkConfig =
        serde_json::from_str(r#"{"model":"m","host":"evil.example:80"}"#).unwrap();
    assert_eq!(bare.model.as_deref(), Some("m"));

    let converse: ConverseConfig =
        serde_json::from_str(r#"{"model":"m","host":"evil.example:80","voice":"v"}"#).unwrap();
    assert_eq!(converse.model.as_deref(), Some("m"));
    assert_eq!(converse.voice.as_deref(), Some("v"));
}

#[test]
fn cancel_parses_the_v1_body_and_the_v2_fields() {
    let msg: LiveTalkClientMsg = serde_json::from_str(r#"{"type":"cancel"}"#).unwrap();
    assert!(matches!(
        msg,
        LiveTalkClientMsg::Cancel {
            turn_id: None,
            heard_sentences: None
        }
    ));

    let msg: LiveTalkClientMsg =
        serde_json::from_str(r#"{"type":"cancel","heard_sentences":2}"#).unwrap();
    assert!(matches!(
        msg,
        LiveTalkClientMsg::Cancel {
            turn_id: None,
            heard_sentences: Some(2)
        }
    ));

    let msg: LiveTalkClientMsg =
        serde_json::from_str(r#"{"type":"cancel","turn_id":5,"heard_sentences":0}"#).unwrap();
    assert!(matches!(
        msg,
        LiveTalkClientMsg::Cancel {
            turn_id: Some(5),
            heard_sentences: Some(0)
        }
    ));
}

#[test]
fn playback_done_and_forget_memory_deserialize() {
    let msg: LiveTalkClientMsg =
        serde_json::from_str(r#"{"type":"playback_done","turn_id":3,"heard_sentences":4}"#)
            .unwrap();
    assert!(matches!(
        msg,
        LiveTalkClientMsg::PlaybackDone {
            turn_id: 3,
            heard_sentences: Some(4)
        }
    ));
    // `turn_id` is required: without it the server cannot tell which
    // playback ended.
    assert!(serde_json::from_str::<LiveTalkClientMsg>(r#"{"type":"playback_done"}"#).is_err());

    let msg: LiveTalkClientMsg = serde_json::from_str(r#"{"type":"forget_memory"}"#).unwrap();
    assert!(matches!(msg, LiveTalkClientMsg::ForgetMemory));
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

// ─── Fakes ───────────────────────────────────────────────────

const FRAME: usize = 512;

#[derive(Default)]
struct VadStats {
    resets: AtomicUsize,
    frames: AtomicUsize,
}

/// A VAD that calls a frame speech when its mean absolute level is above
/// 0.5. Speech starts at the first such frame and ends at the first frame
/// below it.
struct FakeVad {
    stats: Arc<VadStats>,
    speech: Option<Vec<f32>>,
}

impl FakeVad {
    fn new(stats: &Arc<VadStats>) -> Box<Self> {
        Box::new(Self {
            stats: Arc::clone(stats),
            speech: None,
        })
    }
}

#[async_trait]
impl vox::traits::VadBackend for FakeVad {
    async fn process_frame(
        &mut self,
        frame: &vox::AudioChunk,
    ) -> Result<Vec<vox::VadEvent>, vox::VoxError> {
        assert_eq!(frame.samples.len(), FRAME);
        self.stats.frames.fetch_add(1, Ordering::SeqCst);
        let level = frame.samples.iter().map(|s| s.abs()).sum::<f32>() / FRAME as f32;
        let loud = level > 0.5;
        let mut events = Vec::new();
        match (&mut self.speech, loud) {
            (None, true) => {
                self.speech = Some(frame.samples.clone());
                events.push(vox::VadEvent::SpeechStart);
            }
            (Some(buffer), true) => buffer.extend_from_slice(&frame.samples),
            (Some(_), false) => {
                let samples = self.speech.take().unwrap_or_default();
                let duration_ms = samples.len() as u64 * 1000 / 16_000;
                events.push(vox::VadEvent::SpeechEnd(vox::Utterance {
                    audio: vox::AudioChunk {
                        samples,
                        sample_rate: 16_000,
                        channels: 1,
                    },
                    duration_ms,
                    #[cfg(feature = "diarization")]
                    speaker_id: None,
                }));
            }
            (None, false) => events.push(vox::VadEvent::Silence),
        }
        Ok(events)
    }

    fn reset(&mut self) {
        self.stats.resets.fetch_add(1, Ordering::SeqCst);
        self.speech = None;
    }

    fn frame_size(&self) -> usize {
        FRAME
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }
}

/// An STT that answers "utterance 1", "utterance 2", ... and records how
/// many samples each call got. `delays` are consumed one per call.
#[derive(Default)]
struct FakeStt {
    calls: Mutex<Vec<usize>>,
    delays: Mutex<VecDeque<Duration>>,
}

#[async_trait]
impl vox::traits::SttBackend for FakeStt {
    async fn transcribe(&self, audio: &vox::Utterance) -> Result<vox::SttResult, vox::VoxError> {
        let n = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(audio.audio.samples.len());
            calls.len()
        };
        let delay = self.delays.lock().unwrap().pop_front();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        Ok(vox::SttResult {
            text: format!("utterance {n}"),
            language: None,
            duration_ms: audio.duration_ms,
            processing_time_ms: 1,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        })
    }
}

/// A TTS that returns 10 ms of silence and records what it was asked for.
#[derive(Default)]
struct FakeTts {
    requests: Mutex<Vec<(String, Option<String>)>>,
}

#[async_trait]
impl vox::traits::TtsBackend for FakeTts {
    async fn synthesize(&self, request: &vox::TtsRequest) -> Result<vox::TtsOutput, vox::VoxError> {
        self.requests
            .lock()
            .unwrap()
            .push((request.text.clone(), request.voice.clone()));
        Ok(vox::TtsOutput::from_mono(vec![0.0; 160], 16_000))
    }

    fn backend_name(&self) -> &str {
        "fake"
    }
}

/// What the fake Ollama does with one request.
#[derive(Clone, Default)]
struct Script {
    /// Reply text, one stream chunk per entry.
    tokens: Vec<&'static str>,
    /// Pause before each chunk after the first.
    token_delay: Duration,
    /// Pause before the first chunk.
    start_delay: Duration,
    /// Accept the request and never answer.
    silent: bool,
}

fn reply(tokens: &[&'static str]) -> Script {
    Script {
        tokens: tokens.to_vec(),
        ..Script::default()
    }
}

/// One sentence, then the start of a second one that a script can pause
/// in. The segmenter confirms a sentence end only once it has seen what
/// follows, so the first chunk already carries the next word.
const PAUSE_AFTER_FIRST_SENTENCE: [&str; 2] = ["Sentence one is here. And", " then a second one."];

const THREE_SENTENCES: [&str; 3] = [
    "Sentence one is here.",
    " Sentence two follows.",
    " Sentence three ends.",
];

/// Read one HTTP request and return its JSON body. `None` when the client
/// went away first (a turn that was cancelled while sending).
async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Value> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        if let Some(head_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
            let len: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            let body_start = head_end + 4;
            if buf.len() >= body_start + len {
                return serde_json::from_slice(&buf[body_start..body_start + len]).ok();
            }
        }
        match sock.read(&mut tmp).await {
            Ok(n) if n > 0 => buf.extend_from_slice(&tmp[..n]),
            _ => return None,
        }
    }
}

/// Fake Ollama: answers request N with script N. Returns its `host:port`
/// and the request bodies it received.
async fn fake_ollama(scripts: Vec<Script>) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let host = listener.local_addr().unwrap().to_string();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    tokio::spawn(async move {
        for script in scripts {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                let Some(request) = read_request(&mut sock).await else {
                    return;
                };
                seen.lock().unwrap().push(request);
                if script.silent {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    return;
                }
                tokio::time::sleep(script.start_delay).await;
                let head = "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\nconnection: close\r\n\r\n";
                if sock.write_all(head.as_bytes()).await.is_err() {
                    return;
                }
                for (i, token) in script.tokens.iter().enumerate() {
                    if i > 0 {
                        tokio::time::sleep(script.token_delay).await;
                    }
                    let line = json!({
                        "message": { "role": "assistant", "content": token },
                        "done": false,
                    });
                    if sock
                        .write_all(format!("{line}\n").as_bytes())
                        .await
                        .is_err()
                    {
                        return;
                    }
                    let _ = sock.flush().await;
                }
                let done = json!({ "done": true, "prompt_eval_count": 5, "eval_count": 7 });
                let _ = sock.write_all(format!("{done}\n").as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (host, requests)
}

// ─── Test client ─────────────────────────────────────────────

/// Everything a test wants to look at afterwards.
struct Rig {
    vad: Arc<VadStats>,
    stt: Arc<FakeStt>,
    tts: Arc<FakeTts>,
    sessions: Arc<SessionStore>,
    host: String,
    llm_requests: Arc<Mutex<Vec<Value>>>,
    stream_options: StreamOptions,
    playback_grace: Duration,
}

impl Rig {
    async fn new(scripts: Vec<Script>) -> Self {
        let (host, llm_requests) = fake_ollama(scripts).await;
        Self {
            vad: Arc::new(VadStats::default()),
            stt: Arc::new(FakeStt::default()),
            tts: Arc::new(FakeTts::default()),
            sessions: Arc::new(SessionStore::new(Duration::from_secs(600), 16)),
            host,
            llm_requests,
            stream_options: StreamOptions {
                first_token_timeout: Duration::from_secs(10),
                idle_timeout: Duration::from_secs(10),
            },
            // Long, so a test that goes on after a spoken turn has to say
            // `playback_done` (or cancel) like a real client.
            playback_grace: Duration::from_secs(60),
        }
    }

    /// Open a Live Talk connection and send `config` as the first message.
    fn live(&self, config: Value) -> Client {
        let (client, sink, stream) = Client::pipe();
        let deps = LiveTalkDeps {
            stt: self.stt.clone(),
            tts: self.tts.clone(),
            vad: FakeVad::new(&self.vad),
            http_client: reqwest::Client::new(),
            ollama_host: self.host.clone(),
            sessions: Arc::clone(&self.sessions),
            stream_options: self.stream_options,
            playback_grace: self.playback_grace,
        };
        tokio::spawn(run_live_talk(sink, stream, deps));
        client.send(config);
        client
    }

    /// Open a converse connection and send `config` as the first message.
    fn converse(&self, config: Value) -> Client {
        let (client, sink, stream) = Client::pipe();
        let deps = ConverseDeps {
            stt: self.stt.clone(),
            tts: self.tts.clone(),
            vad: FakeVad::new(&self.vad),
            http_client: reqwest::Client::new(),
            ollama_host: self.host.clone(),
            stream_options: self.stream_options,
            post_turn_mute: Duration::ZERO,
        };
        tokio::spawn(run_converse(sink, stream, deps));
        client.send(config);
        client
    }
}

type ServerSink = std::pin::Pin<Box<dyn futures_util::Sink<Message, Error = ()> + Send>>;
type ServerStream = tokio_stream::wrappers::UnboundedReceiverStream<Result<Message, Infallible>>;

/// The client end of an in-memory connection.
struct Client {
    tx: mpsc::UnboundedSender<Result<Message, Infallible>>,
    rx: mpsc::UnboundedReceiver<Message>,
}

impl Client {
    fn pipe() -> (Self, ServerSink, ServerStream) {
        let (to_server, from_client) = mpsc::unbounded_channel();
        let (to_client, from_server) = mpsc::unbounded_channel::<Message>();
        let sink: ServerSink = Box::pin(futures_util::sink::unfold(
            to_client,
            |tx, msg: Message| async move {
                tx.send(msg).map_err(|_| ())?;
                Ok::<_, ()>(tx)
            },
        ));
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(from_client);
        (
            Self {
                tx: to_server,
                rx: from_server,
            },
            sink,
            stream,
        )
    }

    fn send(&self, msg: Value) {
        self.tx
            .send(Ok(Message::Text(msg.to_string().into())))
            .expect("server is gone");
    }

    fn send_audio(&self, samples: &[f32]) {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.tx
            .send(Ok(Message::Binary(bytes.into())))
            .expect("server is gone");
    }

    /// Three loud frames and a quiet one: one utterance for [`FakeVad`].
    fn speak(&self) {
        let mut samples = vec![0.9f32; FRAME * 3];
        samples.extend(std::iter::repeat_n(0.0, FRAME));
        self.send_audio(&samples);
    }

    /// The next message. A binary frame comes back as
    /// `{"type":"<binary>","len":n}`. `None` when nothing arrives in `wait`.
    async fn next_within(&mut self, wait: Duration) -> Option<Value> {
        match tokio::time::timeout(wait, self.rx.recv()).await {
            Ok(Some(Message::Text(text))) => {
                Some(serde_json::from_str(text.as_str()).expect("server sent invalid JSON"))
            }
            Ok(Some(Message::Binary(data))) => Some(json!({"type": "<binary>", "len": data.len()})),
            Ok(Some(_)) | Ok(None) | Err(_) => None,
        }
    }

    /// Read until an event of type `ty`; returns everything read, that event
    /// last. Panics if it does not come.
    async fn until(&mut self, ty: &str) -> Vec<Value> {
        let mut seen = Vec::new();
        loop {
            let Some(event) = self.next_within(Duration::from_secs(5)).await else {
                panic!("no `{ty}` event; got {seen:#?}");
            };
            let found = event["type"] == ty;
            seen.push(event);
            if found {
                return seen;
            }
        }
    }

    /// Read until an event of type `ty` and return it.
    async fn expect(&mut self, ty: &str) -> Value {
        self.until(ty).await.pop().expect("until returns the event")
    }

    /// Everything that arrives within `wait`.
    async fn drain(&mut self, wait: Duration) -> Vec<Value> {
        let deadline = tokio::time::Instant::now() + wait;
        let mut seen = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                return seen;
            }
            match self.next_within(left).await {
                Some(event) => seen.push(event),
                None => return seen,
            }
        }
    }
}

fn types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap_or("?"))
        .collect()
}

fn of_type<'a>(events: &'a [Value], ty: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == ty).collect()
}

fn vad_config() -> Value {
    json!({"type": "config", "model": "test-model"})
}

/// Give the session loop time to handle what was sent.
async fn settle() {
    tokio::time::sleep(Duration::from_millis(100)).await;
}

// ─── Live Talk sessions ──────────────────────────────────────

#[tokio::test]
async fn a_turn_emits_the_v2_event_sequence() {
    let rig = Rig::new(vec![reply(&THREE_SENTENCES)]).await;
    let mut client = rig.live(vad_config());

    let ready = client.expect("ready").await;
    assert_eq!(ready["protocol_version"], 2);
    assert_eq!(ready["model"], "test-model");
    assert_eq!(ready["mode"], "vad");
    let session_id = ready["session_id"].as_str().unwrap();
    assert!(SessionStore::is_valid_id(session_id));
    // A new session starts with an empty memory.
    let memory = client.expect("memory").await;
    assert_eq!(memory["turns"].as_array().unwrap().len(), 0);
    assert!(memory["budget_tokens"].as_u64().unwrap() > 0);

    client.speak();
    let events = client.until("turn_done").await;
    let kinds = types(&events);

    // speech_start, speech_end, transcript and thinking come in this order,
    // before any reply.
    let position = |ty: &str| {
        kinds
            .iter()
            .position(|k| *k == ty)
            .unwrap_or_else(|| panic!("no `{ty}` in {kinds:?}"))
    };
    assert!(position("speech_start") < position("speech_end"));
    assert!(position("speech_end") < position("transcript"));
    assert!(position("transcript") < position("thinking"));
    assert!(position("thinking") < position("token"));
    assert!(position("token") < position("sentence"));
    assert!(position("turn_metrics") < position("turn_done"));

    assert_eq!(of_type(&events, "transcript")[0]["text"], "utterance 1");

    // Raw tokens are forwarded as they came.
    let tokens: Vec<&str> = of_type(&events, "token")
        .iter()
        .map(|t| t["text"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, THREE_SENTENCES);

    // Each sentence is followed by its audio header, which is followed by
    // exactly one binary frame.
    let sentences = of_type(&events, "sentence");
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0]["text"], "Sentence one is here.");
    for (i, kind) in kinds.iter().enumerate() {
        if *kind == "audio_chunk" {
            assert_eq!(kinds[i - 1], "sentence");
            assert_eq!(kinds[i + 1], "<binary>");
            assert_eq!(events[i]["sentence_index"], events[i - 1]["index"]);
            assert_eq!(events[i]["sample_rate"], 16_000);
        }
    }
    assert_eq!(of_type(&events, "audio_chunk").len(), 3);
    assert_eq!(of_type(&events, "<binary>").len(), 3);

    // Every turn-scoped event carries the turn id.
    for event in &events {
        let ty = event["type"].as_str().unwrap();
        if [
            "transcript",
            "thinking",
            "token",
            "sentence",
            "audio_chunk",
            "turn_metrics",
            "turn_done",
        ]
        .contains(&ty)
        {
            assert_eq!(event["turn_id"], 1, "{event}");
        }
    }

    let metrics = of_type(&events, "turn_metrics")[0];
    assert!(metrics["stt_ms"].is_u64());
    assert!(metrics["llm_first_token_ms"].is_u64());
    assert!(metrics["tts_first_audio_ms"].is_u64());

    assert_eq!(events.last().unwrap()["sentences"], 3);

    // The turn is in memory, as heard so far (everything that was sent).
    let memory = of_type(&events, "memory");
    let memory = memory.last().expect("a memory event after the commit");
    assert_eq!(memory["turns"][0]["user"], "utterance 1");
    assert_eq!(
        memory["turns"][0]["assistant"],
        "Sentence one is here. Sentence two follows. Sentence three ends."
    );
    assert_eq!(memory["turns"][0]["interrupted"], false);

    // The LLM was asked with the configured model, through the server's
    // host.
    let requests = rig.llm_requests.lock().unwrap();
    assert_eq!(requests[0]["model"], "test-model");
}

#[tokio::test]
async fn turn_ids_increase_and_the_mic_waits_for_playback_done() {
    let rig = Rig::new(vec![reply(&THREE_SENTENCES), reply(&["Second reply."])]).await;
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    let first = client.until("turn_done").await;
    assert_eq!(first.last().unwrap()["turn_id"], 1);

    // The browser is still playing: the microphone hears the loudspeaker,
    // so this must not start a turn or even reach the VAD.
    let frames_before = rig.vad.frames.load(Ordering::SeqCst);
    client.speak();
    let quiet = client.drain(Duration::from_millis(300)).await;
    assert!(quiet.is_empty(), "gated audio caused {quiet:#?}");
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), frames_before);

    // Playback is over: listening again.
    client.send(json!({"type": "playback_done", "turn_id": 1, "heard_sentences": 3}));
    settle().await;
    client.speak();
    let second = client.until("turn_done").await;
    for event in &second {
        if let Some(turn_id) = event.get("turn_id") {
            assert_eq!(turn_id, 2, "{event}");
        }
    }
    assert_eq!(of_type(&second, "transcript")[0]["text"], "utterance 2");
    assert_eq!(of_type(&second, "speech_end").len(), 1);

    // Having heard everything, the first turn stays whole in memory.
    let memory = of_type(&second, "memory");
    let turns = memory.last().unwrap()["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0]["interrupted"], false);
    assert_eq!(turns[1]["assistant"], "Second reply.");

    // The second request carried the first turn as history.
    let requests = rig.llm_requests.lock().unwrap();
    let messages = requests[1]["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
    assert_eq!(messages[1]["content"], "utterance 1");
    assert_eq!(messages[3]["content"], "utterance 2");
}

#[tokio::test]
async fn the_gate_opens_by_itself_when_the_audio_must_be_over() {
    let mut rig = Rig::new(vec![reply(&["One reply."]), reply(&["Second reply."])]).await;
    // 10 ms of audio + 150 ms margin, and no `playback_done` from the client.
    rig.playback_grace = Duration::from_millis(150);
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    client.until("turn_done").await;
    tokio::time::sleep(Duration::from_millis(400)).await;

    client.speak();
    let second = client.until("turn_done").await;
    assert_eq!(second.last().unwrap()["turn_id"], 2);
}

#[tokio::test]
async fn cancel_after_turn_done_cuts_memory_to_what_was_heard() {
    let rig = Rig::new(vec![reply(&THREE_SENTENCES)]).await;
    let mut client = rig.live(vad_config());
    let session_id = client.expect("ready").await["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    client.speak();
    let events = client.until("turn_done").await;
    let memory = of_type(&events, "memory");
    assert_eq!(memory.last().unwrap()["turns"][0]["interrupted"], false);
    let tokens_before = memory.last().unwrap()["total_tokens"].as_u64().unwrap();

    // The user interrupts while the browser is still on the first sentence.
    // The server finished the turn long ago.
    client.send(json!({"type": "cancel", "turn_id": 1, "heard_sentences": 1}));
    let events = client.until("cancelled").await;

    let cancelled = events.last().unwrap();
    assert_eq!(cancelled["turn_id"], 1);
    assert_eq!(cancelled["reason"], "client_request");

    let memory = of_type(&events, "memory");
    assert_eq!(memory.len(), 1, "one memory event for the correction");
    let turns = memory[0]["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0]["assistant"], "Sentence one is here.");
    assert_eq!(turns[0]["interrupted"], true);
    assert!(memory[0]["total_tokens"].as_u64().unwrap() < tokens_before);

    // The same correction again changes nothing and says nothing new.
    client.send(json!({"type": "cancel", "heard_sentences": 1}));
    let events = client.until("cancelled").await;
    assert!(of_type(&events, "memory").is_empty());

    // What the model will see next time.
    let handle = rig.sessions.attach(Some(&session_id));
    assert!(handle.resumed);
    let conversation = handle.conversation.lock().unwrap();
    assert_eq!(
        conversation.messages_for(None, "next")[1].content,
        format!(
            "Sentence one is here. {}",
            vox::conversation::INTERRUPTED_MARKER
        )
    );
}

#[tokio::test]
async fn cancel_without_a_turn_leaves_the_vad_alone() {
    let rig = Rig::new(vec![]).await;
    let mut client = rig.live(vad_config());
    client.expect("ready").await;
    client.expect("memory").await;

    // 300 loud samples: not a whole frame yet, so they wait.
    client.send_audio(&[0.9; 300]);
    settle().await;
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), 0);
    let resets_before = rig.vad.resets.load(Ordering::SeqCst);

    client.send(json!({"type": "cancel"}));
    let cancelled = client.expect("cancelled").await;
    assert_eq!(cancelled["reason"], "client_request");
    // No turn ever existed.
    assert_eq!(cancelled["turn_id"], 0);

    // The VAD was not reset and nothing was fed to it.
    assert_eq!(rig.vad.resets.load(Ordering::SeqCst), resets_before);
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), 0);

    // The waiting samples are still there: 212 more loud samples complete a
    // loud frame. Had the 300 been dropped, the frame would be 212 loud +
    // 300 quiet samples, which is below the fake VAD's threshold.
    let mut rest = vec![0.9f32; 212];
    rest.extend(std::iter::repeat_n(0.0, 300));
    client.send_audio(&rest);
    let events = client.drain(Duration::from_millis(300)).await;
    assert_eq!(types(&events), ["speech_start"]);
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), 1);
    assert_eq!(rig.vad.resets.load(Ordering::SeqCst), resets_before);
}

#[tokio::test]
async fn cancel_during_a_reply_stops_it_and_records_what_was_heard() {
    let script = Script {
        tokens: vec![
            "Sentence one is here.",
            " Sentence two follows.",
            " Sentence three follows.",
            " Sentence four follows.",
            " Sentence five follows.",
            " Sentence six ends.",
        ],
        token_delay: Duration::from_millis(150),
        ..Script::default()
    };
    let rig = Rig::new(vec![script, reply(&["Second reply."])]).await;
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    // Two sentences have been sent when the user interrupts; the browser
    // had only started the first.
    let mut events = client.until("<binary>").await;
    events.extend(client.until("<binary>").await);
    client.send(json!({"type": "cancel", "turn_id": 1, "heard_sentences": 1}));
    events.extend(client.until("cancelled").await);
    let cancelled = events.last().unwrap();
    assert_eq!(cancelled["turn_id"], 1);
    assert_eq!(cancelled["reason"], "client_request");

    // Nothing of the cancelled turn arrives any more.
    let after = client.drain(Duration::from_millis(500)).await;
    events.extend(after.iter().cloned());
    for event in &after {
        assert!(
            event.get("turn_id").is_none(),
            "event of a cancelled turn: {event}"
        );
        assert_ne!(event["type"], "<binary>");
    }
    assert!(of_type(&events, "turn_done").is_empty());

    // Memory has the turn as interrupted, with the one sentence heard.
    let memory = of_type(&events, "memory");
    let turns = memory.last().expect("memory after the cancel")["turns"]
        .as_array()
        .unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0]["assistant"], "Sentence one is here.");
    assert_eq!(turns[0]["interrupted"], true);

    // The session goes on: the cancel opened the microphone again.
    client.speak();
    let second = client.until("turn_done").await;
    assert_eq!(second.last().unwrap()["turn_id"], 2);
}

#[tokio::test]
async fn cancel_while_speaking_replays_the_gated_audio() {
    let script = Script {
        tokens: PAUSE_AFTER_FIRST_SENTENCE.to_vec(),
        token_delay: Duration::from_secs(5),
        ..Script::default()
    };
    let rig = Rig::new(vec![script]).await;
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    client.until("<binary>").await;
    let frames_before = rig.vad.frames.load(Ordering::SeqCst);

    // The user starts talking over the reply. The server keeps this audio
    // away from the VAD (it cannot tell it from the loudspeaker)...
    client.send_audio(&[0.9f32; FRAME * 2]);
    let quiet = client.drain(Duration::from_millis(200)).await;
    assert!(of_type(&quiet, "speech_start").is_empty(), "{quiet:#?}");
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), frames_before);

    // ...until the client's own detection fires and cancels. Then the held
    // audio goes to the VAD, so the start of the sentence is not lost.
    client.send(json!({"type": "cancel", "heard_sentences": 1}));
    let events = client.until("speech_start").await;
    assert_eq!(of_type(&events, "cancelled").len(), 1);
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), frames_before + 2);
}

#[tokio::test]
async fn speech_while_thinking_is_a_voice_barge_in() {
    let slow = Script {
        tokens: vec!["Too late."],
        start_delay: Duration::from_secs(5),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow, reply(&["Second reply."])]).await;
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    client.until("thinking").await;
    settle().await;

    // No reply audio yet, so the microphone is live.
    client.speak();
    let events = client.until("turn_done").await;
    let cancelled = of_type(&events, "cancelled");
    assert_eq!(cancelled.len(), 1);
    assert_eq!(cancelled[0]["turn_id"], 1);
    assert_eq!(cancelled[0]["reason"], "user_barge_in");
    let kinds = types(&events);
    let position = |ty: &str| kinds.iter().position(|k| *k == ty).unwrap();
    assert!(position("cancelled") < position("speech_start"));
    assert_eq!(events.last().unwrap()["turn_id"], 2);

    // The unanswered question stays in memory as interrupted, so the model
    // knows what came before.
    let requests = rig.llm_requests.lock().unwrap();
    let messages = requests[1]["messages"].as_array().unwrap();
    assert_eq!(messages[1]["content"], "utterance 1");
    assert_eq!(
        messages[2]["content"],
        vox::conversation::INTERRUPTED_MARKER
    );
    assert_eq!(messages[3]["content"], "utterance 2");
}

#[tokio::test]
async fn barge_in_can_be_switched_off() {
    let slow = Script {
        tokens: vec!["Late reply."],
        start_delay: Duration::from_millis(400),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow]).await;
    let mut client =
        rig.live(json!({"type": "config", "model": "test-model", "barge_in_enabled": false}));
    client.expect("ready").await;

    client.speak();
    client.until("thinking").await;
    client.speak();
    let events = client.until("turn_done").await;
    assert!(of_type(&events, "cancelled").is_empty());
    assert!(of_type(&events, "speech_start").is_empty());
    assert_eq!(events.last().unwrap()["turn_id"], 1);
}

#[tokio::test]
async fn cancel_during_transcription_does_not_wait_for_the_stt() {
    let rig = Rig::new(vec![reply(&["Second reply."])]).await;
    rig.stt
        .delays
        .lock()
        .unwrap()
        .push_back(Duration::from_secs(30));
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    client.speak();
    client.until("speech_end").await;
    settle().await;

    let started = std::time::Instant::now();
    client.send(json!({"type": "cancel"}));
    let cancelled = client.expect("cancelled").await;
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(cancelled["turn_id"], 1);

    // Nothing was transcribed, so nothing was remembered, and the next
    // utterance is its own turn.
    client.speak();
    let events = client.until("turn_done").await;
    assert_eq!(events.last().unwrap()["turn_id"], 2);
    assert_eq!(of_type(&events, "transcript")[0]["text"], "utterance 2");
    let memory = of_type(&events, "memory");
    assert_eq!(memory.last().unwrap()["turns"].as_array().unwrap().len(), 1);
    assert_eq!(*rig.stt.calls.lock().unwrap(), [FRAME * 3, FRAME * 3]);
}

#[tokio::test]
async fn speech_during_transcription_is_merged_into_the_next_turn() {
    let rig = Rig::new(vec![reply(&["One reply."])]).await;
    rig.stt
        .delays
        .lock()
        .unwrap()
        .push_back(Duration::from_secs(30));
    let mut client = rig.live(vad_config());
    client.expect("ready").await;

    // The VAD ended the utterance at a pause, but the user goes on.
    client.speak();
    client.until("speech_end").await;
    settle().await;
    client.speak();

    let events = client.until("turn_done").await;
    assert_eq!(of_type(&events, "cancelled")[0]["reason"], "user_barge_in");
    assert_eq!(events.last().unwrap()["turn_id"], 2);
    // The second transcription got both parts.
    assert_eq!(*rig.stt.calls.lock().unwrap(), [FRAME * 3, FRAME * 6]);
}

#[tokio::test]
async fn a_stalled_llm_ends_the_turn_and_the_mic_is_accepted_again() {
    let stalled = Script {
        silent: true,
        ..Script::default()
    };
    let mut rig = Rig::new(vec![stalled, reply(&["Second reply."])]).await;
    rig.stream_options = StreamOptions {
        first_token_timeout: Duration::from_millis(300),
        idle_timeout: Duration::from_millis(300),
    };
    // Barge-in off: while the turn hangs the microphone is gated, so the
    // only way the next utterance can start a turn is the gate lifting.
    let mut client =
        rig.live(json!({"type": "config", "model": "test-model", "barge_in_enabled": false}));
    client.expect("ready").await;
    client.expect("memory").await;

    client.speak();
    let events = client.until("error").await;
    let error = events.last().unwrap();
    assert_eq!(error["fatal"], false);
    assert_eq!(error["turn_id"], 1);
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("did not start responding"),
        "{error}"
    );
    // The metrics say how far the turn got.
    let metrics = of_type(&events, "turn_metrics")[0];
    assert!(metrics["stt_ms"].is_u64());
    assert!(metrics["llm_first_token_ms"].is_null());
    assert!(metrics["tts_first_audio_ms"].is_null());
    // A turn that produced nothing is not remembered.
    assert!(of_type(&events, "memory").is_empty());

    settle().await;
    client.speak();
    let events = client.until("turn_done").await;
    assert_eq!(of_type(&events, "speech_start").len(), 1);
    assert_eq!(events.last().unwrap()["turn_id"], 2);
    assert_eq!(events.last().unwrap()["sentences"], 1);
}

#[tokio::test]
async fn reconnecting_with_the_session_id_keeps_memory_until_forget() {
    let rig = Rig::new(vec![reply(&["First reply."]), reply(&["Second reply."])]).await;

    let mut client = rig.live(vad_config());
    let ready = client.expect("ready").await;
    let session_id = ready["session_id"].as_str().unwrap().to_string();
    client.speak();
    client.until("turn_done").await;
    // The microphone is toggled off: the socket closes.
    drop(client);
    settle().await;

    let mut client = rig.live(json!({
        "type": "config",
        "model": "test-model",
        "session_id": session_id,
    }));
    let ready = client.expect("ready").await;
    assert_eq!(ready["session_id"], session_id.as_str());
    // The memory is sent right away so the client can show it.
    let memory = client.expect("memory").await;
    let turns = memory["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0]["user"], "utterance 1");
    assert_eq!(turns[0]["assistant"], "First reply.");

    // Turn ids restart with the connection; the history does not.
    client.speak();
    let events = client.until("turn_done").await;
    assert_eq!(events.last().unwrap()["turn_id"], 1);
    {
        let requests = rig.llm_requests.lock().unwrap();
        let messages = requests[1]["messages"].as_array().unwrap();
        assert_eq!(messages[1]["content"], "utterance 1");
        assert_eq!(messages[2]["content"], "First reply.");
    }
    let memory = of_type(&events, "memory");
    assert_eq!(memory.last().unwrap()["turns"].as_array().unwrap().len(), 2);

    client.send(json!({"type": "forget_memory"}));
    let memory = client.expect("memory").await;
    assert_eq!(memory["turns"].as_array().unwrap().len(), 0);
    assert_eq!(memory["total_tokens"], 0);

    // Forgotten for good: a third connection finds the session, empty.
    drop(client);
    settle().await;
    let mut client = rig.live(json!({"type": "config", "session_id": session_id}));
    assert_eq!(
        client.expect("ready").await["session_id"],
        session_id.as_str()
    );
    let memory = client.expect("memory").await;
    assert_eq!(memory["turns"].as_array().unwrap().len(), 0);
    assert_eq!(rig.sessions.len(), 1);
}

#[tokio::test]
async fn a_connection_without_session_id_gets_its_own_memory() {
    let rig = Rig::new(vec![reply(&["First reply."])]).await;
    let mut first = rig.live(vad_config());
    let first_id = first.expect("ready").await["session_id"].clone();
    first.speak();
    first.until("turn_done").await;

    let mut second = rig.live(vad_config());
    let ready = second.expect("ready").await;
    assert_ne!(ready["session_id"], first_id);
    let memory = second.expect("memory").await;
    assert_eq!(memory["turns"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn config_changes_apply_to_the_next_live_talk_turn() {
    let rig = Rig::new(vec![reply(&["First reply."]), reply(&["Second reply."])]).await;
    let mut client = rig.live(json!({"type": "config", "model": "model-a", "voice": "voice-a"}));
    client.expect("ready").await;

    client.speak();
    client.until("turn_done").await;
    client.send(json!({"type": "config", "model": "model-b", "voice": "voice-b"}));
    client.send(json!({"type": "playback_done", "turn_id": 1}));
    settle().await;
    client.speak();
    client.until("turn_done").await;

    let requests = rig.llm_requests.lock().unwrap();
    assert_eq!(requests[0]["model"], "model-a");
    assert_eq!(requests[1]["model"], "model-b");
    let voices: Vec<Option<String>> = rig
        .tts
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|(_, voice)| voice.clone())
        .collect();
    assert_eq!(
        voices,
        [Some("voice-a".to_string()), Some("voice-b".to_string())]
    );
}

#[tokio::test]
async fn push_to_talk_runs_a_turn_and_ptt_start_interrupts_it() {
    let slow = Script {
        tokens: vec!["Too late."],
        start_delay: Duration::from_secs(5),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow, reply(&["Second reply."])]).await;
    let mut client =
        rig.live(json!({"type": "config", "model": "test-model", "mode": "push_to_talk"}));
    assert_eq!(client.expect("ready").await["mode"], "push_to_talk");

    // Audio outside a key press is ignored.
    client.send_audio(&[0.9f32; FRAME * 2]);
    client.send(json!({"type": "ptt_start"}));
    client.send_audio(&[0.9f32; FRAME * 4]);
    client.send(json!({"type": "ptt_end"}));
    let events = client.until("thinking").await;
    let kinds = types(&events);
    let position = |ty: &str| kinds.iter().position(|k| *k == ty).unwrap();
    assert!(position("speech_start") < position("speech_end"));
    assert!(position("speech_end") < position("transcript"));

    // Pressing the key again while the reply is pending interrupts it.
    settle().await;
    client.send(json!({"type": "ptt_start"}));
    client.send_audio(&[0.9f32; FRAME]);
    client.send(json!({"type": "ptt_end"}));
    let events = client.until("turn_done").await;
    let cancelled = of_type(&events, "cancelled");
    assert_eq!(cancelled[0]["turn_id"], 1);
    assert_eq!(cancelled[0]["reason"], "user_barge_in");
    assert_eq!(events.last().unwrap()["turn_id"], 2);

    // The VAD is not involved in push-to-talk.
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), 0);
    assert_eq!(*rig.stt.calls.lock().unwrap(), [FRAME * 4, FRAME]);
}

#[tokio::test]
async fn the_session_ends_when_the_client_leaves_mid_turn() {
    let slow = Script {
        tokens: PAUSE_AFTER_FIRST_SENTENCE.to_vec(),
        token_delay: Duration::from_secs(5),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow]).await;
    let mut client = rig.live(vad_config());
    let session_id = client.expect("ready").await["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    client.speak();
    client.until("<binary>").await;

    // Closing the socket ends the session: the server's sending half is
    // dropped, which the client sees as the end of the stream.
    let Client { tx, mut rx } = client;
    drop(tx);
    let closed = tokio::time::timeout(Duration::from_secs(3), async {
        while rx.recv().await.is_some() {}
    })
    .await;
    assert!(closed.is_ok(), "session did not end after the client left");

    // What had been sent is remembered as an interrupted turn.
    let handle = rig.sessions.attach(Some(&session_id));
    assert!(handle.resumed);
    let snapshot = handle.conversation.lock().unwrap().snapshot();
    assert_eq!(snapshot.turns.len(), 1);
    assert_eq!(snapshot.turns[0].assistant, "Sentence one is here.");
    assert!(snapshot.turns[0].interrupted);
}

// ─── Converse sessions ───────────────────────────────────────

#[tokio::test]
async fn converse_applies_a_mid_turn_config_to_the_next_turn() {
    let slow = Script {
        tokens: vec!["First reply."],
        start_delay: Duration::from_millis(400),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow, reply(&["Second reply."])]).await;
    let mut client = rig.converse(json!({"model": "model-a", "voice": "voice-a"}));
    let ready = client.expect("ready").await;
    assert_eq!(ready["model"], "model-a");
    assert_eq!(ready["voice"], "voice-a");

    client.speak();
    client.until("thinking").await;
    // The turn is running; the socket is still being read.
    client.send(json!({"model": "model-b", "voice": "voice-b"}));
    let events = client.until("done").await;
    assert_eq!(of_type(&events, "sentence")[0]["text"], "First reply.");
    assert!(
        of_type(&events, "sentence")[0]["audio_b64"]
            .as_str()
            .unwrap()
            .starts_with("UklGR"),
        "audio is a base64 RIFF file"
    );

    settle().await;
    client.speak();
    let events = client.until("done").await;
    assert_eq!(of_type(&events, "transcript")[0]["text"], "utterance 2");

    let requests = rig.llm_requests.lock().unwrap();
    // The running turn kept what it started with; the next one has the new
    // model and voice.
    assert_eq!(requests[0]["model"], "model-a");
    assert_eq!(requests[1]["model"], "model-b");
    let voices: Vec<Option<String>> = rig
        .tts
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|(_, voice)| voice.clone())
        .collect();
    assert_eq!(
        voices,
        [Some("voice-a".to_string()), Some("voice-b".to_string())]
    );

    // Converse has a memory now: the second request carries the first turn.
    let messages = requests[1]["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
    assert_eq!(messages[1]["content"], "utterance 1");
    assert_eq!(messages[2]["content"], "First reply.");
}

#[tokio::test]
async fn converse_ignores_the_microphone_while_a_turn_runs() {
    let slow = Script {
        tokens: vec!["First reply."],
        start_delay: Duration::from_millis(300),
        ..Script::default()
    };
    let rig = Rig::new(vec![slow]).await;
    let mut client = rig.converse(json!({}));
    client.expect("ready").await;

    client.speak();
    client.until("thinking").await;
    let frames_before = rig.vad.frames.load(Ordering::SeqCst);
    client.speak();
    let events = client.until("done").await;
    assert!(of_type(&events, "speech_start").is_empty());
    assert_eq!(rig.vad.frames.load(Ordering::SeqCst), frames_before);
    assert_eq!(rig.stt.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn converse_reports_a_stalled_llm_and_takes_the_next_turn() {
    let stalled = Script {
        silent: true,
        ..Script::default()
    };
    let mut rig = Rig::new(vec![stalled, reply(&["Second reply."])]).await;
    rig.stream_options = StreamOptions {
        first_token_timeout: Duration::from_millis(300),
        idle_timeout: Duration::from_millis(300),
    };
    let mut client = rig.converse(json!({}));
    client.expect("ready").await;

    client.speak();
    let error = client.expect("error").await;
    assert!(
        error["message"].as_str().unwrap().contains("LLM/TTS error"),
        "{error}"
    );

    settle().await;
    client.speak();
    let events = client.until("done").await;
    assert_eq!(events.last().unwrap()["sentences"], 1);
    // The failed turn left nothing in memory.
    let requests = rig.llm_requests.lock().unwrap();
    assert_eq!(requests[1]["messages"].as_array().unwrap().len(), 2);
}

// ─── Cancellable streaming ───────────────────────────────────

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
