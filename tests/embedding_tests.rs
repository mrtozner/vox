//! Vox as a library inside another application: the engine fed from a
//! caller-owned audio source, stopped by a caller-owned future, and (with the
//! `server` feature) the HTTP server reached through `vox::server`.
//!
//! None of these tests opens an audio device or installs a signal handler.
//! They also run with the microphone compiled out:
//! `cargo test --test embedding_tests --no-default-features`

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use vox::{AudioChunk, SttBackend, SttResult, Utterance, VadBackend, VadEvent, Vox, VoxError};

/// Emits one utterance at frame `trigger_at`, silence otherwise.
struct TriggerVad {
    seen: Arc<Mutex<usize>>,
    trigger_at: usize,
}

impl TriggerVad {
    fn new(trigger_at: usize) -> (Self, Arc<Mutex<usize>>) {
        let seen = Arc::new(Mutex::new(0));
        (
            Self {
                seen: seen.clone(),
                trigger_at,
            },
            seen,
        )
    }
}

#[async_trait]
impl VadBackend for TriggerVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        let count = {
            let mut seen = self.seen.lock().unwrap();
            *seen += 1;
            *seen
        };
        if count == self.trigger_at {
            let utterance = Utterance {
                audio: frame.clone(),
                duration_ms: 32,
                #[cfg(feature = "diarization")]
                speaker_id: None,
            };
            Ok(vec![VadEvent::SpeechStart, VadEvent::SpeechEnd(utterance)])
        } else {
            Ok(vec![VadEvent::Silence])
        }
    }

    fn reset(&mut self) {}

    fn frame_size(&self) -> usize {
        512
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }
}

struct FixedStt(&'static str);

#[async_trait]
impl SttBackend for FixedStt {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        Ok(SttResult {
            text: self.0.to_string(),
            language: Some("en".into()),
            duration_ms: audio.duration_ms,
            processing_time_ms: 1,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        })
    }
}

fn silence(samples: usize) -> AudioChunk {
    AudioChunk {
        samples: vec![0.0; samples],
        sample_rate: 16_000,
        channels: 1,
    }
}

#[tokio::test]
async fn channel_source_delivers_utterances_and_ends_with_the_sender() {
    let (vad, frames) = TriggerVad::new(2);
    let heard = Arc::new(Mutex::new(Vec::new()));
    let sink = heard.clone();

    let (vox, tx) = Vox::builder()
        .vad(vad)
        .stt(FixedStt("from a channel"))
        .on_utterance(move |result, _ctx| sink.lock().unwrap().push(result.text))
        .build_with_channel(16_000, 4)
        .expect("a channel source needs no audio device");

    let feeder = tokio::spawn(async move {
        // Three frames, cut at odd boundaries on purpose.
        for len in [700, 300, 536] {
            tx.send(silence(len)).await.unwrap();
        }
        // Dropping the sender is the end of the stream.
    });

    tokio::time::timeout(Duration::from_secs(5), vox.listen())
        .await
        .expect("listen() returns when the sender is dropped")
        .unwrap();
    feeder.await.unwrap();

    assert_eq!(*frames.lock().unwrap(), 3);
    assert_eq!(*heard.lock().unwrap(), vec!["from a channel".to_string()]);
}

#[tokio::test]
async fn channel_capacity_of_zero_is_accepted() {
    let (vad, _frames) = TriggerVad::new(usize::MAX);
    let (vox, tx) = Vox::builder()
        .vad(vad)
        .stt(FixedStt("unused"))
        .build_with_channel(16_000, 0)
        .expect("capacity 0 is rounded up instead of panicking");
    drop(tx);
    vox.listen().await.unwrap();
}

#[tokio::test]
async fn listen_until_stops_on_the_callers_future_while_the_source_is_open() {
    let (vad, frames) = TriggerVad::new(usize::MAX);
    let (vox, tx) = Vox::builder()
        .vad(vad)
        .stt(FixedStt("unused"))
        .build_with_channel(16_000, 4)
        .unwrap();

    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    // Polled on this task, not spawned: with the `mic` feature `Vox` holds a
    // cpal stream handle and is not `Send`.
    let pipeline = vox.listen_until(async move {
        let _ = stop_rx.await;
    });
    let driver = async {
        tx.send(silence(512)).await.unwrap();
        // Wait until the pipeline has consumed the frame, then ask it to stop.
        while *frames.lock().unwrap() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        stop_tx.send(()).unwrap();
    };

    let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(pipeline, driver)
    })
    .await
    .expect("listen_until() returns once the shutdown future resolves");
    result.unwrap();

    // `tx` was never dropped, so the stop came from the future and not from
    // the end of the source; the stopped pipeline has released its receiver.
    assert!(tx.is_closed(), "the pipeline dropped its receiver");
    assert_eq!(*frames.lock().unwrap(), 1);
}

#[tokio::test]
async fn listen_until_flushes_the_tail_when_the_source_ends_first() {
    let (vad, frames) = TriggerVad::new(usize::MAX);
    let (vox, tx) = Vox::builder()
        .vad(vad)
        .stt(FixedStt("unused"))
        .build_with_channel(16_000, 4)
        .unwrap();

    // One and a half frames: the half is padded and flushed at end of stream.
    tx.send(silence(768)).await.unwrap();
    drop(tx);

    vox.listen_until(std::future::pending()).await.unwrap();
    assert_eq!(*frames.lock().unwrap(), 2);
}

/// Without the microphone there is no audio driver handle in `Vox`, so the
/// pipeline can be moved to another task or thread.
#[cfg(not(feature = "mic"))]
#[tokio::test]
async fn pipeline_is_spawnable_without_the_mic_feature() {
    let (vad, frames) = TriggerVad::new(usize::MAX);
    let (vox, tx) = Vox::builder()
        .vad(vad)
        .stt(FixedStt("unused"))
        .build_with_channel(16_000, 4)
        .unwrap();

    let pipeline = tokio::spawn(vox.listen());
    tx.send(silence(1024)).await.unwrap();
    drop(tx);
    pipeline.await.unwrap().unwrap();
    assert_eq!(*frames.lock().unwrap(), 2);
}

#[cfg(not(feature = "mic"))]
#[test]
fn build_without_the_mic_feature_points_at_the_source_constructors() {
    let (vad, _frames) = TriggerVad::new(1);
    let err = match Vox::builder().vad(vad).stt(FixedStt("unused")).build() {
        Ok(_) => panic!("there is no capture code to build with"),
        Err(e) => e,
    };
    assert!(matches!(err, VoxError::Audio(_)), "got: {err}");
    let msg = err.to_string();
    assert!(msg.contains("mic"), "got: {msg}");
    assert!(msg.contains("build_with_source"), "got: {msg}");
}

/// The binary is a client of the library: nothing is compiled into it by path.
#[test]
fn binary_does_not_include_modules_from_the_library_tree() {
    let bin = include_str!("../src/bin/vox.rs");
    assert!(!bin.contains("../server/mod.rs"));
    for file in [
        "error_edge_tests.rs",
        "live_talk_integration.rs",
        "offline_server_tests.rs",
        "privacy_ledger_tests.rs",
        "server_tests.rs",
        "wav_safety_tests.rs",
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join(file);
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains("#[path"),
            "{file} must use vox::server, not a #[path] include"
        );
    }
}

#[cfg(feature = "server")]
mod server {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use vox::server::{self, ServerOptions, ServerState};

    #[tokio::test]
    async fn router_is_reachable_through_the_library() {
        let state = Arc::new(ServerState::empty());
        let response = server::build_router(state)
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    /// The embedding recipe from the `vox::server` module docs: own the
    /// listener, own the state, stop the server with your own future.
    #[tokio::test]
    async fn serve_runs_on_a_caller_owned_listener_until_the_callers_future() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(ServerState::with_options(
            "127.0.0.1",
            port,
            &ServerOptions::default(),
        ));

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let running = tokio::spawn(server::serve(
            listener,
            state,
            Duration::from_secs(2),
            async move {
                let _ = stop_rx.await;
            },
        ));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let response = client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        drop(client);

        stop_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), running)
            .await
            .expect("serve() returns after the shutdown future resolves")
            .unwrap()
            .unwrap();
    }
}
