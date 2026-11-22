//! Streaming text-to-speech WebSocket endpoint (`/v1/speak`).

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};

use super::ServerState;
use super::error::ServerError;
use super::models::{SpeakWsRequest, TtsWsEvent};
use super::ws::encode_wav_chunk;
use crate::traits::StreamingTtsBackend;
use crate::tts::SentenceStreamingAdapter;
use crate::types::TtsRequest;

/// Upgrade HTTP connection to WebSocket for streaming text-to-speech.
pub async fn speak_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, ServerError> {
    let tts = state
        .tts
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("TTS backend not loaded"))?;
    let tts = Arc::clone(tts);

    Ok(ws.on_upgrade(move |socket| handle_speak_ws(socket, tts, state)))
}

async fn handle_speak_ws(
    socket: WebSocket,
    tts: Arc<dyn crate::traits::TtsBackend>,
    state: Arc<ServerState>,
) {
    // Counted as an open session until this function returns, so shutdown
    // can wait for it.
    let _session = state.ws_session();
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Wait for the first text message containing the TTS request.
    let req: SpeakWsRequest = loop {
        let msg = tokio::select! {
            msg = ws_rx.next() => msg,
            () = state.shutdown.cancelled() => {
                let _ = ws_tx.send(super::shutdown_close_message()).await;
                return;
            }
        };
        match msg {
            Some(Ok(Message::Text(text))) => match serde_json::from_str::<SpeakWsRequest>(&text) {
                Ok(r) => break r,
                Err(e) => {
                    let _ = send_tts_event(
                        &mut ws_tx,
                        &TtsWsEvent::Error {
                            message: format!("invalid request: {e}"),
                        },
                    )
                    .await;
                    return;
                }
            },
            Some(Ok(Message::Close(_))) | None => return,
            Some(Err(_)) => return,
            _ => continue,
        }
    };

    if req.text.trim().is_empty() {
        let _ = send_tts_event(
            &mut ws_tx,
            &TtsWsEvent::Error {
                message: "empty text".into(),
            },
        )
        .await;
        return;
    }

    // One inference slot for the whole synthesis; waits while the server
    // is at its limit instead of failing the session.
    let Some(_permit) = state.inference_permit_wait().await else {
        let _ = ws_tx.send(super::shutdown_close_message()).await;
        return;
    };

    let handle = tokio::runtime::Handle::current();
    let adapter = SentenceStreamingAdapter::new(Arc::clone(&tts), handle);

    let tts_request = TtsRequest {
        text: req.text,
        voice: req.voice,
        seed: None,
    };

    let mut session = match adapter.create_tts_session(&tts_request) {
        Ok(s) => s,
        Err(e) => {
            let _ = send_tts_event(
                &mut ws_tx,
                &TtsWsEvent::Error {
                    message: format!("failed to create TTS session: {e}"),
                },
            )
            .await;
            return;
        }
    };

    let _ = send_tts_event(&mut ws_tx, &TtsWsEvent::Start {}).await;

    // Pull chunks in a blocking thread since pull_chunk() calls mpsc::recv().
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel(4);

    // Use std::thread::spawn (not spawn_blocking) because pull_chunk internally
    // may wait indefinitely and we don't want to tie up the blocking pool.
    std::thread::spawn(move || {
        loop {
            match session.pull_chunk() {
                Ok(Some(chunk)) => {
                    if chunk_tx.blocking_send(Ok(chunk)).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = chunk_tx.blocking_send(Err(e));
                    break;
                }
            }
        }
    });

    let mut chunk_index: usize = 0;

    loop {
        let result = tokio::select! {
            result = chunk_rx.recv() => match result {
                Some(result) => result,
                None => break,
            },
            () = state.shutdown.cancelled() => {
                let _ = ws_tx.send(super::shutdown_close_message()).await;
                return;
            }
        };
        match result {
            Ok(chunk) => {
                chunk_index += 1;
                let progress = chunk.progress;

                // Encode chunk as WAV
                match encode_wav_chunk(&chunk.samples, chunk.sample_rate) {
                    Ok(wav_bytes) => {
                        if ws_tx.send(Message::Binary(wav_bytes.into())).await.is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = send_tts_event(
                            &mut ws_tx,
                            &TtsWsEvent::Error {
                                message: format!("WAV encode error: {e}"),
                            },
                        )
                        .await;
                        return;
                    }
                }

                if send_tts_event(
                    &mut ws_tx,
                    &TtsWsEvent::Progress {
                        chunk: chunk_index,
                        progress,
                    },
                )
                .await
                .is_err()
                {
                    return;
                }
            }
            Err(e) => {
                let _ = send_tts_event(
                    &mut ws_tx,
                    &TtsWsEvent::Error {
                        message: format!("TTS error: {e}"),
                    },
                )
                .await;
                return;
            }
        }
    }

    let _ = send_tts_event(
        &mut ws_tx,
        &TtsWsEvent::Done {
            chunks: chunk_index,
        },
    )
    .await;
}

async fn send_tts_event(
    tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    event: &TtsWsEvent,
) -> Result<(), axum::Error> {
    let json = serde_json::to_string(event).unwrap_or_default();
    tx.send(Message::Text(json.into()))
        .await
        .map_err(axum::Error::new)
}
