//! WebSocket streaming endpoint for real-time voice transcription.
//!
//! Clients connect via WebSocket, send raw PCM audio frames (f32 LE, 16kHz mono),
//! and receive JSON events: speech_start, transcript, speech_end, error.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;

use super::ServerState;
use super::error::ServerError;

/// JSON events sent from server to client.
#[derive(Serialize)]
#[serde(tag = "type")]
enum WsEvent {
    #[serde(rename = "speech_start")]
    SpeechStart,
    #[serde(rename = "transcript")]
    Transcript {
        text: String,
        duration_ms: u64,
        processing_time_ms: u64,
    },
    #[serde(rename = "transcribing")]
    Transcribing,
    #[serde(rename = "partial")]
    Partial {
        text: String,
        is_final: bool,
        stability: f32,
        duration_ms: u64,
        processing_time_ms: u64,
    },
    #[serde(rename = "speech_end")]
    SpeechEnd,
    #[serde(rename = "error")]
    Error { message: String },
}

/// Upgrade HTTP connection to WebSocket for real-time voice transcription.
pub async fn listen_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, ServerError> {
    let stt = state
        .stt
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("STT backend not loaded"))?;
    let stt = Arc::clone(stt);

    let vad_path = state
        .vad_model_path
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("VAD model not available"))?
        .clone();

    let streaming_stt = state.streaming_stt.clone();

    Ok(ws.on_upgrade(move |socket| handle_ws(socket, stt, vad_path, streaming_stt)))
}

/// Handle a single WebSocket connection.
async fn handle_ws(
    socket: WebSocket,
    stt: Arc<dyn vox::traits::SttBackend>,
    vad_path: std::path::PathBuf,
    streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>>,
) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    #[cfg(feature = "silero")]
    let mut vad: Box<dyn vox::traits::VadBackend> = match vox::SileroVad::new(&vad_path) {
        Ok(v) => Box::new(v),
        Err(e) => {
            let msg = serde_json::to_string(&WsEvent::Error {
                message: format!("failed to initialize VAD: {e}"),
            })
            .unwrap_or_default();
            let _ = ws_tx.send(Message::Text(msg.into())).await;
            return;
        }
    };

    #[cfg(not(feature = "silero"))]
    {
        let _ = vad_path;
        let _ = streaming_stt;
        let msg = serde_json::to_string(&WsEvent::Error {
            message: "server compiled without silero feature".into(),
        })
        .unwrap_or_default();
        let _ = ws_tx.send(Message::Text(msg.into())).await;
        return;
    }

    #[cfg(feature = "silero")]
    {
        let frame_size = vad.frame_size();
        let has_streaming = streaming_stt.is_some();

        tracing::info!(
            streaming = has_streaming,
            "WebSocket client connected, VAD frame_size={frame_size}"
        );

        let mut session: Option<Box<dyn vox::traits::SttSession>> = None;

        loop {
            let ws_msg = ws_rx.next().await;
            let Some(msg) = ws_msg else { break };
            match msg {
                Ok(Message::Binary(data)) => {
                    let samples = bytes_to_f32_samples(&data);
                    let chunk = vox::AudioChunk {
                        samples,
                        sample_rate: 16000,
                        channels: 1,
                    };

                    for frame_samples in chunk.samples.chunks(frame_size) {
                        if frame_samples.len() < frame_size {
                            continue;
                        }

                        let frame = vox::AudioChunk {
                            samples: frame_samples.to_vec(),
                            sample_rate: 16000,
                            channels: 1,
                        };

                        if let Some(s) = &mut session {
                            match s.push_audio(&frame.samples, 16000) {
                                Ok(Some(text)) => {
                                    if let Ok(json) = serde_json::to_string(&WsEvent::Partial {
                                        text,
                                        is_final: false,
                                        stability: 0.8,
                                        duration_ms: 0,
                                        processing_time_ms: 0,
                                    }) {
                                        if ws_tx.send(Message::Text(json.into())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    tracing::warn!("WS: push_audio failed: {e}, dropping session");
                                    session = None;
                                }
                            }
                        }

                        match vad.process_frame(&frame).await {
                            Ok(events) => {
                                for event in events {
                                    let json = match event {
                                        vox::VadEvent::SpeechStart => {
                                            if let Some(streaming) = &streaming_stt {
                                                match streaming.create_session() {
                                                    Ok(s) => session = Some(s),
                                                    Err(e) => tracing::warn!(
                                                        "WS: streaming session failed: {e}"
                                                    ),
                                                }
                                            }
                                            serde_json::to_string(&WsEvent::SpeechStart)
                                        }
                                        vox::VadEvent::SpeechEnd(utterance) => {
                                            let stt_result = if let Some(mut s) = session.take() {
                                                match s.finish() {
                                                    Ok(result) => Some(result),
                                                    Err(e) => {
                                                        tracing::warn!(
                                                            "WS: streaming finish failed: {e}, batch fallback"
                                                        );
                                                        None
                                                    }
                                                }
                                            } else {
                                                None
                                            };

                                            let stt_result = match stt_result {
                                                Some(r) if !r.text.is_empty() => r,
                                                _ => {
                                                    if let Ok(t_json) = serde_json::to_string(
                                                        &WsEvent::Transcribing,
                                                    ) {
                                                        let _ = ws_tx
                                                            .send(Message::Text(t_json.into()))
                                                            .await;
                                                    }
                                                    match stt.transcribe(&utterance).await {
                                                        Ok(result) => result,
                                                        Err(e) => {
                                                            let _ = ws_tx
                                                                .send(Message::Text(
                                                                    serde_json::to_string(
                                                                        &WsEvent::Error {
                                                                            message: format!(
                                                                                "STT error: {e}"
                                                                            ),
                                                                        },
                                                                    )
                                                                    .unwrap_or_default()
                                                                    .into(),
                                                                ))
                                                                .await;
                                                            continue;
                                                        }
                                                    }
                                                }
                                            };

                                            if !stt_result.text.is_empty() {
                                                let transcript_json =
                                                    serde_json::to_string(&WsEvent::Transcript {
                                                        text: stt_result.text,
                                                        duration_ms: stt_result.duration_ms,
                                                        processing_time_ms: stt_result
                                                            .processing_time_ms,
                                                    })
                                                    .unwrap_or_default();
                                                let _ = ws_tx
                                                    .send(Message::Text(transcript_json.into()))
                                                    .await;
                                            }

                                            serde_json::to_string(&WsEvent::SpeechEnd)
                                        }
                                        vox::VadEvent::Silence => continue,
                                    };

                                    if let Ok(json) = json {
                                        if ws_tx.send(Message::Text(json.into())).await.is_err() {
                                            return; // client disconnected
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let msg = serde_json::to_string(&WsEvent::Error {
                                    message: format!("VAD error: {e}"),
                                })
                                .unwrap_or_default();
                                let _ = ws_tx.send(Message::Text(msg.into())).await;
                                return;
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {} // ignore text, ping, pong
            }
        }

        tracing::info!("WebSocket client disconnected");
    }
}

/// Interpret raw bytes as f32 little-endian PCM samples.
fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
