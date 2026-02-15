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
    // Verify STT backend is loaded
    let stt = state
        .stt
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("STT backend not loaded"))?;
    let stt = Arc::clone(stt);

    // Verify VAD model path is available
    let vad_path = state
        .vad_model_path
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("VAD model not available"))?
        .clone();

    Ok(ws.on_upgrade(move |socket| handle_ws(socket, stt, vad_path)))
}

/// Handle a single WebSocket connection.
async fn handle_ws(
    socket: WebSocket,
    stt: Arc<dyn vox::traits::SttBackend>,
    vad_path: std::path::PathBuf,
) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Create per-connection VAD instance
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

        tracing::info!("WebSocket client connected, VAD frame_size={frame_size}");

        while let Some(msg) = ws_rx.next().await {
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

                        match vad.process_frame(&frame).await {
                            Ok(events) => {
                                for event in events {
                                    let json = match event {
                                        vox::VadEvent::SpeechStart => {
                                            serde_json::to_string(&WsEvent::SpeechStart)
                                        }
                                        vox::VadEvent::SpeechEnd(utterance) => {
                                            match stt.transcribe(&utterance).await {
                                                Ok(result) if !result.text.is_empty() => {
                                                    // Send transcript
                                                    let transcript_json = serde_json::to_string(
                                                        &WsEvent::Transcript {
                                                            text: result.text,
                                                            duration_ms: result.duration_ms,
                                                            processing_time_ms: result
                                                                .processing_time_ms,
                                                        },
                                                    )
                                                    .unwrap_or_default();
                                                    let _ = ws_tx
                                                        .send(Message::Text(transcript_json.into()))
                                                        .await;

                                                    // Then send speech_end
                                                    serde_json::to_string(&WsEvent::SpeechEnd)
                                                }
                                                Ok(_) => {
                                                    // Empty transcription, just send speech_end
                                                    serde_json::to_string(&WsEvent::SpeechEnd)
                                                }
                                                Err(e) => serde_json::to_string(&WsEvent::Error {
                                                    message: format!("STT error: {e}"),
                                                }),
                                            }
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
