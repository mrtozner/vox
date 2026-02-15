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
use tokio::sync::mpsc;
use tokio::time::Instant;

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

        // Channel for partial transcription results from background tasks.
        let (partial_tx, mut partial_rx) = mpsc::channel::<WsEvent>(4);

        // Partial transcription state.
        let mut in_speech = false;
        let mut last_partial_time = Instant::now();
        let mut partial_running = false;

        /// Minimum interval between partial transcription attempts.
        const PARTIAL_INTERVAL_MS: u64 = 1000;

        loop {
            tokio::select! {
                ws_msg = ws_rx.next() => {
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

                                match vad.process_frame(&frame).await {
                                    Ok(events) => {
                                        for event in events {
                                            let json = match event {
                                                vox::VadEvent::SpeechStart => {
                                                    in_speech = true;
                                                    last_partial_time = Instant::now();
                                                    partial_running = false;
                                                    serde_json::to_string(&WsEvent::SpeechStart)
                                                }
                                                vox::VadEvent::SpeechEnd(utterance) => {
                                                    in_speech = false;
                                                    partial_running = false;
                                                    // Notify client that transcription is in progress
                                                    if let Ok(t_json) = serde_json::to_string(&WsEvent::Transcribing) {
                                                        let _ = ws_tx.send(Message::Text(t_json.into())).await;
                                                    }
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

                            // After processing frames, check if we should fire a partial transcription.
                            if in_speech
                                && !partial_running
                                && last_partial_time.elapsed().as_millis() >= PARTIAL_INTERVAL_MS as u128
                            {
                                if let Some(buffer) = vad.current_speech_buffer() {
                                    partial_running = true;
                                    last_partial_time = Instant::now();
                                    let stt_clone = Arc::clone(&stt);
                                    let tx = partial_tx.clone();
                                    tokio::spawn(async move {
                                        let start = Instant::now();
                                        let duration_ms = (buffer.samples.len() as u64 * 1000)
                                            / u64::from(buffer.sample_rate);
                                        let utterance = vox::Utterance {
                                            audio: buffer,
                                            duration_ms,
                                        };
                                        match stt_clone.transcribe(&utterance).await {
                                            Ok(result) if !result.text.is_empty() => {
                                                let _ = tx.send(WsEvent::Partial {
                                                    text: result.text,
                                                    is_final: false,
                                                    stability: 0.5,
                                                    duration_ms: result.duration_ms,
                                                    processing_time_ms: start.elapsed().as_millis() as u64,
                                                }).await;
                                            }
                                            _ => {
                                                // Empty or error -- silently skip partial.
                                            }
                                        }
                                    });
                                }
                            }
                        }
                        Ok(Message::Close(_)) => break,
                        Err(_) => break,
                        _ => {} // ignore text, ping, pong
                    }
                }
                Some(partial_event) = partial_rx.recv() => {
                    partial_running = false;
                    if let Ok(json) = serde_json::to_string(&partial_event) {
                        if ws_tx.send(Message::Text(json.into())).await.is_err() {
                            return; // client disconnected
                        }
                    }
                }
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
