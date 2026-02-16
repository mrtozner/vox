//! WebSocket streaming endpoint for real-time voice transcription.
//!
//! Clients connect via WebSocket, send raw PCM audio frames (f32 LE, 16kHz mono),
//! and receive JSON events: speech_start, transcript, speech_end, error.

use std::sync::Arc;
use std::time::Instant;

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
                    let t_msg = Instant::now();
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
                                                    let t_fallback = Instant::now();
                                                    match stt.transcribe(&utterance).await {
                                                        Ok(result) => {
                                                            tracing::debug!(
                                                                elapsed_us = t_fallback
                                                                    .elapsed()
                                                                    .as_micros(),
                                                                "ws stt fallback transcribe"
                                                            );
                                                            result
                                                        }
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
                    tracing::debug!(
                        elapsed_us = t_msg.elapsed().as_micros(),
                        "ws binary message processed"
                    );
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

// ─── TTS WebSocket (/v1/speak) ───────────────────────────────

use super::models::{SpeakWsRequest, TtsWsEvent};
use vox::traits::StreamingTtsBackend;
use vox::tts::SentenceStreamingAdapter;
use vox::types::TtsRequest;

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

    Ok(ws.on_upgrade(move |socket| handle_speak_ws(socket, tts)))
}

async fn handle_speak_ws(socket: WebSocket, tts: Arc<dyn vox::traits::TtsBackend>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Wait for the first text message containing the TTS request.
    let req: SpeakWsRequest = loop {
        match ws_rx.next().await {
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

    while let Some(result) = chunk_rx.recv().await {
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
        .map_err(|e| axum::Error::new(e))
}

fn encode_wav_chunk(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer =
            hound::WavWriter::new(&mut buf, spec).map_err(|e| format!("WAV init: {e}"))?;
        for &s in samples {
            writer
                .write_sample(s)
                .map_err(|e| format!("WAV write: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("WAV finalize: {e}"))?;
    }
    Ok(buf.into_inner())
}
