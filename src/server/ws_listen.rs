//! WebSocket streaming endpoint for real-time voice transcription (`/v1/listen`).
//!
//! Clients connect via WebSocket, send raw PCM audio frames (f32 LE, 16kHz mono),
//! and receive JSON events: speech_start, transcript, speech_end, error.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};

use super::ServerState;
use super::error::ServerError;
use super::speakers::{
    SpeakerPublic, current_speaker_list, diarize_utterance, enrollment_blocked_notice,
    forget_speaker, rename_speaker,
};
use super::ws::bytes_to_f32_samples;

/// JSON events sent from server to client.
#[derive(Serialize)]
#[serde(tag = "type")]
enum WsEvent {
    #[serde(rename = "listen_ready")]
    ListenReady {
        /// Whether the server can perform speaker diarization. When `false`,
        /// the client should hide/disable the diarization toggle.
        diarization_available: bool,
    },
    #[serde(rename = "speech_start")]
    SpeechStart,
    #[serde(rename = "transcript")]
    Transcript {
        text: String,
        duration_ms: u64,
        processing_time_ms: u64,
        /// Stable identifier for the speaker (e.g. `speaker_0`).
        /// Only populated when diarization is enabled for this session.
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_id: Option<String>,
        /// Human-readable label (e.g. `Speaker 1` or `Alice`).
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_name: Option<String>,
        /// Deterministic hex color for the speaker.
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker_color: Option<String>,
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
    /// Full snapshot of all known speakers. Sent after toggling diarization
    /// on, after a rename, and after a forget.
    #[serde(rename = "speaker_list")]
    SpeakerList { speakers: Vec<SpeakerPublic> },
    /// Single-speaker change (new enrollment, rename). The UI should patch
    /// its local list rather than re-render from scratch.
    #[serde(rename = "speaker_update")]
    SpeakerUpdate { speaker: SpeakerPublic },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Client → server text-frame commands. Unknown variants are ignored so
/// older clients stay compatible with newer servers.
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum WsClientCommand {
    #[serde(rename = "config")]
    Config {
        #[serde(default)]
        diarization: Option<bool>,
    },
    #[serde(rename = "rename_speaker")]
    RenameSpeaker { id: String, name: String },
    #[serde(rename = "forget_speaker")]
    ForgetSpeaker { id: String },
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

    Ok(ws.on_upgrade(move |socket| handle_ws(socket, stt, vad_path, streaming_stt, state)))
}

/// Handle a single WebSocket connection.
async fn handle_ws(
    socket: WebSocket,
    stt: Arc<dyn vox::traits::SttBackend>,
    vad_path: std::path::PathBuf,
    streaming_stt: Option<Arc<dyn vox::traits::StreamingSttBackend>>,
    state: Arc<ServerState>,
) {
    // Counted as an open session until this function returns, so shutdown
    // can wait for it.
    let _session = state.ws_session();
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

        // Diarization is session-scoped (client can toggle on/off). Server-side
        // availability is set by whether the pipeline loaded at startup.
        #[cfg(feature = "diarization")]
        let diarization_available = state.diarization.is_some() && state.speaker_db.is_some();
        #[cfg(not(feature = "diarization"))]
        let diarization_available = false;
        // Start with diarization OFF by default — biometric features are
        // opt-in. The client will send a `config` command to enable it.
        let mut diarization_on = false;

        tracing::info!(
            streaming = has_streaming,
            diarization_available,
            "WebSocket client connected, VAD frame_size={frame_size}"
        );

        // Send a handshake event so the UI can distinguish "connected, listening"
        // from "WS open but stuck". The UI was previously silent until the first
        // VAD event, making hangs indistinguishable from idle.
        if let Ok(ready_json) = serde_json::to_string(&WsEvent::ListenReady {
            diarization_available,
        }) && ws_tx.send(Message::Text(ready_json.into())).await.is_err()
        {
            return;
        }

        let mut session: Option<Box<dyn vox::traits::SttSession>> = None;
        // Residual buffer preserves samples that don't fit a full VAD frame so
        // they carry over to the next binary message instead of being dropped
        // at the ~4%-per-chunk rate of the old `continue`-based implementation.
        let mut residual: Vec<f32> = Vec::with_capacity(frame_size * 2);

        loop {
            let ws_msg = tokio::select! {
                msg = ws_rx.next() => msg,
                () = state.shutdown.cancelled() => {
                    let _ = ws_tx.send(super::shutdown_close_message()).await;
                    break;
                }
            };
            let Some(msg) = ws_msg else { break };
            match msg {
                Ok(Message::Binary(data)) => {
                    let t_msg = Instant::now();
                    let samples = bytes_to_f32_samples(&data);

                    // Diagnostic: first message in a connection dumps sample stats
                    // so we can tell instantly whether the bytes-to-f32 decode is
                    // producing sane values (range ~[-1,1]) or garbage.
                    let peek_min = samples.iter().copied().fold(f32::INFINITY, f32::min);
                    let peek_max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let peek_rms = {
                        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
                        (sum_sq / samples.len().max(1) as f32).sqrt()
                    };
                    tracing::debug!(
                        bytes = data.len(),
                        samples = samples.len(),
                        min = peek_min,
                        max = peek_max,
                        rms = peek_rms,
                        "ws binary message received"
                    );

                    // Prepend residual to new samples, then process in exact
                    // frame-size chunks. Keep the tail for next message.
                    residual.extend_from_slice(&samples);
                    let full_frames = residual.len() / frame_size;
                    let consumed = full_frames * frame_size;
                    let frames: Vec<f32> = residual.drain(..consumed).collect();

                    for frame_samples in frames.chunks_exact(frame_size) {
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
                                    }) && ws_tx.send(Message::Text(json.into())).await.is_err()
                                    {
                                        return;
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
                                                    // Waits for an inference slot; `None`
                                                    // only while shutting down.
                                                    let _permit =
                                                        state.inference_permit_wait().await;
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

                                            // Run diarization in parallel with the transcript
                                            // send if enabled. We only care about the registry
                                            // snapshot after it runs, so it's cheap to block
                                            // here on the same task.
                                            let (
                                                speaker_id,
                                                speaker_name,
                                                speaker_color,
                                                new_speaker_update,
                                            ) = diarize_utterance(
                                                &state,
                                                diarization_on,
                                                &utterance,
                                            )
                                            .await;

                                            // Always send a Transcript event, even when empty,
                                            // so the UI can clear its "Transcribing..." placeholder.
                                            // An empty string is a valid signal that "nothing was
                                            // heard"; suppressing it leaves the UI stuck.
                                            let transcript_json =
                                                serde_json::to_string(&WsEvent::Transcript {
                                                    text: stt_result.text,
                                                    duration_ms: stt_result.duration_ms,
                                                    processing_time_ms: stt_result
                                                        .processing_time_ms,
                                                    speaker_id,
                                                    speaker_name,
                                                    speaker_color,
                                                })
                                                .unwrap_or_default();
                                            let _ = ws_tx
                                                .send(Message::Text(transcript_json.into()))
                                                .await;

                                            // If diarization auto-enrolled a new speaker,
                                            // push a SpeakerUpdate so the sidebar appears
                                            // without the client having to poll.
                                            if let Some(update) = new_speaker_update
                                                && let Ok(json) =
                                                    serde_json::to_string(&WsEvent::SpeakerUpdate {
                                                        speaker: update,
                                                    })
                                            {
                                                let _ =
                                                    ws_tx.send(Message::Text(json.into())).await;
                                            }

                                            serde_json::to_string(&WsEvent::SpeechEnd)
                                        }
                                        vox::VadEvent::Silence => continue,
                                    };

                                    if let Ok(json) = json
                                        && ws_tx.send(Message::Text(json.into())).await.is_err()
                                    {
                                        return; // client disconnected
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
                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<WsClientCommand>(&text) {
                        Ok(WsClientCommand::Config { diarization }) => {
                            if let Some(enabled) = diarization {
                                if enabled && !diarization_available {
                                    let _ = ws_tx
                                        .send(Message::Text(
                                            serde_json::to_string(&WsEvent::Error {
                                                message: "diarization unavailable on server".into(),
                                            })
                                            .unwrap_or_default()
                                            .into(),
                                        ))
                                        .await;
                                } else {
                                    diarization_on = enabled;
                                    tracing::info!(
                                        diarization = diarization_on,
                                        "WS: diarization toggled"
                                    );
                                    if diarization_on {
                                        // Enrollment is gated on consent recorded
                                        // server-side; say so rather than silently
                                        // never creating speakers.
                                        if let Some(message) =
                                            enrollment_blocked_notice(&state).await
                                            && let Ok(json) =
                                                serde_json::to_string(&WsEvent::Error { message })
                                        {
                                            let _ = ws_tx.send(Message::Text(json.into())).await;
                                        }
                                        // Send full snapshot so the UI can
                                        // populate the speaker sidebar.
                                        if let Some(list) = current_speaker_list(&state).await
                                            && let Ok(json) =
                                                serde_json::to_string(&WsEvent::SpeakerList {
                                                    speakers: list,
                                                })
                                        {
                                            let _ = ws_tx.send(Message::Text(json.into())).await;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(WsClientCommand::RenameSpeaker { id, name }) => {
                            if let Err(e) = rename_speaker(&state, &id, &name).await {
                                let _ = ws_tx
                                    .send(Message::Text(
                                        serde_json::to_string(&WsEvent::Error {
                                            message: format!("rename failed: {e}"),
                                        })
                                        .unwrap_or_default()
                                        .into(),
                                    ))
                                    .await;
                            } else if let Some(list) = current_speaker_list(&state).await
                                && let Ok(json) =
                                    serde_json::to_string(&WsEvent::SpeakerList { speakers: list })
                            {
                                let _ = ws_tx.send(Message::Text(json.into())).await;
                            }
                        }
                        Ok(WsClientCommand::ForgetSpeaker { id }) => {
                            if let Err(e) = forget_speaker(&state, &id).await {
                                let _ = ws_tx
                                    .send(Message::Text(
                                        serde_json::to_string(&WsEvent::Error {
                                            message: format!("forget failed: {e}"),
                                        })
                                        .unwrap_or_default()
                                        .into(),
                                    ))
                                    .await;
                            } else if let Some(list) = current_speaker_list(&state).await
                                && let Ok(json) =
                                    serde_json::to_string(&WsEvent::SpeakerList { speakers: list })
                            {
                                let _ = ws_tx.send(Message::Text(json.into())).await;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("WS: ignoring unknown text message: {e}");
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {} // ignore ping, pong
            }
        }

        tracing::info!("WebSocket client disconnected");
    }
}
