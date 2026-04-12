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
use serde::{Deserialize, Serialize};

use super::ServerState;
use super::error::ServerError;

/// Client-facing view of an enrolled speaker.
///
/// The reference embedding is never serialized over the wire — it's a
/// biometric identifier and the client has no use for it.
#[derive(Serialize, Clone, Debug)]
struct SpeakerPublic {
    id: String,
    name: String,
    /// Deterministic hex color derived from `id` via DJB2.
    color: String,
}

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

/// Colorblind-friendly palette for visually distinct speaker colors.
const SPEAKER_PALETTE: &[&str] = &[
    "#4e79a7", // steel blue
    "#f28e2c", // orange
    "#e15759", // coral red
    "#76b7b2", // teal
    "#59a14f", // green
    "#edc949", // gold
    "#af7aa1", // purple
    "#ff9da7", // pink
    "#9c755f", // brown
    "#bab0ab", // warm gray
];

fn color_for(id: &str) -> String {
    // For auto-enrolled speaker_N IDs, use index for maximum distinction
    if let Some(n) = id.strip_prefix("speaker_").and_then(|n| n.parse::<usize>().ok()) {
        let idx = n.saturating_sub(1) % SPEAKER_PALETTE.len();
        return SPEAKER_PALETTE[idx].to_string();
    }
    // For custom/renamed IDs, DJB2 hash into the palette
    let mut hash: u32 = 5381;
    for b in id.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(*b as u32);
    }
    SPEAKER_PALETTE[(hash as usize) % SPEAKER_PALETTE.len()].to_string()
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
        }) {
            if ws_tx.send(Message::Text(ready_json.into())).await.is_err() {
                return;
            }
        }

        let mut session: Option<Box<dyn vox::traits::SttSession>> = None;
        // Residual buffer preserves samples that don't fit a full VAD frame so
        // they carry over to the next binary message instead of being dropped
        // at the ~4%-per-chunk rate of the old `continue`-based implementation.
        let mut residual: Vec<f32> = Vec::with_capacity(frame_size * 2);

        loop {
            let ws_msg = ws_rx.next().await;
            let Some(msg) = ws_msg else { break };
            match msg {
                Ok(Message::Binary(data)) => {
                    let t_msg = Instant::now();
                    let samples = bytes_to_f32_samples(&data);

                    // Diagnostic: first message in a connection dumps sample stats
                    // so we can tell instantly whether the bytes-to-f32 decode is
                    // producing sane values (range ~[-1,1]) or garbage.
                    let peek_min = samples
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, f32::min);
                    let peek_max = samples
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
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
                                            if let Some(update) = new_speaker_update {
                                                if let Ok(json) = serde_json::to_string(
                                                    &WsEvent::SpeakerUpdate { speaker: update },
                                                ) {
                                                    let _ = ws_tx
                                                        .send(Message::Text(json.into()))
                                                        .await;
                                                }
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
                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<WsClientCommand>(&text) {
                        Ok(WsClientCommand::Config { diarization }) => {
                            if let Some(enabled) = diarization {
                                if enabled && !diarization_available {
                                    let _ = ws_tx
                                        .send(Message::Text(
                                            serde_json::to_string(&WsEvent::Error {
                                                message: "diarization unavailable on server"
                                                    .into(),
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
                                        // Send full snapshot so the UI can
                                        // populate the speaker sidebar.
                                        if let Some(list) = current_speaker_list(&state).await {
                                            if let Ok(json) =
                                                serde_json::to_string(&WsEvent::SpeakerList {
                                                    speakers: list,
                                                })
                                            {
                                                let _ = ws_tx
                                                    .send(Message::Text(json.into()))
                                                    .await;
                                            }
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
                            } else if let Some(list) = current_speaker_list(&state).await {
                                if let Ok(json) = serde_json::to_string(&WsEvent::SpeakerList {
                                    speakers: list,
                                }) {
                                    let _ = ws_tx.send(Message::Text(json.into())).await;
                                }
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
                            } else if let Some(list) = current_speaker_list(&state).await {
                                if let Ok(json) = serde_json::to_string(&WsEvent::SpeakerList {
                                    speakers: list,
                                }) {
                                    let _ = ws_tx.send(Message::Text(json.into())).await;
                                }
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

/// Run speaker diarization on a completed utterance and return the speaker
/// metadata to attach to the next `Transcript` event.
///
/// Returns `(speaker_id, speaker_name, speaker_color, optional SpeakerPublic
/// describing a newly enrolled speaker)`. The fourth element is `Some` only
/// when the pipeline auto-enrolled a brand-new speaker during this call so
/// the caller can emit a `SpeakerUpdate`.
async fn diarize_utterance(
    state: &Arc<ServerState>,
    diarization_on: bool,
    utterance: &vox::Utterance,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<SpeakerPublic>,
) {
    if !diarization_on {
        return (None, None, None, None);
    }
    #[cfg(feature = "diarization")]
    {
        let Some(pipeline) = state.diarization.as_ref() else {
            return (None, None, None, None);
        };
        let Some(db) = state.speaker_db.as_ref() else {
            return (None, None, None, None);
        };

        // Count speakers before processing to detect auto-enrollment below.
        let before_count = {
            let reg = pipeline.registry();
            let guard = reg.lock().unwrap();
            guard.speaker_count()
        };

        // process_utterance consumes the utterance; clone to avoid mutating
        // the caller's copy.
        let processed = match pipeline.process_utterance(utterance.clone()).await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(error = %e, "WS: diarization failed, continuing without");
                return (None, None, None, None);
            }
        };

        let Some(id) = processed.speaker_id.clone() else {
            return (None, None, None, None);
        };
        // "unknown" is the sentinel returned for too-short utterances or
        // extraction failures — don't surface it as a real speaker.
        if id == "unknown" {
            return (None, None, None, None);
        }
        let color = color_for(&id);

        // Snapshot the registry to look up the human-readable name and check
        // for new enrollments.
        let (name, new_speaker, new_embedding) = {
            let reg = pipeline.registry();
            let guard = reg.lock().unwrap();
            let name = guard
                .get_speaker(&id)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| id.clone());
            let new_speaker = if guard.speaker_count() > before_count {
                // New speaker enrolled during this call.
                guard.get_speaker(&id).map(|s| SpeakerPublic {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    color: color_for(&s.id),
                })
            } else {
                None
            };
            let new_embedding = if new_speaker.is_some() {
                guard.get_speaker(&id).map(|s| s.embedding.clone())
            } else {
                None
            };
            (name, new_speaker, new_embedding)
        };

        // Persist new enrollments to SQLite so they survive a restart.
        if let (Some(speaker_public), Some(emb)) = (&new_speaker, new_embedding) {
            let persisted = vox::Speaker {
                id: speaker_public.id.clone(),
                name: speaker_public.name.clone(),
                embedding: emb,
            };
            if let Err(e) = db.store_speaker(&persisted).await {
                tracing::warn!(error = %e, "WS: failed to persist new speaker");
            }
        } else if let Err(e) = db.update_last_seen(&id).await {
            tracing::debug!(error = %e, "WS: update_last_seen failed");
        }

        (Some(id), Some(name), Some(color), new_speaker)
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, utterance);
        (None, None, None, None)
    }
}

/// Snapshot the current list of enrolled speakers for `SpeakerList` events.
async fn current_speaker_list(state: &Arc<ServerState>) -> Option<Vec<SpeakerPublic>> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state.diarization.as_ref()?;
        let reg = pipeline.registry();
        let guard = reg.lock().unwrap();
        let list: Vec<SpeakerPublic> = guard
            .list_speakers()
            .into_iter()
            .map(|s| SpeakerPublic {
                id: s.id.clone(),
                name: s.name.clone(),
                color: color_for(&s.id),
            })
            .collect();
        Some(list)
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = state;
        None
    }
}

/// Rename a speaker in both the in-memory registry and the SQLite database.
async fn rename_speaker(
    state: &Arc<ServerState>,
    id: &str,
    name: &str,
) -> Result<(), vox::VoxError> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state
            .diarization
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("diarization disabled".into()))?;
        let db = state
            .speaker_db
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("speaker db disabled".into()))?;

        {
            let reg = pipeline.registry();
            let mut guard = reg.lock().unwrap();
            guard.rename(id, name)?;
        }
        db.update_speaker_name(id, name).await?;
        Ok(())
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, id, name);
        Err(vox::VoxError::Diarization("diarization disabled".into()))
    }
}

/// Forget a speaker from both the in-memory registry and the SQLite database.
async fn forget_speaker(state: &Arc<ServerState>, id: &str) -> Result<(), vox::VoxError> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state
            .diarization
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("diarization disabled".into()))?;
        let db = state
            .speaker_db
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("speaker db disabled".into()))?;

        {
            let reg = pipeline.registry();
            let mut guard = reg.lock().unwrap();
            guard.forget(id)?;
        }
        db.delete_speaker(id).await?;
        Ok(())
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, id);
        Err(vox::VoxError::Diarization("diarization disabled".into()))
    }
}

/// Interpret raw bytes as f32 little-endian PCM samples.
pub(super) fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
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
        .map_err(axum::Error::new)
}

pub(super) fn encode_wav_chunk(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    // 16-bit PCM: Chrome's HTMLAudioElement cannot decode 32-bit float WAV.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer =
            hound::WavWriter::new(&mut buf, spec).map_err(|e| format!("WAV init: {e}"))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample_i16 = (clamped * i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| format!("WAV write: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("WAV finalize: {e}"))?;
    }
    Ok(buf.into_inner())
}

// ─── Continuous Voice Chat WebSocket (/v1/converse) ──────────

use super::models::{ConverseConfig, ConverseWsEvent};

/// Minimal std-lib base64 encoder to avoid pulling in a new dependency.
///
/// Emits standard base64 (RFC 4648) with `=` padding. This is used to
/// embed synthesized audio inside the JSON wire protocol used by
/// [`converse_ws`], keeping the wire messages self-contained.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push(ALPHABET[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

/// Upgrade HTTP connection to the continuous voice-chat WebSocket.
///
/// This endpoint unifies VAD → STT → LLM → TTS into a single long-lived
/// WebSocket, eliminating the per-turn reconnect latency that results
/// from orchestrating `/v1/listen`, `/v1/chat`, and `/v1/speak`
/// separately. The LLM response is streamed sentence-by-sentence so
/// audio playback can begin while later sentences are still being
/// generated.
///
/// See [`ConverseWsEvent`] for the server → client wire protocol.
pub async fn converse_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, ServerError> {
    let stt = state
        .stt
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("STT backend not loaded"))?;
    let stt = Arc::clone(stt);

    let tts = state
        .conversation_tts.as_ref().or(state.tts.as_ref())
        .ok_or_else(|| ServerError::service_unavailable("TTS backend not loaded"))?;
    let tts = Arc::clone(tts);

    let vad_path = state
        .vad_model_path
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("VAD model not available"))?
        .clone();

    Ok(ws.on_upgrade(move |socket| handle_converse_ws(socket, stt, tts, vad_path, state)))
}

/// Shared state sent from the STT task to the per-turn LLM/TTS dispatcher.
#[cfg(feature = "silero")]
async fn handle_converse_ws(
    socket: WebSocket,
    stt: Arc<dyn vox::traits::SttBackend>,
    tts: Arc<dyn vox::traits::TtsBackend>,
    vad_path: std::path::PathBuf,
    state: Arc<ServerState>,
) {
    let (ws_tx, mut ws_rx) = socket.split();
    let ws_tx = Arc::new(tokio::sync::Mutex::new(ws_tx));

    // Initialize VAD.
    let mut vad: Box<dyn vox::traits::VadBackend> = match vox::SileroVad::new(&vad_path) {
        Ok(v) => Box::new(v),
        Err(e) => {
            let _ = send_converse_event(
                &ws_tx,
                &ConverseWsEvent::Error {
                    message: format!("failed to initialize VAD: {e}"),
                },
            )
            .await;
            return;
        }
    };
    let frame_size = vad.frame_size();

    // Read optional config as the first text message, non-blocking.
    // If no config arrives before the first binary frame, use defaults.
    let mut config = ConverseConfig::default();
    let mut pending_audio: Vec<u8> = Vec::new();
    match tokio::time::timeout(std::time::Duration::from_millis(250), ws_rx.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<ConverseConfig>(&text) {
            Ok(c) => config = c,
            Err(e) => {
                tracing::warn!("converse_ws: invalid config JSON, using defaults: {e}");
            }
        },
        Ok(Some(Ok(Message::Binary(data)))) => {
            // Client started streaming audio immediately.
            pending_audio = data.to_vec();
        }
        Ok(Some(Ok(Message::Close(_)))) | Ok(None) => return,
        Ok(Some(Err(_))) => return,
        _ => {}
    }

    let model = config
        .model
        .clone()
        .unwrap_or_else(|| "llama3.2".to_string());
    let host = config
        .host
        .clone()
        .unwrap_or_else(|| state.ollama_host.clone());
    let voice = config.voice.clone();

    tracing::info!(
        model = %model,
        host = %host,
        voice = ?voice,
        "converse_ws: client connected"
    );

    if send_converse_event(
        &ws_tx,
        &ConverseWsEvent::Ready {
            model: model.clone(),
            voice: voice.clone(),
        },
    )
    .await
    .is_err()
    {
        return;
    }

    let http_client = state.http_client.clone();
    let system_prompt = vox::build_system_prompt(vox::VoicePromptMode::Voice);

    // Main loop: VAD frames → transcribe → stream LLM → TTS → repeat.
    let mut current_utterance: Option<vox::Utterance> = None;
    // Persistent audio residual — preserves samples that don't fit a full VAD
    // frame so they carry over to the next binary message (instead of being
    // silently dropped at the 4%-per-chunk rate of the old implementation).
    let mut audio_residual: Vec<f32> = Vec::new();

    // Process any audio captured before config parse finished.
    if !pending_audio.is_empty() {
        let samples = bytes_to_f32_samples(&pending_audio);
        if let Err(()) = handle_audio_samples(
            &samples,
            frame_size,
            vad.as_mut(),
            &ws_tx,
            &mut current_utterance,
            &mut audio_residual,
        )
        .await
        {
            return;
        }
    }

    loop {
        // If an utterance was captured, dispatch STT + LLM + TTS.
        if let Some(utterance) = current_utterance.take() {
            if let Err(()) = run_turn(
                utterance,
                &stt,
                &tts,
                &http_client,
                &host,
                &model,
                &system_prompt,
                voice.as_deref(),
                &ws_tx,
            )
            .await
            {
                return;
            }

            // Reset VAD LSTM/lookback state so the next turn starts fresh.
            vad.reset();
            audio_residual.clear();

            // Drain any audio frames buffered by the client while run_turn
            // was synthesizing TTS — these almost certainly captured our own
            // playback via the user's mic and would otherwise self-trigger
            // a feedback loop.
            let drain_deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(400);
            while std::time::Instant::now() < drain_deadline {
                match tokio::time::timeout(
                    std::time::Duration::from_millis(20),
                    ws_rx.next(),
                )
                .await
                {
                    Ok(Some(Ok(Message::Binary(_)))) => {
                        // discard stale mic frames from during TTS playback
                    }
                    Ok(Some(Ok(Message::Text(text)))) => {
                        if let Ok(new_cfg) = serde_json::from_str::<ConverseConfig>(&text) {
                            if new_cfg.voice.is_some() {
                                tracing::info!(
                                    voice = ?new_cfg.voice,
                                    "converse_ws: voice updated"
                                );
                            }
                        }
                    }
                    Ok(Some(Ok(Message::Close(_)))) | Ok(None) => return,
                    Ok(Some(Err(_))) => return,
                    Err(_) => break, // no message within 20ms → drain complete
                    _ => {}
                }
            }
            // Extra safety: reset VAD a second time after drain in case any
            // frame slipped through and primed the LSTM state.
            vad.reset();
            audio_residual.clear();
        }

        // Wait for the next audio frame.
        let Some(msg) = ws_rx.next().await else { break };
        match msg {
            Ok(Message::Binary(data)) => {
                let samples = bytes_to_f32_samples(&data);
                if handle_audio_samples(
                    &samples,
                    frame_size,
                    vad.as_mut(),
                    &ws_tx,
                    &mut current_utterance,
                    &mut audio_residual,
                )
                .await
                .is_err()
                {
                    return;
                }
            }
            Ok(Message::Text(text)) => {
                // Allow mid-session config update (e.g. voice switch).
                if let Ok(new_cfg) = serde_json::from_str::<ConverseConfig>(&text) {
                    if new_cfg.voice.is_some() {
                        tracing::info!(voice = ?new_cfg.voice, "converse_ws: voice updated");
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => {}
        }
    }

    tracing::info!("converse_ws: client disconnected");
}

/// Feed audio samples through VAD, emit events, and latch any completed
/// utterance into `current_utterance` for downstream processing.
///
/// `residual` holds samples that didn't fit a full VAD frame on the previous
/// call; they're prepended to `samples` so no audio is dropped at chunk
/// boundaries. The leftover tail is written back to `residual` on return.
#[cfg(feature = "silero")]
async fn handle_audio_samples(
    samples: &[f32],
    frame_size: usize,
    vad: &mut dyn vox::traits::VadBackend,
    ws_tx: &Arc<tokio::sync::Mutex<futures_util::stream::SplitSink<WebSocket, Message>>>,
    current_utterance: &mut Option<vox::Utterance>,
    residual: &mut Vec<f32>,
) -> Result<(), ()> {
    residual.extend_from_slice(samples);
    let full_frames = residual.len() / frame_size;
    let consumed = full_frames * frame_size;
    // Drain the full-frame prefix; whatever remains in `residual` after this
    // is the tail (< frame_size) we carry to the next call.
    let frames: Vec<f32> = residual.drain(..consumed).collect();

    for frame_samples in frames.chunks_exact(frame_size) {
        let frame = vox::AudioChunk {
            samples: frame_samples.to_vec(),
            sample_rate: 16000,
            channels: 1,
        };
        match vad.process_frame(&frame).await {
            Ok(events) => {
                for event in events {
                    match event {
                        vox::VadEvent::SpeechStart => {
                            if send_converse_event(ws_tx, &ConverseWsEvent::SpeechStart)
                                .await
                                .is_err()
                            {
                                return Err(());
                            }
                        }
                        vox::VadEvent::SpeechEnd(utt) => {
                            if send_converse_event(ws_tx, &ConverseWsEvent::SpeechEnd)
                                .await
                                .is_err()
                            {
                                return Err(());
                            }
                            *current_utterance = Some(utt);
                            // Return early — main loop will dispatch the turn.
                            // Any frames beyond this point in the current
                            // burst are likely just post-speech silence and
                            // are safely ignored.
                            return Ok(());
                        }
                        vox::VadEvent::Silence => {}
                    }
                }
            }
            Err(e) => {
                let _ = send_converse_event(
                    ws_tx,
                    &ConverseWsEvent::Error {
                        message: format!("VAD error: {e}"),
                    },
                )
                .await;
                return Err(());
            }
        }
    }
    Ok(())
}

/// Execute one conversation turn: STT → streaming LLM → TTS per sentence.
#[cfg(feature = "silero")]
#[allow(clippy::too_many_arguments)]
async fn run_turn(
    utterance: vox::Utterance,
    stt: &Arc<dyn vox::traits::SttBackend>,
    tts: &Arc<dyn vox::traits::TtsBackend>,
    http_client: &reqwest::Client,
    host: &str,
    model: &str,
    system_prompt: &str,
    voice: Option<&str>,
    ws_tx: &Arc<tokio::sync::Mutex<futures_util::stream::SplitSink<WebSocket, Message>>>,
) -> Result<(), ()> {
    // 1. STT
    let stt_result = match stt.transcribe(&utterance).await {
        Ok(r) => r,
        Err(e) => {
            let _ = send_converse_event(
                ws_tx,
                &ConverseWsEvent::Error {
                    message: format!("STT error: {e}"),
                },
            )
            .await;
            return Ok(());
        }
    };

    if stt_result.text.trim().is_empty() {
        // Nothing transcribed; skip the LLM round-trip but keep the socket open.
        let _ = send_converse_event(ws_tx, &ConverseWsEvent::Done { sentences: 0 }).await;
        return Ok(());
    }

    if send_converse_event(
        ws_tx,
        &ConverseWsEvent::Transcript {
            text: stt_result.text.clone(),
            duration_ms: stt_result.duration_ms,
            processing_time_ms: stt_result.processing_time_ms,
        },
    )
    .await
    .is_err()
    {
        return Err(());
    }

    if send_converse_event(ws_tx, &ConverseWsEvent::Thinking)
        .await
        .is_err()
    {
        return Err(());
    }

    // 2. Streaming LLM → TTS per sentence
    let sentence_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tts_for_cb = Arc::clone(tts);
    let ws_tx_for_cb = Arc::clone(ws_tx);
    let voice_owned = voice.map(|v| v.to_string());
    let counter_for_cb = Arc::clone(&sentence_counter);

    let on_sentence = move |sentence: &str| {
        let tts = Arc::clone(&tts_for_cb);
        let ws_tx = Arc::clone(&ws_tx_for_cb);
        let voice = voice_owned.clone();
        let counter = Arc::clone(&counter_for_cb);
        let text = sentence.to_string();
        async move {
            if text.trim().is_empty() {
                return Ok(());
            }
            let request = vox::TtsRequest {
                text: text.clone(),
                voice,
                seed: None,
            };
            let output = tts.synthesize(&request).await?;
            let wav_bytes = encode_wav_chunk(&output.audio.samples, output.audio.sample_rate)
                .map_err(vox::VoxError::Tts)?;
            let audio_b64 = base64_encode(&wav_bytes);
            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let event = ConverseWsEvent::Sentence {
                index,
                text,
                audio_b64,
                sample_rate: output.audio.sample_rate,
            };
            if send_converse_event(&ws_tx, &event).await.is_err() {
                return Err(vox::VoxError::Pipeline("client disconnected".to_string()));
            }
            Ok(())
        }
    };

    let llm_result = vox::streaming_chat::stream_chat_with_tts(
        http_client,
        host,
        model,
        &stt_result.text,
        Arc::clone(tts),
        Some(system_prompt.to_string()),
        voice.map(|v| v.to_string()),
        on_sentence,
    )
    .await;

    if let Err(e) = llm_result {
        let _ = send_converse_event(
            ws_tx,
            &ConverseWsEvent::Error {
                message: format!("LLM/TTS error: {e}"),
            },
        )
        .await;
        return Ok(());
    }

    let total = sentence_counter.load(std::sync::atomic::Ordering::Relaxed);
    if send_converse_event(ws_tx, &ConverseWsEvent::Done { sentences: total })
        .await
        .is_err()
    {
        return Err(());
    }
    Ok(())
}

/// Fallback when the `silero` feature is disabled: immediately report an error.
#[cfg(not(feature = "silero"))]
async fn handle_converse_ws(
    socket: WebSocket,
    _stt: Arc<dyn vox::traits::SttBackend>,
    _tts: Arc<dyn vox::traits::TtsBackend>,
    _vad_path: std::path::PathBuf,
    _state: Arc<ServerState>,
) {
    let (ws_tx, _ws_rx) = socket.split();
    let ws_tx = Arc::new(tokio::sync::Mutex::new(ws_tx));
    let _ = send_converse_event(
        &ws_tx,
        &ConverseWsEvent::Error {
            message: "server compiled without silero feature".into(),
        },
    )
    .await;
}

/// Serialize and send a `ConverseWsEvent` over the shared WebSocket sink.
async fn send_converse_event(
    ws_tx: &Arc<tokio::sync::Mutex<futures_util::stream::SplitSink<WebSocket, Message>>>,
    event: &ConverseWsEvent,
) -> Result<(), axum::Error> {
    let json = serde_json::to_string(event).unwrap_or_default();
    let mut guard = ws_tx.lock().await;
    guard
        .send(Message::Text(json.into()))
        .await
        .map_err(axum::Error::new)
}
