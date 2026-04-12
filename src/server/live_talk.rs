//! Full-duplex voice chat with barge-in. Client can interrupt assistant mid-sentence;
//! server cancels in-flight LLM/TTS via [`tokio_util::sync::CancellationToken`] and
//! immediately captures the new utterance. Protocol: JSON text events + binary f32 PCM.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::Mutex;
#[cfg(feature = "silero")]
use tokio_util::sync::CancellationToken;
#[cfg(feature = "silero")]
use tracing::{debug, info};

use super::error::ServerError;
use super::models::{
    CancelReason, LiveTalkClientMsg, LiveTalkConfig, LiveTalkMode, LiveTalkWsEvent,
};
use super::ServerState;

/// Shared WebSocket sender for interleaved events from main loop and worker tasks.
type WsSink = Arc<Mutex<futures_util::stream::SplitSink<WebSocket, Message>>>;

/// Active turn handle. Held in `Option` so barge-in can `take()` and cancel cleanly.
#[cfg(feature = "silero")]
struct ActiveTurn {
    handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
}

/// Upgrade HTTP connection to Live Talk WebSocket. Returns 503 if STT/TTS/VAD unavailable.
pub async fn live_talk_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, ServerError> {
    if state.stt.is_none() {
        return Err(ServerError::service_unavailable("STT backend not loaded"));
    }
    if state.tts.is_none() {
        return Err(ServerError::service_unavailable("TTS backend not loaded"));
    }
    if state.vad_model_path.is_none() {
        return Err(ServerError::service_unavailable("VAD model not available"));
    }

    Ok(ws.on_upgrade(move |socket| handle_live_talk(socket, state)))
}

#[cfg(feature = "silero")]
#[allow(clippy::too_many_lines)]
async fn handle_live_talk(socket: WebSocket, state: Arc<ServerState>) {
    let stt = match state.stt.as_ref() {
        Some(s) => Arc::clone(s),
        None => {
            let (ws_tx, _) = socket.split();
            let ws_tx: WsSink = Arc::new(Mutex::new(ws_tx));
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: "STT backend not loaded".into(),
                    fatal: true,
                },
            )
            .await;
            return;
        }
    };
    let tts = match state.conversation_tts.as_ref().or(state.tts.as_ref()) {
        Some(t) => Arc::clone(t),
        None => {
            let (ws_tx, _) = socket.split();
            let ws_tx: WsSink = Arc::new(Mutex::new(ws_tx));
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: "TTS backend not loaded".into(),
                    fatal: true,
                },
            )
            .await;
            return;
        }
    };
    let vad_path = match state.vad_model_path.as_ref() {
        Some(p) => p.clone(),
        None => {
            let (ws_tx, _) = socket.split();
            let ws_tx: WsSink = Arc::new(Mutex::new(ws_tx));
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: "VAD model not available".into(),
                    fatal: true,
                },
            )
            .await;
            return;
        }
    };

    let (ws_tx, mut ws_rx) = socket.split();
    let ws_tx: WsSink = Arc::new(Mutex::new(ws_tx));

    let mut vad: Box<dyn vox::traits::VadBackend> = match vox::SileroVad::new(&vad_path) {
        Ok(v) => Box::new(v),
        Err(e) => {
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: format!("failed to initialize VAD: {e}"),
                    fatal: true,
                },
            )
            .await;
            return;
        }
    };
    let frame_size = vad.frame_size();

    let mut config = LiveTalkConfig {
        model: None,
        host: None,
        voice: None,
        mode: super::models::LiveTalkModeConfig::default(),
        barge_in_enabled: true,
        system_prompt_override: None,
    };
    let mut pending_audio: Vec<u8> = Vec::new();
    match tokio::time::timeout(Duration::from_millis(250), ws_rx.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => {
            if let Ok(LiveTalkClientMsg::Config(c)) = serde_json::from_str(&text) {
                config = c;
            } else if let Ok(c) = serde_json::from_str::<LiveTalkConfig>(&text) {
                config = c;
            } else {
                tracing::warn!("live_talk: unrecognized first text msg, using defaults");
            }
        }
        Ok(Some(Ok(Message::Binary(data)))) => {
            pending_audio = data.to_vec();
        }
        Ok(Some(Ok(Message::Close(_)))) | Ok(None) => return,
        Ok(Some(Err(_))) => return,
        _ => {}
    }

    let mut model = config
        .model
        .clone()
        .unwrap_or_else(|| "llama3.2".to_string());
    let host = config
        .host
        .clone()
        .unwrap_or_else(|| state.ollama_host.clone());
    let mut voice = config.voice.clone();
    let mut mode: LiveTalkMode = config.mode.into();
    let barge_in_enabled = config.barge_in_enabled;
    let system_prompt = config
        .system_prompt_override
        .clone()
        .unwrap_or_else(|| vox::build_system_prompt(vox::VoicePromptMode::Voice));

    info!(
        model = %model,
        host = %host,
        voice = ?voice,
        mode = ?mode,
        barge_in = barge_in_enabled,
        "live_talk: client connected"
    );

    if !send_live_event(
        &ws_tx,
        &LiveTalkWsEvent::Ready {
            model: model.clone(),
            voice: voice.clone(),
            mode,
        },
    )
    .await
    {
        return;
    }

    let http_client = state.http_client.clone();

    let mut active_turn: Option<ActiveTurn> = None;
    let mut audio_residual: Vec<f32> = Vec::with_capacity(frame_size * 2);
    let mut ptt_active = false;
    let mut ptt_buffer: Vec<f32> = Vec::new();

    if !pending_audio.is_empty() {
        let samples = super::ws::bytes_to_f32_samples(&pending_audio);
        if handle_live_frames(
            &samples,
            frame_size,
            vad.as_mut(),
            &ws_tx,
            &mut audio_residual,
            &mut active_turn,
            barge_in_enabled,
            mode,
            ptt_active,
            &mut ptt_buffer,
            &Arc::clone(&stt),
            &Arc::clone(&tts),
            &http_client,
            &host,
            &model,
            &system_prompt,
            voice.as_deref(),
        )
        .await
        .is_err()
        {
            return;
        }
    }

    loop {
        let Some(msg) = ws_rx.next().await else { break };
        match msg {
            Ok(Message::Binary(data)) => {
                if matches!(mode, LiveTalkMode::PushToTalk) {
                    if !ptt_active {
                        continue;
                    }
                    let samples = super::ws::bytes_to_f32_samples(&data);
                    ptt_buffer.extend_from_slice(&samples);
                    continue;
                }

                // Detect completed turns: if the spawned task has finished,
                // clean up and reset VAD + residual so the next utterance
                // starts from a clean state (no echo contamination).
                if let Some(ref turn) = active_turn {
                    if turn.handle.is_finished() {
                        active_turn.take();
                        vad.reset();
                        audio_residual.clear();
                    }
                }

                let samples = super::ws::bytes_to_f32_samples(&data);
                if handle_live_frames(
                    &samples,
                    frame_size,
                    vad.as_mut(),
                    &ws_tx,
                    &mut audio_residual,
                    &mut active_turn,
                    barge_in_enabled,
                    mode,
                    ptt_active,
                    &mut ptt_buffer,
                    &Arc::clone(&stt),
                    &Arc::clone(&tts),
                    &http_client,
                    &host,
                    &model,
                    &system_prompt,
                    voice.as_deref(),
                )
                .await
                .is_err()
                {
                    break;
                }
            }
            Ok(Message::Text(text)) => {
                let parsed: Result<LiveTalkClientMsg, _> = serde_json::from_str(&text);
                match parsed {
                    Ok(LiveTalkClientMsg::Config(new_cfg)) => {
                        if let Some(m) = new_cfg.model {
                            model = m;
                        }
                        if let Some(v) = new_cfg.voice {
                            voice = Some(v);
                        }
                        mode = new_cfg.mode.into();
                        info!(
                            model = %model,
                            voice = ?voice,
                            mode = ?mode,
                            "live_talk: config updated"
                        );
                    }
                    Ok(LiveTalkClientMsg::Cancel) => {
                        if let Some(turn) = active_turn.take() {
                            turn.cancel.cancel();
                            let _ = tokio::time::timeout(
                                Duration::from_millis(100),
                                turn.handle,
                            )
                            .await;
                        }
                        // Reset VAD after cancellation so echo-primed LSTM
                        // state doesn't bleed into the next utterance.
                        vad.reset();
                        audio_residual.clear();
                        let _ = send_live_event(
                            &ws_tx,
                            &LiveTalkWsEvent::Cancelled {
                                reason: CancelReason::ClientRequest,
                            },
                        )
                        .await;
                    }
                    Ok(LiveTalkClientMsg::PttStart) => {
                        ptt_active = true;
                        ptt_buffer.clear();
                        if let Some(turn) = active_turn.take() {
                            turn.cancel.cancel();
                            let _ = tokio::time::timeout(
                                Duration::from_millis(100),
                                turn.handle,
                            )
                            .await;
                            vad.reset();
                            audio_residual.clear();
                            let _ = send_live_event(
                                &ws_tx,
                                &LiveTalkWsEvent::Cancelled {
                                    reason: CancelReason::UserBargeIn,
                                },
                            )
                            .await;
                        }
                        let _ = send_live_event(&ws_tx, &LiveTalkWsEvent::SpeechStart).await;
                    }
                    Ok(LiveTalkClientMsg::PttEnd) => {
                        ptt_active = false;
                        if matches!(mode, LiveTalkMode::PushToTalk) && !ptt_buffer.is_empty() {
                            let samples = std::mem::take(&mut ptt_buffer);
                            let duration_ms =
                                (samples.len() as u64 * 1000) / 16_000;
                            let utterance = vox::Utterance {
                                audio: vox::AudioChunk {
                                    samples,
                                    sample_rate: 16000,
                                    channels: 1,
                                },
                                duration_ms,
                                #[cfg(feature = "diarization")]
                                speaker_id: None,
                            };
                            dispatch_turn(
                                utterance,
                                &mut active_turn,
                                &ws_tx,
                                &Arc::clone(&stt),
                                &Arc::clone(&tts),
                                &http_client,
                                &host,
                                &model,
                                &system_prompt,
                                voice.as_deref(),
                            );
                        }
                    }
                    Err(e) => {
                        debug!("live_talk: ignoring unknown text msg: {e}");
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if let Some(turn) = active_turn.take() {
        turn.cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_millis(200), turn.handle).await;
    }

    info!("live_talk: client disconnected");
}

/// Feed audio through VAD and dispatch turn/barge-in events.
///
/// When a turn is active (TTS playing), VAD processing is **skipped** to
/// prevent echo feedback: the mic picks up TTS audio from the speaker,
/// Silero classifies it as speech, and the resulting echo-contaminated
/// utterance degrades STT quality (producing "[INAUDIBLE]" or hallucinated
/// text). Barge-in detection during active turns relies on the client-side
/// RMS check instead.
#[cfg(feature = "silero")]
#[allow(clippy::too_many_arguments)]
async fn handle_live_frames(
    samples: &[f32],
    frame_size: usize,
    vad: &mut dyn vox::traits::VadBackend,
    ws_tx: &WsSink,
    residual: &mut Vec<f32>,
    active_turn: &mut Option<ActiveTurn>,
    _barge_in_enabled: bool,
    mode: LiveTalkMode,
    _ptt_active: bool,
    _ptt_buffer: &mut [f32],
    stt: &Arc<dyn vox::traits::SttBackend>,
    tts: &Arc<dyn vox::traits::TtsBackend>,
    http_client: &reqwest::Client,
    host: &str,
    model: &str,
    system_prompt: &str,
    voice: Option<&str>,
) -> Result<(), ()> {
    if matches!(mode, LiveTalkMode::PushToTalk) {
        return Ok(());
    }

    // Skip VAD processing while a turn is active (TTS playing) to prevent
    // echo feedback. The mic captures TTS audio from the speaker; without
    // this gate, Silero would classify the echo as speech, buffer it, and
    // eventually send echo-contaminated audio to STT. Client-side RMS
    // barge-in (clientBargeInCheck in ui.html) handles interruption instead.
    if active_turn.is_some() {
        return Ok(());
    }

    residual.extend_from_slice(samples);
    let full_frames = residual.len() / frame_size;
    let consumed = full_frames * frame_size;
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
                            if !send_live_event(ws_tx, &LiveTalkWsEvent::SpeechStart).await {
                                return Err(());
                            }
                        }
                        vox::VadEvent::SpeechEnd(utt) => {
                            dispatch_turn(
                                utt,
                                active_turn,
                                ws_tx,
                                stt,
                                tts,
                                http_client,
                                host,
                                model,
                                system_prompt,
                                voice,
                            );
                        }
                        vox::VadEvent::Silence => {}
                    }
                }
            }
            Err(e) => {
                let _ = send_live_event(
                    ws_tx,
                    &LiveTalkWsEvent::Error {
                        message: format!("VAD error: {e}"),
                        fatal: false,
                    },
                )
                .await;
                return Err(());
            }
        }
    }
    Ok(())
}

/// Spawn per-turn worker task. Runs independently so main loop can observe new SpeechStart and cancel.
#[cfg(feature = "silero")]
#[allow(clippy::too_many_arguments)]
fn dispatch_turn(
    utterance: vox::Utterance,
    active_turn: &mut Option<ActiveTurn>,
    ws_tx: &WsSink,
    stt: &Arc<dyn vox::traits::SttBackend>,
    tts: &Arc<dyn vox::traits::TtsBackend>,
    http_client: &reqwest::Client,
    host: &str,
    model: &str,
    system_prompt: &str,
    voice: Option<&str>,
) {
    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let ws_tx_for_task = Arc::clone(ws_tx);
    let stt_for_task = Arc::clone(stt);
    let tts_for_task = Arc::clone(tts);
    let http_for_task = http_client.clone();
    let host_for_task = host.to_string();
    let model_for_task = model.to_string();
    let system_prompt_for_task = system_prompt.to_string();
    let voice_for_task = voice.map(|v| v.to_string());

    let handle = tokio::spawn(async move {
        run_turn_cancellable(
            utterance,
            cancel_for_task,
            ws_tx_for_task,
            stt_for_task,
            tts_for_task,
            http_for_task,
            host_for_task,
            model_for_task,
            system_prompt_for_task,
            voice_for_task,
        )
        .await;
    });

    *active_turn = Some(ActiveTurn { handle, cancel });
}

/// STT → streaming LLM → TTS per sentence, cancellable. Runs on detached task;
/// cancellation drops future at next await. Non-fatal errors keep socket open.
#[cfg(feature = "silero")]
#[allow(clippy::too_many_arguments)]
async fn run_turn_cancellable(
    utterance: vox::Utterance,
    cancel: CancellationToken,
    ws_tx: WsSink,
    stt: Arc<dyn vox::traits::SttBackend>,
    tts: Arc<dyn vox::traits::TtsBackend>,
    http_client: reqwest::Client,
    host: String,
    model: String,
    system_prompt: String,
    voice: Option<String>,
) {
    if cancel.is_cancelled() {
        return;
    }
    let stt_result = match stt.transcribe(&utterance).await {
        Ok(r) => r,
        Err(e) => {
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: format!("STT: {e}"),
                    fatal: false,
                },
            )
            .await;
            return;
        }
    };
    if cancel.is_cancelled() {
        return;
    }

    if stt_result.text.trim().is_empty() {
        let _ = send_live_event(&ws_tx, &LiveTalkWsEvent::TurnDone { sentences: 0 }).await;
        return;
    }

    if !send_live_event(
        &ws_tx,
        &LiveTalkWsEvent::Transcript {
            text: stt_result.text.clone(),
            duration_ms: stt_result.duration_ms,
            processing_time_ms: stt_result.processing_time_ms,
        },
    )
    .await
    {
        return;
    }
    if !send_live_event(&ws_tx, &LiveTalkWsEvent::Thinking).await {
        return;
    }

    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tts_cb = Arc::clone(&tts);
    let ws_cb = Arc::clone(&ws_tx);
    let voice_cb = voice.clone();
    let counter_cb = Arc::clone(&counter);
    let cancel_cb = cancel.clone();

    let on_sentence = move |sentence: &str| {
        let tts = Arc::clone(&tts_cb);
        let ws = Arc::clone(&ws_cb);
        let voice = voice_cb.clone();
        let counter = Arc::clone(&counter_cb);
        let cancel = cancel_cb.clone();
        let text = sentence.to_string();
        async move {
            if text.trim().is_empty() {
                return Ok(());
            }
            if cancel.is_cancelled() {
                return Err(vox::VoxError::Pipeline("cancelled".into()));
            }
            let req = vox::TtsRequest {
                text: text.clone(),
                voice,
                seed: None,
            };
            let output = tokio::select! {
                _ = cancel.cancelled() => {
                    return Err(vox::VoxError::Pipeline("cancelled".into()));
                }
                r = tts.synthesize(&req) => r?,
            };
            if cancel.is_cancelled() {
                return Err(vox::VoxError::Pipeline("cancelled".into()));
            }

            let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if !send_live_event(&ws, &LiveTalkWsEvent::Sentence {
                index,
                text: text.clone(),
            })
            .await
            {
                return Err(vox::VoxError::Pipeline("client disconnected".into()));
            }
            if !send_live_event(
                &ws,
                &LiveTalkWsEvent::AudioChunk {
                    sentence_index: index,
                    sample_rate: output.audio.sample_rate,
                },
            )
            .await
            {
                return Err(vox::VoxError::Pipeline("client disconnected".into()));
            }

            let wav_bytes = super::ws::encode_wav_chunk(
                &output.audio.samples,
                output.audio.sample_rate,
            )
            .map_err(vox::VoxError::Tts)?;
            {
                let mut sink = ws.lock().await;
                if sink
                    .send(Message::Binary(wav_bytes.into()))
                    .await
                    .is_err()
                {
                    return Err(vox::VoxError::Pipeline("client disconnected".into()));
                }
            }
            Ok(())
        }
    };

    let stop = vox::streaming_chat::stream_chat_with_tts_cancellable(
        &http_client,
        &host,
        &model,
        &stt_result.text,
        Arc::clone(&tts),
        Some(system_prompt),
        voice,
        cancel.clone(),
        on_sentence,
    )
    .await;

    match stop {
        Ok(vox::streaming_chat::StopReason::Finished) => {
            let total = counter.load(std::sync::atomic::Ordering::Relaxed);
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::TurnDone { sentences: total },
            )
            .await;
        }
        Ok(vox::streaming_chat::StopReason::Cancelled) => {
            debug!("live_talk: turn cancelled");
        }
        Err(e) => {
            let _ = send_live_event(
                &ws_tx,
                &LiveTalkWsEvent::Error {
                    message: format!("LLM: {e}"),
                    fatal: false,
                },
            )
            .await;
        }
    }
}

/// Send LiveTalkWsEvent. Returns false if client disconnected.
async fn send_live_event(ws_tx: &WsSink, event: &LiveTalkWsEvent) -> bool {
    let Ok(json) = serde_json::to_string(event) else {
        return false;
    };
    let mut sink = ws_tx.lock().await;
    sink.send(Message::Text(json.into())).await.is_ok()
}

#[cfg(not(feature = "silero"))]
async fn handle_live_talk(socket: WebSocket, _state: Arc<ServerState>) {
    let (ws_tx, _) = socket.split();
    let ws_tx: WsSink = Arc::new(Mutex::new(ws_tx));
    let _ = send_live_event(
        &ws_tx,
        &LiveTalkWsEvent::Error {
            message: "Live Talk requires the `silero` feature".into(),
            fatal: true,
        },
    )
    .await;
}
