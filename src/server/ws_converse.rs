//! Continuous voice-chat WebSocket endpoint (`/v1/converse`).

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};

use super::ServerState;
use super::error::ServerError;
use super::models::{ConverseConfig, ConverseWsEvent};
use super::ws::{bytes_to_f32_samples, encode_wav_chunk};

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
        .conversation_tts
        .as_ref()
        .or(state.tts.as_ref())
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
            let drain_deadline = std::time::Instant::now() + std::time::Duration::from_millis(400);
            while std::time::Instant::now() < drain_deadline {
                match tokio::time::timeout(std::time::Duration::from_millis(20), ws_rx.next()).await
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
