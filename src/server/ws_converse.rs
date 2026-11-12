//! Continuous voice-chat WebSocket endpoint (`/v1/converse`).
//!
//! One long-lived socket carries VAD → STT → LLM → TTS. Compared to
//! `/v1/live-talk` there is no barge-in: while a reply is being produced the
//! microphone is ignored. What the connection does guarantee:
//!
//! - The socket is read while a turn runs (the turn is its own task), so a
//!   config change or a close is seen immediately.
//! - Config is mutable: a `model` or `voice` sent at any time applies from
//!   the next turn on.
//! - The connection has a conversation memory, so later turns see earlier
//!   ones.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use vox::conversation::{Conversation, SharedConversation};
use vox::streaming_chat::{StopReason, StreamOptions};

use super::ServerState;
use super::error::ServerError;
use super::models::{ConverseConfig, ConverseWsEvent};
use super::ws::{bytes_to_f32_samples, encode_wav_chunk};

/// Model used when the client names none.
const DEFAULT_MODEL: &str = "llama3.2";

/// How long the connection waits for the first message to be a config.
const CONFIG_WAIT: Duration = Duration::from_millis(250);

/// Default for [`ConverseDeps::post_turn_mute`].
pub const DEFAULT_POST_TURN_MUTE: Duration = Duration::from_millis(400);

/// Everything a converse session needs. The transport is passed separately
/// so the session can be driven without a real socket.
pub struct ConverseDeps {
    pub stt: Arc<dyn vox::traits::SttBackend>,
    pub tts: Arc<dyn vox::traits::TtsBackend>,
    pub vad: Box<dyn vox::traits::VadBackend>,
    pub http_client: reqwest::Client,
    /// Ollama `host:port`. Server configuration; never taken from the client.
    pub ollama_host: String,
    pub stream_options: StreamOptions,
    /// Microphone audio is ignored for this long after a turn ends: those
    /// frames were captured while the reply was being produced and mostly
    /// hold the loudspeaker.
    pub post_turn_mute: Duration,
}

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
    if state.stt.is_none() {
        return Err(ServerError::service_unavailable("STT backend not loaded"));
    }
    if state.conversation_tts.is_none() && state.tts.is_none() {
        return Err(ServerError::service_unavailable("TTS backend not loaded"));
    }
    if state.vad_model_path.is_none() {
        return Err(ServerError::service_unavailable("VAD model not available"));
    }

    Ok(ws.on_upgrade(move |socket| handle_converse_ws(socket, state)))
}

/// Tell the client why the session cannot start, then give up.
async fn refuse(mut socket: WebSocket, message: String) {
    if let Ok(json) = serde_json::to_string(&ConverseWsEvent::Error { message }) {
        let _ = socket.send(Message::Text(json.into())).await;
    }
}

#[cfg(feature = "silero")]
async fn handle_converse_ws(socket: WebSocket, state: Arc<ServerState>) {
    let Some(stt) = state.stt.as_ref().map(Arc::clone) else {
        return refuse(socket, "STT backend not loaded".into()).await;
    };
    let Some(tts) = state
        .conversation_tts
        .as_ref()
        .or(state.tts.as_ref())
        .map(Arc::clone)
    else {
        return refuse(socket, "TTS backend not loaded".into()).await;
    };
    let Some(vad_path) = state.vad_model_path.clone() else {
        return refuse(socket, "VAD model not available".into()).await;
    };
    let vad: Box<dyn vox::traits::VadBackend> = match vox::SileroVad::new(&vad_path) {
        Ok(v) => Box::new(v),
        Err(e) => return refuse(socket, format!("failed to initialize VAD: {e}")).await,
    };

    let (sink, stream) = socket.split();
    run_converse(
        sink,
        stream,
        ConverseDeps {
            stt,
            tts,
            vad,
            http_client: state.http_client.clone(),
            ollama_host: state.ollama_host.clone(),
            stream_options: StreamOptions::default(),
            post_turn_mute: DEFAULT_POST_TURN_MUTE,
        },
    )
    .await;
}

/// Fallback when the `silero` feature is disabled: immediately report an error.
#[cfg(not(feature = "silero"))]
async fn handle_converse_ws(socket: WebSocket, _state: Arc<ServerState>) {
    refuse(socket, "server compiled without silero feature".into()).await;
}

/// The socket's sending half, shared by the connection loop and turn tasks.
type SharedSink<S> = Arc<tokio::sync::Mutex<S>>;

/// Serialize and send one event. Returns `false` when the client is gone.
async fn send_event<S>(sink: &SharedSink<S>, event: &ConverseWsEvent) -> bool
where
    S: Sink<Message> + Unpin,
{
    let Ok(json) = serde_json::to_string(event) else {
        return false;
    };
    sink.lock()
        .await
        .send(Message::Text(json.into()))
        .await
        .is_ok()
}

/// What stays the same for every turn of a connection.
struct TurnCtx<S> {
    sink: SharedSink<S>,
    stt: Arc<dyn vox::traits::SttBackend>,
    tts: Arc<dyn vox::traits::TtsBackend>,
    http_client: reqwest::Client,
    ollama_host: String,
    stream_options: StreamOptions,
    system_prompt: String,
    conversation: SharedConversation,
}

/// Run one converse session over any message transport until the client
/// leaves.
pub async fn run_converse<S, R, E>(sink: S, mut stream: R, deps: ConverseDeps)
where
    S: Sink<Message> + Send + Unpin + 'static,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    let sink: SharedSink<S> = Arc::new(tokio::sync::Mutex::new(sink));
    let mut vad = deps.vad;
    let frame_size = vad.frame_size().max(1);

    // The first message may be a config. If audio comes first, defaults are
    // used until a config arrives.
    let mut model = DEFAULT_MODEL.to_string();
    let mut voice: Option<String> = None;
    let mut pending_audio: Vec<f32> = Vec::new();
    match tokio::time::timeout(CONFIG_WAIT, stream.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<ConverseConfig>(&text) {
            Ok(config) => apply_config(config, &mut model, &mut voice),
            Err(e) => tracing::warn!("converse_ws: invalid config JSON, using defaults: {e}"),
        },
        Ok(Some(Ok(Message::Binary(data)))) => pending_audio = bytes_to_f32_samples(&data),
        Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) | Ok(None) => return,
        _ => {}
    }

    tracing::info!(
        model = %model,
        host = %deps.ollama_host,
        voice = ?voice,
        "converse_ws: client connected"
    );

    if !send_event(
        &sink,
        &ConverseWsEvent::Ready {
            model: model.clone(),
            voice: voice.clone(),
        },
    )
    .await
    {
        return;
    }

    let ctx = Arc::new(TurnCtx {
        sink: Arc::clone(&sink),
        stt: deps.stt,
        tts: deps.tts,
        http_client: deps.http_client,
        ollama_host: deps.ollama_host,
        stream_options: deps.stream_options,
        system_prompt: vox::build_system_prompt(vox::VoicePromptMode::Voice),
        conversation: Arc::new(std::sync::Mutex::new(Conversation::for_num_ctx(
            vox::conversation::ollama_num_ctx(),
        ))),
    });

    // Cancelled when the client leaves, so a running turn stops.
    let closed = CancellationToken::new();
    let mut active: Option<tokio::task::JoinHandle<()>> = None;
    let mut muted_until: Option<Instant> = None;
    // Samples that did not fill a whole VAD frame yet.
    let mut residual: Vec<f32> = Vec::new();

    let mut next_audio = Some(pending_audio).filter(|a| !a.is_empty());
    loop {
        let samples = match next_audio.take() {
            Some(samples) => samples,
            None => {
                let Some(Ok(msg)) = stream.next().await else {
                    break;
                };
                match msg {
                    Message::Binary(data) => bytes_to_f32_samples(&data),
                    Message::Text(text) => {
                        // Applies from the next turn on; a running turn
                        // keeps the values it started with.
                        match serde_json::from_str::<ConverseConfig>(&text) {
                            Ok(config) => {
                                apply_config(config, &mut model, &mut voice);
                                tracing::info!(
                                    model = %model,
                                    voice = ?voice,
                                    "converse_ws: config updated"
                                );
                            }
                            Err(e) => tracing::debug!("converse_ws: ignoring text msg: {e}"),
                        }
                        continue;
                    }
                    Message::Close(_) => break,
                    _ => continue,
                }
            }
        };

        if active.as_ref().is_some_and(|turn| turn.is_finished()) {
            active = None;
            // Start the next utterance from a clean state.
            vad.reset();
            residual.clear();
            muted_until = Some(Instant::now() + deps.post_turn_mute);
        }
        if active.is_some() {
            // No barge-in here: the microphone is ignored during a turn.
            continue;
        }
        if muted_until.is_some_and(|until| Instant::now() < until) {
            continue;
        }
        muted_until = None;

        let utterance = match detect_utterance(
            &samples,
            frame_size,
            vad.as_mut(),
            &sink,
            &mut residual,
        )
        .await
        {
            Ok(Some(utterance)) => utterance,
            Ok(None) => continue,
            Err(()) => break,
        };
        active = Some(tokio::spawn(run_turn(
            Arc::clone(&ctx),
            utterance,
            model.clone(),
            voice.clone(),
            closed.clone(),
        )));
    }

    closed.cancel();
    if let Some(turn) = active {
        let _ = tokio::time::timeout(Duration::from_millis(250), turn).await;
    }
    tracing::info!("converse_ws: client disconnected");
}

fn apply_config(config: ConverseConfig, model: &mut String, voice: &mut Option<String>) {
    if let Some(m) = config.model {
        *model = m;
    }
    if let Some(v) = config.voice {
        *voice = Some(v);
    }
}

/// Feed audio through the VAD and emit `speech_start` / `speech_end`.
/// Returns the utterance when one was completed, `Err` when the session
/// cannot go on.
///
/// `residual` holds samples that did not fill a whole VAD frame on the
/// previous call, so no audio is dropped at chunk boundaries.
async fn detect_utterance<S>(
    samples: &[f32],
    frame_size: usize,
    vad: &mut dyn vox::traits::VadBackend,
    sink: &SharedSink<S>,
    residual: &mut Vec<f32>,
) -> Result<Option<vox::Utterance>, ()>
where
    S: Sink<Message> + Unpin,
{
    residual.extend_from_slice(samples);
    let consumed = (residual.len() / frame_size) * frame_size;
    let frames: Vec<f32> = residual.drain(..consumed).collect();

    for frame_samples in frames.chunks_exact(frame_size) {
        let frame = vox::AudioChunk {
            samples: frame_samples.to_vec(),
            sample_rate: 16000,
            channels: 1,
        };
        let events = match vad.process_frame(&frame).await {
            Ok(events) => events,
            Err(e) => {
                let _ = send_event(
                    sink,
                    &ConverseWsEvent::Error {
                        message: format!("VAD error: {e}"),
                    },
                )
                .await;
                return Err(());
            }
        };
        for event in events {
            match event {
                vox::VadEvent::SpeechStart => {
                    if !send_event(sink, &ConverseWsEvent::SpeechStart).await {
                        return Err(());
                    }
                }
                vox::VadEvent::SpeechEnd(utterance) => {
                    if !send_event(sink, &ConverseWsEvent::SpeechEnd).await {
                        return Err(());
                    }
                    // The rest of this batch is post-speech audio that the
                    // turn would ignore anyway.
                    return Ok(Some(utterance));
                }
                vox::VadEvent::Silence => {}
            }
        }
    }
    Ok(None)
}

/// One conversation turn: STT → streaming LLM → TTS per sentence. Errors are
/// reported to the client and end the turn, not the connection.
async fn run_turn<S>(
    ctx: Arc<TurnCtx<S>>,
    utterance: vox::Utterance,
    model: String,
    voice: Option<String>,
    closed: CancellationToken,
) where
    S: Sink<Message> + Send + Unpin + 'static,
{
    let sink = &ctx.sink;

    let stt_result = tokio::select! {
        biased;
        _ = closed.cancelled() => return,
        r = ctx.stt.transcribe(&utterance) => r,
    };
    let stt_result = match stt_result {
        Ok(r) => r,
        Err(e) => {
            let _ = send_event(
                sink,
                &ConverseWsEvent::Error {
                    message: format!("STT error: {e}"),
                },
            )
            .await;
            return;
        }
    };

    let user_text = stt_result.text.trim().to_string();
    if user_text.is_empty() {
        // Nothing transcribed; skip the LLM round-trip.
        let _ = send_event(sink, &ConverseWsEvent::Done { sentences: 0 }).await;
        return;
    }

    if !send_event(
        sink,
        &ConverseWsEvent::Transcript {
            text: user_text.clone(),
            duration_ms: stt_result.duration_ms,
            processing_time_ms: stt_result.processing_time_ms,
        },
    )
    .await
        || !send_event(sink, &ConverseWsEvent::Thinking).await
    {
        return;
    }

    let messages = ctx
        .conversation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .messages_for(Some(&ctx.system_prompt), &user_text);

    // Sentences whose audio reached the client; these go into memory.
    let spoken: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let on_sentence = {
        let ctx = Arc::clone(&ctx);
        let spoken = Arc::clone(&spoken);
        move |sentence: &str| {
            let ctx = Arc::clone(&ctx);
            let spoken = Arc::clone(&spoken);
            let voice = voice.clone();
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
                let output = ctx.tts.synthesize(&request).await?;
                let wav = encode_wav_chunk(&output.audio.samples, output.audio.sample_rate)
                    .map_err(vox::VoxError::Tts)?;
                let index = spoken.lock().unwrap_or_else(|e| e.into_inner()).len();
                let event = ConverseWsEvent::Sentence {
                    index,
                    text: text.clone(),
                    audio_b64: base64_encode(&wav),
                    sample_rate: output.audio.sample_rate,
                };
                if !send_event(&ctx.sink, &event).await {
                    return Err(vox::VoxError::Pipeline("client disconnected".to_string()));
                }
                spoken.lock().unwrap_or_else(|e| e.into_inner()).push(text);
                Ok(())
            }
        }
    };

    let stop = vox::streaming_chat::stream_chat(
        &ctx.http_client,
        &ctx.ollama_host,
        &model,
        &messages,
        &ctx.stream_options,
        closed,
        |_token: &str| {},
        on_sentence,
    )
    .await;

    let spoken = std::mem::take(&mut *spoken.lock().unwrap_or_else(|e| e.into_inner()));
    let interrupted = match &stop {
        Ok(StopReason::Finished) => Some(false),
        Ok(StopReason::Cancelled) => Some(true),
        // A reply that failed half-way is remembered as cut short; one that
        // never produced audio is not remembered at all.
        Err(_) => (!spoken.is_empty()).then_some(true),
    };
    if let Some(interrupted) = interrupted {
        ctx.conversation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .commit(&user_text, &spoken, interrupted);
    }

    match stop {
        Ok(StopReason::Finished) => {
            let _ = send_event(
                sink,
                &ConverseWsEvent::Done {
                    sentences: spoken.len(),
                },
            )
            .await;
        }
        Ok(StopReason::Cancelled) => {}
        Err(e) => {
            let _ = send_event(
                sink,
                &ConverseWsEvent::Error {
                    message: format!("LLM/TTS error: {e}"),
                },
            )
            .await;
        }
    }
}
