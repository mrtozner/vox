//! Full-duplex voice chat with barge-in (`/v1/live-talk`, protocol v2).
//!
//! The wire protocol is documented on [`LiveTalkWsEvent`] and
//! [`LiveTalkClientMsg`]. This module is the server side of it:
//!
//! - **One writer.** Everything sent to the client goes through one queue
//!   ([`Outbox`]) drained by one task, so an `audio_chunk` header is always
//!   followed by its binary frame, and events of a cancelled turn that are
//!   still queued are dropped instead of sent.
//! - **Turns run detached.** A turn (STT → LLM → TTS) is a spawned task; the
//!   connection loop keeps reading the socket and can cancel it at any point,
//!   including while it is transcribing.
//! - **Microphone gate.** While reply audio is playing, the microphone hears
//!   the loudspeaker, so that audio is kept away from the VAD. It is held in
//!   a short ring buffer instead of being thrown away; when the client
//!   cancels (its own barge-in detection fired) the ring is replayed into the
//!   VAD, so the start of what the user said is not lost. The gate stays
//!   closed after `turn_done` until the client reports `playback_done` (or
//!   the audio that was sent must have finished playing).
//! - **Voice barge-in.** Before the first reply audio there is nothing to
//!   echo, so the VAD keeps listening; speech in that phase cancels the turn
//!   with reason `user_barge_in`.
//! - **Memory records what was heard.** A turn is stored with the sentences
//!   that were sent; `cancel` and `playback_done` carry `heard_sentences`,
//!   which cuts the stored turn down to what was really played, also after
//!   `turn_done`.
//! - **Sessions.** Memory lives in the server's
//!   [`SessionStore`](vox::conversation::SessionStore), keyed by the
//!   `session_id` from `ready`, so closing the socket (for example by
//!   toggling the microphone) does not lose the conversation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use vox::conversation::{SessionStore, SharedConversation};
use vox::streaming_chat::{StopReason, StreamOptions};

use super::ServerState;
use super::error::ServerError;
use super::models::{
    CancelReason, LIVE_TALK_PROTOCOL_VERSION, LiveTalkClientMsg, LiveTalkConfig, LiveTalkMode,
    LiveTalkWsEvent,
};

/// Sample rate of the microphone stream.
const MIC_SAMPLE_RATE: u64 = 16_000;

/// Gated microphone audio kept for replay after a cancel (~400 ms).
const RING_SAMPLES: usize = 6_400;

/// Longest push-to-talk recording that is kept (2 minutes).
const MAX_PTT_SAMPLES: usize = 16_000 * 120;

/// Longest audio carried over from a turn that was interrupted by voice
/// while it was still being transcribed (30 s).
const MAX_CARRY_SAMPLES: usize = 16_000 * 30;

/// How long the connection waits for the first message to be a config.
const CONFIG_WAIT: Duration = Duration::from_millis(250);

/// How long a cancel waits for the turn task to wind down. The task stops at
/// its next await point, so this is normally not reached.
const CANCEL_JOIN: Duration = Duration::from_millis(250);

/// Default for [`LiveTalkDeps::playback_grace`].
pub const DEFAULT_PLAYBACK_GRACE: Duration = Duration::from_millis(800);

/// Model used when the client names none.
const DEFAULT_MODEL: &str = "llama3.2";

const PHASE_STT: u8 = 0;
const PHASE_THINKING: u8 = 1;
const PHASE_SPEAKING: u8 = 2;

/// Everything a Live Talk session needs. The transport is passed separately
/// so the session can be driven without a real socket.
pub struct LiveTalkDeps {
    pub stt: Arc<dyn vox::traits::SttBackend>,
    pub tts: Arc<dyn vox::traits::TtsBackend>,
    pub vad: Box<dyn vox::traits::VadBackend>,
    pub http_client: reqwest::Client,
    /// Ollama `host:port`. Server configuration; never taken from the client.
    pub ollama_host: String,
    pub sessions: Arc<SessionStore>,
    /// LLM stream timeouts. A stalled LLM ends the turn with a non-fatal
    /// error instead of holding the session.
    pub stream_options: StreamOptions,
    /// How long after the expected end of playback the microphone gate
    /// opens by itself, for a client that sends no `playback_done`.
    pub playback_grace: Duration,
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

/// Tell the client why the session cannot start, then give up.
async fn refuse(mut socket: WebSocket, message: String) {
    let event = LiveTalkWsEvent::Error {
        message,
        fatal: true,
        turn_id: None,
    };
    if let Ok(json) = serde_json::to_string(&event) {
        let _ = socket.send(Message::Text(json.into())).await;
    }
}

#[cfg(feature = "silero")]
async fn handle_live_talk(socket: WebSocket, state: Arc<ServerState>) {
    // Counted as an open session until this function returns, so shutdown
    // can wait for it.
    let _session = state.ws_session();
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

    // Every STT/TTS call of the session takes a server inference slot.
    let stt: Arc<dyn vox::traits::SttBackend> = Arc::new(super::ws::LimitedStt {
        inner: stt,
        state: Arc::clone(&state),
    });
    let tts: Arc<dyn vox::traits::TtsBackend> = Arc::new(super::ws::LimitedTts {
        inner: tts,
        state: Arc::clone(&state),
    });

    // On shutdown the session sees the client as gone and the relay sends
    // the 1001 close frame.
    let (sink, stream) = socket.split();
    let (sink, stream, relay) = super::ws::with_shutdown(sink, stream, state.shutdown.clone());
    run_live_talk(
        sink,
        stream,
        LiveTalkDeps {
            stt,
            tts,
            vad,
            http_client: state.http_client.clone(),
            ollama_host: state.ollama_host.clone(),
            sessions: Arc::clone(&state.sessions),
            stream_options: StreamOptions::default(),
            playback_grace: DEFAULT_PLAYBACK_GRACE,
        },
    )
    .await;
    let _ = relay.await;
}

#[cfg(not(feature = "silero"))]
async fn handle_live_talk(socket: WebSocket, _state: Arc<ServerState>) {
    refuse(socket, "Live Talk requires the `silero` feature".into()).await;
}

// ─── Outgoing queue ──────────────────────────────────────────

enum Out {
    /// `turn` is set for events that are worthless once that turn is dead.
    Event {
        turn: Option<u64>,
        event: LiveTalkWsEvent,
    },
    /// An `audio_chunk` header and its binary frame, written back to back.
    Audio {
        turn: u64,
        header: LiveTalkWsEvent,
        wav: Vec<u8>,
    },
}

/// Handle for queueing messages to the client.
#[derive(Clone)]
struct Outbox {
    tx: mpsc::UnboundedSender<Out>,
    /// Turns up to and including this id are dead: whatever is still queued
    /// for them is dropped by the writer.
    dead_upto: Arc<AtomicU64>,
}

impl Outbox {
    /// Queue an event that must reach the client whatever happens to turns.
    fn send(&self, event: LiveTalkWsEvent) -> bool {
        self.tx.send(Out::Event { turn: None, event }).is_ok()
    }

    /// Queue an event that belongs to `turn`.
    fn send_turn(&self, turn: u64, event: LiveTalkWsEvent) -> bool {
        self.tx
            .send(Out::Event {
                turn: Some(turn),
                event,
            })
            .is_ok()
    }

    fn send_audio(&self, turn: u64, header: LiveTalkWsEvent, wav: Vec<u8>) -> bool {
        self.tx.send(Out::Audio { turn, header, wav }).is_ok()
    }

    /// Stop sending anything that belongs to `turn` or an earlier turn.
    fn kill_turn(&self, turn: u64) {
        self.dead_upto.fetch_max(turn, Ordering::SeqCst);
    }

    /// `true` once the client is gone.
    fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

/// Drain the queue into the socket. Ends when every [`Outbox`] is dropped or
/// the client stops accepting messages.
async fn write_loop<S>(mut sink: S, mut rx: mpsc::UnboundedReceiver<Out>, dead_upto: Arc<AtomicU64>)
where
    S: Sink<Message> + Unpin,
{
    while let Some(out) = rx.recv().await {
        let is_dead = |turn: u64| turn <= dead_upto.load(Ordering::SeqCst);
        match out {
            Out::Event { turn, event } => {
                if turn.is_some_and(is_dead) {
                    continue;
                }
                let Ok(json) = serde_json::to_string(&event) else {
                    continue;
                };
                if sink.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
            Out::Audio { turn, header, wav } => {
                if is_dead(turn) {
                    continue;
                }
                let Ok(json) = serde_json::to_string(&header) else {
                    continue;
                };
                if sink.send(Message::Text(json.into())).await.is_err()
                    || sink.send(Message::Binary(wav.into())).await.is_err()
                {
                    break;
                }
            }
        }
    }
    let _ = sink.close().await;
}

// ─── Turn task ───────────────────────────────────────────────

/// What the connection loop and a turn task share about one turn.
struct TurnShared {
    phase: AtomicU8,
    /// When the browser will have played everything queued so far, assuming
    /// it plays each chunk as soon as the one before it ends. `None` until
    /// the first audio.
    playback_end: std::sync::Mutex<Option<Instant>>,
    /// Number of sentences the client says it heard; `usize::MAX` = unknown.
    heard_limit: AtomicUsize,
}

/// The values a turn starts with. Later config changes do not affect it.
#[derive(Clone)]
struct TurnConfig {
    model: String,
    voice: Option<String>,
    system_prompt: String,
}

/// Per-connection state shared with turn tasks.
struct TurnCtx {
    stt: Arc<dyn vox::traits::SttBackend>,
    tts: Arc<dyn vox::traits::TtsBackend>,
    http_client: reqwest::Client,
    ollama_host: String,
    stream_options: StreamOptions,
    out: Outbox,
    conversation: SharedConversation,
    /// The turn of this connection that is the newest entry in
    /// `conversation`, if any. Locked only while `conversation` is locked.
    last_committed: std::sync::Mutex<Option<u64>>,
}

impl TurnCtx {
    /// Store a turn and tell the client what memory looks like now.
    fn commit(&self, turn_id: u64, user_text: &str, spoken: &[String], interrupted: bool) {
        let stored = !user_text.trim().is_empty()
            && (interrupted || spoken.iter().any(|s| !s.trim().is_empty()));
        if !stored {
            return;
        }
        let snapshot = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            let dropped = conv.commit(user_text, spoken, interrupted);
            *self
                .last_committed
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(turn_id);
            debug!(
                turn_id,
                turns = conv.len(),
                history_tokens = conv.history_tokens(),
                dropped,
                interrupted,
                "live_talk: conversation updated"
            );
            conv.snapshot()
        };
        self.out.send(LiveTalkWsEvent::Memory(snapshot));
    }

    /// Cut the stored turn `turn_id` down to the `heard` sentences that were
    /// played. Does nothing unless that turn is the newest one in memory and
    /// has more sentences than that.
    fn amend(&self, turn_id: u64, heard: usize) -> bool {
        let snapshot = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            let is_last = *self
                .last_committed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                == Some(turn_id);
            if !is_last || !conv.amend_last_spoken(heard) {
                return false;
            }
            conv.snapshot()
        };
        debug!(turn_id, heard, "live_talk: memory cut to what was heard");
        self.out.send(LiveTalkWsEvent::Memory(snapshot));
        true
    }

    fn forget(&self) {
        let snapshot = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            conv.clear();
            *self
                .last_committed
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
            conv.snapshot()
        };
        self.out.send(LiveTalkWsEvent::Memory(snapshot));
    }
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// `None` while the stage the cell times has not happened.
fn read_metric(cell: &AtomicU64) -> Option<u64> {
    match cell.load(Ordering::Relaxed) {
        u64::MAX => None,
        ms => Some(ms),
    }
}

/// STT → streaming LLM → TTS per sentence. Runs on its own task; when
/// `cancel` fires it stops at the next await point. Errors end the turn but
/// never the session.
async fn run_turn(
    ctx: Arc<TurnCtx>,
    cfg: TurnConfig,
    turn_id: u64,
    utterance: Arc<vox::Utterance>,
    cancel: CancellationToken,
    shared: Arc<TurnShared>,
) {
    let out = &ctx.out;

    // Transcription is raced against the cancel token: a backend that takes
    // seconds must not hold a cancelled turn.
    let stt_started = Instant::now();
    let stt_result = tokio::select! {
        biased;
        _ = cancel.cancelled() => return,
        r = ctx.stt.transcribe(&utterance) => r,
    };
    let stt_ms = elapsed_ms(stt_started);
    let stt_result = match stt_result {
        Ok(r) => r,
        Err(e) => {
            out.send_turn(
                turn_id,
                LiveTalkWsEvent::Error {
                    message: format!("STT: {e}"),
                    fatal: false,
                    turn_id: Some(turn_id),
                },
            );
            return;
        }
    };
    if cancel.is_cancelled() {
        return;
    }

    let user_text = stt_result.text.trim().to_string();
    if user_text.is_empty() {
        out.send_turn(
            turn_id,
            LiveTalkWsEvent::TurnDone {
                turn_id,
                sentences: 0,
            },
        );
        return;
    }

    out.send_turn(
        turn_id,
        LiveTalkWsEvent::Transcript {
            turn_id,
            text: user_text.clone(),
            duration_ms: stt_result.duration_ms,
            processing_time_ms: stt_result.processing_time_ms,
            speaker_id: None,
            speaker_name: None,
            speaker_color: None,
        },
    );
    shared.phase.store(PHASE_THINKING, Ordering::SeqCst);
    if !out.send_turn(turn_id, LiveTalkWsEvent::Thinking { turn_id }) {
        return;
    }

    let messages = ctx
        .conversation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .messages_for(Some(&cfg.system_prompt), &user_text);

    let llm_started = Instant::now();
    let first_token_ms = Arc::new(AtomicU64::new(u64::MAX));
    let first_audio_ms = Arc::new(AtomicU64::new(u64::MAX));
    let sentence_count = Arc::new(AtomicUsize::new(0));
    // Sentences whose audio was queued for the client. Only these go into
    // memory, so the model never assumes unheard content.
    let spoken: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    let on_token = {
        let out = ctx.out.clone();
        let first_token_ms = Arc::clone(&first_token_ms);
        move |token: &str| {
            let _ = first_token_ms.compare_exchange(
                u64::MAX,
                elapsed_ms(llm_started),
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            out.send_turn(
                turn_id,
                LiveTalkWsEvent::Token {
                    turn_id,
                    text: token.to_string(),
                },
            );
        }
    };

    let on_sentence = {
        let ctx = Arc::clone(&ctx);
        let shared = Arc::clone(&shared);
        let voice = cfg.voice.clone();
        let spoken = Arc::clone(&spoken);
        let sentence_count = Arc::clone(&sentence_count);
        let first_audio_ms = Arc::clone(&first_audio_ms);
        // `stream_chat` races this future against the cancel token, so a
        // slow synthesis does not delay a cancel either.
        move |sentence: &str| {
            let ctx = Arc::clone(&ctx);
            let shared = Arc::clone(&shared);
            let voice = voice.clone();
            let spoken = Arc::clone(&spoken);
            let sentence_count = Arc::clone(&sentence_count);
            let first_audio_ms = Arc::clone(&first_audio_ms);
            let text = sentence.to_string();
            async move {
                if text.trim().is_empty() {
                    return Ok(());
                }
                let tts_started = Instant::now();
                let request = vox::TtsRequest {
                    text: text.clone(),
                    voice,
                    seed: None,
                };
                let output = ctx.tts.synthesize(&request).await?;
                let wav =
                    super::ws::encode_wav_chunk(&output.audio.samples, output.audio.sample_rate)
                        .map_err(vox::VoxError::Tts)?;

                let index = sentence_count.fetch_add(1, Ordering::SeqCst);
                if index == 0 {
                    first_audio_ms.store(elapsed_ms(tts_started), Ordering::Relaxed);
                }
                {
                    let now = Instant::now();
                    let length = Duration::from_millis(output.audio.duration_ms());
                    let mut end = shared
                        .playback_end
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    *end = Some((*end).map_or(now, |end| end.max(now)) + length);
                }
                // From here on the microphone hears the reply: the
                // connection loop gates it.
                shared.phase.store(PHASE_SPEAKING, Ordering::SeqCst);

                let sent = ctx.out.send_turn(
                    turn_id,
                    LiveTalkWsEvent::Sentence {
                        turn_id,
                        index,
                        text: text.clone(),
                    },
                ) && ctx.out.send_audio(
                    turn_id,
                    LiveTalkWsEvent::AudioChunk {
                        turn_id,
                        sentence_index: index,
                        sample_rate: output.audio.sample_rate,
                    },
                    wav,
                );
                if !sent {
                    return Err(vox::VoxError::Pipeline("client disconnected".into()));
                }
                spoken.lock().unwrap_or_else(|e| e.into_inner()).push(text);
                Ok(())
            }
        }
    };

    let stop = vox::streaming_chat::stream_chat(
        &ctx.http_client,
        &ctx.ollama_host,
        &cfg.model,
        &messages,
        &ctx.stream_options,
        cancel.clone(),
        on_token,
        on_sentence,
    )
    .await;

    // Record the turn. Finished: the whole reply. Cancelled, or failed after
    // part of it went out: what was sent, marked interrupted. Failed before
    // anything went out: not recorded.
    let mut spoken = std::mem::take(&mut *spoken.lock().unwrap_or_else(|e| e.into_inner()));
    let mut interrupted = match &stop {
        Ok(StopReason::Finished) => Some(false),
        Ok(StopReason::Cancelled) => Some(true),
        Err(_) => (!spoken.is_empty()).then_some(true),
    };
    // The client may already have said how much of this turn it played.
    let heard = shared.heard_limit.load(Ordering::SeqCst);
    if heard < spoken.len() {
        spoken.truncate(heard);
        interrupted = interrupted.map(|_| true);
    }
    if let Some(interrupted) = interrupted {
        ctx.commit(turn_id, &user_text, &spoken, interrupted);
    }

    let metrics = LiveTalkWsEvent::TurnMetrics {
        turn_id,
        stt_ms,
        llm_first_token_ms: read_metric(&first_token_ms),
        tts_first_audio_ms: read_metric(&first_audio_ms),
    };
    match stop {
        Ok(StopReason::Finished) => {
            out.send_turn(turn_id, metrics);
            out.send_turn(
                turn_id,
                LiveTalkWsEvent::TurnDone {
                    turn_id,
                    sentences: sentence_count.load(Ordering::SeqCst),
                },
            );
        }
        Ok(StopReason::Cancelled) => {
            // The connection loop already told the client.
            debug!(turn_id, "live_talk: turn cancelled");
        }
        Err(e) => {
            // Non-fatal: the turn is over, the session goes on and the
            // microphone is accepted again.
            out.send_turn(turn_id, metrics);
            out.send_turn(
                turn_id,
                LiveTalkWsEvent::Error {
                    message: e.to_string(),
                    fatal: false,
                    turn_id: Some(turn_id),
                },
            );
        }
    }
}

// ─── Connection ──────────────────────────────────────────────

/// A running turn. Held in an `Option` so a cancel can `take()` it.
struct ActiveTurn {
    id: u64,
    handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
    shared: Arc<TurnShared>,
    utterance: Arc<vox::Utterance>,
}

/// The turn task is done but the browser is still playing its audio.
struct PlaybackGate {
    turn_id: u64,
    until: Instant,
}

struct Session {
    ctx: Arc<TurnCtx>,
    out: Outbox,
    vad: Box<dyn vox::traits::VadBackend>,
    frame_size: usize,
    playback_grace: Duration,

    cfg: TurnConfig,
    mode: LiveTalkMode,
    barge_in_enabled: bool,

    /// Samples waiting for a full VAD frame.
    residual: Vec<f32>,
    /// The last [`RING_SAMPLES`] of microphone audio that was gated.
    ring: VecDeque<f32>,
    /// Whether the previous audio was gated (edge detection for the gate).
    was_gated: bool,
    /// Audio of a turn that was interrupted by voice before it was
    /// transcribed; the next utterance is appended to it.
    carry: Vec<f32>,

    active: Option<ActiveTurn>,
    playback: Option<PlaybackGate>,
    /// The newest turn the client reported `playback_done` for.
    played_upto: u64,
    /// Id of the newest turn of this connection; 0 before the first.
    last_turn_id: u64,

    ptt_active: bool,
    ptt_buffer: Vec<f32>,
}

impl Session {
    /// Notice a finished turn task and an expired playback gate.
    fn reap(&mut self) {
        if self.active.as_ref().is_some_and(|t| t.handle.is_finished()) {
            if let Some(turn) = self.active.take() {
                let playback_end = *turn
                    .shared
                    .playback_end
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(playback_end) = playback_end {
                    // The browser is still playing. Keep the microphone
                    // gated until it says `playback_done`, or until the
                    // audio must be over.
                    if turn.id > self.played_upto {
                        self.playback = Some(PlaybackGate {
                            turn_id: turn.id,
                            until: playback_end + self.playback_grace,
                        });
                    }
                }
            }
        }
        if self
            .playback
            .as_ref()
            .is_some_and(|gate| Instant::now() >= gate.until)
        {
            self.playback = None;
        }
    }

    /// Whether microphone audio must be kept away from the VAD right now.
    fn gated(&self) -> bool {
        if self.playback.is_some() {
            return true;
        }
        self.active.as_ref().is_some_and(|turn| {
            !self.barge_in_enabled || turn.shared.phase.load(Ordering::SeqCst) == PHASE_SPEAKING
        })
    }

    /// Hold gated audio in the ring buffer.
    fn gate_audio(&mut self, samples: &[f32]) {
        if !self.was_gated {
            self.was_gated = true;
            // Nothing is fed while gated: start from a clean VAD when the
            // gate opens again.
            self.vad.reset();
            self.residual.clear();
            self.ring.clear();
        }
        self.ring.extend(samples.iter().copied());
        let excess = self.ring.len().saturating_sub(RING_SAMPLES);
        self.ring.drain(..excess);
    }

    /// The gate was opened by a cancel: what the ring holds is most likely
    /// the start of the user's speech, so give it to the VAD.
    async fn replay_ring(&mut self) {
        if !self.was_gated {
            return;
        }
        self.was_gated = false;
        let held: Vec<f32> = self.ring.drain(..).collect();
        if self.mode == LiveTalkMode::Vad && !held.is_empty() {
            debug!(samples = held.len(), "live_talk: replaying gated audio");
            self.feed_vad(&held).await;
        }
    }

    async fn on_audio(&mut self, samples: &[f32]) {
        self.reap();
        if self.mode == LiveTalkMode::PushToTalk {
            if self.ptt_active {
                let room = MAX_PTT_SAMPLES.saturating_sub(self.ptt_buffer.len());
                self.ptt_buffer
                    .extend_from_slice(&samples[..samples.len().min(room)]);
            }
            return;
        }
        if self.gated() {
            self.gate_audio(samples);
            return;
        }
        if self.was_gated {
            // The gate opened by itself (playback ended): what the ring
            // holds is the loudspeaker, not the user.
            self.was_gated = false;
            self.ring.clear();
        }
        self.feed_vad(samples).await;
    }

    /// Run audio through the VAD and act on what it finds.
    async fn feed_vad(&mut self, samples: &[f32]) {
        self.residual.extend_from_slice(samples);
        let mut offset = 0;
        while self.residual.len() - offset >= self.frame_size {
            if self.gated() {
                // A turn started, or reached its first audio, inside this
                // batch. The rest is handled below.
                break;
            }
            let frame = vox::AudioChunk {
                samples: self.residual[offset..offset + self.frame_size].to_vec(),
                sample_rate: MIC_SAMPLE_RATE as u32,
                channels: 1,
            };
            offset += self.frame_size;
            let events = match self.vad.process_frame(&frame).await {
                Ok(events) => events,
                Err(e) => {
                    self.out.send(LiveTalkWsEvent::Error {
                        message: format!("VAD error: {e}"),
                        fatal: false,
                        turn_id: None,
                    });
                    self.vad.reset();
                    self.residual.clear();
                    return;
                }
            };
            for event in events {
                match event {
                    vox::VadEvent::SpeechStart => {
                        // Speech while a turn is still transcribing or
                        // thinking: the user goes on, or changed their mind.
                        self.interrupt(CancelReason::UserBargeIn, None).await;
                        self.out.send(LiveTalkWsEvent::SpeechStart);
                    }
                    vox::VadEvent::SpeechEnd(utterance) => {
                        self.interrupt(CancelReason::UserBargeIn, None).await;
                        self.out.send(LiveTalkWsEvent::SpeechEnd);
                        self.dispatch(utterance);
                    }
                    vox::VadEvent::Silence => {}
                }
            }
        }
        self.residual.drain(..offset);
        if self.gated() {
            let rest = std::mem::take(&mut self.residual);
            self.gate_audio(&rest);
        }
    }

    /// Cancel the running turn, if any, and tell the client. Returns its id.
    async fn interrupt(&mut self, reason: CancelReason, heard: Option<usize>) -> Option<u64> {
        let turn = self.active.take()?;
        if let Some(heard) = heard {
            turn.shared.heard_limit.store(heard, Ordering::SeqCst);
        }
        turn.cancel.cancel();
        self.out.kill_turn(turn.id);

        if reason == CancelReason::UserBargeIn
            && turn.shared.phase.load(Ordering::SeqCst) == PHASE_STT
        {
            // The user kept talking before the first part was transcribed.
            // Keep that audio and transcribe it together with what follows,
            // so nothing they said is dropped.
            let samples = &turn.utterance.audio.samples;
            let start = samples.len().saturating_sub(MAX_CARRY_SAMPLES);
            self.carry = samples[start..].to_vec();
        }

        // The task stops at its next await point and records the turn.
        let _ = tokio::time::timeout(CANCEL_JOIN, turn.handle).await;
        if let Some(heard) = heard {
            // Covers a task that had recorded the turn before it saw the
            // limit. A no-op otherwise.
            self.ctx.amend(turn.id, heard);
        }
        self.out.send(LiveTalkWsEvent::Cancelled {
            turn_id: turn.id,
            reason,
        });
        Some(turn.id)
    }

    /// Start a turn for `utterance` on its own task.
    fn dispatch(&mut self, mut utterance: vox::Utterance) {
        if !self.carry.is_empty() {
            let mut samples = std::mem::take(&mut self.carry);
            samples.extend_from_slice(&utterance.audio.samples);
            utterance.duration_ms = (samples.len() as u64 * 1000) / MIC_SAMPLE_RATE;
            utterance.audio.samples = samples;
        }
        self.playback = None;
        self.last_turn_id += 1;
        let id = self.last_turn_id;

        let cancel = CancellationToken::new();
        let shared = Arc::new(TurnShared {
            phase: AtomicU8::new(PHASE_STT),
            playback_end: std::sync::Mutex::new(None),
            heard_limit: AtomicUsize::new(usize::MAX),
        });
        let utterance = Arc::new(utterance);
        let handle = tokio::spawn(run_turn(
            Arc::clone(&self.ctx),
            self.cfg.clone(),
            id,
            Arc::clone(&utterance),
            cancel.clone(),
            Arc::clone(&shared),
        ));
        self.active = Some(ActiveTurn {
            id,
            handle,
            cancel,
            shared,
            utterance,
        });
    }

    async fn on_cancel(&mut self, turn_id: Option<u64>, heard: Option<usize>) {
        self.reap();
        let active_id = self.active.as_ref().map(|turn| turn.id);
        match active_id {
            Some(active) if turn_id.is_none_or(|id| id == active) => {
                self.interrupt(CancelReason::ClientRequest, heard).await;
                self.carry.clear();
                self.replay_ring().await;
            }
            Some(active) => {
                // A cancel for an earlier turn that arrived late. That turn
                // is over; only its memory entry can still be corrected.
                debug!(?turn_id, active, "live_talk: cancel for an earlier turn");
                if let (Some(id), Some(heard)) = (turn_id, heard) {
                    self.ctx.amend(id, heard);
                }
            }
            None => {
                // No turn is running. This is the normal case right after
                // `turn_done`, while the browser is still playing. The VAD
                // and the samples waiting for it are left alone: the user
                // may be in the middle of a word.
                let target = turn_id.unwrap_or(self.last_turn_id);
                if target != 0 && target == self.last_turn_id {
                    self.out.kill_turn(target);
                    if let Some(heard) = heard {
                        self.ctx.amend(target, heard);
                    }
                    if self.playback.take().is_some() {
                        self.replay_ring().await;
                    }
                }
                self.out.send(LiveTalkWsEvent::Cancelled {
                    turn_id: target,
                    reason: CancelReason::ClientRequest,
                });
            }
        }
    }

    fn on_playback_done(&mut self, turn_id: u64, heard: Option<usize>) {
        self.reap();
        if let Some(heard) = heard {
            self.ctx.amend(turn_id, heard);
        }
        self.played_upto = self.played_upto.max(turn_id);
        if self
            .playback
            .as_ref()
            .is_some_and(|gate| gate.turn_id <= turn_id)
        {
            self.playback = None;
        }
    }

    fn on_config(&mut self, new: LiveTalkConfig) {
        if let Some(model) = new.model {
            self.cfg.model = model;
        }
        if let Some(voice) = new.voice {
            self.cfg.voice = Some(voice);
        }
        if let Some(prompt) = new.system_prompt_override {
            self.cfg.system_prompt = prompt;
        }
        self.barge_in_enabled = new.barge_in_enabled;
        let mode: LiveTalkMode = new.mode.into();
        if mode != self.mode {
            self.mode = mode;
            self.ptt_active = false;
            self.ptt_buffer.clear();
            self.vad.reset();
            self.residual.clear();
            self.ring.clear();
            self.was_gated = false;
        }
        info!(
            model = %self.cfg.model,
            voice = ?self.cfg.voice,
            mode = ?self.mode,
            barge_in = self.barge_in_enabled,
            "live_talk: config updated"
        );
    }

    async fn on_ptt_start(&mut self) {
        self.reap();
        self.ptt_active = true;
        self.ptt_buffer.clear();
        self.interrupt(CancelReason::UserBargeIn, None).await;
        if self.playback.take().is_some() {
            self.out.kill_turn(self.last_turn_id);
        }
        self.out.send(LiveTalkWsEvent::SpeechStart);
    }

    async fn on_ptt_end(&mut self) {
        if !self.ptt_active {
            return;
        }
        self.ptt_active = false;
        self.out.send(LiveTalkWsEvent::SpeechEnd);
        if self.mode != LiveTalkMode::PushToTalk || self.ptt_buffer.is_empty() {
            return;
        }
        let samples = std::mem::take(&mut self.ptt_buffer);
        let duration_ms = (samples.len() as u64 * 1000) / MIC_SAMPLE_RATE;
        self.interrupt(CancelReason::UserBargeIn, None).await;
        self.dispatch(vox::Utterance {
            audio: vox::AudioChunk {
                samples,
                sample_rate: MIC_SAMPLE_RATE as u32,
                channels: 1,
            },
            duration_ms,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        });
    }

    async fn on_text(&mut self, text: &str) {
        match serde_json::from_str::<LiveTalkClientMsg>(text) {
            Ok(LiveTalkClientMsg::Config(new)) => self.on_config(new),
            Ok(LiveTalkClientMsg::Cancel {
                turn_id,
                heard_sentences,
            }) => self.on_cancel(turn_id, heard_sentences).await,
            Ok(LiveTalkClientMsg::PlaybackDone {
                turn_id,
                heard_sentences,
            }) => self.on_playback_done(turn_id, heard_sentences),
            Ok(LiveTalkClientMsg::ForgetMemory) => {
                self.carry.clear();
                self.ctx.forget();
            }
            Ok(LiveTalkClientMsg::PttStart) => self.on_ptt_start().await,
            Ok(LiveTalkClientMsg::PttEnd) => self.on_ptt_end().await,
            Err(e) => debug!("live_talk: ignoring unknown text msg: {e}"),
        }
    }
}

/// The first text message may be `{"type":"config",...}` or a bare config
/// object.
fn parse_first_config(text: &str) -> Option<LiveTalkConfig> {
    if let Ok(LiveTalkClientMsg::Config(config)) = serde_json::from_str(text) {
        return Some(config);
    }
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if value.get("type").is_some() {
        // Some other command; not a config.
        return None;
    }
    serde_json::from_value(value).ok()
}

/// Run one Live Talk session over any message transport until the client
/// leaves.
pub async fn run_live_talk<S, R, E>(sink: S, mut stream: R, deps: LiveTalkDeps)
where
    S: Sink<Message> + Send + Unpin + 'static,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let dead_upto = Arc::new(AtomicU64::new(0));
    let out = Outbox {
        tx,
        dead_upto: Arc::clone(&dead_upto),
    };
    let mut writer = tokio::spawn(write_loop(sink, rx, dead_upto));

    let mut config = LiveTalkConfig::default();
    let mut first_text: Option<String> = None;
    let mut pending_audio: Vec<f32> = Vec::new();
    match tokio::time::timeout(CONFIG_WAIT, stream.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => match parse_first_config(&text) {
            Some(c) => config = c,
            // Not a config: handle it as a command once the session is up.
            None => first_text = Some(text.to_string()),
        },
        Ok(Some(Ok(Message::Binary(data)))) => {
            pending_audio = super::ws::bytes_to_f32_samples(&data);
        }
        Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) | Ok(None) => {
            drop(out);
            let _ = writer.await;
            return;
        }
        _ => {}
    }

    let session_handle = deps.sessions.attach(config.session_id.as_deref());
    let mode: LiveTalkMode = config.mode.into();
    let cfg = TurnConfig {
        model: config.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        voice: config.voice,
        system_prompt: config
            .system_prompt_override
            .unwrap_or_else(|| vox::build_system_prompt(vox::VoicePromptMode::Voice)),
    };

    info!(
        model = %cfg.model,
        host = %deps.ollama_host,
        voice = ?cfg.voice,
        mode = ?mode,
        barge_in = config.barge_in_enabled,
        session = %session_handle.id,
        resumed = session_handle.resumed,
        "live_talk: client connected"
    );

    out.send(LiveTalkWsEvent::Ready {
        model: cfg.model.clone(),
        voice: cfg.voice.clone(),
        mode,
        protocol_version: LIVE_TALK_PROTOCOL_VERSION,
        session_id: session_handle.id.clone(),
    });
    let snapshot = session_handle
        .conversation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .snapshot();
    out.send(LiveTalkWsEvent::Memory(snapshot));

    let frame_size = deps.vad.frame_size().max(1);
    let mut session = Session {
        ctx: Arc::new(TurnCtx {
            stt: deps.stt,
            tts: deps.tts,
            http_client: deps.http_client,
            ollama_host: deps.ollama_host,
            stream_options: deps.stream_options,
            out: out.clone(),
            conversation: session_handle.conversation,
            last_committed: std::sync::Mutex::new(None),
        }),
        out,
        vad: deps.vad,
        frame_size,
        playback_grace: deps.playback_grace,
        cfg,
        mode,
        barge_in_enabled: config.barge_in_enabled,
        residual: Vec::with_capacity(frame_size * 2),
        ring: VecDeque::with_capacity(RING_SAMPLES),
        was_gated: false,
        carry: Vec::new(),
        active: None,
        playback: None,
        played_upto: 0,
        last_turn_id: 0,
        ptt_active: false,
        ptt_buffer: Vec::new(),
    };

    if let Some(text) = first_text {
        session.on_text(&text).await;
    }
    if !pending_audio.is_empty() {
        session.on_audio(&pending_audio).await;
    }

    while !session.out.is_closed() {
        let Some(Ok(msg)) = stream.next().await else {
            break;
        };
        match msg {
            Message::Binary(data) => {
                let samples = super::ws::bytes_to_f32_samples(&data);
                session.on_audio(&samples).await;
            }
            Message::Text(text) => session.on_text(&text).await,
            Message::Close(_) => break,
            _ => {}
        }
    }

    // The client is gone. Stop the turn; what it sent so far stays in memory
    // (marked interrupted) for a reconnect with the same session id.
    if let Some(turn) = session.active.take() {
        turn.cancel.cancel();
        let _ = tokio::time::timeout(CANCEL_JOIN, turn.handle).await;
    }
    deps.sessions.touch(&session_handle.id);
    // Dropping the session drops the last queue handles; the writer then
    // flushes what is left and closes the socket.
    drop(session);
    if tokio::time::timeout(Duration::from_secs(1), &mut writer)
        .await
        .is_err()
    {
        writer.abort();
    }

    info!(session = %session_handle.id, "live_talk: client disconnected");
}
