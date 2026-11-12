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
//! - **Speakers.** With `config.diarization` on, the speaker of an utterance
//!   is looked up while it is being transcribed ([`SpeakerService`]). The
//!   lookup is read-only: talking never enrolls or adapts a voiceprint, so
//!   the assistant's own voice or a television cannot end up in the speaker
//!   database. It gets a short grace period after the transcript is ready;
//!   if it is not done by then the turn goes on unlabelled. A known speaker
//!   is named in `transcript` and in the message the model gets
//!   (`[Name] text`, stored exactly like that). A voice is enrolled only by
//!   the explicit `enroll_speaker` command.
//! - **Long-term memory.** For a person who switched memory on, each stored
//!   turn is also written to the [`MemoryStore`], and the first message of
//!   theirs in a session carries a block of what is remembered about them.
//!   Unidentified speakers are never written.
//! - **Distill on evict.** When a finished turn pushes older turns out of
//!   the context window, the local model is asked, in the pause that
//!   follows, what in those turns is worth keeping about each person who
//!   has memory on; the answer is stored as facts and reported with
//!   `memory_distilled`. It runs right after a trim because the model's
//!   cached prompt prefix is void at that moment anyway. It never competes
//!   with the conversation: the user starting to speak, a `cancel`,
//!   `ptt_start`, `forget_memory` or the socket closing aborts the request,
//!   and the turns go back in the queue for the next trim.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use vox::conversation::{
    ChatMessage, EvictedTurn, RecallNeed, SessionStore, SharedConversation, SpeakerLabel,
    UserMessage,
};
use vox::diarization::IdentifyMode;
use vox::memory::{ExchangeRef, HeardExchange, MemoryStore};
use vox::streaming_chat::{StopReason, StreamOptions};

use super::ServerState;
use super::error::ServerError;
use super::models::{
    CancelReason, LIVE_TALK_PROTOCOL_VERSION, LiveTalkClientMsg, LiveTalkConfig, LiveTalkMode,
    LiveTalkWsEvent,
};
use super::speakers::{self, SpeakerCtx};

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

/// Default for [`LiveTalkDeps::identify_grace`].
pub const DEFAULT_IDENTIFY_GRACE: Duration = Duration::from_millis(150);

/// Longest wait for the lookup of what is remembered about a speaker. It is
/// a local database read; when it takes longer the turn goes on without.
const RECALL_TIMEOUT: Duration = Duration::from_millis(250);

/// Stored facts shown to the model in a distillation request, so it does
/// not repeat them.
const DISTILL_KNOWN_FACTS: u32 = 40;

/// A distillation reply is a handful of short lines. One that grows past
/// this is cut off and read as far as it got.
const MAX_DISTILL_REPLY_BYTES: usize = 4_000;

/// Environment variable that switches distillation of trimmed turns off
/// (`0`, `false`, `off` or `no`).
pub const DISTILL_ENV: &str = "VOX_MEMORY_DISTILL";

/// Whether trimmed turns are distilled, from [`DISTILL_ENV`]. On by default.
pub fn distill_enabled_from_env() -> bool {
    let value = std::env::var(DISTILL_ENV).unwrap_or_default();
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no"
    )
}

/// Model used when the client names none.
const DEFAULT_MODEL: &str = "llama3.2";

const PHASE_STT: u8 = 0;
const PHASE_THINKING: u8 = 1;
const PHASE_SPEAKING: u8 = 2;

/// A speaker as Live Talk reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerTag {
    pub id: String,
    pub name: String,
    /// Hex color for the UI.
    pub color: String,
}

/// Outcome of an explicit enrollment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enrollment {
    pub speaker: SpeakerTag,
    /// `false` when the voice was already known: nothing new was stored and
    /// `speaker` is who it belongs to.
    pub created: bool,
}

/// What a Live Talk session needs from the speaker subsystem.
#[async_trait]
pub trait SpeakerService: Send + Sync {
    /// Who spoke `audio`, if it is a known voice. Must not change anything:
    /// no enrollment, no adaptation.
    async fn identify(&self, audio: &vox::AudioChunk) -> Option<SpeakerTag>;

    /// Store the voice in `audio` as a speaker, on the user's explicit
    /// request. A new speaker gets `name` when one is given. The error is a
    /// message for the user.
    async fn enroll(
        &self,
        audio: &vox::AudioChunk,
        name: Option<&str>,
    ) -> Result<Enrollment, String>;
}

/// The server's speaker pipeline and database behind [`SpeakerService`].
struct ServerSpeakers(SpeakerCtx);

#[async_trait]
impl SpeakerService for ServerSpeakers {
    async fn identify(&self, audio: &vox::AudioChunk) -> Option<SpeakerTag> {
        let found = speakers::identify_speaker(&self.0, audio, IdentifyMode::IdentifyOnly).await?;
        Some(SpeakerTag {
            id: found.id,
            name: found.name,
            color: found.color,
        })
    }

    async fn enroll(
        &self,
        audio: &vox::AudioChunk,
        name: Option<&str>,
    ) -> Result<Enrollment, String> {
        let name = match name.map(str::trim).filter(|n| !n.is_empty()) {
            Some(name) => {
                Some(vox::diarization::validate_speaker_name(name).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        // The same gate as everywhere else: no voiceprint without consent.
        if !speakers::enrollment_consent(&self.0).await.granted {
            return Err(
                "Speaker enrollment consent is not recorded on the server. Grant it with \
                 PUT /v1/speakers/consent {\"granted\": true} and try again."
                    .to_string(),
            );
        }
        let found = speakers::identify_speaker(&self.0, audio, IdentifyMode::AdaptAndEnroll)
            .await
            .ok_or_else(|| {
                format!(
                    "The voice could not be stored: the last utterance is shorter than {} ms \
                     or could not be analysed.",
                    vox::diarization::MIN_SPEAKER_AUDIO_MS
                )
            })?;
        let created = found.new_speaker.is_some();
        let mut speaker = SpeakerTag {
            id: found.id,
            name: found.name,
            color: found.color,
        };
        // Only a speaker made by this call is named here. A voice that was
        // already known keeps its name: a wrong match must not rename
        // someone else.
        if let (true, Some(name)) = (created, name) {
            match speakers::rename(&self.0, &speaker.id, &name).await {
                Ok(detail) => speaker.name = detail.name,
                Err(e) => {
                    tracing::warn!(error = %e, speaker_id = %speaker.id, "enrolled speaker could not be named");
                }
            }
        }
        Ok(Enrollment { speaker, created })
    }
}

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
    /// Speaker identification and enrollment; `None` when the server has no
    /// speaker model.
    pub speakers: Option<Arc<dyn SpeakerService>>,
    /// Long-term memory of known speakers; `None` when there is no speaker
    /// database.
    pub memory: Option<MemoryStore>,
    /// How long a turn waits for the speaker lookup after the transcript is
    /// ready. The lookup runs alongside transcription, so this is extra
    /// latency only when identification is the slower of the two.
    pub identify_grace: Duration,
    /// Distill turns that were trimmed out of the context into long-term
    /// facts while the session is idle. Needs `memory`.
    pub distill: bool,
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
    let speaker_ctx = SpeakerCtx::from_state(&state);
    let speakers: Option<Arc<dyn SpeakerService>> =
        if speaker_ctx.pipeline.is_some() && speaker_ctx.db.is_some() {
            Some(Arc::new(ServerSpeakers(speaker_ctx)))
        } else {
            None
        };
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
            speakers,
            memory: state.speaker_db.as_deref().map(MemoryStore::new),
            identify_grace: DEFAULT_IDENTIFY_GRACE,
            distill: distill_enabled_from_env(),
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
    /// The prompt as sent: with the speaker paragraph when `diarization`.
    system_prompt: String,
    /// Identify the speaker of each turn. Only `true` when the server can.
    diarization: bool,
}

/// The system prompt for a session: `base`, plus the explanation of speaker
/// tags and recall blocks when speakers are identified.
fn effective_prompt(base: &str, diarization: bool) -> String {
    if diarization {
        vox::prompts::with_speaker_awareness(base)
    } else {
        base.to_string()
    }
}

/// A write to long-term memory, done in order by [`memory_worker`].
enum MemoryOp {
    /// A turn was stored in the conversation.
    Record {
        turn_id: u64,
        speaker_id: String,
        user_text: String,
        sentences: Vec<String>,
        interrupted: bool,
    },
    /// Only the first `heard` sentences of that turn were played.
    Amend { turn_id: u64, heard: usize },
}

/// Apply memory writes one after the other, off the turn task. Ends when the
/// session and its turns are gone.
async fn memory_worker(store: MemoryStore, mut rx: mpsc::UnboundedReceiver<MemoryOp>) {
    // The newest stored turn; the only one that can still be corrected.
    let mut last: Option<(u64, ExchangeRef, Vec<String>)> = None;
    while let Some(op) = rx.recv().await {
        match op {
            MemoryOp::Record {
                turn_id,
                speaker_id,
                user_text,
                sentences,
                interrupted,
            } => {
                let reply = sentences.join(" ");
                last = match store
                    .record_exchange(&speaker_id, &user_text, Some(&reply), interrupted)
                    .await
                {
                    // `None`: memory is off for this person; nothing written.
                    Ok(stored) => stored.map(|exchange| (turn_id, exchange, sentences)),
                    Err(e) => {
                        tracing::warn!(error = %e, "live_talk: turn not written to long-term memory");
                        None
                    }
                };
            }
            MemoryOp::Amend { turn_id, heard } => {
                let Some((id, exchange, sentences)) = last.as_mut() else {
                    continue;
                };
                if *id != turn_id || heard >= sentences.len() {
                    continue;
                }
                sentences.truncate(heard);
                match store
                    .amend_exchange(*exchange, Some(&sentences.join(" ")))
                    .await
                {
                    Ok(amended) => *exchange = amended,
                    Err(e) => {
                        tracing::warn!(error = %e, "live_talk: long-term memory not corrected");
                    }
                }
            }
        }
    }
}

// ─── Distill on evict ────────────────────────────────────────

/// A running distillation of trimmed turns.
struct DistillJob {
    cancel: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

/// Trimmed turns of one person that go into one request.
struct DistillBatch {
    speaker: SpeakerLabel,
    turns: Vec<EvictedTurn>,
}

/// What became of a batch.
enum Distilled {
    /// The facts that were new and are stored now (possibly none).
    Stored(Vec<String>),
    /// Nothing to do for this person: memory is off, or the session was
    /// told to forget meanwhile.
    Skipped,
    /// Aborted before an answer was there; the turns are still to do.
    Cancelled,
}

/// Longest excerpt in one distillation request, in characters: about a
/// quarter of the context window at the conservative three characters per
/// token.
fn distill_batch_chars() -> usize {
    vox::conversation::ollama_num_ctx() * 3 / 4
}

/// Group trimmed turns by speaker, in order of first appearance, and cut
/// each group into batches of at most `max_chars` of excerpt. A turn is
/// never split; one that is larger than `max_chars` gets a batch of its own.
fn distill_batches(turns: Vec<EvictedTurn>, max_chars: usize) -> VecDeque<DistillBatch> {
    let mut groups: Vec<DistillBatch> = Vec::new();
    for turn in turns {
        match groups.iter_mut().find(|g| g.speaker.id == turn.speaker.id) {
            Some(group) => {
                // The newest name the person was known by.
                group.speaker.name.clone_from(&turn.speaker.name);
                group.turns.push(turn);
            }
            None => groups.push(DistillBatch {
                speaker: turn.speaker.clone(),
                turns: vec![turn],
            }),
        }
    }
    let line = |text: &str| {
        text.chars()
            .count()
            .min(vox::memory::MAX_DISTILL_LINE_CHARS)
            + 16
    };
    let mut batches = VecDeque::new();
    for group in groups {
        let mut batch = Vec::new();
        let mut size = 0;
        for turn in group.turns {
            let cost = line(&turn.user) + line(&turn.assistant);
            if !batch.is_empty() && size + cost > max_chars {
                batches.push_back(DistillBatch {
                    speaker: group.speaker.clone(),
                    turns: std::mem::take(&mut batch),
                });
                size = 0;
            }
            size += cost;
            batch.push(turn);
        }
        if !batch.is_empty() {
            batches.push_back(DistillBatch {
                speaker: group.speaker,
                turns: batch,
            });
        }
    }
    batches
}

/// Ask the model what is worth keeping from one batch and store it.
async fn distill_batch(
    ctx: &TurnCtx,
    store: &MemoryStore,
    model: &str,
    batch: &DistillBatch,
    epoch: u64,
    cancel: &CancellationToken,
) -> Result<Distilled, vox::VoxError> {
    if cancel.is_cancelled() {
        return Ok(Distilled::Cancelled);
    }
    let speaker_id = &batch.speaker.id;
    // Nothing about a person who has memory off is sent anywhere, not even
    // to the local model.
    if !store.is_enabled(speaker_id).await? {
        return Ok(Distilled::Skipped);
    }
    let known = store.facts(speaker_id, DISTILL_KNOWN_FACTS).await?;
    let exchanges: Vec<HeardExchange> = batch
        .turns
        .iter()
        .map(|turn| HeardExchange {
            user: turn.user.clone(),
            assistant: turn.assistant.clone(),
            interrupted: turn.interrupted,
        })
        .collect();
    let (system, user) = vox::memory::distill_messages(&batch.speaker.name, &known, &exchanges);
    let messages = [ChatMessage::system(system), ChatMessage::user(user)];

    // A child token, so a reply that runs away can be cut off without
    // that looking like the session cancelling the work.
    let stop = cancel.child_token();
    let mut reply = String::new();
    let stopped = vox::streaming_chat::stream_chat(
        &ctx.http_client,
        &ctx.ollama_host,
        model,
        &messages,
        &ctx.stream_options,
        stop.clone(),
        |token: &str| {
            reply.push_str(token);
            if reply.len() > MAX_DISTILL_REPLY_BYTES {
                stop.cancel();
            }
        },
        |_: &str| async { Ok(()) },
    )
    .await?;
    if stopped == StopReason::Cancelled && cancel.is_cancelled() {
        return Ok(Distilled::Cancelled);
    }
    if ctx.forget_epoch.load(Ordering::SeqCst) != epoch {
        return Ok(Distilled::Skipped);
    }
    let facts = vox::memory::parse_distilled_facts(&reply);
    // The store checks the person's setting once more when it writes.
    store
        .add_facts(speaker_id, &facts)
        .await
        .map(Distilled::Stored)
}

/// Distill `turns`, one request per batch. When `cancel` fires, the turns
/// that are not done go back to the conversation's queue, unless the session
/// was told to forget since `epoch`.
async fn run_distill(
    ctx: Arc<TurnCtx>,
    store: MemoryStore,
    model: String,
    turns: Vec<EvictedTurn>,
    epoch: u64,
    cancel: CancellationToken,
) {
    let mut batches = distill_batches(turns, distill_batch_chars());
    while let Some(batch) = batches.pop_front() {
        match distill_batch(&ctx, &store, &model, &batch, epoch, &cancel).await {
            Ok(Distilled::Stored(facts)) => {
                debug!(
                    speaker_id = %batch.speaker.id,
                    turns = batch.turns.len(),
                    facts = facts.len(),
                    "live_talk: trimmed turns distilled"
                );
                if !facts.is_empty() {
                    ctx.out.send(LiveTalkWsEvent::MemoryDistilled {
                        speaker_id: batch.speaker.id,
                        speaker_name: batch.speaker.name,
                        facts,
                    });
                }
            }
            Ok(Distilled::Skipped) => {}
            Ok(Distilled::Cancelled) => {
                let mut rest = batch.turns;
                rest.extend(batches.into_iter().flat_map(|b| b.turns));
                debug!(
                    turns = rest.len(),
                    "live_talk: distillation interrupted, turns kept for later"
                );
                let mut conv = ctx.conversation.lock().unwrap_or_else(|e| e.into_inner());
                if ctx.forget_epoch.load(Ordering::SeqCst) == epoch {
                    conv.restore_evicted(rest);
                }
                return;
            }
            Err(e) => {
                // The turns of this batch are given up; what was said is
                // still in the person's stored exchanges.
                tracing::warn!(error = %e, "live_talk: trimmed turns could not be distilled");
            }
        }
    }
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
    /// Counts `forget_memory` commands. A turn that started before one does
    /// not get recorded after it. Changed only while `conversation` is
    /// locked.
    forget_epoch: AtomicU64,
    speakers: Option<Arc<dyn SpeakerService>>,
    identify_grace: Duration,
    memory: Option<MemoryStore>,
    /// Queue of [`memory_worker`]; `None` without a memory store.
    memory_tx: Option<mpsc::UnboundedSender<MemoryOp>>,
    /// The distillation of trimmed turns that is running, if any. `None`
    /// when the session does not distill (no memory store, or switched off).
    distill: Option<std::sync::Mutex<Option<DistillJob>>>,
}

impl TurnCtx {
    /// Start distilling the turns that were trimmed out of the context,
    /// unless a distillation is already running or there are none.
    fn start_distill(self: &Arc<Self>, model: &str) {
        let (Some(slot), Some(store)) = (&self.distill, &self.memory) else {
            return;
        };
        let mut job = slot.lock().unwrap_or_else(|e| e.into_inner());
        if job.as_ref().is_some_and(|job| !job.handle.is_finished()) {
            return;
        }
        let (turns, epoch) = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            (
                conv.take_evicted(),
                self.forget_epoch.load(Ordering::SeqCst),
            )
        };
        if turns.is_empty() {
            return;
        }
        debug!(turns = turns.len(), "live_talk: distilling trimmed turns");
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_distill(
            Arc::clone(self),
            store.clone(),
            model.to_string(),
            turns,
            epoch,
            cancel.clone(),
        ));
        *job = Some(DistillJob { cancel, handle });
    }

    /// Abort a running distillation: the model is needed for the
    /// conversation. Its unfinished turns go back in the queue.
    fn cancel_distill(&self) {
        if let Some(slot) = &self.distill
            && let Some(job) = slot.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
        {
            job.cancel.cancel();
        }
    }

    /// Abort a running distillation and hand out its task, to wait for it.
    fn stop_distill(&self) -> Option<tokio::task::JoinHandle<()>> {
        let job = self
            .distill
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        job.cancel.cancel();
        Some(job.handle)
    }

    /// Store a turn and tell the client what memory looks like now. Returns
    /// whether older turns were trimmed out of the context to make room.
    ///
    /// `epoch` is the value of `forget_epoch` when the turn started: if the
    /// user said "forget" in between, the turn is not stored.
    fn commit(
        &self,
        turn_id: u64,
        epoch: u64,
        user: &UserMessage,
        spoken: &[String],
        interrupted: bool,
    ) -> bool {
        let stored =
            !user.text().is_empty() && (interrupted || spoken.iter().any(|s| !s.trim().is_empty()));
        if !stored {
            return false;
        }
        let trimmed;
        let snapshot = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            if self.forget_epoch.load(Ordering::SeqCst) != epoch {
                debug!(
                    turn_id,
                    "live_talk: turn not stored, memory was cleared meanwhile"
                );
                return false;
            }
            let dropped = conv.commit_message(user, spoken, interrupted);
            trimmed = dropped > 0;
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

        // Long-term memory: only for an identified speaker. Whether that
        // person switched memory on is checked by the store when it writes.
        if let (Some(tx), Some(speaker)) = (&self.memory_tx, user.speaker()) {
            let _ = tx.send(MemoryOp::Record {
                turn_id,
                speaker_id: speaker.id.clone(),
                user_text: user.text().to_string(),
                sentences: spoken
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                interrupted,
            });
        }
        trimmed
    }

    /// The block of remembered context to put in front of this speaker's
    /// message, if one is due (see [`RecallNeed`]).
    async fn recall_block(&self, speaker: &SpeakerTag, said: &str) -> Option<String> {
        let memory = self.memory.as_ref()?;
        let need = || {
            self.conversation
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .recall_need(&speaker.id)
        };
        match need() {
            RecallNeed::Inject(block) => return Some(block),
            RecallNeed::Nothing => return None,
            RecallNeed::Lookup => {}
        }
        let lookup = memory.recall_block(&speaker.id, &speaker.name, Some(said));
        match tokio::time::timeout(RECALL_TIMEOUT, lookup).await {
            Ok(Ok(block)) => {
                // Frozen for the session, "nothing" included: later turns of
                // this person do not look again.
                self.conversation
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .freeze_recall(&speaker.id, block);
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "live_talk: recall failed, continuing without");
                return None;
            }
            Err(_) => {
                debug!("live_talk: recall not ready in time, continuing without");
                return None;
            }
        }
        match need() {
            RecallNeed::Inject(block) => Some(block),
            _ => None,
        }
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
        if let Some(tx) = &self.memory_tx {
            let _ = tx.send(MemoryOp::Amend { turn_id, heard });
        }
        true
    }

    fn forget(&self) {
        // Turns that were waiting to be distilled are forgotten too; a
        // request that is under way is dropped without storing anything.
        self.cancel_distill();
        let snapshot = {
            let mut conv = self.conversation.lock().unwrap_or_else(|e| e.into_inner());
            conv.clear();
            self.forget_epoch.fetch_add(1, Ordering::SeqCst);
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
    let epoch = ctx.forget_epoch.load(Ordering::SeqCst);

    // Who is speaking is looked up while the utterance is transcribed. The
    // lookup is read-only, so it is simply left to finish if the turn ends
    // first.
    let identify = match (&ctx.speakers, cfg.diarization) {
        (Some(service), true) => {
            let service = Arc::clone(service);
            let utterance = Arc::clone(&utterance);
            Some(tokio::spawn(async move {
                service.identify(&utterance.audio).await
            }))
        }
        _ => None,
    };

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

    // The lookup had the whole transcription to finish and gets a short
    // grace period on top. Not done by then: the turn goes on unlabelled.
    let speaker = match identify {
        Some(lookup) => {
            let waited = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                r = tokio::time::timeout(ctx.identify_grace, lookup) => r,
            };
            match waited {
                Ok(Ok(speaker)) => speaker,
                Ok(Err(e)) => {
                    debug!(turn_id, error = %e, "live_talk: speaker lookup failed");
                    None
                }
                Err(_) => {
                    debug!(
                        turn_id,
                        "live_talk: speaker lookup not ready, turn is unlabelled"
                    );
                    None
                }
            }
        }
        None => None,
    };

    out.send_turn(
        turn_id,
        LiveTalkWsEvent::Transcript {
            turn_id,
            text: user_text.clone(),
            duration_ms: stt_result.duration_ms,
            processing_time_ms: stt_result.processing_time_ms,
            speaker_id: speaker.as_ref().map(|s| s.id.clone()),
            speaker_name: speaker.as_ref().map(|s| s.name.clone()),
            speaker_color: speaker.as_ref().map(|s| s.color.clone()),
        },
    );
    shared.phase.store(PHASE_THINKING, Ordering::SeqCst);
    if !out.send_turn(turn_id, LiveTalkWsEvent::Thinking { turn_id }) {
        return;
    }

    let recall = match &speaker {
        Some(speaker) => tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            block = ctx.recall_block(speaker, &user_text) => block,
        },
        None => None,
    };
    // Built once: these exact bytes go to the model now and into the
    // history afterwards.
    let user = UserMessage::new(
        &user_text,
        speaker.map(|s| SpeakerLabel {
            id: s.id,
            name: s.name,
        }),
        recall.as_deref(),
    );

    let messages = ctx
        .conversation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .messages_for(Some(&cfg.system_prompt), user.content());

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
    let trimmed = match interrupted {
        Some(interrupted) => ctx.commit(turn_id, epoch, &user, &spoken, interrupted),
        None => false,
    };

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
            // The model is idle until the user speaks again, and its cached
            // prefix went with the trimmed turns: the moment to distill
            // them. Turns trimmed by a turn that did not finish wait for
            // the next one that does.
            if trimmed && !cancel.is_cancelled() {
                ctx.start_distill(&cfg.model);
            }
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
    /// The system prompt without the speaker paragraph.
    base_prompt: String,
    mode: LiveTalkMode,
    barge_in_enabled: bool,
    /// The utterance of the newest turn, for `enroll_speaker`.
    last_utterance: Option<Arc<vox::Utterance>>,

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
        if self.active.as_ref().is_some_and(|t| t.handle.is_finished())
            && let Some(turn) = self.active.take()
        {
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
                        // The model is about to be needed.
                        self.ctx.cancel_distill();
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
        // Whatever started it, a turn never shares the model with a
        // distillation.
        self.ctx.cancel_distill();
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
        self.last_utterance = Some(Arc::clone(&utterance));
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
        // A cancel is the client's barge-in: the user is speaking.
        self.ctx.cancel_distill();
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
            self.base_prompt = prompt;
        }
        self.cfg.diarization = new.diarization && self.ctx.speakers.is_some();
        self.cfg.system_prompt = effective_prompt(&self.base_prompt, self.cfg.diarization);
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
            diarization = self.cfg.diarization,
            "live_talk: config updated"
        );
    }

    /// Store the voice of the last utterance as a speaker. The work runs on
    /// its own task (extracting a voiceprint takes a moment) and answers
    /// with `speaker_enrolled` or a non-fatal error.
    fn on_enroll(&mut self, name: Option<String>) {
        let refuse = |message: &str| {
            self.out.send(LiveTalkWsEvent::Error {
                message: message.to_string(),
                fatal: false,
                turn_id: None,
            });
        };
        let Some(service) = self.ctx.speakers.clone() else {
            return refuse("Speaker recognition is not available on this server.");
        };
        let Some(utterance) = self.last_utterance.clone() else {
            return refuse("Say something first: the voice is taken from the last utterance.");
        };
        let out = self.out.clone();
        tokio::spawn(async move {
            match service.enroll(&utterance.audio, name.as_deref()).await {
                Ok(enrollment) => {
                    info!(
                        speaker_id = %enrollment.speaker.id,
                        created = enrollment.created,
                        "live_talk: speaker enrolled on request"
                    );
                    out.send(LiveTalkWsEvent::SpeakerEnrolled {
                        speaker_id: enrollment.speaker.id,
                        speaker_name: enrollment.speaker.name,
                        speaker_color: enrollment.speaker.color,
                        created: enrollment.created,
                    });
                }
                Err(message) => {
                    out.send(LiveTalkWsEvent::Error {
                        message,
                        fatal: false,
                        turn_id: None,
                    });
                }
            }
        });
    }

    async fn on_ptt_start(&mut self) {
        self.reap();
        self.ptt_active = true;
        self.ptt_buffer.clear();
        self.ctx.cancel_distill();
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
            Ok(LiveTalkClientMsg::EnrollSpeaker { name }) => self.on_enroll(name),
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
    let base_prompt = config
        .system_prompt_override
        .unwrap_or_else(|| vox::build_system_prompt(vox::VoicePromptMode::Voice));
    let diarization = config.diarization && deps.speakers.is_some();
    let cfg = TurnConfig {
        model: config.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        voice: config.voice,
        system_prompt: effective_prompt(&base_prompt, diarization),
        diarization,
    };

    info!(
        model = %cfg.model,
        host = %deps.ollama_host,
        voice = ?cfg.voice,
        mode = ?mode,
        barge_in = config.barge_in_enabled,
        diarization,
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
        diarization,
    });
    let snapshot = session_handle
        .conversation
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .snapshot();
    out.send(LiveTalkWsEvent::Memory(snapshot));

    let memory_tx = deps.memory.clone().map(|store| {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(memory_worker(store, rx));
        tx
    });

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
            forget_epoch: AtomicU64::new(0),
            speakers: deps.speakers,
            identify_grace: deps.identify_grace,
            distill: (deps.distill && deps.memory.is_some()).then(|| std::sync::Mutex::new(None)),
            memory: deps.memory,
            memory_tx,
        }),
        out,
        vad: deps.vad,
        frame_size,
        playback_grace: deps.playback_grace,
        cfg,
        base_prompt,
        mode,
        barge_in_enabled: config.barge_in_enabled,
        last_utterance: None,
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
    // A distillation holds the outgoing queue open. Stop it; its turns go
    // back to the session and are picked up after a reconnect.
    if let Some(distill) = session.ctx.stop_distill() {
        let _ = tokio::time::timeout(CANCEL_JOIN, distill).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::speakers::tests::{enroll, test_ctx};

    fn two_seconds() -> vox::AudioChunk {
        vox::AudioChunk {
            samples: vec![0.1; 32_000],
            sample_rate: 16_000,
            channels: 1,
        }
    }

    #[tokio::test]
    async fn explicit_enrollment_is_refused_without_consent() {
        let ctx = test_ctx().await;
        let service = ServerSpeakers(ctx.clone());

        let refused = service.enroll(&two_seconds(), Some("Mert")).await;
        let message = refused.expect_err("no consent, no voiceprint");
        assert!(message.contains("consent"), "{message}");
        assert!(speakers::list_speakers(&ctx).await.unwrap().is_empty());

        // A name that could not be stored is rejected before anything else.
        let refused = service.enroll(&two_seconds(), Some("<b>")).await;
        assert!(refused.is_err());

        // With consent the request gets as far as the encoder, which this
        // test context does not have: still nothing is stored.
        speakers::set_enrollment_consent(&ctx, true).await.unwrap();
        let refused = service.enroll(&two_seconds(), Some("Mert")).await;
        let message = refused.expect_err("no encoder model");
        assert!(message.contains("could not be stored"), "{message}");
        assert!(speakers::list_speakers(&ctx).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn identification_without_an_encoder_labels_nobody() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        let service = ServerSpeakers(ctx);
        assert_eq!(service.identify(&two_seconds()).await, None);
    }

    fn evicted(id: &str, name: &str, user: &str) -> EvictedTurn {
        EvictedTurn {
            speaker: SpeakerLabel {
                id: id.into(),
                name: name.into(),
            },
            user: user.into(),
            assistant: "ok.".into(),
            interrupted: false,
        }
    }

    #[test]
    fn distill_batches_are_per_speaker_and_bounded() {
        let turns = vec![
            evicted("speaker_1", "Mert", "one"),
            evicted("speaker_2", "Ada", "two"),
            evicted("speaker_1", "Mert Ozoner", "three"),
            evicted("speaker_1", "Mert Ozoner", &"x".repeat(5_000)),
            evicted("speaker_1", "Mert Ozoner", "five"),
        ];
        // Room for two short turns (each costs 3 + 3 + 2 * 16) or for one
        // long one, whose text counts as cut to the line limit.
        let batches = distill_batches(turns, 700);
        let shape: Vec<(&str, &str, Vec<usize>)> = batches
            .iter()
            .map(|b| {
                (
                    b.speaker.id.as_str(),
                    b.speaker.name.as_str(),
                    b.turns.iter().map(|t| t.user.len()).collect(),
                )
            })
            .collect();
        assert_eq!(
            shape,
            [
                // Renamed meanwhile: the newest name is used throughout.
                ("speaker_1", "Mert Ozoner", vec![3, 5]),
                ("speaker_1", "Mert Ozoner", vec![5_000, 4]),
                ("speaker_2", "Ada", vec![3]),
            ]
        );
        assert!(distill_batches(Vec::new(), 700).is_empty());
        // A turn that is larger than the limit still gets through, alone.
        let batches = distill_batches(vec![evicted("speaker_1", "Mert", "one")], 1);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn the_speaker_paragraph_is_only_added_for_diarization() {
        assert_eq!(effective_prompt("base", false), "base");
        let aware = effective_prompt("base", true);
        assert!(aware.starts_with("base\n\n"));
        assert!(aware.contains("[Memory"));
    }
}
