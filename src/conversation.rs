//! Lightweight per-session conversation memory for voice chat.
//!
//! Design goals (see the omp / T3 Code harnesses for prior art):
//!
//! - **Append-only, byte-stable prefix.** Messages are never rewritten on a
//!   normal turn, so Ollama/llama.cpp can reuse the KV cache for the whole
//!   history and only prefill the new user message.
//! - **Token budget, not turn count.** History is bounded by an estimated
//!   token budget. Trimming drops whole turns from the front.
//! - **Hysteresis.** When the budget is exceeded we trim down to a lower
//!   watermark in one go, so the cache prefix is invalidated rarely instead of
//!   on every turn once the window is full.
//! - **Voice truth.** On barge-in only the sentences that were actually sent
//!   to the listener are recorded, followed by an interruption marker, so the
//!   model never assumes the user heard something they did not. Because the
//!   server only knows what it *sent*, a client can later report how many
//!   sentences were really played ([`Conversation::amend_last_spoken`]).
//!
//! - **Speakers are part of the text.** When the speaker of a turn is known
//!   the user message is stored as `[Name] text`, exactly as it was sent to
//!   the model ([`UserMessage`]). A later rename does not touch stored turns,
//!   so the prefix stays byte-stable.
//! - **Recall is frozen.** What is remembered about a person from earlier
//!   sessions is looked up once per session, frozen, and placed in front of
//!   that person's first message. It is put in again only when the turn that
//!   carried it has been trimmed away.
//!
//! [`SessionStore`] keeps conversations alive across reconnects, keyed by a
//! session id and dropped after an idle time-to-live.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

/// Default Ollama context window used by every vox chat request.
///
/// Every vox code path must send the same value: Ollama reloads the model when
/// `num_ctx` changes between requests. Override with `VOX_OLLAMA_NUM_CTX`.
pub const DEFAULT_NUM_CTX: usize = 8192;

/// How long Ollama should keep the model loaded between turns.
pub const OLLAMA_KEEP_ALIVE: &str = "30m";

/// Marker appended to an assistant reply that the user interrupted.
pub const INTERRUPTED_MARKER: &str = "[interrupted by user]";

/// Context window shared by all vox → Ollama requests.
pub fn ollama_num_ctx() -> usize {
    std::env::var("VOX_OLLAMA_NUM_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 512)
        .unwrap_or(DEFAULT_NUM_CTX)
}

/// Ollama request options shared by all vox chat requests.
pub fn ollama_options() -> serde_json::Value {
    serde_json::json!({ "num_ctx": ollama_num_ctx() })
}

/// One message in Ollama `/api/chat` format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: &'static str,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system",
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant",
            content: content.into(),
        }
    }
}

/// Who said a user message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeakerLabel {
    /// Stable speaker id (the key of the speaker database).
    pub id: String,
    /// The name as it was when the message was said.
    pub name: String,
}

/// Longest speaker name placed in a `[Name] ` tag.
const MAX_TAG_NAME_CHARS: usize = 64;

/// The name as it may appear inside a `[Name] ` tag: no brackets, no control
/// characters, bounded length. Empty when nothing usable is left.
fn tag_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_control() || c == '[' || c == ']' {
                ' '
            } else {
                c
            }
        })
        .collect();
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MAX_TAG_NAME_CHARS)
        .collect::<String>()
        .trim_end()
        .to_string()
}

/// One user message: what was said, by whom, and the text the model gets.
///
/// The model-facing text is built once, here, and then both sent and stored
/// verbatim, so the stored history can never differ from what was sent:
///
/// ```text
/// <recall block>            (only when one is injected with this message)
///
/// [Name] what was said      (the tag only when the speaker is known)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessage {
    text: String,
    speaker: Option<SpeakerLabel>,
    has_recall: bool,
    content: String,
}

impl UserMessage {
    /// A message from an unidentified speaker, with no recall block.
    pub fn plain(text: &str) -> Self {
        Self::new(text, None, None)
    }

    /// `recall` is a block of remembered context to put in front of the
    /// message (see [`Conversation::recall_need`]); an empty one is ignored.
    pub fn new(text: &str, speaker: Option<SpeakerLabel>, recall: Option<&str>) -> Self {
        let text = text.trim();
        let recall = recall.map(str::trim).filter(|r| !r.is_empty());
        let mut content = String::new();
        if let Some(recall) = recall {
            content.push_str(recall);
            content.push_str("\n\n");
        }
        if let Some(speaker) = &speaker {
            let name = tag_name(&speaker.name);
            if !name.is_empty() {
                content.push('[');
                content.push_str(&name);
                content.push_str("] ");
            }
        }
        content.push_str(text);
        Self {
            text: text.to_string(),
            speaker,
            has_recall: recall.is_some(),
            content,
        }
    }

    /// What was said, without tag or recall block.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn speaker(&self) -> Option<&SpeakerLabel> {
        self.speaker.as_ref()
    }

    /// The text the model gets, and that is stored.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Whether a recall block is part of [`content`](Self::content).
    pub fn has_recall(&self) -> bool {
        self.has_recall
    }
}

/// What to do about remembered context before a known person's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallNeed {
    /// Nothing was looked up for this person in this session yet: look it up
    /// and hand the result to [`Conversation::freeze_recall`].
    Lookup,
    /// Put this frozen block in front of the message.
    Inject(String),
    /// Nothing to add: the block is already in the history, there is none,
    /// or the session was told to forget.
    Nothing,
}

#[derive(Debug, Clone)]
struct RecallEntry {
    /// `None`: looked up, nothing to recall.
    block: Option<String>,
    /// A stored turn carries the block.
    in_context: bool,
}

#[derive(Debug, Clone)]
struct Turn {
    /// The user message as the model sees it (tag and recall block included).
    user: String,
    /// What the user said, for display.
    user_text: String,
    speaker: Option<SpeakerLabel>,
    /// This turn's user message carries the recall block of its speaker.
    carries_recall: bool,
    /// Sentences that reached the listener, trimmed, in order.
    sentences: Vec<String>,
    interrupted: bool,
    /// What the model sees as its reply: the sentences joined, plus the
    /// interruption marker when `interrupted`.
    assistant: String,
    tokens: usize,
}

impl Turn {
    fn new(user: &UserMessage, sentences: Vec<String>, interrupted: bool) -> Self {
        let mut turn = Self {
            user: user.content.clone(),
            user_text: user.text.clone(),
            speaker: user.speaker.clone(),
            carries_recall: user.has_recall && user.speaker.is_some(),
            sentences,
            interrupted,
            assistant: String::new(),
            tokens: 0,
        };
        turn.render();
        turn
    }

    /// Recompute `assistant` and `tokens` from the sentences.
    fn render(&mut self) {
        let mut assistant = self.sentences.join(" ");
        if self.interrupted {
            if !assistant.is_empty() {
                assistant.push(' ');
            }
            assistant.push_str(INTERRUPTED_MARKER);
        }
        self.tokens = estimate_tokens(&self.user) + estimate_tokens(&assistant);
        self.assistant = assistant;
    }
}

/// One remembered turn, as shown to a client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TurnSnapshot {
    /// What the user said, without the speaker tag or a recall block.
    pub user: String,
    /// Who said it, when the speaker was identified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    /// The reply the listener heard, without the interruption marker.
    pub assistant: String,
    pub interrupted: bool,
    /// Estimated tokens this turn takes in the model's context.
    pub tokens: usize,
}

/// The whole memory of a session, as shown to a client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationSnapshot {
    pub turns: Vec<TurnSnapshot>,
    pub total_tokens: usize,
    pub budget_tokens: usize,
    pub low_watermark_tokens: usize,
    /// Turns dropped from the front by trimming since the last
    /// [`Conversation::clear`].
    pub dropped_turns: usize,
}

/// Rough token estimate. Deliberately conservative (~3 chars/token) so
/// non-English text such as Turkish does not overflow the window.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(3) + 4
}

/// Bounded, append-only conversation history for one session.
#[derive(Debug, Clone)]
pub struct Conversation {
    turns: Vec<Turn>,
    total_tokens: usize,
    budget_tokens: usize,
    low_watermark_tokens: usize,
    dropped_turns: usize,
    /// Recall state by speaker id; see [`Conversation::recall_need`].
    recall: HashMap<String, RecallEntry>,
    /// Set by [`Conversation::clear`]: no recall for the rest of the session.
    recall_suppressed: bool,
}

impl Conversation {
    /// History budget derived from the Ollama context window: half of the
    /// window is left for the system prompt, the new message and the reply.
    pub fn for_num_ctx(num_ctx: usize) -> Self {
        Self::with_budget(num_ctx / 2)
    }

    /// `budget_tokens` is the history ceiling; trimming goes down to 60% of it.
    pub fn with_budget(budget_tokens: usize) -> Self {
        Self {
            turns: Vec::new(),
            total_tokens: 0,
            budget_tokens,
            low_watermark_tokens: budget_tokens * 3 / 5,
            dropped_turns: 0,
            recall: HashMap::new(),
            recall_suppressed: false,
        }
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn history_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Forget every turn.
    ///
    /// The frozen recall blocks go too, and none is looked up or injected
    /// for the rest of the session: the user asked to forget, so context
    /// from earlier sessions must not come straight back. Stored long-term
    /// memory is not touched by this.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.total_tokens = 0;
        self.dropped_turns = 0;
        self.recall.clear();
        self.recall_suppressed = true;
    }

    /// What remembered context, if any, belongs in front of the next message
    /// of the person `speaker_id`.
    pub fn recall_need(&self, speaker_id: &str) -> RecallNeed {
        if self.recall_suppressed {
            return RecallNeed::Nothing;
        }
        match self.recall.get(speaker_id) {
            None => RecallNeed::Lookup,
            Some(RecallEntry {
                block: Some(block),
                in_context: false,
            }) => RecallNeed::Inject(block.clone()),
            Some(_) => RecallNeed::Nothing,
        }
    }

    /// Record the result of a recall lookup for `speaker_id`; `None` (or an
    /// empty block) means there is nothing to recall. The first result for a
    /// person stays for the whole session, so the block is the same bytes
    /// every time it is injected.
    pub fn freeze_recall(&mut self, speaker_id: &str, block: Option<String>) {
        if self.recall_suppressed {
            return;
        }
        let block = block
            .map(|b| b.trim().to_string())
            .filter(|b| !b.is_empty());
        self.recall
            .entry(speaker_id.to_string())
            .or_insert(RecallEntry {
                block,
                in_context: false,
            });
    }

    /// Number of sentences recorded for the newest turn.
    pub fn last_spoken_len(&self) -> Option<usize> {
        self.turns.last().map(|t| t.sentences.len())
    }

    /// The current memory, for display.
    pub fn snapshot(&self) -> ConversationSnapshot {
        ConversationSnapshot {
            turns: self
                .turns
                .iter()
                .map(|t| TurnSnapshot {
                    user: t.user_text.clone(),
                    speaker_id: t.speaker.as_ref().map(|s| s.id.clone()),
                    speaker_name: t.speaker.as_ref().map(|s| s.name.clone()),
                    assistant: t.sentences.join(" "),
                    interrupted: t.interrupted,
                    tokens: t.tokens,
                })
                .collect(),
            total_tokens: self.total_tokens,
            budget_tokens: self.budget_tokens,
            low_watermark_tokens: self.low_watermark_tokens,
            dropped_turns: self.dropped_turns,
        }
    }

    /// Correct the newest turn after the fact: the listener heard only its
    /// first `heard` sentences.
    ///
    /// Cuts the turn down to those sentences and marks it interrupted.
    /// Returns `false` and changes nothing when there is no turn or when the
    /// turn has no more than `heard` sentences, so calling it again with the
    /// same value is harmless.
    pub fn amend_last_spoken(&mut self, heard: usize) -> bool {
        let Some(turn) = self.turns.last_mut() else {
            return false;
        };
        if heard >= turn.sentences.len() {
            return false;
        }
        let before = turn.tokens;
        turn.sentences.truncate(heard);
        turn.interrupted = true;
        turn.render();
        self.total_tokens = self.total_tokens - before + turn.tokens;
        true
    }

    /// Build the message list for a new turn: system prompt, history, then
    /// the new user message. Volatile context belongs in `user_text`, never in
    /// the system prompt, so the cached prefix stays valid.
    ///
    /// For a message built with [`UserMessage`], pass its
    /// [`content`](UserMessage::content).
    pub fn messages_for(&self, system_prompt: Option<&str>, user_text: &str) -> Vec<ChatMessage> {
        let mut out = Vec::with_capacity(self.turns.len() * 2 + 2);
        if let Some(sys) = system_prompt {
            out.push(ChatMessage::system(sys));
        }
        for t in &self.turns {
            out.push(ChatMessage::user(t.user.clone()));
            out.push(ChatMessage::assistant(t.assistant.clone()));
        }
        out.push(ChatMessage::user(user_text));
        out
    }

    /// Record a completed or interrupted turn.
    ///
    /// `spoken` are the sentences the listener actually received. When
    /// `interrupted` is true an interruption marker is appended. Returns the
    /// number of turns dropped by trimming (0 when the prefix stayed intact).
    pub fn commit(&mut self, user_text: &str, spoken: &[String], interrupted: bool) -> usize {
        self.commit_message(&UserMessage::plain(user_text), spoken, interrupted)
    }

    /// [`commit`](Self::commit) for a message that may carry a speaker and a
    /// recall block. The message's [`content`](UserMessage::content) is
    /// stored as it is.
    pub fn commit_message(
        &mut self,
        user: &UserMessage,
        spoken: &[String],
        interrupted: bool,
    ) -> usize {
        if user.text.is_empty() {
            return 0;
        }
        let sentences: Vec<String> = spoken
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if sentences.is_empty() && !interrupted {
            // Nothing reached the listener and nothing was interrupted
            // (e.g. empty LLM reply): skip to keep the history meaningful.
            return 0;
        }
        let turn = Turn::new(user, sentences, interrupted);
        let tokens = turn.tokens;
        if turn.carries_recall
            && let Some(entry) = turn
                .speaker
                .as_ref()
                .and_then(|speaker| self.recall.get_mut(&speaker.id))
        {
            entry.in_context = true;
        }
        self.turns.push(turn);
        self.total_tokens += tokens;
        self.trim()
    }

    fn trim(&mut self) -> usize {
        if self.total_tokens <= self.budget_tokens {
            return 0;
        }
        let mut drop = 0;
        let mut remaining = self.total_tokens;
        // Always keep the newest turn, even if it alone exceeds the watermark.
        while drop + 1 < self.turns.len() && remaining > self.low_watermark_tokens {
            remaining -= self.turns[drop].tokens;
            drop += 1;
        }
        for turn in self.turns.drain(..drop) {
            if !turn.carries_recall {
                continue;
            }
            // The block left the context with this turn: the person's next
            // message brings it back.
            if let Some(entry) = turn
                .speaker
                .as_ref()
                .and_then(|speaker| self.recall.get_mut(&speaker.id))
            {
                entry.in_context = false;
            }
        }
        self.total_tokens = remaining;
        self.dropped_turns += drop;
        drop
    }
}

/// A conversation shared between a connection and its turn tasks.
///
/// A `std` mutex is enough: it is never held across an `.await`.
pub type SharedConversation = Arc<Mutex<Conversation>>;

/// Default idle time after which a session's memory is dropped.
pub const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(30 * 60);

/// Default cap on the number of sessions kept at once.
pub const DEFAULT_MAX_SESSIONS: usize = 256;

struct SessionEntry {
    conversation: SharedConversation,
    last_seen: Instant,
}

/// What [`SessionStore::attach`] hands to a connection.
#[derive(Debug, Clone)]
pub struct SessionHandle {
    /// The id to send back to the client; reconnect with it to continue.
    pub id: String,
    pub conversation: SharedConversation,
    /// `true` when an existing session was found for the requested id.
    pub resumed: bool,
}

/// In-memory conversations keyed by session id, so that a client that
/// reconnects (for example after toggling the microphone) keeps its memory.
///
/// Sessions idle for longer than the time-to-live are dropped the next time
/// the store is used. Nothing is written to disk. The id is the only thing
/// protecting a session's memory: ids made here are unguessable, and a
/// client-chosen id is accepted only if it is 8-64 characters of
/// `[A-Za-z0-9_-]`.
pub struct SessionStore {
    sessions: Mutex<HashMap<String, SessionEntry>>,
    ttl: Duration,
    max_sessions: usize,
}

impl Default for SessionStore {
    /// Time-to-live from `VOX_SESSION_TTL_SECS` (default 30 minutes).
    fn default() -> Self {
        let ttl = std::env::var("VOX_SESSION_TTL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_SESSION_TTL);
        Self::new(ttl, DEFAULT_MAX_SESSIONS)
    }
}

impl SessionStore {
    pub fn new(ttl: Duration, max_sessions: usize) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            ttl,
            max_sessions: max_sessions.max(1),
        }
    }

    /// Whether `id` is acceptable as a client-chosen session id.
    pub fn is_valid_id(id: &str) -> bool {
        (8..=64).contains(&id.len())
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    }

    /// Find the session for `requested`, or start a new one.
    ///
    /// A valid `requested` id that is unknown (never seen, or expired) starts
    /// an empty session under that id; an invalid or missing one gets a new
    /// id.
    pub fn attach(&self, requested: Option<&str>) -> SessionHandle {
        self.attach_at(requested, Instant::now())
    }

    fn attach_at(&self, requested: Option<&str>, now: Instant) -> SessionHandle {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let ttl = self.ttl;
        sessions.retain(|_, s| now.saturating_duration_since(s.last_seen) < ttl);

        let requested = requested.filter(|id| Self::is_valid_id(id));
        if let Some(id) = requested
            && let Some(entry) = sessions.get_mut(id)
        {
            entry.last_seen = now;
            return SessionHandle {
                id: id.to_string(),
                conversation: Arc::clone(&entry.conversation),
                resumed: true,
            };
        }

        while sessions.len() >= self.max_sessions {
            let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, s)| s.last_seen)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            sessions.remove(&oldest);
        }

        let id = match requested {
            Some(id) => id.to_string(),
            None => loop {
                let id = new_session_id();
                if !sessions.contains_key(&id) {
                    break id;
                }
            },
        };
        let conversation: SharedConversation =
            Arc::new(Mutex::new(Conversation::for_num_ctx(ollama_num_ctx())));
        sessions.insert(
            id.clone(),
            SessionEntry {
                conversation: Arc::clone(&conversation),
                last_seen: now,
            },
        );
        SessionHandle {
            id,
            conversation,
            resumed: false,
        }
    }

    /// Restart the idle clock of a session (call when a connection ends, so
    /// the time-to-live counts from the last use).
    pub fn touch(&self, id: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = sessions.get_mut(id) {
            entry.last_seen = Instant::now();
        }
    }

    /// Number of sessions currently kept (expired ones included until the
    /// next [`attach`](Self::attach)).
    pub fn len(&self) -> usize {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A 128-bit random id in hex.
///
/// The randomness comes from the operating system through the standard
/// library's randomly keyed SipHash ([`std::collections::hash_map::RandomState`]):
/// two independently keyed hashers are applied to a counter and the clock.
fn new_session_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut out = String::with_capacity(32);
    for _ in 0..2 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(n);
        hasher.write_u128(nanos);
        out.push_str(&format!("{:016x}", hasher.finish()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn messages_include_system_history_and_new_user() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("hi", &s(&["Hello.", "How can I help?"]), false);
        let m = c.messages_for(Some("sys"), "what's up");
        let roles: Vec<_> = m.iter().map(|x| x.role).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(m[2].content, "Hello. How can I help?");
        assert_eq!(m[3].content, "what's up");
    }

    #[test]
    fn prefix_is_stable_across_turns() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("a", &s(&["one."]), false);
        let first = c.messages_for(Some("sys"), "b");
        c.commit("b", &s(&["two."]), false);
        let second = c.messages_for(Some("sys"), "c");
        // Everything before the new user message is unchanged.
        assert_eq!(&second[..first.len() - 1], &first[..first.len() - 1]);
    }

    #[test]
    fn interrupted_turn_records_only_spoken_part() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("tell me a story", &s(&["Once upon a time."]), true);
        let m = c.messages_for(None, "stop");
        assert_eq!(
            m[1].content,
            format!("Once upon a time. {INTERRUPTED_MARKER}")
        );
    }

    #[test]
    fn interrupted_before_any_audio_still_recorded() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("hey", &[], true);
        assert_eq!(c.len(), 1);
        assert_eq!(c.messages_for(None, "x")[1].content, INTERRUPTED_MARKER);
    }

    #[test]
    fn empty_reply_or_empty_user_is_skipped() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("hey", &[], false);
        c.commit("   ", &s(&["x"]), false);
        assert!(c.is_empty());
    }

    #[test]
    fn trims_whole_turns_with_hysteresis() {
        // Each turn: user 30 chars (14 tok) + assistant 30 chars (14 tok) = 28.
        let mut c = Conversation::with_budget(100);
        let u = "u".repeat(30);
        let a = s(&[&"a".repeat(30)]);
        for _ in 0..3 {
            assert_eq!(c.commit(&u, &a, false), 0);
        }
        assert_eq!(c.history_tokens(), 84);
        // 4th turn -> 112 > 100: trim down to <= 60 (2 turns = 56).
        assert_eq!(c.commit(&u, &a, false), 2);
        assert_eq!(c.len(), 2);
        assert_eq!(c.history_tokens(), 56);
        // Next turn fits again without trimming (84 <= 100): prefix untouched.
        assert_eq!(c.commit(&u, &a, false), 0);
    }

    #[test]
    fn newest_turn_is_always_kept() {
        let mut c = Conversation::with_budget(10);
        c.commit(&"x".repeat(300), &s(&["y"]), false);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn num_ctx_env_override_is_validated() {
        // Only checks the parsing helper's fallback; env is not mutated here.
        assert!(ollama_num_ctx() >= 512);
        assert_eq!(Conversation::for_num_ctx(8192).budget_tokens, 4096);
    }

    #[test]
    fn amend_last_spoken_cuts_the_newest_turn_and_marks_it_interrupted() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("first", &s(&["Kept."]), false);
        c.commit(
            "story",
            &s(&[
                "One.",
                "Two is a much longer sentence than one.",
                "Three is long as well, and never heard.",
            ]),
            false,
        );
        let before = c.history_tokens();

        assert!(c.amend_last_spoken(1));

        let snap = c.snapshot();
        assert_eq!(snap.turns.len(), 2);
        assert_eq!(snap.turns[0].assistant, "Kept.");
        assert!(!snap.turns[0].interrupted);
        assert_eq!(snap.turns[1].assistant, "One.");
        assert!(snap.turns[1].interrupted);
        assert_eq!(c.last_spoken_len(), Some(1));
        assert_eq!(
            c.messages_for(None, "x")[3].content,
            format!("One. {INTERRUPTED_MARKER}")
        );
        // Token accounting follows the shorter reply.
        assert!(c.history_tokens() < before);
        assert_eq!(
            snap.total_tokens,
            snap.turns.iter().map(|t| t.tokens).sum::<usize>()
        );
    }

    #[test]
    fn amend_last_spoken_is_a_no_op_when_everything_was_heard() {
        let mut c = Conversation::with_budget(10_000);
        assert!(!c.amend_last_spoken(0));
        c.commit("hi", &s(&["One.", "Two."]), false);
        let before = c.snapshot();
        assert!(!c.amend_last_spoken(2));
        assert!(!c.amend_last_spoken(7));
        assert_eq!(c.snapshot(), before);
        // Amending twice with the same value changes nothing the second time.
        assert!(c.amend_last_spoken(1));
        assert!(!c.amend_last_spoken(1));
    }

    #[test]
    fn amend_to_zero_keeps_the_turn_as_interrupted() {
        let mut c = Conversation::with_budget(10_000);
        c.commit("hi", &s(&["One."]), false);
        assert!(c.amend_last_spoken(0));
        let snap = c.snapshot();
        assert_eq!(snap.turns[0].assistant, "");
        assert!(snap.turns[0].interrupted);
        assert_eq!(c.messages_for(None, "x")[1].content, INTERRUPTED_MARKER);
    }

    #[test]
    fn snapshot_reports_budget_and_dropped_turns() {
        let mut c = Conversation::with_budget(100);
        let u = "u".repeat(30);
        let a = s(&[&"a".repeat(30)]);
        for _ in 0..4 {
            c.commit(&u, &a, false);
        }
        let snap = c.snapshot();
        assert_eq!(snap.budget_tokens, 100);
        assert_eq!(snap.low_watermark_tokens, 60);
        assert_eq!(snap.dropped_turns, 2);
        assert_eq!(snap.turns.len(), 2);
        assert_eq!(snap.total_tokens, 56);

        c.clear();
        let snap = c.snapshot();
        assert!(snap.turns.is_empty());
        assert_eq!(snap.total_tokens, 0);
        assert_eq!(snap.dropped_turns, 0);
    }

    #[test]
    fn snapshot_serializes_the_wire_shape() {
        let mut c = Conversation::with_budget(1000);
        c.commit("hi", &s(&["Hello."]), true);
        let json = serde_json::to_value(c.snapshot()).unwrap();
        assert_eq!(json["turns"][0]["user"], "hi");
        assert_eq!(json["turns"][0]["assistant"], "Hello.");
        assert_eq!(json["turns"][0]["interrupted"], true);
        assert!(json["turns"][0]["tokens"].as_u64().unwrap() > 0);
        assert_eq!(json["budget_tokens"], 1000);
        assert_eq!(json["low_watermark_tokens"], 600);
        assert_eq!(json["dropped_turns"], 0);
    }

    #[test]
    fn session_store_resumes_a_known_id() {
        let store = SessionStore::new(Duration::from_secs(60), 8);
        let first = store.attach(None);
        assert!(!first.resumed);
        assert!(SessionStore::is_valid_id(&first.id));
        first
            .conversation
            .lock()
            .unwrap()
            .commit("hi", &s(&["Hello."]), false);

        let again = store.attach(Some(&first.id));
        assert!(again.resumed);
        assert_eq!(again.id, first.id);
        assert_eq!(again.conversation.lock().unwrap().len(), 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn session_store_ids_are_unique_and_client_ids_are_validated() {
        let store = SessionStore::new(Duration::from_secs(60), 8);
        let a = store.attach(None);
        let b = store.attach(None);
        assert_ne!(a.id, b.id);
        assert_eq!(a.id.len(), 32);

        // A well-formed unknown id is adopted; a malformed one is replaced.
        let chosen = store.attach(Some("my-session_01"));
        assert_eq!(chosen.id, "my-session_01");
        assert!(!chosen.resumed);
        let too_long = "x".repeat(65);
        for bad in [
            "short",
            "has space in it",
            "semi;colon;id",
            too_long.as_str(),
        ] {
            let handle = store.attach(Some(bad));
            assert_ne!(handle.id, bad);
            assert!(!handle.resumed);
        }
    }

    #[test]
    fn session_store_drops_idle_sessions_after_the_ttl() {
        let store = SessionStore::new(Duration::from_secs(10), 8);
        let t0 = Instant::now();
        let first = store.attach_at(None, t0);
        first
            .conversation
            .lock()
            .unwrap()
            .commit("hi", &s(&["Hello."]), false);

        // Still there just before the deadline; that use restarts the clock.
        let kept = store.attach_at(Some(&first.id), t0 + Duration::from_secs(9));
        assert!(kept.resumed);
        let kept = store.attach_at(Some(&first.id), t0 + Duration::from_secs(18));
        assert!(kept.resumed);

        // Gone once idle for the whole time-to-live: same id, empty memory.
        let late = store.attach_at(Some(&first.id), t0 + Duration::from_secs(28));
        assert!(!late.resumed);
        assert!(late.conversation.lock().unwrap().is_empty());
    }

    #[test]
    fn session_store_evicts_the_oldest_when_full() {
        let store = SessionStore::new(Duration::from_secs(600), 2);
        let t0 = Instant::now();
        let a = store.attach_at(None, t0);
        let b = store.attach_at(None, t0 + Duration::from_secs(1));
        let _c = store.attach_at(None, t0 + Duration::from_secs(2));
        assert_eq!(store.len(), 2);
        // `a` was the oldest and made room for `_c`.
        let a_again = store.attach_at(Some(&a.id), t0 + Duration::from_secs(3));
        assert!(!a_again.resumed);
        // Re-creating `a` evicted `b`, the oldest one left.
        let b_again = store.attach_at(Some(&b.id), t0 + Duration::from_secs(4));
        assert!(!b_again.resumed);
        assert_eq!(store.len(), 2);
    }

    fn label(id: &str, name: &str) -> Option<SpeakerLabel> {
        Some(SpeakerLabel {
            id: id.to_string(),
            name: name.to_string(),
        })
    }

    #[test]
    fn user_message_is_tagged_with_the_speaker_name() {
        let m = UserMessage::new("  what's the time?  ", label("speaker_1", "Mert"), None);
        assert_eq!(m.content(), "[Mert] what's the time?");
        assert_eq!(m.text(), "what's the time?");
        assert!(!m.has_recall());

        assert_eq!(UserMessage::plain("hi").content(), "hi");
        // A name cannot break out of the tag or smuggle in a line.
        let m = UserMessage::new("hi", label("x", "A]\n[system"), None);
        assert_eq!(m.content(), "[A system] hi");
        // Nothing usable in the name: no tag at all.
        let m = UserMessage::new("hi", label("x", " ][ "), None);
        assert_eq!(m.content(), "hi");
    }

    #[test]
    fn recall_block_goes_in_front_of_the_tag() {
        let m = UserMessage::new("hi", label("speaker_1", "Mert"), Some("[Memory]\n"));
        assert_eq!(m.content(), "[Memory]\n\n[Mert] hi");
        assert!(m.has_recall());
        assert!(!UserMessage::new("hi", None, Some("  ")).has_recall());
    }

    #[test]
    fn stored_messages_are_byte_identical_between_calls_and_after_a_rename() {
        let mut c = Conversation::with_budget(10_000);
        let first = UserMessage::new("hello", label("speaker_1", "Mert"), None);
        let sent_first = c.messages_for(Some("sys"), first.content());
        c.commit_message(&first, &s(&["Hi Mert."]), false);
        let second = UserMessage::new("and me", label("speaker_2", "Ada"), None);
        c.commit_message(&second, &s(&["Hi Ada."]), false);

        let a = serde_json::to_vec(&c.messages_for(Some("sys"), "next")).unwrap();
        let b = serde_json::to_vec(&c.messages_for(Some("sys"), "next")).unwrap();
        assert_eq!(a, b);

        // What was sent for the first turn is what is stored for it.
        let now = c.messages_for(Some("sys"), "next");
        assert_eq!(now[..2], sent_first[..]);
        assert_eq!(now[1].content, "[Mert] hello");
        assert_eq!(now[3].content, "[Ada] and me");

        // The speaker is renamed: the stored turns keep the old tag, only the
        // new message has the new one.
        let third = UserMessage::new("again", label("speaker_1", "Mert Ozoner"), None);
        let before = serde_json::to_vec(&now[..now.len() - 1]).unwrap();
        c.commit_message(&third, &s(&["Sure."]), false);
        let later = c.messages_for(Some("sys"), "next");
        let prefix = serde_json::to_vec(&later[..now.len() - 1]).unwrap();
        assert_eq!(before, prefix);
        assert_eq!(later[5].content, "[Mert Ozoner] again");
    }

    #[test]
    fn snapshot_shows_plain_text_and_the_speaker() {
        let mut c = Conversation::with_budget(10_000);
        let m = UserMessage::new("hello", label("speaker_1", "Mert"), Some("[Memory] x"));
        c.commit_message(&m, &s(&["Hi."]), false);
        c.commit("anonymous", &s(&["Hi."]), false);
        let snap = c.snapshot();
        assert_eq!(snap.turns[0].user, "hello");
        assert_eq!(snap.turns[0].speaker_id.as_deref(), Some("speaker_1"));
        assert_eq!(snap.turns[0].speaker_name.as_deref(), Some("Mert"));
        // The tag and the block count towards the context.
        assert_eq!(
            snap.turns[0].tokens,
            estimate_tokens("[Memory] x\n\n[Mert] hello") + estimate_tokens("Hi.")
        );
        let json = serde_json::to_value(&snap.turns[1]).unwrap();
        assert!(json.get("speaker_id").is_none());
        assert!(json.get("speaker_name").is_none());
    }

    #[test]
    fn recall_is_looked_up_once_and_injected_once() {
        let mut c = Conversation::with_budget(10_000);
        assert_eq!(c.recall_need("speaker_1"), RecallNeed::Lookup);
        c.freeze_recall("speaker_1", Some("[Memory] likes tea".into()));
        // A later lookup result does not replace the frozen block.
        c.freeze_recall("speaker_1", Some("[Memory] something else".into()));
        let RecallNeed::Inject(block) = c.recall_need("speaker_1") else {
            panic!("expected the frozen block");
        };
        assert_eq!(block, "[Memory] likes tea");

        // Not committed (the turn was dropped): still to be injected.
        assert!(matches!(c.recall_need("speaker_1"), RecallNeed::Inject(_)));
        let m = UserMessage::new("hi", label("speaker_1", "Mert"), Some(&block));
        c.commit_message(&m, &s(&["Hello."]), false);
        assert_eq!(c.recall_need("speaker_1"), RecallNeed::Nothing);

        // Another person is independent; "nothing to recall" is frozen too.
        assert_eq!(c.recall_need("speaker_2"), RecallNeed::Lookup);
        c.freeze_recall("speaker_2", None);
        assert_eq!(c.recall_need("speaker_2"), RecallNeed::Nothing);
    }

    #[test]
    fn recall_comes_back_when_its_turn_is_trimmed_away() {
        let mut c = Conversation::with_budget(60);
        c.freeze_recall("speaker_1", Some("[Memory] likes tea".into()));
        let m = UserMessage::new("hi", label("speaker_1", "Mert"), Some("[Memory] likes tea"));
        c.commit_message(&m, &s(&["Hello."]), false);
        assert_eq!(c.recall_need("speaker_1"), RecallNeed::Nothing);
        let long = "x".repeat(90);
        let mut dropped = 0;
        for _ in 0..4 {
            dropped += c.commit(&long, &s(&["ok."]), false);
        }
        assert!(dropped > 0);
        assert_eq!(
            c.recall_need("speaker_1"),
            RecallNeed::Inject("[Memory] likes tea".into())
        );
    }

    #[test]
    fn clear_stops_recall_for_the_rest_of_the_session() {
        let mut c = Conversation::with_budget(10_000);
        c.freeze_recall("speaker_1", Some("[Memory] likes tea".into()));
        c.clear();
        assert_eq!(c.recall_need("speaker_1"), RecallNeed::Nothing);
        c.freeze_recall("speaker_1", Some("[Memory] likes tea".into()));
        assert_eq!(c.recall_need("speaker_1"), RecallNeed::Nothing);
        assert_eq!(c.recall_need("speaker_2"), RecallNeed::Nothing);
    }
}
