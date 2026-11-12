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
//!   model never assumes the user heard something they did not.

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

#[derive(Debug, Clone)]
struct Turn {
    user: String,
    assistant: String,
    tokens: usize,
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

    pub fn clear(&mut self) {
        self.turns.clear();
        self.total_tokens = 0;
    }

    /// Build the message list for a new turn: system prompt, history, then
    /// the new user message. Volatile context belongs in `user_text`, never in
    /// the system prompt, so the cached prefix stays valid.
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
        let user = user_text.trim();
        if user.is_empty() {
            return 0;
        }
        let mut assistant = spoken
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if interrupted {
            if !assistant.is_empty() {
                assistant.push(' ');
            }
            assistant.push_str(INTERRUPTED_MARKER);
        }
        if assistant.is_empty() {
            // Nothing reached the listener and nothing was interrupted
            // (e.g. empty LLM reply): skip to keep the history meaningful.
            return 0;
        }
        let tokens = estimate_tokens(user) + estimate_tokens(&assistant);
        self.turns.push(Turn {
            user: user.to_string(),
            assistant,
            tokens,
        });
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
        self.turns.drain(..drop);
        self.total_tokens = remaining;
        drop
    }
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
}
