//! Streaming LLM→TTS integration.
//!
//! Streams an Ollama `/api/chat` reply and hands it to the caller in two
//! forms:
//!
//! - `on_token`: every raw token as it arrives, for display. Nothing in this
//!   module writes to stdout.
//! - `on_sentence`: speakable segments from the shared
//!   [`Segmenter`](crate::tts::segment::Segmenter), already sanitized for TTS.
//!   The string passed here is the one to synthesize *and* the one to store in
//!   conversation memory.
//!
//! A stream that stops producing data fails with a timeout instead of holding
//! the turn forever (see [`StreamOptions`]).
//!
//! This module is compiled with the `cli` or `server` feature.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::conversation::{ChatMessage, Conversation};
use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::tts::segment::Segmenter;

/// Ollama `/api/chat` streaming response chunk.
#[derive(Deserialize)]
struct OllamaChatChunk {
    #[serde(default)]
    message: Option<OllamaChatDelta>,
    #[serde(default)]
    done: bool,
    /// Prompt tokens actually evaluated (excludes KV-cache hits). Only on the
    /// final chunk; small values across turns mean the prefix cache works.
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct OllamaChatDelta {
    #[serde(default)]
    content: String,
}

/// Why streaming stopped (normal completion or cancellation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Finished,
    Cancelled,
}

/// Timeouts for one streamed LLM reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamOptions {
    /// Longest wait from sending the request to the first response data.
    /// Generous by default because Ollama may have to load the model first.
    pub first_token_timeout: Duration,
    /// Longest gap between two pieces of response data once the stream runs.
    pub idle_timeout: Duration,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            first_token_timeout: Duration::from_secs(120),
            idle_timeout: Duration::from_secs(30),
        }
    }
}

/// Stream a single-prompt reply (no history) and call `on_sentence` per
/// speakable segment.
///
/// The `tts` and `voice` arguments are unused and kept for source
/// compatibility; synthesis happens in `on_sentence`.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat_with_tts<F, Fut>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    tts: Arc<dyn TtsBackend>,
    system_prompt: Option<String>,
    voice: Option<String>,
    on_sentence: F,
) -> Result<(), VoxError>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    stream_chat_with_tts_cancellable(
        client,
        host,
        model,
        prompt,
        tts,
        system_prompt,
        voice,
        CancellationToken::new(),
        on_sentence,
    )
    .await
    .map(|_| ())
}

/// Like `stream_chat_with_tts` but returns early if `cancel` fires.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat_with_tts_cancellable<F, Fut>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    _tts: Arc<dyn TtsBackend>,
    system_prompt: Option<String>,
    _voice: Option<String>,
    cancel: CancellationToken,
    on_sentence: F,
) -> Result<StopReason, VoxError>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    let messages = Conversation::with_budget(0).messages_for(system_prompt.as_deref(), prompt);
    stream_chat_messages_cancellable(client, host, model, &messages, cancel, on_sentence).await
}

/// Stream an Ollama `/api/chat` reply for a full message list (system,
/// history, new user message), calling `on_sentence` per speakable segment.
///
/// Uses the default [`StreamOptions`] and ignores raw tokens. See
/// [`stream_chat`] for the full form.
pub async fn stream_chat_messages_cancellable<F, Fut>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    messages: &[ChatMessage],
    cancel: CancellationToken,
    on_sentence: F,
) -> Result<StopReason, VoxError>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    stream_chat(
        client,
        host,
        model,
        messages,
        &StreamOptions::default(),
        cancel,
        |_| {},
        on_sentence,
    )
    .await
}

/// One parsed NDJSON line of the Ollama stream.
struct Delta {
    content: String,
    done: bool,
}

fn parse_line(line: &[u8], model: &str, messages: usize) -> Result<Option<Delta>, VoxError> {
    if line.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(None);
    }
    let chunk: OllamaChatChunk = match serde_json::from_slice(line) {
        Ok(c) => c,
        Err(_) => {
            let done = serde_json::from_slice::<serde_json::Value>(line)
                .ok()
                .and_then(|raw| raw["done"].as_bool())
                .unwrap_or(false);
            if !done {
                tracing::debug!(
                    line = ?String::from_utf8_lossy(line),
                    "streaming_chat: skipping malformed NDJSON line"
                );
            }
            return Ok(done.then(|| Delta {
                content: String::new(),
                done: true,
            }));
        }
    };
    if let Some(err) = chunk.error {
        return Err(VoxError::Pipeline(format!("Ollama error: {err}")));
    }
    if chunk.done {
        tracing::info!(
            model,
            messages,
            prompt_eval_count = ?chunk.prompt_eval_count,
            eval_count = ?chunk.eval_count,
            "llm turn finished"
        );
    }
    Ok(Some(Delta {
        content: chunk.message.map(|m| m.content).unwrap_or_default(),
        done: chunk.done,
    }))
}

/// Hand segments to `on_sentence`. Returns `true` if cancelled meanwhile.
async fn deliver<F, Fut>(
    segments: Vec<String>,
    cancel: &CancellationToken,
    on_sentence: &mut F,
) -> Result<bool, VoxError>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    for segment in segments {
        if cancel.is_cancelled() {
            return Ok(true);
        }
        let fut = on_sentence(&segment);
        tokio::select! {
            _ = cancel.cancelled() => return Ok(true),
            r = fut => r?,
        }
    }
    Ok(false)
}

/// Stream an Ollama `/api/chat` reply.
///
/// - `on_token` receives every raw content token (including markdown and
///   `<think>` text) for display.
/// - `on_sentence` receives sanitized, speakable segments. Time spent
///   awaiting it does not count against the timeouts.
///
/// Fails with [`VoxError::Pipeline`] when no response data arrives within
/// `options.first_token_timeout`, or when the running stream is silent for
/// longer than `options.idle_timeout`.
///
/// All vox requests share `num_ctx`/`keep_alive` from [`crate::conversation`]
/// so they hit the same loaded model and can reuse its KV cache for an
/// unchanged message prefix.
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat<T, F, Fut>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    messages: &[ChatMessage],
    options: &StreamOptions,
    cancel: CancellationToken,
    mut on_token: T,
    mut on_sentence: F,
) -> Result<StopReason, VoxError>
where
    T: FnMut(&str) + Send,
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "keep_alive": crate::conversation::OLLAMA_KEEP_ALIVE,
        "options": crate::conversation::ollama_options(),
    });
    let (request, egress) = crate::privacy::ollama_http_request(
        client,
        reqwest::Method::POST,
        host,
        "/api/chat",
        "llm chat",
        crate::privacy::json_len(&body),
    )
    .map_err(|e| VoxError::Pipeline(e.to_string()))?;

    let first_token_timeout = options.first_token_timeout;
    let idle_timeout = options.idle_timeout;
    let first_deadline = tokio::time::Instant::now() + first_token_timeout;
    let no_start = move || {
        VoxError::Pipeline(format!(
            "Ollama did not start responding within {:.1}s",
            first_token_timeout.as_secs_f32()
        ))
    };

    let response = tokio::select! {
        _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
        r = tokio::time::timeout_at(first_deadline, request.json(&body).send()) => {
            r.map_err(|_| no_start())?
                .map_err(|e| VoxError::Pipeline(format!("Ollama request failed: {e}")))?
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        let detail = detail.trim();
        return Err(VoxError::Pipeline(if detail.is_empty() {
            format!("Ollama returned HTTP {status}")
        } else {
            format!("Ollama returned HTTP {status}: {detail}")
        }));
    }

    let mut stream = response.bytes_stream();
    let mut segmenter = Segmenter::new();
    let mut line_buffer: Vec<u8> = Vec::new();
    let mut started = false;
    let mut eof = false;
    let mut finished = false;

    while !finished && !eof {
        let deadline = if started {
            tokio::time::Instant::now() + idle_timeout
        } else {
            first_deadline
        };
        tokio::select! {
            // Data that is already there wins over an expired timer, so a
            // caller that was busy (e.g. playing audio) does not turn a
            // healthy stream into a timeout.
            biased;
            _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
            maybe = stream.next() => match maybe {
                Some(chunk) => {
                    let chunk =
                        chunk.map_err(|e| VoxError::Pipeline(format!("Stream error: {e}")))?;
                    started |= !chunk.is_empty();
                    egress.add_received(chunk.len() as u64);
                    line_buffer.extend_from_slice(&chunk);
                }
                None => eof = true,
            },
            _ = tokio::time::sleep_until(deadline) => {
                return Err(if started {
                    VoxError::Pipeline(format!(
                        "Ollama stream stalled: no data for {:.1}s",
                        idle_timeout.as_secs_f32()
                    ))
                } else {
                    no_start()
                });
            }
        }

        loop {
            let line: Vec<u8> = match line_buffer.iter().position(|&b| b == b'\n') {
                Some(pos) => line_buffer.drain(..=pos).collect(),
                // A final line without a trailing newline.
                None if eof && !line_buffer.is_empty() => std::mem::take(&mut line_buffer),
                None => break,
            };
            let Some(delta) = parse_line(&line, model, messages.len())? else {
                continue;
            };
            if !delta.content.is_empty() {
                on_token(&delta.content);
                let segments = segmenter.push(&delta.content);
                if deliver(segments, &cancel, &mut on_sentence).await? {
                    return Ok(StopReason::Cancelled);
                }
            }
            if delta.done {
                finished = true;
                break;
            }
        }
    }

    if deliver(segmenter.flush(), &cancel, &mut on_sentence).await? {
        return Ok(StopReason::Cancelled);
    }

    Ok(StopReason::Finished)
}

/// A multi-turn chat with one Ollama model: keeps the [`Conversation`] and
/// records each turn after it was spoken.
pub struct ChatSession {
    client: reqwest::Client,
    host: String,
    model: String,
    system_prompt: Option<String>,
    conversation: Conversation,
    options: StreamOptions,
}

impl ChatSession {
    /// New session with an empty history sized for the shared Ollama context
    /// window.
    pub fn new(
        client: reqwest::Client,
        host: impl Into<String>,
        model: impl Into<String>,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            client,
            host: host.into(),
            model: model.into(),
            system_prompt,
            conversation: Conversation::for_num_ctx(crate::conversation::ollama_num_ctx()),
            options: StreamOptions::default(),
        }
    }

    pub fn with_options(mut self, options: StreamOptions) -> Self {
        self.options = options;
        self
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    pub fn conversation_mut(&mut self) -> &mut Conversation {
        &mut self.conversation
    }

    /// Run one turn: send the history plus `user_text`, stream the reply, and
    /// record the turn.
    ///
    /// Only segments for which `on_sentence` returned `Ok` are recorded. A
    /// turn that was cancelled, or that failed after part of it was delivered,
    /// is recorded with the interruption marker; a turn that failed before
    /// anything was delivered is not recorded.
    pub async fn turn<T, F, Fut>(
        &mut self,
        user_text: &str,
        cancel: CancellationToken,
        on_token: T,
        mut on_sentence: F,
    ) -> Result<StopReason, VoxError>
    where
        T: FnMut(&str) + Send,
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
    {
        let messages = self
            .conversation
            .messages_for(self.system_prompt.as_deref(), user_text);
        let delivered: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let delivered_ref = &delivered;

        let result = stream_chat(
            &self.client,
            &self.host,
            &self.model,
            &messages,
            &self.options,
            cancel,
            on_token,
            |segment: &str| {
                let fut = on_sentence(segment);
                let segment = segment.to_string();
                async move {
                    fut.await?;
                    delivered_ref
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(segment);
                    Ok(())
                }
            },
        )
        .await;

        let delivered = delivered.into_inner().unwrap_or_else(|e| e.into_inner());
        match &result {
            Ok(StopReason::Finished) => {
                self.conversation.commit(user_text, &delivered, false);
            }
            Ok(StopReason::Cancelled) => {
                self.conversation.commit(user_text, &delivered, true);
            }
            Err(_) if !delivered.is_empty() => {
                self.conversation.commit(user_text, &delivered, true);
            }
            Err(_) => {}
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// What the fake Ollama does with one connection.
    #[derive(Clone, Default)]
    struct Script {
        /// NDJSON lines to send, one write each.
        lines: Vec<String>,
        /// Sent last, without a trailing newline.
        tail: String,
        /// Keep the connection open and silent after the lines.
        stall: bool,
        /// Accept the request but never answer.
        silent: bool,
    }

    fn token(content: &str) -> String {
        serde_json::json!({
            "message": { "role": "assistant", "content": content },
            "done": false,
        })
        .to_string()
    }

    fn done() -> String {
        serde_json::json!({ "done": true, "prompt_eval_count": 5, "eval_count": 7 }).to_string()
    }

    fn reply(tokens: &[&str]) -> Script {
        let mut lines: Vec<String> = tokens.iter().map(|t| token(t)).collect();
        lines.push(done());
        Script {
            lines,
            ..Script::default()
        }
    }

    async fn read_request(sock: &mut tokio::net::TcpStream) -> serde_json::Value {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            if let Some(head_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
                let len: usize = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let body_start = head_end + 4;
                if buf.len() >= body_start + len {
                    return serde_json::from_slice(&buf[body_start..body_start + len])
                        .unwrap_or(serde_json::Value::Null);
                }
            }
            let n = sock.read(&mut tmp).await.expect("read request");
            assert!(n > 0, "client closed before sending a full request");
            buf.extend_from_slice(&tmp[..n]);
        }
    }

    /// Fake Ollama: serves one script per connection, in order. Returns its
    /// `host:port` and the request bodies it received.
    async fn fake_ollama(scripts: Vec<Script>) -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let host = listener.local_addr().unwrap().to_string();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        tokio::spawn(async move {
            for script in scripts {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let seen = Arc::clone(&seen);
                tokio::spawn(async move {
                    let request = read_request(&mut sock).await;
                    seen.lock().unwrap().push(request);
                    if script.silent {
                        tokio::time::sleep(Duration::from_secs(30)).await;
                        return;
                    }
                    let head = "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\nconnection: close\r\n\r\n";
                    let _ = sock.write_all(head.as_bytes()).await;
                    for line in &script.lines {
                        let _ = sock.write_all(format!("{line}\n").as_bytes()).await;
                        let _ = sock.flush().await;
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    let _ = sock.write_all(script.tail.as_bytes()).await;
                    if script.stall {
                        tokio::time::sleep(Duration::from_secs(30)).await;
                    }
                    let _ = sock.shutdown().await;
                });
            }
        });
        (host, requests)
    }

    fn short_timeouts() -> StreamOptions {
        StreamOptions {
            first_token_timeout: Duration::from_millis(300),
            idle_timeout: Duration::from_millis(300),
        }
    }

    /// Run `stream_chat` against `host`, returning the result, the tokens and
    /// the segments.
    async fn run(
        host: &str,
        options: &StreamOptions,
    ) -> (Result<StopReason, VoxError>, Vec<String>, Vec<String>) {
        let client = reqwest::Client::new();
        let messages = vec![ChatMessage::user("hi")];
        let mut tokens = Vec::new();
        let segments = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&segments);
        let result = stream_chat(
            &client,
            host,
            "test-model",
            &messages,
            options,
            CancellationToken::new(),
            |t| tokens.push(t.to_string()),
            move |s: &str| {
                sink.lock().unwrap().push(s.to_string());
                async { Ok(()) }
            },
        )
        .await;
        let segments = segments.lock().unwrap().clone();
        (result, tokens, segments)
    }

    #[tokio::test]
    async fn on_token_is_called_for_every_token() {
        let sent = ["Hello", " there", ".", " How", " are", " you", "?"];
        let (host, requests) = fake_ollama(vec![reply(&sent)]).await;

        let (result, tokens, segments) = run(&host, &StreamOptions::default()).await;

        assert_eq!(result.unwrap(), StopReason::Finished);
        assert_eq!(tokens, sent);
        assert_eq!(segments, vec!["Hello there.", "How are you?"]);
        let request = requests.lock().unwrap()[0].clone();
        assert_eq!(request["stream"], true);
        assert_eq!(request["model"], "test-model");
    }

    #[tokio::test]
    async fn think_block_and_markdown_reach_tokens_but_not_tts() {
        let sent = [
            "<think>",
            "The user said hi. ",
            "Be brief.",
            "</think>",
            "**Hi**",
            " there!",
            " See https://example.com/docs",
            " for more.",
        ];
        let (host, _) = fake_ollama(vec![reply(&sent)]).await;

        let (result, tokens, segments) = run(&host, &StreamOptions::default()).await;

        assert_eq!(result.unwrap(), StopReason::Finished);
        assert_eq!(tokens.concat(), sent.concat());
        assert_eq!(segments, vec!["Hi there!", "See for more."]);
    }

    #[tokio::test]
    async fn final_line_without_newline_is_processed() {
        // No `done` line and no trailing newline: the stream just ends.
        let (host, _) = fake_ollama(vec![Script {
            lines: vec![token("Bye.")],
            tail: token(" Now"),
            ..Script::default()
        }])
        .await;
        let (result, tokens, segments) = run(&host, &StreamOptions::default()).await;
        assert_eq!(result.unwrap(), StopReason::Finished);
        assert_eq!(tokens, vec!["Bye.", " Now"]);
        assert_eq!(segments, vec!["Bye.", "Now"]);
    }

    #[tokio::test]
    async fn stalled_stream_times_out() {
        let (host, _) = fake_ollama(vec![Script {
            lines: vec![token("One sentence."), token(" And then")],
            stall: true,
            ..Script::default()
        }])
        .await;

        let start = Instant::now();
        let (result, tokens, segments) = run(&host, &short_timeouts()).await;

        let err = result.expect_err("a stalled stream must fail").to_string();
        assert!(err.contains("stalled"), "unexpected error: {err}");
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_eq!(tokens.len(), 2);
        assert_eq!(segments, vec!["One sentence."]);
    }

    #[tokio::test]
    async fn server_that_never_answers_times_out() {
        let (host, _) = fake_ollama(vec![Script {
            silent: true,
            ..Script::default()
        }])
        .await;

        let start = Instant::now();
        let (result, tokens, segments) = run(&host, &short_timeouts()).await;

        let err = result.expect_err("a silent server must fail").to_string();
        assert!(err.contains("did not start"), "unexpected error: {err}");
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(tokens.is_empty());
        assert!(segments.is_empty());
    }

    #[tokio::test]
    async fn slow_on_sentence_does_not_trip_the_idle_timeout() {
        let (host, _) = fake_ollama(vec![reply(&["First.", " Second."])]).await;
        let client = reqwest::Client::new();
        let messages = vec![ChatMessage::user("hi")];
        let options = StreamOptions {
            first_token_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_millis(100),
        };
        let result = stream_chat(
            &client,
            &host,
            "m",
            &messages,
            &options,
            CancellationToken::new(),
            |_| {},
            |_s: &str| async {
                tokio::time::sleep(Duration::from_millis(250)).await;
                Ok(())
            },
        )
        .await;
        assert_eq!(result.unwrap(), StopReason::Finished);
    }

    #[tokio::test]
    async fn ollama_error_line_is_returned() {
        let (host, _) = fake_ollama(vec![Script {
            lines: vec![serde_json::json!({ "error": "model not found" }).to_string()],
            ..Script::default()
        }])
        .await;
        let (result, _, _) = run(&host, &StreamOptions::default()).await;
        let err = result.unwrap_err().to_string();
        assert!(err.contains("model not found"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn chat_session_sends_previous_turn_on_second_turn() {
        let (host, requests) = fake_ollama(vec![
            reply(&["**Paris**", " is the capital.", " 🎉"]),
            reply(&["About two million."]),
        ])
        .await;
        let mut session = ChatSession::new(
            reqwest::Client::new(),
            host,
            "test-model",
            Some("You are a voice assistant.".to_string()),
        );

        for question in [
            "What is the capital of France?",
            "How many people live there?",
        ] {
            let stop = session
                .turn(
                    question,
                    CancellationToken::new(),
                    |_| {},
                    |_s: &str| async { Ok(()) },
                )
                .await
                .unwrap();
            assert_eq!(stop, StopReason::Finished);
        }

        let requests = requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0]["messages"],
            serde_json::json!([
                { "role": "system", "content": "You are a voice assistant." },
                { "role": "user", "content": "What is the capital of France?" },
            ])
        );
        // Turn two carries turn one, with the assistant text exactly as it
        // was handed to TTS (no markdown, no emoji).
        assert_eq!(
            requests[1]["messages"],
            serde_json::json!([
                { "role": "system", "content": "You are a voice assistant." },
                { "role": "user", "content": "What is the capital of France?" },
                { "role": "assistant", "content": "Paris is the capital." },
                { "role": "user", "content": "How many people live there?" },
            ])
        );
        // Same options on every request, or Ollama reloads the model.
        assert_eq!(requests[0]["options"], requests[1]["options"]);
        assert_eq!(session.conversation().len(), 2);
    }

    #[tokio::test]
    async fn chat_session_records_only_delivered_segments() {
        let (host, _) = fake_ollama(vec![
            Script {
                lines: vec![token("Heard this."), token(" Never")],
                stall: true,
                ..Script::default()
            },
            Script {
                silent: true,
                ..Script::default()
            },
        ])
        .await;
        let mut session = ChatSession::new(reqwest::Client::new(), host, "m", None)
            .with_options(short_timeouts());
        let ok = |_s: &str| async { Ok(()) };

        // Fails after one segment was spoken: recorded as interrupted.
        assert!(
            session
                .turn("first", CancellationToken::new(), |_| {}, ok)
                .await
                .is_err()
        );
        assert_eq!(session.conversation().len(), 1);
        let messages = session.conversation().messages_for(None, "x");
        assert_eq!(
            messages[1].content,
            format!("Heard this. {}", crate::conversation::INTERRUPTED_MARKER)
        );

        // Fails before anything was spoken: not recorded.
        assert!(
            session
                .turn("second", CancellationToken::new(), |_| {}, ok)
                .await
                .is_err()
        );
        assert_eq!(session.conversation().len(), 1);
    }
}
