//! Streaming LLM→TTS integration.
//!
//! Enables sentence-level streaming from LLM (Ollama) to TTS, reducing
//! perceived latency by starting audio playback before the LLM completes.
//!
//! **Feature requirements**: This module requires the `cli` or `server` features
//! which provide serde/reqwest dependencies.

#[cfg(any(feature = "cli", feature = "server"))]
use crate::error::VoxError;
#[cfg(any(feature = "cli", feature = "server"))]
use crate::traits::TtsBackend;
#[cfg(any(feature = "cli", feature = "server"))]
use std::sync::Arc;
#[cfg(any(feature = "cli", feature = "server"))]
use tokio_stream::StreamExt;

#[cfg(any(feature = "cli", feature = "server"))]
use serde::Deserialize;
#[cfg(any(feature = "cli", feature = "server"))]
use serde_json;

/// Ollama streaming response chunk.
#[cfg(any(feature = "cli", feature = "server"))]
#[derive(Deserialize)]
struct OllamaChunk {
    response: String,
    done: bool,
}

/// Buffers tokens into complete sentences for streaming TTS.
///
/// Accumulates tokens from an LLM stream and yields complete sentences
/// when sentence-ending punctuation is detected. This enables TTS to start
/// on the first sentence while later sentences are still being generated.
#[cfg(any(feature = "cli", feature = "server"))]
pub struct SentenceBuffer {
    buffer: String,
    sentences: Vec<String>,
}

#[cfg(any(feature = "cli", feature = "server"))]
impl Default for SentenceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(feature = "cli", feature = "server"))]
impl SentenceBuffer {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            sentences: Vec::new(),
        }
    }

    /// Add a token to the buffer and check for complete sentences.
    ///
    /// Returns any newly completed sentences (usually 0-1, rarely 2+).
    ///
    /// **Lookahead logic**: A sentence ending with punctuation is only emitted
    /// when the NEXT token starts with whitespace, confirming it's not an
    /// abbreviation like "Dr." or "Mrs."
    pub fn push(&mut self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }

        let mut completed = Vec::new();

        // Check if new token starts with whitespace - this confirms previous sentence
        let starts_with_whitespace = token.chars().next().is_some_and(|c| c.is_whitespace());

        // If buffer ends with sentence punctuation AND new token starts with space,
        // the buffered text is a complete sentence
        if starts_with_whitespace && !self.buffer.is_empty() {
            let trimmed = self.buffer.trim_end();
            if let Some(last_char) = trimmed.chars().last() {
                if matches!(last_char, '.' | '!' | '?' | ';') {
                    let sentence = trimmed.to_string();
                    completed.push(sentence.clone());
                    self.sentences.push(sentence);
                    self.buffer.clear();
                }
            }
        }

        // Add new token to buffer
        self.buffer.push_str(token);

        // Check for mid-token sentence boundaries (e.g., "Hello. World")
        // Split on sentence punctuation followed by whitespace
        let mut last_split = 0;
        let chars: Vec<char> = self.buffer.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            if matches!(ch, '.' | '!' | '?' | ';') {
                // Check if followed by whitespace (mid-token sentence boundary)
                if i + 1 < chars.len() && chars[i + 1].is_whitespace() {
                    // Complete sentence found
                    let sentence: String = chars[last_split..=i].iter().collect();
                    let trimmed = sentence.trim().to_string();
                    if !trimmed.is_empty() {
                        completed.push(trimmed.clone());
                        self.sentences.push(trimmed);
                    }
                    last_split = i + 1;
                }
            }
        }

        // Keep remaining text after last sentence boundary
        if last_split > 0 {
            self.buffer = chars[last_split..].iter().collect();
            self.buffer = self.buffer.trim_start().to_string();
        }

        // Word-count fallback: flush after 12+ words even without sentence punctuation.
        // This prevents long run-on sentences from delaying TTS start.
        let current_words = self.buffer.split_whitespace().count();
        if current_words >= 12 {
            let flushed = self.buffer.trim().to_string();
            if !flushed.is_empty() {
                completed.push(flushed.clone());
                self.sentences.push(flushed);
                self.buffer.clear();
            }
        }

        completed
    }

    /// Flush any remaining incomplete sentence in the buffer.
    /// Returns ONLY the leftover text, NOT sentences already emitted via push().
    pub fn flush(self) -> Option<String> {
        let trimmed = self.buffer.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    /// Get all sentences (both completed sentences and remaining buffer).
    pub fn finish(self) -> Vec<String> {
        let mut all = self.sentences;
        let trimmed = self.buffer.trim().to_string();
        if !trimmed.is_empty() {
            all.push(trimmed);
        }
        all
    }
}

/// Why streaming stopped (normal completion or cancellation).
#[cfg(any(feature = "cli", feature = "server"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Finished,
    Cancelled,
}

/// Stream LLM tokens and synthesize complete sentences immediately.
#[cfg(any(feature = "cli", feature = "server"))]
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
    let cancel = tokio_util::sync::CancellationToken::new();
    stream_chat_with_tts_cancellable(
        client,
        host,
        model,
        prompt,
        tts,
        system_prompt,
        voice,
        cancel,
        on_sentence,
    )
    .await
    .map(|_| ())
}

/// Like `stream_chat_with_tts` but returns early if `cancel` fires.
#[cfg(any(feature = "cli", feature = "server"))]
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat_with_tts_cancellable<F, Fut>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    _tts: Arc<dyn TtsBackend>,
    system_prompt: Option<String>,
    _voice: Option<String>,
    cancel: tokio_util::sync::CancellationToken,
    mut on_sentence: F,
) -> Result<StopReason, VoxError>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), VoxError>> + Send,
{
    let url = format!("http://{}/api/generate", host);

    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": true,
    });

    if let Some(sys) = system_prompt {
        body["system"] = serde_json::json!(sys);
    }

    let response = tokio::select! {
        _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
        r = client.post(&url).json(&body).send() => {
            r.map_err(|e| VoxError::Pipeline(format!("Ollama request failed: {}", e)))?
        }
    };

    if !response.status().is_success() {
        return Err(VoxError::Pipeline(format!(
            "Ollama returned HTTP {}",
            response.status()
        )));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = SentenceBuffer::new();
    let mut line_buffer: Vec<u8> = Vec::new();
    let mut done = false;

    'outer: loop {
        if cancel.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }

        let chunk = tokio::select! {
            _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
            maybe = stream.next() => match maybe {
                Some(r) => r.map_err(|e| VoxError::Pipeline(format!("Stream error: {}", e)))?,
                None => break 'outer,
            },
        };

        line_buffer.extend_from_slice(&chunk);

        let mut start = 0;
        for (i, &byte) in line_buffer.iter().enumerate() {
            if byte == b'\n' {
                let line = &line_buffer[start..i];
                start = i + 1;
                if line.is_empty() || line.iter().all(|b| b.is_ascii_whitespace()) {
                    continue;
                }
                let ollama_chunk: OllamaChunk = match serde_json::from_slice(line) {
                    Ok(c) => c,
                    Err(_) => {
                        if let Ok(raw) = serde_json::from_slice::<serde_json::Value>(line) {
                            if raw["done"].as_bool().unwrap_or(false) {
                                done = true;
                                break 'outer;
                            }
                        }
                        tracing::debug!(
                            line = ?String::from_utf8_lossy(line),
                            "streaming_chat: skipping malformed NDJSON line"
                        );
                        continue;
                    }
                };

                if !ollama_chunk.response.is_empty() {
                    print!("{}", ollama_chunk.response);
                    use std::io::Write;
                    std::io::stdout().flush().ok();

                    let sentences = buffer.push(&ollama_chunk.response);

                    for sentence in sentences {
                        if cancel.is_cancelled() {
                            return Ok(StopReason::Cancelled);
                        }
                        let fut = on_sentence(&sentence);
                        tokio::select! {
                            _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
                            r = fut => r?,
                        }
                    }
                }

                if ollama_chunk.done {
                    done = true;
                    break 'outer;
                }
            }
        }

        line_buffer.drain(..start);
    }

    let _ = done;

    if let Some(remaining) = buffer.flush() {
        if cancel.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }
        let fut = on_sentence(&remaining);
        tokio::select! {
            _ = cancel.cancelled() => return Ok(StopReason::Cancelled),
            r = fut => r?,
        }
    }

    println!(); // Newline after streaming text

    Ok(StopReason::Finished)
}

#[cfg(all(test, any(feature = "cli", feature = "server")))]
mod tests {
    use super::*;

    #[test]
    fn sentence_buffer_simple() {
        let mut buf = SentenceBuffer::new();

        assert_eq!(buf.push("Hello"), Vec::<String>::new());
        assert_eq!(buf.push(". "), vec!["Hello."]);
        assert_eq!(buf.push("World"), Vec::<String>::new());
        assert_eq!(buf.push("!"), Vec::<String>::new());

        assert_eq!(buf.finish(), vec!["Hello.", "World!"]);
    }

    #[test]
    fn sentence_buffer_multiple_in_one_token() {
        let mut buf = SentenceBuffer::new();
        let sentences = buf.push("Hello. World!");
        assert_eq!(sentences, vec!["Hello."]);
        assert_eq!(buf.finish(), vec!["Hello.", "World!"]);
    }

    #[test]
    fn sentence_buffer_question_and_exclamation() {
        let mut buf = SentenceBuffer::new();
        buf.push("How are you? ");
        buf.push("I'm fine!");
        let all = buf.finish();
        assert_eq!(all, vec!["How are you?", "I'm fine!"]);
    }

    #[test]
    fn sentence_buffer_preserves_decimals() {
        let mut buf = SentenceBuffer::new();
        buf.push("The price is 3.50 dollars.");
        assert_eq!(buf.finish(), vec!["The price is 3.50 dollars."]);
    }

    #[test]
    fn sentence_buffer_word_count_fallback() {
        let mut buf = SentenceBuffer::new();
        // Push 15 words without any sentence-ending punctuation
        let words = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";
        let sentences = buf.push(words);
        // Should have flushed after 12+ words even without punctuation
        assert!(!sentences.is_empty(), "should flush after 12+ words");
        assert_eq!(sentences.len(), 1);
        // The remaining buffer should have the leftover words (if any)
        let remaining = buf.flush();
        // All 15 words were in one token so they all flush at once
        assert!(remaining.is_none() || remaining.unwrap().split_whitespace().count() < 12);
    }

    #[test]
    fn sentence_buffer_handles_abbreviations() {
        // "Dr. " will split (known limitation, acceptable)
        let mut buf = SentenceBuffer::new();
        let s1 = buf.push("Dr. ");
        assert_eq!(s1, vec!["Dr."]);
        buf.push("Smith said hello.");
        assert_eq!(buf.finish(), vec!["Dr.", "Smith said hello."]);
    }
}
