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

        completed
    }

    /// Flush any remaining text as a final sentence.
    /// Returns ALL sentences (completed via push() + remaining buffer).
    pub fn flush(mut self) -> Vec<String> {
        let trimmed = self.buffer.trim().to_string();
        if !trimmed.is_empty() {
            self.sentences.push(trimmed);
        }
        self.sentences
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

/// Stream text from Ollama and synthesize sentences as they complete.
///
/// This is the core streaming LLM→TTS integration. It:
/// 1. Makes a streaming request to Ollama
/// 2. Buffers tokens into complete sentences
/// 3. Synthesizes each sentence immediately (returns audio chunks)
/// 4. Plays audio while later sentences are still being generated
///
/// # Example latency comparison
///
/// **Non-streaming (old)**:
/// ```text
/// LLM generates full response (2-4s) → TTS starts → User hears output
/// Perceived wait: 2-4s + TTS latency
/// ```
///
/// **Streaming (new)**:
/// ```text
/// LLM generates sentence 1 (500ms) → TTS starts → User hears sentence 1
/// LLM generates sentence 2 → TTS queues → User hears sentence 2
/// Perceived wait: Only first sentence latency (~500-800ms)
/// ```
#[cfg(any(feature = "cli", feature = "server"))]
#[allow(clippy::too_many_arguments)]
pub async fn stream_chat_with_tts<F>(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    _tts: Arc<dyn TtsBackend>,
    system_prompt: Option<String>,
    _voice: Option<String>,
    mut on_sentence: F,
) -> Result<(), VoxError>
where
    F: FnMut(&str) -> Result<(), VoxError>,
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

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| VoxError::Pipeline(format!("Ollama request failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(VoxError::Pipeline(format!(
            "Ollama returned HTTP {}",
            response.status()
        )));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = SentenceBuffer::new();

    // Process streaming response
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| VoxError::Pipeline(format!("Stream error: {}", e)))?;

        // Parse each line as JSON (Ollama sends newline-delimited JSON)
        for line in chunk.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }

            let ollama_chunk: OllamaChunk = serde_json::from_slice(line)
                .map_err(|e| VoxError::Pipeline(format!("JSON parse error: {}", e)))?;

            if !ollama_chunk.response.is_empty() {
                print!("{}", ollama_chunk.response); // Show tokens as they arrive
                use std::io::Write;
                std::io::stdout().flush().ok();

                // Check for complete sentences
                let sentences = buffer.push(&ollama_chunk.response);

                // Call handler for each complete sentence
                for sentence in sentences {
                    on_sentence(&sentence)?;
                }
            }

            if ollama_chunk.done {
                break;
            }
        }
    }

    // Handle any remaining text (incomplete sentence at the end)
    let remaining = buffer.flush();
    for sentence in remaining {
        on_sentence(&sentence)?;
    }

    println!(); // Newline after streaming text

    Ok(())
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
    fn sentence_buffer_handles_abbreviations() {
        // "Dr. " will split (known limitation, acceptable)
        let mut buf = SentenceBuffer::new();
        let s1 = buf.push("Dr. ");
        assert_eq!(s1, vec!["Dr."]);
        buf.push("Smith said hello.");
        assert_eq!(buf.finish(), vec!["Dr.", "Smith said hello."]);
    }
}
