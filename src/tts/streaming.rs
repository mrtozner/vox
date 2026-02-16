//! Sentence-level streaming adapter for any TTS backend.
//!
//! Wraps a batch [`TtsBackend`] to provide incremental audio output by splitting
//! text into sentences and synthesizing each one independently. The first sentence
//! can be played while later sentences are still being synthesized.
//!
//! ## Known limitations
//!
//! - **Prosody breaks at boundaries**: Each sentence is synthesized independently,
//!   so cross-sentence intonation context is lost.
//! - **Abbreviations**: "Dr. Smith" splits at the period because it's followed by
//!   a space. Over-splitting is acceptable — both chunks synthesize fine.

use std::sync::Arc;

use crate::error::VoxError;
use crate::traits::{StreamingTtsBackend, TtsBackend, TtsChunk, TtsSession};
use crate::types::TtsRequest;

/// Wraps any [`TtsBackend`] to provide sentence-level streaming.
///
/// Splits input text into sentences, synthesizes each one sequentially,
/// and yields audio chunks as they complete. The first sentence can be
/// played while later sentences are still being synthesized.
pub struct SentenceStreamingAdapter {
    backend: Arc<dyn TtsBackend>,
    runtime: tokio::runtime::Handle,
}

impl SentenceStreamingAdapter {
    pub fn new(backend: Arc<dyn TtsBackend>, runtime: tokio::runtime::Handle) -> Self {
        Self { backend, runtime }
    }
}

impl StreamingTtsBackend for SentenceStreamingAdapter {
    fn create_tts_session(&self, request: &TtsRequest) -> Result<Box<dyn TtsSession>, VoxError> {
        let sentences = split_sentences(&request.text);
        if sentences.is_empty() {
            return Err(VoxError::Tts("empty text".into()));
        }

        let total = sentences.len();
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<TtsChunk, VoxError>>(2);
        let backend = Arc::clone(&self.backend);
        let voice = request.voice.clone();
        let seed = request.seed;

        let handle = self.runtime.clone();
        // Must be std::thread::spawn, NOT tokio::task::spawn_blocking.
        // We call handle.block_on() inside, which panics if called from
        // within a tokio runtime context (spawn_blocking runs on tokio's
        // blocking pool, which is still inside the runtime).
        std::thread::spawn(move || {
            for (i, sentence) in sentences.into_iter().enumerate() {
                let req = TtsRequest {
                    text: sentence,
                    voice: voice.clone(),
                    seed,
                };
                let result = handle.block_on(backend.synthesize(&req));
                let chunk = match result {
                    Ok(output) => Ok(TtsChunk {
                        samples: output.audio.samples,
                        sample_rate: output.audio.sample_rate,
                        progress: (i + 1) as f32 / total as f32,
                    }),
                    Err(e) => Err(e),
                };
                if tx.send(chunk).is_err() {
                    break; // receiver dropped, stop synthesizing
                }
            }
            // tx drops here, signaling completion
        });

        Ok(Box::new(SentenceStreamingSession { rx }))
    }
}

struct SentenceStreamingSession {
    rx: std::sync::mpsc::Receiver<Result<TtsChunk, VoxError>>,
}

impl TtsSession for SentenceStreamingSession {
    fn pull_chunk(&mut self) -> Result<Option<TtsChunk>, VoxError> {
        match self.rx.recv() {
            Ok(Ok(chunk)) => Ok(Some(chunk)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None), // sender dropped = done
        }
    }
}

/// Split text into sentences at sentence-ending punctuation followed by whitespace.
///
/// Splits on `.!?;` only when the next character is whitespace or end-of-string.
/// This preserves decimals like "3.50" but will split abbreviations like "Dr. Smith"
/// (known limitation — over-splitting is acceptable for TTS).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | ';') {
            let next_is_boundary = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
            if next_is_boundary {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::split_sentences;

    #[test]
    fn splits_on_period() {
        assert_eq!(split_sentences("Hello. World."), vec!["Hello.", "World."]);
    }

    #[test]
    fn splits_on_mixed_punctuation() {
        assert_eq!(
            split_sentences("Hello! How are you? Fine."),
            vec!["Hello!", "How are you?", "Fine."]
        );
    }

    #[test]
    fn handles_no_punctuation() {
        assert_eq!(split_sentences("Hello world"), vec!["Hello world"]);
    }

    #[test]
    fn handles_empty_string() {
        assert!(split_sentences("").is_empty());
    }

    #[test]
    fn handles_whitespace_only() {
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn preserves_decimals() {
        // "3.50" — period NOT followed by whitespace, stays together
        assert_eq!(
            split_sentences("The price is 3.50 dollars."),
            vec!["The price is 3.50 dollars."]
        );
    }

    #[test]
    fn abbreviations_followed_by_space_split() {
        // Known limitation: "Dr. " splits because period is followed by space.
        // Over-splitting is acceptable — both chunks synthesize fine.
        assert_eq!(
            split_sentences("Dr. Smith said hello."),
            vec!["Dr.", "Smith said hello."]
        );
    }

    #[test]
    fn handles_trailing_text_without_punctuation() {
        assert_eq!(split_sentences("Hello. World"), vec!["Hello.", "World"]);
    }
}
