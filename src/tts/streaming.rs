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
//!
//! Splitting and text clean-up are done by the shared segmenter in
//! [`super::segment`], the same one the streaming LLM path uses.

use std::sync::Arc;

use super::segment::split_sentences;
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

#[cfg(test)]
mod tests {
    use super::split_sentences;

    // The splitter itself is tested in `tts::segment`; these cover what the
    // adapter relies on.

    #[test]
    fn splits_on_sentence_ends() {
        assert_eq!(
            split_sentences("Hello! How are you? Fine."),
            vec!["Hello!", "How are you?", "Fine."]
        );
    }

    #[test]
    fn empty_text_has_no_segments() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn abbreviations_do_not_split() {
        assert_eq!(
            split_sentences("Dr. Smith said hello."),
            vec!["Dr. Smith said hello."]
        );
        assert_eq!(
            split_sentences("Bring fruit, e.g. apples. 1. First wash them."),
            vec!["Bring fruit, e.g. apples.", "1. First wash them."]
        );
    }

    #[test]
    fn markup_is_not_synthesized() {
        assert_eq!(
            split_sentences("<think>plan</think>**Hi** there."),
            vec!["Hi there."]
        );
    }
}
