//! Backend trait definitions for the pluggable pipeline.
//!
//! Vox uses a trait-based backend system inspired by `embedded-hal`.
//! Implement these traits to plug in custom VAD, STT, or TTS engines.
//!
//! All backends require `Send + Sync`, so they can be shared across
//! threads via `Arc`. STT and TTS backends use `&self` (not `&mut self`),
//! making them safe to call concurrently from multiple tasks.

use async_trait::async_trait;

use crate::error::VoxError;
use crate::types::{AudioChunk, SttResult, TtsOutput, TtsRequest, Utterance};

/// Voice Activity Detection backend.
///
/// Processes audio frames and emits [`VadEvent`]s when speech
/// starts or ends.
#[async_trait]
pub trait VadBackend: Send + Sync {
    /// Process a single audio frame and return VAD events.
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError>;

    /// Reset the internal VAD state.
    fn reset(&mut self);

    /// Expected frame size in samples.
    fn frame_size(&self) -> usize;

    /// Expected sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Return a copy of the current speech audio buffer (if speech is in progress).
    fn current_speech_buffer(&self) -> Option<AudioChunk> {
        None
    }
}

/// Speech-to-Text backend.
///
/// Transcribes a complete [`Utterance`] into text. Takes `&self`,
/// so a single backend instance can serve concurrent requests
/// when wrapped in `Arc`.
#[async_trait]
pub trait SttBackend: Send + Sync {
    /// Transcribe an audio utterance to text.
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError>;
}

/// Text-to-Speech backend.
///
/// Synthesizes text into audio samples. Thread-safe by design:
/// wrap in `Arc<dyn TtsBackend>` and call from multiple tokio tasks
/// concurrently. Backends use `spawn_blocking` internally for
/// CPU-bound inference, keeping the async runtime responsive.
#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize text to audio.
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError>;

    /// List all voices supported by this backend.
    fn list_voices(&self) -> Vec<crate::types::VoiceInfo> {
        vec![]
    }

    /// The backend name (e.g. "kokoro", "pocket", "chatterbox").
    fn backend_name(&self) -> &str {
        "unknown"
    }
}

/// Events emitted by a [`VadBackend`].
#[derive(Debug)]
pub enum VadEvent {
    /// Speech has started.
    SpeechStart,
    /// Speech has ended; contains the complete utterance.
    SpeechEnd(Utterance),
    /// No speech detected in this frame.
    Silence,
}

/// A streaming STT session that accepts audio incrementally.
///
/// Created by [`StreamingSttBackend::create_session`]. Push audio
/// chunks as they arrive; each push may return updated partial text.
/// Call [`finish`](SttSession::finish) when speech ends to get the final result.
pub trait SttSession: Send {
    /// Push audio samples and get optional updated partial text.
    ///
    /// Returns `Ok(Some(text))` when partial text has changed,
    /// `Ok(None)` when nothing new is available yet.
    fn push_audio(&mut self, samples: &[f32], sample_rate: u32)
    -> Result<Option<String>, VoxError>;

    /// Signal that no more audio will arrive and get the final result.
    fn finish(&mut self) -> Result<SttResult, VoxError>;
}

/// Streaming Speech-to-Text backend.
///
/// Creates [`SttSession`]s for incremental audio processing.
/// Unlike [`SttBackend`] which transcribes complete utterances,
/// this processes audio chunk-by-chunk with partial results.
pub trait StreamingSttBackend: Send + Sync {
    /// Create a new streaming session.
    fn create_session(&self) -> Result<Box<dyn SttSession>, VoxError>;
}

/// A chunk of audio from a streaming TTS session.
pub struct TtsChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub progress: f32, // 0.0 to 1.0
}

/// A streaming TTS session that produces audio in chunks.
///
/// Created by [`StreamingTtsBackend::create_tts_session`].
/// Call [`pull_chunk`](TtsSession::pull_chunk) repeatedly until it returns `None`.
pub trait TtsSession: Send {
    /// Pull the next audio chunk. Returns `None` when synthesis is complete.
    fn pull_chunk(&mut self) -> Result<Option<TtsChunk>, VoxError>;
}

/// Streaming Text-to-Speech backend.
///
/// Creates [`TtsSession`]s that produce audio incrementally.
pub trait StreamingTtsBackend: Send + Sync {
    /// Create a streaming session for the given text.
    fn create_tts_session(&self, request: &TtsRequest) -> Result<Box<dyn TtsSession>, VoxError>;
}
