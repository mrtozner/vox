//! Error types for the Vox pipeline.

use std::path::PathBuf;

/// Errors that can occur in the Vox pipeline.
#[derive(thiserror::Error, Debug)]
pub enum VoxError {
    /// Audio device or capture error.
    #[error("audio device error: {0}")]
    Audio(String),

    /// Voice activity detection error.
    #[error("VAD error: {0}")]
    Vad(String),

    /// Speech-to-text error.
    #[error("STT error: {0}")]
    Stt(String),

    /// Text-to-speech error.
    #[error("TTS error: {0}")]
    Tts(String),

    /// No STT backend was configured.
    #[error("no STT backend configured")]
    NoStt,

    /// No VAD backend was configured.
    #[error("no VAD backend configured")]
    NoVad,

    /// A required model file was not found.
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),

    /// General pipeline error.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
