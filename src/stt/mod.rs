//! Speech-to-Text backends.
//!
//! The default backend is Whisper (behind the `whisper` feature flag),
//! using `whisper-rs` bindings to `whisper.cpp`.

#[cfg(feature = "whisper")]
mod whisper;

#[cfg(feature = "whisper")]
pub use self::whisper::{WhisperBackend, WhisperConfig, WhisperModel};
