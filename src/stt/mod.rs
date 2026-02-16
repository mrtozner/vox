//! Speech-to-Text backends.
//!
//! The default backend is Whisper (behind the `whisper` feature flag),
//! using `whisper-rs` bindings to `whisper.cpp`.

#[cfg(feature = "whisper")]
mod whisper;

#[cfg(feature = "whisper")]
pub use self::whisper::{WhisperBackend, WhisperConfig, WhisperModel};

#[cfg(feature = "sherpa")]
mod sherpa;

#[cfg(feature = "sherpa")]
pub use self::sherpa::{SherpaBackend, SherpaConfig, SherpaModel};

#[cfg(feature = "sherpa")]
mod sherpa_streaming;

#[cfg(feature = "sherpa")]
pub use self::sherpa_streaming::{
    SherpaStreamingBackend, SherpaStreamingConfig, SherpaStreamingModel,
};
