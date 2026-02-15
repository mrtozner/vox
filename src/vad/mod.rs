//! Voice Activity Detection backends.
//!
//! The default backend is Silero VAD (behind the `silero` feature flag),
//! which runs a small ONNX model to detect speech in audio frames.

#[cfg(feature = "silero")]
mod silero;

#[cfg(feature = "silero")]
pub use silero::{SileroVad, VadConfig};
