//! Text-to-Speech backends.
//!
//! Available backends:
//! - **Kokoro**: High-quality 82M parameter model via ONNX Runtime (feature `kokoro`)

#[cfg(feature = "kokoro")]
mod kokoro;

#[cfg(feature = "kokoro")]
pub use kokoro::{KokoroBackend, KokoroConfig};
