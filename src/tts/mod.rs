//! Text-to-Speech backends.
//!
//! Available backends:
//! - **Kokoro**: High-quality 82M parameter model via ONNX Runtime (feature `kokoro`)
//! - **Pocket TTS**: Lightweight 100M parameter model via Candle (feature `pocket`)

#[cfg(feature = "kokoro")]
mod kokoro;

#[cfg(feature = "pocket")]
mod pocket;

#[cfg(feature = "kokoro")]
pub use kokoro::{KokoroBackend, KokoroConfig};

#[cfg(feature = "pocket")]
pub use pocket::{PocketTtsBackend, PocketTtsConfig};
