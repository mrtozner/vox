//! Text-to-Speech backends.
//!
//! Available backends:
//! - **Kokoro**: High-quality 82M parameter model via ONNX Runtime (feature `kokoro`)
//! - **Pocket TTS**: Lightweight 100M parameter model via Candle (feature `pocket`)
//! - **Chatterbox Turbo**: Voice cloning 350M parameter model via ONNX Runtime (feature `chatterbox`)

#[cfg(feature = "kokoro")]
mod kokoro;

#[cfg(feature = "pocket")]
mod pocket;

#[cfg(feature = "chatterbox")]
mod chatterbox;

#[cfg(feature = "piper")]
mod piper;

mod streaming;

#[cfg(feature = "kokoro")]
pub use kokoro::{KokoroBackend, KokoroConfig};

#[cfg(feature = "pocket")]
pub use pocket::{PocketTtsBackend, PocketTtsConfig};

#[cfg(feature = "chatterbox")]
pub use chatterbox::{ChatterboxBackend, ChatterboxConfig};

#[cfg(feature = "piper")]
pub use self::piper::{PiperBackend, PiperConfig};

pub use streaming::SentenceStreamingAdapter;
