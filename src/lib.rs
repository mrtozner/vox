//! # Vox — The open-source voice AI framework
//!
//! Vox assembles a local voice pipeline into a single crate:
//!
//! ```text
//! Audio In → VAD (Silero) → STT (Whisper) → [LLM] → [TTS] → Audio Out
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use vox::{Vox, SileroVad, WhisperBackend};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let vox = Vox::builder()
//!         .vad(SileroVad::new("silero_vad.onnx")?)
//!         .stt(WhisperBackend::from_model("ggml-tiny.en.bin")?)
//!         .on_utterance(|result, _ctx| {
//!             println!("{}", result.text);
//!         })
//!         .build()?;
//!
//!     vox.listen().await?;
//!     Ok(())
//! }
//! ```

pub mod audio;
pub mod engine;
pub mod error;
pub mod stt;
pub mod traits;
pub mod tts;
pub mod types;
pub mod vad;

// Public re-exports — the "prelude" surface
pub use engine::{Vox, VoxBuilder, VoxConfig, VoxContext};
pub use error::VoxError;
pub use traits::{StreamingSttBackend, SttBackend, SttSession, TtsBackend, VadBackend, VadEvent};
pub use types::{
    AudioChunk, PipelineStats, SttResult, TtsOutput, TtsRequest, Utterance, VoiceInfo,
};

// Conditional backend re-exports for convenience
#[cfg(feature = "silero")]
pub use vad::{SileroVad, VadConfig};

#[cfg(feature = "whisper")]
pub use stt::{WhisperBackend, WhisperConfig, WhisperModel};

#[cfg(feature = "sherpa")]
pub use stt::{SherpaBackend, SherpaConfig, SherpaModel};

#[cfg(feature = "sherpa")]
pub use stt::{SherpaStreamingBackend, SherpaStreamingConfig, SherpaStreamingModel};

#[cfg(feature = "kokoro")]
pub use tts::{KokoroBackend, KokoroConfig};

#[cfg(feature = "pocket")]
pub use tts::{PocketTtsBackend, PocketTtsConfig};

#[cfg(feature = "chatterbox")]
pub use tts::{ChatterboxBackend, ChatterboxConfig};

#[cfg(feature = "piper")]
pub use tts::{PiperBackend, PiperConfig};

#[cfg(any(
    feature = "kokoro",
    feature = "pocket",
    feature = "chatterbox",
    feature = "piper",
    feature = "tts"
))]
pub use audio::AudioPlayer;
