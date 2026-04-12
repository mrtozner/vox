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
pub mod model_cache;
pub mod streaming_pipeline;
pub mod stt;
pub mod traits;
pub mod tts;
pub mod types;
pub mod vad;

#[cfg(feature = "diarization")]
pub mod diarization;

#[cfg(feature = "intelligence")]
pub mod intelligence;

// Tier 1 quality optimizations
#[cfg(any(feature = "cli", feature = "server"))]
pub mod prompts;
#[cfg(any(feature = "cli", feature = "server"))]
pub mod streaming_chat;

// Environment-aware capability registry (Phase 1: read-only facts).
#[cfg(any(feature = "cli", feature = "server"))]
pub mod capabilities;
#[cfg(any(feature = "cli", feature = "server"))]
pub mod system_profile;

// Public re-exports — the "prelude" surface
pub use engine::{Vox, VoxBuilder, VoxConfig, VoxContext};
pub use error::VoxError;
pub use model_cache::{CacheKey, CacheStats, ModelCache};
pub use traits::{
    StreamingSttBackend, StreamingTtsBackend, SttBackend, SttSession, TtsBackend, TtsChunk,
    TtsSession, VadBackend, VadEvent,
};
pub use types::{
    AudioChunk, PipelineStats, SttResult, TtsOutput, TtsRequest, Utterance, VoiceInfo,
};

// Tier 1: Voice-optimized prompts
#[cfg(any(feature = "cli", feature = "server"))]
pub use prompts::{VoicePromptMode, build_system_prompt};

// Capability registry re-exports
#[cfg(any(feature = "cli", feature = "server"))]
pub use capabilities::{
    Capability, CapabilityRegistry, FeatureFlags, GpuFacts, HardwareFacts, LoadedModel,
    ModelInventory, OllamaModelSummary,
};

// Conditional backend re-exports for convenience
#[cfg(feature = "silero")]
pub use vad::{SileroVad, VadConfig};

#[cfg(feature = "whisper")]
pub use stt::{WhisperBackend, WhisperConfig, WhisperModel};

#[cfg(feature = "distil-whisper")]
pub use stt::{DistilWhisperBackend, DistilWhisperConfig, DistilWhisperModel};

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

#[cfg(feature = "qwen3")]
pub use tts::{Qwen3Backend, Qwen3Config};

pub use tts::SentenceStreamingAdapter;

#[cfg(feature = "diarization")]
pub use diarization::{
    ConversationEntry, DiarizationConfig, DiarizationPipeline, DiarizationPipelineBuilder,
    EmbeddingConfig, Recognition, RecognitionConfig, Speaker, SpeakerDatabase, SpeakerEmbedding,
    SpeakerRegistry, cosine_similarity,
};

// Intelligence layer exports (opt-in feature)
#[cfg(feature = "intelligence")]
pub use intelligence::{
    CacheMetrics, ConversationMetadata, PreferenceType, PrivacyDashboard, SemanticCache,
    SemanticCacheConfig, SpeakerInfo, UserModel, UserProfile, VoiceMemory, VoiceMemoryBuilder,
};

#[cfg(any(
    feature = "kokoro",
    feature = "pocket",
    feature = "chatterbox",
    feature = "piper",
    feature = "qwen3",
    feature = "tts"
))]
pub use audio::AudioPlayer;
