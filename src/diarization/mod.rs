//! Speaker diarization and recognition.
//!
//! This module provides speaker diarization capabilities:
//! - Speaker embedding extraction from audio
//! - Speaker enrollment and identification
//! - Multi-speaker conversation segmentation
//! - Local speaker database with preferences

pub mod database;
pub mod embedding;
pub mod pipeline;
pub mod recognition;

pub use database::{
    ConversationEntry, EmbeddingVersionChange, PreferenceEntry, SpeakerDatabase, SpeakerExport,
    SpeakerRecord,
};
pub use embedding::{EmbeddingConfig, MIN_SPEAKER_AUDIO_MS, SpeakerEmbedding, cosine_similarity};
pub use pipeline::{
    DiarizationConfig, DiarizationPipeline, DiarizationPipelineBuilder, Identification,
    IdentifyMode,
};
pub use recognition::{
    AdaptationConfig, DEFAULT_ADAPT_MARGIN, DEFAULT_ADAPT_THRESHOLD, DEFAULT_THRESHOLD,
    MAX_SAMPLE_COUNT, MAX_SPEAKER_NAME_CHARS, MatchScores, Recognition, RecognitionConfig, Speaker,
    SpeakerRegistry, merge_embeddings, validate_speaker_name,
};

/// Version tag of the embedding front end (mel features, CMVN, model input
/// layout). Voiceprints are only comparable within one version; bump this
/// when the preprocessing changes in a way that moves embeddings, and
/// [`SpeakerDatabase::apply_embedding_version`] will invalidate the old ones.
pub const EMBEDDING_VERSION: &str = "v2-cmvn";
