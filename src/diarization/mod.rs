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

pub use database::{ConversationEntry, SpeakerDatabase};
pub use embedding::{EmbeddingConfig, SpeakerEmbedding, cosine_similarity};
pub use pipeline::{DiarizationConfig, DiarizationPipeline, DiarizationPipelineBuilder};
pub use recognition::{Recognition, RecognitionConfig, Speaker, SpeakerRegistry};
