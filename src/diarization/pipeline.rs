//! Multi-speaker conversation pipeline.
//!
//! This module integrates speaker diarization with the existing VAD+STT pipeline,
//! enabling speaker identification for each utterance in multi-speaker conversations.

use std::sync::{Arc, Mutex};

use crate::error::VoxError;
use crate::types::{AudioChunk, Utterance};

use super::embedding::SpeakerEmbedding;
use super::recognition::{Recognition, SpeakerRegistry};

/// Configuration for the multi-speaker pipeline.
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Whether to automatically enroll unknown speakers.
    pub auto_enroll: bool,
    /// Minimum audio duration (ms) for speaker identification.
    pub min_audio_ms: u64,
    /// Whether to skip identification for very short utterances.
    pub skip_short_utterances: bool,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            auto_enroll: false,
            min_audio_ms: 500,
            skip_short_utterances: true,
        }
    }
}

/// Multi-speaker diarization pipeline.
///
/// Integrates speaker embedding extraction and recognition with the voice pipeline.
/// Processes utterances from VAD and adds speaker identification.
pub struct DiarizationPipeline {
    embedding: Arc<Mutex<SpeakerEmbedding>>,
    registry: Arc<Mutex<SpeakerRegistry>>,
    config: DiarizationConfig,
    unknown_counter: Arc<Mutex<u32>>,
}

impl DiarizationPipeline {
    /// Create a new diarization pipeline.
    ///
    /// # Arguments
    /// * `embedding` - Speaker embedding extractor
    /// * `registry` - Speaker registry for enrollment/identification
    /// * `config` - Pipeline configuration
    pub fn new(
        embedding: SpeakerEmbedding,
        registry: SpeakerRegistry,
        config: DiarizationConfig,
    ) -> Self {
        // Initialize counter from existing speakers to avoid ID collisions
        // when speakers are pre-loaded from the database at startup.
        let max_num = registry
            .list_speakers()
            .iter()
            .filter_map(|s| s.id.strip_prefix("speaker_").and_then(|n| n.parse::<u32>().ok()))
            .max()
            .unwrap_or(0);

        Self {
            embedding: Arc::new(Mutex::new(embedding)),
            registry: Arc::new(Mutex::new(registry)),
            config,
            unknown_counter: Arc::new(Mutex::new(max_num)),
        }
    }

    /// Process an utterance and identify the speaker.
    ///
    /// # Arguments
    /// * `utterance` - The utterance to process (from VAD)
    ///
    /// # Returns
    /// Updated utterance with speaker_id field populated.
    pub async fn process_utterance(&self, mut utterance: Utterance) -> Result<Utterance, VoxError> {
        // Skip short utterances if configured
        if self.config.skip_short_utterances && utterance.duration_ms < self.config.min_audio_ms {
            utterance.speaker_id = Some("unknown".to_string());
            return Ok(utterance);
        }

        // Extract embedding on a blocking thread — ONNX Runtime uses
        // thread-local caching internally and must not run on the async
        // executor where tasks can migrate between OS threads.
        let emb_arc = Arc::clone(&self.embedding);
        let audio = utterance.audio.clone();
        let embedding = match tokio::task::spawn_blocking(move || {
            let mut guard = emb_arc
                .lock()
                .map_err(|e| VoxError::Diarization(format!("embedding mutex poisoned: {e}")))?;
            guard.extract(&audio)
        })
        .await
        {
            Ok(Ok(emb)) => emb,
            Ok(Err(e)) => {
                tracing::warn!("failed to extract speaker embedding: {e}");
                utterance.speaker_id = Some("unknown".to_string());
                return Ok(utterance);
            }
            Err(e) => {
                tracing::warn!("embedding task panicked: {e}");
                utterance.speaker_id = Some("unknown".to_string());
                return Ok(utterance);
            }
        };

        // Identify speaker
        let mut registry = self.registry.lock()
            .map_err(|e| VoxError::Diarization(format!("registry mutex poisoned: {e}")))?;
        let recognition = registry.identify(&embedding)?;

        let speaker_id = match recognition {
            Recognition::Identified {
                speaker_id,
                confidence,
            } => {
                tracing::debug!(
                    speaker_id = %speaker_id,
                    confidence = %confidence,
                    "speaker identified"
                );
                // Adapt reference embedding with EMA (70% old, 30% new)
                registry.update_embedding(&speaker_id, &embedding, 0.7);
                speaker_id
            }
            Recognition::Unknown { best_score } => {
                tracing::debug!(
                    best_score = %best_score,
                    "unknown speaker detected"
                );

                if self.config.auto_enroll {
                    let mut counter = self.unknown_counter.lock()
                        .map_err(|e| VoxError::Diarization(format!("counter mutex poisoned: {e}")))?;
                    *counter += 1;
                    let id = format!("speaker_{}", *counter);
                    let name = format!("Speaker {}", *counter);
                    drop(counter);

                    registry.enroll(&id, &name, embedding)?;
                    tracing::info!(speaker_id = %id, "auto-enrolled new speaker");
                    id
                } else {
                    "unknown".to_string()
                }
            }
        };

        utterance.speaker_id = Some(speaker_id);
        Ok(utterance)
    }

    /// Enroll a speaker from an audio sample.
    ///
    /// # Arguments
    /// * `id` - Unique speaker identifier
    /// * `name` - Human-readable name
    /// * `audio` - Reference audio sample
    ///
    /// # Returns
    /// `Ok(())` if enrollment succeeded.
    pub async fn enroll_speaker(
        &self,
        id: impl Into<String>,
        name: impl Into<String>,
        audio: &AudioChunk,
    ) -> Result<(), VoxError> {
        let emb_arc = Arc::clone(&self.embedding);
        let audio = audio.clone();
        let embedding = tokio::task::spawn_blocking(move || {
            let mut guard = emb_arc
                .lock()
                .map_err(|e| VoxError::Diarization(format!("embedding mutex poisoned: {e}")))?;
            guard.extract(&audio)
        })
        .await
        .map_err(|e| VoxError::Diarization(format!("embedding task panicked: {e}")))??;

        let mut registry = self.registry.lock()
            .map_err(|e| VoxError::Diarization(format!("registry mutex poisoned: {e}")))?;
        registry.enroll(id, name, embedding)?;

        Ok(())
    }

    /// Get the speaker registry (for listing speakers, etc.).
    pub fn registry(&self) -> Arc<Mutex<SpeakerRegistry>> {
        Arc::clone(&self.registry)
    }
}

/// Builder for DiarizationPipeline.
pub struct DiarizationPipelineBuilder {
    embedding: Option<SpeakerEmbedding>,
    registry: Option<SpeakerRegistry>,
    config: DiarizationConfig,
}

impl DiarizationPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            embedding: None,
            registry: None,
            config: DiarizationConfig::default(),
        }
    }

    /// Set the speaker embedding extractor.
    pub fn embedding(mut self, embedding: SpeakerEmbedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the speaker registry.
    pub fn registry(mut self, registry: SpeakerRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set the pipeline configuration.
    pub fn config(mut self, config: DiarizationConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable auto-enrollment of unknown speakers.
    pub fn auto_enroll(mut self, enabled: bool) -> Self {
        self.config.auto_enroll = enabled;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> Result<DiarizationPipeline, VoxError> {
        let embedding = self
            .embedding
            .ok_or_else(|| VoxError::Diarization("speaker embedding extractor not set".into()))?;

        let registry = self.registry.unwrap_or_default();

        Ok(DiarizationPipeline::new(embedding, registry, self.config))
    }
}

impl Default for DiarizationPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DiarizationConfig::default();
        assert!(!config.auto_enroll);
        assert_eq!(config.min_audio_ms, 500);
        assert!(config.skip_short_utterances);
    }

    #[test]
    fn test_builder_missing_embedding() {
        let builder = DiarizationPipelineBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }
}
