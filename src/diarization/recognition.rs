//! Speaker recognition and enrollment.
//!
//! This module provides speaker enrollment and identification using
//! speaker embeddings. Speakers are enrolled with a name and reference
//! embedding, then identified via cosine similarity matching.

use std::collections::HashMap;

use crate::error::VoxError;

use super::embedding::cosine_similarity;

/// Default similarity threshold for speaker identification.
const DEFAULT_THRESHOLD: f32 = 0.35;

/// A registered speaker with their reference embedding.
#[derive(Debug, Clone)]
pub struct Speaker {
    /// Unique speaker identifier (UUID or similar).
    pub id: String,
    /// Human-readable speaker name.
    pub name: String,
    /// Reference embedding for this speaker (L2-normalized).
    pub embedding: Vec<f32>,
}

/// Speaker recognition result.
#[derive(Debug, Clone, PartialEq)]
pub enum Recognition {
    /// Speaker was identified with confidence score.
    Identified { speaker_id: String, confidence: f32 },
    /// No speaker matched above threshold.
    Unknown { best_score: f32 },
}

/// Configuration for speaker recognition.
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Minimum cosine similarity to consider a match (default: 0.35).
    pub threshold: f32,
    /// Whether to return unknown for low confidence (default: true).
    pub require_threshold: bool,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            require_threshold: true,
        }
    }
}

/// In-memory speaker registry for enrollment and recognition.
///
/// Stores speaker embeddings and performs cosine similarity matching
/// for identification. For persistent storage, use SpeakerDatabase.
pub struct SpeakerRegistry {
    speakers: HashMap<String, Speaker>,
    config: RecognitionConfig,
}

impl SpeakerRegistry {
    /// Create a new speaker registry with default configuration.
    pub fn new() -> Self {
        Self::with_config(RecognitionConfig::default())
    }

    /// Create a new speaker registry with custom configuration.
    pub fn with_config(config: RecognitionConfig) -> Self {
        Self {
            speakers: HashMap::new(),
            config,
        }
    }

    /// Enroll a new speaker with a reference embedding.
    ///
    /// # Arguments
    /// * `id` - Unique speaker identifier
    /// * `name` - Human-readable name
    /// * `embedding` - Reference embedding (should be L2-normalized)
    ///
    /// # Returns
    /// `Ok(())` if enrollment succeeded, `Err` if speaker ID already exists.
    pub fn enroll(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Result<(), VoxError> {
        let id = id.into();
        if self.speakers.contains_key(&id) {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' already enrolled"
            )));
        }

        self.speakers.insert(
            id.clone(),
            Speaker {
                id,
                name: name.into(),
                embedding,
            },
        );

        Ok(())
    }

    /// Identify a speaker from an embedding.
    ///
    /// # Arguments
    /// * `embedding` - Query embedding (should be L2-normalized)
    ///
    /// # Returns
    /// Recognition result with speaker ID and confidence, or Unknown if no match.
    pub fn identify(&self, embedding: &[f32]) -> Result<Recognition, VoxError> {
        if self.speakers.is_empty() {
            return Ok(Recognition::Unknown { best_score: 0.0 });
        }

        let mut best_match: Option<(&str, f32)> = None;

        for (id, speaker) in &self.speakers {
            if speaker.embedding.len() != embedding.len() {
                return Err(VoxError::Diarization(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    speaker.embedding.len(),
                    embedding.len()
                )));
            }

            let similarity = cosine_similarity(&speaker.embedding, embedding);

            if let Some((_, best_score)) = best_match {
                if similarity > best_score {
                    best_match = Some((id.as_str(), similarity));
                }
            } else {
                best_match = Some((id.as_str(), similarity));
            }
        }

        match best_match {
            Some((id, score))
                if !self.config.require_threshold || score >= self.config.threshold =>
            {
                Ok(Recognition::Identified {
                    speaker_id: id.to_string(),
                    confidence: score,
                })
            }
            Some((_, score)) => Ok(Recognition::Unknown { best_score: score }),
            None => Ok(Recognition::Unknown { best_score: 0.0 }),
        }
    }

    /// List all enrolled speakers.
    pub fn list_speakers(&self) -> Vec<&Speaker> {
        self.speakers.values().collect()
    }

    /// Get a speaker by ID.
    pub fn get_speaker(&self, id: &str) -> Option<&Speaker> {
        self.speakers.get(id)
    }

    /// Remove a speaker from the registry.
    ///
    /// # Returns
    /// `Ok(())` if removal succeeded, `Err` if speaker ID not found.
    pub fn forget(&mut self, id: &str) -> Result<(), VoxError> {
        self.speakers
            .remove(id)
            .ok_or_else(|| VoxError::Diarization(format!("speaker with ID '{id}' not found")))?;
        Ok(())
    }

    /// Rename an enrolled speaker in place.
    ///
    /// Changes the human-readable `name` without touching the embedding or ID.
    /// Used by the `/v1/listen` WebSocket to let users label `Speaker 1` → `Alice`.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if the speaker ID is unknown.
    pub fn rename(
        &mut self,
        id: &str,
        new_label: impl Into<String>,
    ) -> Result<(), VoxError> {
        let speaker = self
            .speakers
            .get_mut(id)
            .ok_or_else(|| VoxError::Diarization(format!("speaker with ID '{id}' not found")))?;
        speaker.name = new_label.into();
        Ok(())
    }

    /// Update a speaker's embedding using exponential moving average.
    ///
    /// Blends the new observation with the stored reference:
    /// `new_ref = alpha * old + (1 - alpha) * observation`, then L2-normalizes.
    /// This makes identification more robust over time as the reference
    /// adapts to the speaker's natural voice variation.
    pub fn update_embedding(&mut self, id: &str, observation: &[f32], alpha: f32) {
        if let Some(speaker) = self.speakers.get_mut(id) {
            if speaker.embedding.len() == observation.len() {
                for (stored, &obs) in speaker.embedding.iter_mut().zip(observation.iter()) {
                    *stored = alpha * *stored + (1.0 - alpha) * obs;
                }
                // L2-normalize
                let norm: f32 = speaker.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for v in &mut speaker.embedding {
                        *v /= norm;
                    }
                }
            }
        }
    }

    /// Update the recognition threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.config.threshold = threshold;
    }

    /// Get the current recognition threshold.
    pub fn threshold(&self) -> f32 {
        self.config.threshold
    }

    /// Get the number of enrolled speakers.
    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    /// Clear all enrolled speakers.
    pub fn clear(&mut self) {
        self.speakers.clear();
    }
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f32]) -> Vec<f32> {
        // L2-normalize
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        values.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_enroll_and_identify() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb1.clone()).unwrap();

        let result = registry.identify(&emb1).unwrap();
        assert!(matches!(result, Recognition::Identified { .. }));
        if let Recognition::Identified {
            speaker_id,
            confidence,
        } = result
        {
            assert_eq!(speaker_id, "alice");
            assert!((confidence - 1.0).abs() < 1e-6); // Perfect match
        }
    }

    #[test]
    fn test_identify_unknown_no_speakers() {
        let registry = SpeakerRegistry::new();
        let emb = make_embedding(&[1.0, 0.0, 0.0]);
        let result = registry.identify(&emb).unwrap();
        assert!(matches!(result, Recognition::Unknown { best_score: 0.0 }));
    }

    #[test]
    fn test_identify_below_threshold() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb1).unwrap();

        // Very different embedding
        let emb2 = make_embedding(&[0.0, 1.0, 0.0]);
        let result = registry.identify(&emb2).unwrap();

        // Should be Unknown because similarity is below threshold
        assert!(matches!(result, Recognition::Unknown { .. }));
    }

    #[test]
    fn test_enroll_duplicate_id() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb1.clone()).unwrap();

        let result = registry.enroll("alice", "Alice2", emb1);
        assert!(result.is_err());
    }

    #[test]
    fn test_forget_speaker() {
        let mut registry = SpeakerRegistry::new();

        let emb = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb.clone()).unwrap();

        assert_eq!(registry.speaker_count(), 1);

        registry.forget("alice").unwrap();
        assert_eq!(registry.speaker_count(), 0);

        let result = registry.identify(&emb).unwrap();
        assert!(matches!(result, Recognition::Unknown { .. }));
    }

    #[test]
    fn test_forget_nonexistent_speaker() {
        let mut registry = SpeakerRegistry::new();
        let result = registry.forget("alice");
        assert!(result.is_err());
    }

    #[test]
    fn test_rename_speaker() {
        let mut registry = SpeakerRegistry::new();
        let emb = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("speaker_1", "Speaker 1", emb).unwrap();

        registry.rename("speaker_1", "Alice").unwrap();
        assert_eq!(registry.get_speaker("speaker_1").unwrap().name, "Alice");

        // Renaming a non-existent speaker should error out.
        assert!(registry.rename("speaker_99", "Bob").is_err());
    }

    #[test]
    fn test_list_speakers() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        let emb2 = make_embedding(&[0.0, 1.0, 0.0]);

        registry.enroll("alice", "Alice", emb1).unwrap();
        registry.enroll("bob", "Bob", emb2).unwrap();

        let speakers = registry.list_speakers();
        assert_eq!(speakers.len(), 2);
    }

    #[test]
    fn test_get_speaker() {
        let mut registry = SpeakerRegistry::new();

        let emb = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb).unwrap();

        let speaker = registry.get_speaker("alice").unwrap();
        assert_eq!(speaker.id, "alice");
        assert_eq!(speaker.name, "Alice");

        assert!(registry.get_speaker("bob").is_none());
    }

    #[test]
    fn test_set_threshold() {
        let mut registry = SpeakerRegistry::new();
        assert_eq!(registry.threshold(), DEFAULT_THRESHOLD);

        registry.set_threshold(0.85);
        assert_eq!(registry.threshold(), 0.85);
    }

    #[test]
    fn test_clear() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        let emb2 = make_embedding(&[0.0, 1.0, 0.0]);

        registry.enroll("alice", "Alice", emb1).unwrap();
        registry.enroll("bob", "Bob", emb2).unwrap();

        assert_eq!(registry.speaker_count(), 2);

        registry.clear();
        assert_eq!(registry.speaker_count(), 0);
    }

    #[test]
    fn test_update_embedding() {
        let mut registry = SpeakerRegistry::new();
        let emb = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb).unwrap();

        // Update with a noticeably different embedding
        let obs = make_embedding(&[0.5, 0.5, 0.0]);
        registry.update_embedding("alice", &obs, 0.7);

        // Should still identify as Alice
        let query = make_embedding(&[0.9, 0.1, 0.0]);
        let result = registry.identify(&query).unwrap();
        assert!(matches!(result, Recognition::Identified { speaker_id, .. } if speaker_id == "alice"));

        // Verify embedding was actually modified (not identical to original)
        let speaker = registry.get_speaker("alice").unwrap();
        assert!((speaker.embedding[0] - 1.0).abs() > 0.01, "embedding should have shifted");
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut registry = SpeakerRegistry::new();

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb1).unwrap();

        let emb2 = make_embedding(&[1.0, 0.0, 0.0, 0.0]); // Different dimension
        let result = registry.identify(&emb2);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_speakers_best_match() {
        let mut registry = SpeakerRegistry::new();

        let emb_alice = make_embedding(&[1.0, 0.0, 0.0]);
        let emb_bob = make_embedding(&[0.0, 1.0, 0.0]);
        let emb_charlie = make_embedding(&[0.0, 0.0, 1.0]);

        registry.enroll("alice", "Alice", emb_alice).unwrap();
        registry.enroll("bob", "Bob", emb_bob).unwrap();
        registry.enroll("charlie", "Charlie", emb_charlie).unwrap();

        // Query similar to Bob
        let query = make_embedding(&[0.1, 0.9, 0.0]);
        let result = registry.identify(&query).unwrap();

        if let Recognition::Identified { speaker_id, .. } = result {
            assert_eq!(speaker_id, "bob");
        } else {
            panic!("Expected Identified, got {:?}", result);
        }
    }

    #[test]
    fn test_default_threshold() {
        let config = RecognitionConfig::default();
        assert_eq!(config.threshold, DEFAULT_THRESHOLD);
        assert!(config.require_threshold);
    }

    #[test]
    fn test_custom_config() {
        let config = RecognitionConfig {
            threshold: 0.85,
            require_threshold: false,
        };
        let mut registry = SpeakerRegistry::with_config(config);

        assert_eq!(registry.threshold(), 0.85);

        let emb1 = make_embedding(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb1).unwrap();

        // Very different embedding (low similarity)
        let emb2 = make_embedding(&[0.0, 1.0, 0.0]);
        let result = registry.identify(&emb2).unwrap();

        // With require_threshold=false, should still identify (even if score is low)
        assert!(matches!(result, Recognition::Identified { .. }));
    }
}
