//! Speaker recognition and enrollment.
//!
//! This module provides speaker enrollment and identification using
//! speaker embeddings. Speakers are enrolled with a name and reference
//! embedding, then identified via cosine similarity matching.
//!
//! Two thresholds are in play. [`DEFAULT_THRESHOLD`] decides whether a
//! match is good enough to *label* an utterance. The stricter
//! [`AdaptationConfig`] decides whether the utterance may also *move* the
//! stored reference; a borderline match never does.

use std::collections::HashMap;

use crate::error::VoxError;

use super::embedding::cosine_similarity;

/// Default similarity threshold for accepting an identification.
pub const DEFAULT_THRESHOLD: f32 = 0.35;

/// Default minimum similarity before an observation may move a speaker's
/// reference embedding. Deliberately well above [`DEFAULT_THRESHOLD`]: a
/// borderline match is good enough to label a transcript, but letting it
/// rewrite the reference is how one voice slowly drifts into another.
pub const DEFAULT_ADAPT_THRESHOLD: f32 = 0.55;

/// Default minimum lead over the runner-up speaker before adapting.
pub const DEFAULT_ADAPT_MARGIN: f32 = 0.1;

/// Cap on the observation count used as the weight of the stored reference.
/// Without a cap a long-lived speaker would stop adapting altogether.
pub const MAX_SAMPLE_COUNT: u32 = 50;

/// Longest accepted speaker name, in characters.
pub const MAX_SPEAKER_NAME_CHARS: usize = 64;

/// Validate and normalize a human-readable speaker name.
///
/// Leading/trailing whitespace is trimmed. The result must be 1 to
/// [`MAX_SPEAKER_NAME_CHARS`] characters of letters, digits, spaces and
/// `-`, `_`, `.`, `'` (any script). Markup and control characters are
/// rejected so a name can be shown in a UI or placed in a prompt as-is.
pub fn validate_speaker_name(name: &str) -> Result<String, VoxError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(VoxError::Diarization("speaker name is empty".into()));
    }
    let chars = trimmed.chars().count();
    if chars > MAX_SPEAKER_NAME_CHARS {
        return Err(VoxError::Diarization(format!(
            "speaker name is too long: {chars} characters (maximum: {MAX_SPEAKER_NAME_CHARS})"
        )));
    }
    if let Some(bad) = trimmed
        .chars()
        .find(|c| !(c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.' | '\'')))
    {
        return Err(VoxError::Diarization(format!(
            "speaker name contains an unsupported character: {bad:?}"
        )));
    }
    Ok(trimmed.to_string())
}

/// Count-weighted mean of two reference embeddings, L2-normalized.
///
/// Returns `None` when the dimensions differ or the embeddings are empty.
pub fn merge_embeddings(a: &[f32], a_count: u32, b: &[f32], b_count: u32) -> Option<Vec<f32>> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let wa = a_count.max(1) as f32;
    let wb = b_count.max(1) as f32;
    let mut merged: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (wa * x + wb * y) / (wa + wb))
        .collect();
    l2_normalize(&mut merged);
    Some(merged)
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

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

/// When an accepted identification is allowed to move the stored reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptationConfig {
    /// Minimum cosine similarity to the matched speaker.
    pub min_score: f32,
    /// Minimum lead over the second-best speaker.
    pub min_margin: f32,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            min_score: DEFAULT_ADAPT_THRESHOLD,
            min_margin: DEFAULT_ADAPT_MARGIN,
        }
    }
}

/// Raw scores from comparing one embedding against every enrolled speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchScores {
    /// Best-matching speaker and its cosine similarity, if anyone is enrolled.
    pub best: Option<(String, f32)>,
    /// Similarity of the second-best speaker (`-1.0` when there is none).
    pub runner_up: f32,
}

/// In-memory speaker registry for enrollment and recognition.
///
/// Stores speaker embeddings and performs cosine similarity matching
/// for identification. For persistent storage, use SpeakerDatabase.
pub struct SpeakerRegistry {
    speakers: HashMap<String, Speaker>,
    /// Number of observations folded into each reference embedding.
    counts: HashMap<String, u32>,
    config: RecognitionConfig,
    adaptation: AdaptationConfig,
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
            counts: HashMap::new(),
            config,
            adaptation: AdaptationConfig::default(),
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
        self.enroll_with_count(id, name, embedding, 1)
    }

    /// Enroll a speaker whose reference already aggregates `sample_count`
    /// observations (used when hydrating from the database).
    pub fn enroll_with_count(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        embedding: Vec<f32>,
        sample_count: u32,
    ) -> Result<(), VoxError> {
        let id = id.into();
        if self.speakers.contains_key(&id) {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' already enrolled"
            )));
        }

        self.counts.insert(id.clone(), sample_count.max(1));
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
        let scores = self.score(embedding)?;
        match scores.best {
            Some((id, score)) if self.accepts(score) => Ok(Recognition::Identified {
                speaker_id: id,
                confidence: score,
            }),
            Some((_, score)) => Ok(Recognition::Unknown { best_score: score }),
            None => Ok(Recognition::Unknown { best_score: 0.0 }),
        }
    }

    /// Compare an embedding against every enrolled speaker and return the
    /// best and second-best similarity, without applying any threshold.
    pub fn score(&self, embedding: &[f32]) -> Result<MatchScores, VoxError> {
        let mut best: Option<(&str, f32)> = None;
        let mut runner_up = -1.0f32;

        for (id, speaker) in &self.speakers {
            if speaker.embedding.len() != embedding.len() {
                return Err(VoxError::Diarization(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    speaker.embedding.len(),
                    embedding.len()
                )));
            }

            let similarity = cosine_similarity(&speaker.embedding, embedding);
            match best {
                Some((_, best_score)) if similarity <= best_score => {
                    runner_up = runner_up.max(similarity);
                }
                Some((_, best_score)) => {
                    runner_up = best_score;
                    best = Some((id.as_str(), similarity));
                }
                None => best = Some((id.as_str(), similarity)),
            }
        }

        Ok(MatchScores {
            best: best.map(|(id, score)| (id.to_string(), score)),
            runner_up,
        })
    }

    /// Whether a best-match score counts as an identification.
    pub fn accepts(&self, score: f32) -> bool {
        !self.config.require_threshold || score >= self.config.threshold
    }

    /// Whether a match with these scores is confident enough to adapt on.
    pub fn should_adapt(&self, score: f32, runner_up: f32) -> bool {
        score >= self.adaptation.min_score && score - runner_up >= self.adaptation.min_margin
    }

    /// Fold an observation into a speaker's reference if the match is
    /// confident enough (see [`AdaptationConfig`]).
    ///
    /// The reference is the count-weighted mean of everything folded in so
    /// far: `new = (n * old + observation) / (n + 1)`, L2-normalized, with
    /// `n` capped at [`MAX_SAMPLE_COUNT`]. A single odd sample therefore
    /// moves an established reference very little.
    ///
    /// Returns `true` if the reference changed.
    pub fn adapt(&mut self, id: &str, observation: &[f32], score: f32, runner_up: f32) -> bool {
        if !self.should_adapt(score, runner_up) {
            return false;
        }
        let Some(speaker) = self.speakers.get_mut(id) else {
            return false;
        };
        if speaker.embedding.len() != observation.len() {
            return false;
        }
        let count = self.counts.entry(id.to_string()).or_insert(1);
        let n = (*count).clamp(1, MAX_SAMPLE_COUNT) as f32;
        for (stored, &obs) in speaker.embedding.iter_mut().zip(observation.iter()) {
            *stored = (n * *stored + obs) / (n + 1.0);
        }
        l2_normalize(&mut speaker.embedding);
        *count = count.saturating_add(1).min(MAX_SAMPLE_COUNT);
        true
    }

    /// Number of observations folded into a speaker's reference.
    pub fn sample_count(&self, id: &str) -> Option<u32> {
        self.speakers
            .contains_key(id)
            .then(|| self.counts.get(id).copied().unwrap_or(1))
    }

    /// Replace a speaker's reference embedding and observation count.
    pub fn set_embedding(
        &mut self,
        id: &str,
        embedding: Vec<f32>,
        sample_count: u32,
    ) -> Result<(), VoxError> {
        let speaker = self
            .speakers
            .get_mut(id)
            .ok_or_else(|| VoxError::Diarization(format!("speaker with ID '{id}' not found")))?;
        speaker.embedding = embedding;
        self.counts.insert(id.to_string(), sample_count.max(1));
        Ok(())
    }

    /// Merge `source` into `target`: the target keeps its ID and name, its
    /// reference becomes the count-weighted mean of both, and the source is
    /// removed.
    ///
    /// Returns the merged `(embedding, sample_count)`.
    pub fn merge(&mut self, source: &str, target: &str) -> Result<(Vec<f32>, u32), VoxError> {
        if source == target {
            return Err(VoxError::Diarization(
                "cannot merge a speaker into itself".into(),
            ));
        }
        let not_found =
            |id: &str| VoxError::Diarization(format!("speaker with ID '{id}' not found"));
        let src = self.speakers.get(source).ok_or_else(|| not_found(source))?;
        let dst = self.speakers.get(target).ok_or_else(|| not_found(target))?;
        let src_count = self.counts.get(source).copied().unwrap_or(1);
        let dst_count = self.counts.get(target).copied().unwrap_or(1);
        let merged = merge_embeddings(&dst.embedding, dst_count, &src.embedding, src_count)
            .ok_or_else(|| {
                VoxError::Diarization("cannot merge speakers: embedding dimensions differ".into())
            })?;
        let count = src_count.saturating_add(dst_count).min(MAX_SAMPLE_COUNT);

        self.speakers.remove(source);
        self.counts.remove(source);
        if let Some(dst) = self.speakers.get_mut(target) {
            dst.embedding = merged.clone();
        }
        self.counts.insert(target.to_string(), count);
        Ok((merged, count))
    }

    /// Set when an accepted identification may adapt the reference.
    pub fn set_adaptation(&mut self, adaptation: AdaptationConfig) {
        self.adaptation = adaptation;
    }

    /// Current adaptation thresholds.
    pub fn adaptation(&self) -> AdaptationConfig {
        self.adaptation
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
        self.counts.remove(id);
        Ok(())
    }

    /// Rename an enrolled speaker in place.
    ///
    /// Changes the human-readable `name` without touching the embedding or ID.
    /// Used by the `/v1/listen` WebSocket to let users label `Speaker 1` → `Alice`.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if the speaker ID is unknown.
    pub fn rename(&mut self, id: &str, new_label: impl Into<String>) -> Result<(), VoxError> {
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
    ///
    /// Unconditional: the caller decides whether the observation is
    /// trustworthy. Prefer [`SpeakerRegistry::adapt`], which only moves the
    /// reference on a confident, unambiguous match.
    pub fn update_embedding(&mut self, id: &str, observation: &[f32], alpha: f32) {
        if let Some(speaker) = self.speakers.get_mut(id)
            && speaker.embedding.len() == observation.len()
        {
            for (stored, &obs) in speaker.embedding.iter_mut().zip(observation.iter()) {
                *stored = alpha * *stored + (1.0 - alpha) * obs;
            }
            l2_normalize(&mut speaker.embedding);
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
        self.counts.clear();
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
        assert!(
            matches!(result, Recognition::Identified { speaker_id, .. } if speaker_id == "alice")
        );

        // Verify embedding was actually modified (not identical to original)
        let speaker = registry.get_speaker("alice").unwrap();
        assert!(
            (speaker.embedding[0] - 1.0).abs() > 0.01,
            "embedding should have shifted"
        );
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

    /// Unit vector with cosine similarity `score` to `[1, 0, 0]`.
    fn at_score(score: f32) -> Vec<f32> {
        vec![score, (1.0 - score * score).sqrt(), 0.0]
    }

    #[test]
    fn test_borderline_match_does_not_adapt() {
        let mut registry = SpeakerRegistry::new();
        let reference = make_embedding(&[1.0, 0.0, 0.0]);
        registry
            .enroll("alice", "Alice", reference.clone())
            .unwrap();

        // 0.36 clears the acceptance threshold (0.35) but not the
        // adaptation threshold (0.55).
        let obs = at_score(0.36);
        let scores = registry.score(&obs).unwrap();
        let (id, score) = scores.best.clone().unwrap();
        assert_eq!(id, "alice");
        assert!(matches!(
            registry.identify(&obs).unwrap(),
            Recognition::Identified { .. }
        ));
        assert!(!registry.adapt(&id, &obs, score, scores.runner_up));
        assert_eq!(registry.get_speaker("alice").unwrap().embedding, reference);
        assert_eq!(registry.sample_count("alice"), Some(1));
    }

    #[test]
    fn test_confident_match_adapts_with_count_weighting() {
        let mut registry = SpeakerRegistry::new();
        registry
            .enroll("alice", "Alice", make_embedding(&[1.0, 0.0, 0.0]))
            .unwrap();

        let obs = at_score(0.6);
        let scores = registry.score(&obs).unwrap();
        let (id, score) = scores.best.clone().unwrap();
        assert!(registry.adapt(&id, &obs, score, scores.runner_up));
        assert_eq!(registry.sample_count("alice"), Some(2));

        // The first observation counts as much as the enrollment sample.
        let first_shift = registry.get_speaker("alice").unwrap().embedding[1];
        assert!(first_shift > 0.3, "reference should have moved");

        // With many samples behind it the same observation moves it far less.
        let mut settled = SpeakerRegistry::new();
        settled
            .enroll_with_count("alice", "Alice", make_embedding(&[1.0, 0.0, 0.0]), 40)
            .unwrap();
        assert!(settled.adapt("alice", &obs, 0.6, -1.0));
        let settled_shift = settled.get_speaker("alice").unwrap().embedding[1];
        assert!(settled_shift < first_shift / 5.0);

        // Still unit length.
        let norm: f32 = settled
            .get_speaker("alice")
            .unwrap()
            .embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ambiguous_match_does_not_adapt() {
        let mut registry = SpeakerRegistry::new();
        registry
            .enroll("alice", "Alice", make_embedding(&[1.0, 0.0, 0.0]))
            .unwrap();
        registry
            .enroll("bob", "Bob", make_embedding(&[0.9, 0.3, 0.0]))
            .unwrap();

        // Close to both: a high score, but the lead over the runner-up is
        // below the required margin.
        let obs = make_embedding(&[0.97, 0.15, 0.0]);
        let scores = registry.score(&obs).unwrap();
        let (id, score) = scores.best.clone().unwrap();
        assert!(score > DEFAULT_ADAPT_THRESHOLD);
        assert!(score - scores.runner_up < DEFAULT_ADAPT_MARGIN);
        let before = registry.get_speaker(&id).unwrap().embedding.clone();
        assert!(!registry.adapt(&id, &obs, score, scores.runner_up));
        assert_eq!(registry.get_speaker(&id).unwrap().embedding, before);
    }

    #[test]
    fn test_score_reports_runner_up() {
        let mut registry = SpeakerRegistry::new();
        assert_eq!(
            registry.score(&[1.0, 0.0, 0.0]).unwrap(),
            MatchScores {
                best: None,
                runner_up: -1.0
            }
        );

        registry.enroll("a", "A", vec![1.0, 0.0, 0.0]).unwrap();
        registry.enroll("b", "B", vec![0.0, 1.0, 0.0]).unwrap();
        registry.enroll("c", "C", vec![0.0, 0.0, 1.0]).unwrap();
        let scores = registry.score(&make_embedding(&[0.8, 0.6, 0.0])).unwrap();
        let (id, best) = scores.best.unwrap();
        assert_eq!(id, "a");
        assert!((best - 0.8).abs() < 1e-5);
        assert!((scores.runner_up - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_sample_count_is_capped() {
        let mut registry = SpeakerRegistry::new();
        let reference = make_embedding(&[1.0, 0.0, 0.0]);
        registry
            .enroll_with_count("alice", "Alice", reference.clone(), MAX_SAMPLE_COUNT)
            .unwrap();
        assert!(registry.adapt("alice", &reference, 1.0, -1.0));
        assert_eq!(registry.sample_count("alice"), Some(MAX_SAMPLE_COUNT));
    }

    #[test]
    fn test_merge_combines_two_identities() {
        let mut registry = SpeakerRegistry::new();
        registry
            .enroll_with_count("speaker_1", "Alice", make_embedding(&[1.0, 0.0, 0.0]), 3)
            .unwrap();
        registry
            .enroll_with_count(
                "speaker_2",
                "Speaker 2",
                make_embedding(&[0.0, 1.0, 0.0]),
                1,
            )
            .unwrap();

        let (merged, count) = registry.merge("speaker_2", "speaker_1").unwrap();
        assert_eq!(count, 4);
        assert_eq!(registry.speaker_count(), 1);
        assert!(registry.get_speaker("speaker_2").is_none());
        let target = registry.get_speaker("speaker_1").unwrap();
        assert_eq!(target.name, "Alice");
        assert_eq!(target.embedding, merged);
        // Weighted 3:1 towards the target's original reference.
        assert!(merged[0] > merged[1] * 2.5);

        assert!(registry.merge("speaker_1", "speaker_1").is_err());
        assert!(registry.merge("speaker_9", "speaker_1").is_err());
    }

    #[test]
    fn test_validate_speaker_name() {
        assert_eq!(validate_speaker_name("  Alice  ").unwrap(), "Alice");
        assert_eq!(
            validate_speaker_name("Zoë O'Neil-2").unwrap(),
            "Zoë O'Neil-2"
        );
        assert_eq!(validate_speaker_name("Çağrı").unwrap(), "Çağrı");
        assert!(validate_speaker_name("").is_err());
        assert!(validate_speaker_name("   ").is_err());
        assert!(validate_speaker_name("<script>").is_err());
        assert!(validate_speaker_name("a\nb").is_err());
        assert!(validate_speaker_name("a\u{0}b").is_err());
        assert!(validate_speaker_name(&"x".repeat(MAX_SPEAKER_NAME_CHARS)).is_ok());
        assert!(validate_speaker_name(&"x".repeat(MAX_SPEAKER_NAME_CHARS + 1)).is_err());
    }
}
