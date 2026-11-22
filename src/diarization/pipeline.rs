//! Multi-speaker conversation pipeline.
//!
//! This module integrates speaker diarization with the existing VAD+STT pipeline,
//! enabling speaker identification for each utterance in multi-speaker conversations.
//!
//! Callers choose how much an utterance may change the registry through
//! [`IdentifyMode`]: nothing at all ([`DiarizationPipeline::identify_only`]),
//! adapting known speakers, or also enrolling unknown ones.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::VoxError;
use crate::types::{AudioChunk, Utterance};

use super::embedding::{MIN_SPEAKER_AUDIO_MS, SpeakerEmbedding};
use super::recognition::{Speaker, SpeakerRegistry};

/// Configuration for the multi-speaker pipeline.
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Whether [`DiarizationPipeline::process_utterance`] enrolls unknown
    /// speakers automatically.
    pub auto_enroll: bool,
    /// Minimum audio duration (ms) for speaker identification. Defaults to
    /// [`MIN_SPEAKER_AUDIO_MS`]; the extractor rejects anything shorter.
    pub min_audio_ms: u64,
    /// Whether to skip identification for very short utterances.
    pub skip_short_utterances: bool,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            auto_enroll: false,
            min_audio_ms: MIN_SPEAKER_AUDIO_MS,
            skip_short_utterances: true,
        }
    }
}

/// How much an identification is allowed to change the speaker registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentifyMode {
    /// Read-only: never adapts a reference and never enrolls anyone.
    IdentifyOnly,
    /// Known speakers adapt on a confident match; unknown voices stay unknown.
    Adapt,
    /// As [`IdentifyMode::Adapt`], and an unknown voice is enrolled as a new
    /// `speaker_N`.
    AdaptAndEnroll,
}

/// Outcome of identifying one utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Identification {
    /// The matched (or newly enrolled) speaker, or `None` if unknown.
    pub speaker_id: Option<String>,
    /// Cosine similarity to the closest enrolled speaker (0.0 when nobody is
    /// enrolled or the audio was too short). For a new enrollment this is
    /// how close the nearest *existing* speaker was.
    pub confidence: f32,
    /// A new speaker was enrolled by this call.
    pub enrolled: bool,
    /// The matched speaker's reference embedding was adapted by this call.
    pub adapted: bool,
}

impl Identification {
    fn unknown(confidence: f32) -> Self {
        Self {
            speaker_id: None,
            confidence,
            enrolled: false,
            adapted: false,
        }
    }
}

/// Multi-speaker diarization pipeline.
///
/// Integrates speaker embedding extraction and recognition with the voice pipeline.
/// Processes utterances from VAD and adds speaker identification.
pub struct DiarizationPipeline {
    /// `None` only for a pipeline built with
    /// [`DiarizationPipeline::without_encoder`].
    embedding: Option<Arc<Mutex<SpeakerEmbedding>>>,
    registry: Arc<Mutex<SpeakerRegistry>>,
    config: DiarizationConfig,
    /// Highest `speaker_N` number handed out so far.
    unknown_counter: Mutex<u32>,
    /// Speakers whose adapted reference has not been persisted yet.
    dirty: Mutex<HashSet<String>>,
    /// Whether a flush of `dirty` is already scheduled.
    flush_scheduled: AtomicBool,
}

/// Lock a mutex, recovering the data if another thread panicked while
/// holding it. The registry and counters stay internally consistent across
/// every operation here, so a poisoned lock is not a reason to take the
/// whole server down.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Highest `N` among `speaker_N` IDs in the registry.
fn max_auto_number(registry: &SpeakerRegistry) -> u32 {
    registry
        .list_speakers()
        .iter()
        .filter_map(|s| {
            s.id.strip_prefix("speaker_")
                .and_then(|n| n.parse::<u32>().ok())
        })
        .max()
        .unwrap_or(0)
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
        Self::from_parts(Some(embedding), registry, config)
    }

    /// Create a pipeline with no embedding model.
    ///
    /// Everything that works on embeddings or on the registry is available
    /// ([`resolve_embedding`](Self::resolve_embedding), rename, merge, ...);
    /// anything that needs audio returns an error. Intended for tests and
    /// for callers that compute embeddings elsewhere.
    #[doc(hidden)]
    pub fn without_encoder(registry: SpeakerRegistry, config: DiarizationConfig) -> Self {
        Self::from_parts(None, registry, config)
    }

    fn from_parts(
        embedding: Option<SpeakerEmbedding>,
        registry: SpeakerRegistry,
        config: DiarizationConfig,
    ) -> Self {
        // Initialize counter from existing speakers to avoid ID collisions
        // when speakers are pre-loaded from the database at startup.
        let max_num = max_auto_number(&registry);

        Self {
            embedding: embedding.map(|e| Arc::new(Mutex::new(e))),
            registry: Arc::new(Mutex::new(registry)),
            config,
            unknown_counter: Mutex::new(max_num),
            dirty: Mutex::new(HashSet::new()),
            flush_scheduled: AtomicBool::new(false),
        }
    }

    /// Never hand out a `speaker_N` ID with `N <= floor`.
    ///
    /// The registry only knows the speakers that still exist. Pass the
    /// database's high-water mark here so the ID of a deleted speaker is not
    /// issued again to somebody else.
    pub fn reserve_speaker_numbers(&self, floor: u32) {
        let mut counter = lock(&self.unknown_counter);
        *counter = (*counter).max(floor);
    }

    /// Extract a speaker embedding on a blocking thread — ONNX Runtime uses
    /// thread-local caching internally and must not run on the async
    /// executor where tasks can migrate between OS threads.
    async fn embed(&self, audio: &AudioChunk) -> Result<Vec<f32>, VoxError> {
        let emb_arc =
            self.embedding.as_ref().map(Arc::clone).ok_or_else(|| {
                VoxError::Diarization("no speaker embedding model is loaded".into())
            })?;
        let audio = audio.clone();
        tokio::task::spawn_blocking(move || lock(&emb_arc).extract(&audio))
            .await
            .map_err(|e| VoxError::Diarization(format!("embedding task panicked: {e}")))?
    }

    /// Whether audio of this length is long enough to identify a speaker.
    fn long_enough(&self, duration_ms: u64) -> bool {
        !self.config.skip_short_utterances || duration_ms >= self.config.min_audio_ms
    }

    /// Identify who is speaking without changing anything: no enrollment,
    /// no adaptation of stored references.
    ///
    /// Audio shorter than the configured minimum yields an unknown speaker
    /// with confidence 0.
    pub async fn identify_only(&self, audio: &AudioChunk) -> Result<Identification, VoxError> {
        self.identify(audio, IdentifyMode::IdentifyOnly).await
    }

    /// Identify who is speaking, updating the registry as far as `mode`
    /// allows. Returns the match together with its confidence score.
    pub async fn identify(
        &self,
        audio: &AudioChunk,
        mode: IdentifyMode,
    ) -> Result<Identification, VoxError> {
        let Some(embedding) = self.identification_embedding(audio).await? else {
            return Ok(Identification::unknown(0.0));
        };
        self.resolve_embedding(embedding, mode)
    }

    /// Extract the embedding for identification without altering the registry.
    pub(crate) async fn identification_embedding(
        &self,
        audio: &AudioChunk,
    ) -> Result<Option<Vec<f32>>, VoxError> {
        if !self.long_enough(audio.duration_ms()) {
            return Ok(None);
        }
        self.embed(audio).await.map(Some)
    }

    /// The decision step of [`identify`](Self::identify), for an embedding
    /// that has already been extracted.
    pub fn resolve_embedding(
        &self,
        embedding: Vec<f32>,
        mode: IdentifyMode,
    ) -> Result<Identification, VoxError> {
        let mut registry = lock(&self.registry);
        let scores = registry.score(&embedding)?;

        let best_score = match scores.best {
            Some((speaker_id, score)) if registry.accepts(score) => {
                tracing::debug!(
                    speaker_id = %speaker_id,
                    confidence = %score,
                    "speaker identified"
                );
                // Only a confident, unambiguous match may move the stored
                // reference; a borderline one just labels the utterance.
                let adapted = mode != IdentifyMode::IdentifyOnly
                    && registry.adapt(&speaker_id, &embedding, score, scores.runner_up);
                if adapted {
                    lock(&self.dirty).insert(speaker_id.clone());
                }
                return Ok(Identification {
                    speaker_id: Some(speaker_id),
                    confidence: score,
                    enrolled: false,
                    adapted,
                });
            }
            Some((_, score)) => score,
            None => 0.0,
        };

        tracing::debug!(best_score = %best_score, "unknown speaker detected");
        if mode != IdentifyMode::AdaptAndEnroll {
            return Ok(Identification::unknown(best_score));
        }

        let number = {
            let mut counter = lock(&self.unknown_counter);
            *counter += 1;
            *counter
        };
        let id = format!("speaker_{number}");
        registry.enroll(&id, format!("Speaker {number}"), embedding)?;
        tracing::info!(speaker_id = %id, "auto-enrolled new speaker");
        Ok(Identification {
            speaker_id: Some(id),
            confidence: best_score,
            enrolled: true,
            adapted: false,
        })
    }

    /// Process an utterance and identify the speaker.
    ///
    /// Unknown speakers are enrolled when `auto_enroll` is configured.
    ///
    /// # Arguments
    /// * `utterance` - The utterance to process (from VAD)
    ///
    /// # Returns
    /// Updated utterance with speaker_id field populated (`"unknown"` when
    /// nobody matched, the audio was too short, or extraction failed).
    pub async fn process_utterance(&self, mut utterance: Utterance) -> Result<Utterance, VoxError> {
        // Skip short utterances if configured
        if !self.long_enough(utterance.duration_ms) {
            utterance.speaker_id = Some("unknown".to_string());
            return Ok(utterance);
        }

        let embedding = match self.embed(&utterance.audio).await {
            Ok(emb) => emb,
            Err(e) => {
                tracing::warn!("failed to extract speaker embedding: {e}");
                utterance.speaker_id = Some("unknown".to_string());
                return Ok(utterance);
            }
        };

        let mode = if self.config.auto_enroll {
            IdentifyMode::AdaptAndEnroll
        } else {
            IdentifyMode::Adapt
        };
        let identification = self.resolve_embedding(embedding, mode)?;
        utterance.speaker_id = Some(
            identification
                .speaker_id
                .unwrap_or_else(|| "unknown".to_string()),
        );
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
        let embedding = self.embed(audio).await?;
        lock(&self.registry).enroll(id, name, embedding)?;
        Ok(())
    }

    /// Get the speaker registry (for listing speakers, etc.).
    ///
    /// Prefer [`with_registry`](Self::with_registry), which cannot fail on a
    /// poisoned lock.
    pub fn registry(&self) -> Arc<Mutex<SpeakerRegistry>> {
        Arc::clone(&self.registry)
    }

    /// Run `f` with exclusive access to the speaker registry.
    ///
    /// Do not call back into the pipeline from `f`.
    pub fn with_registry<R>(&self, f: impl FnOnce(&mut SpeakerRegistry) -> R) -> R {
        f(&mut lock(&self.registry))
    }

    /// A speaker's current reference and the observation count behind it.
    pub fn voiceprint(&self, id: &str) -> Option<(Speaker, u32)> {
        let registry = lock(&self.registry);
        let speaker = registry.get_speaker(id)?.clone();
        let count = registry.sample_count(id).unwrap_or(1);
        Some((speaker, count))
    }

    /// Whether any adapted reference is waiting to be persisted.
    pub fn has_unsaved_adaptations(&self) -> bool {
        !lock(&self.dirty).is_empty()
    }

    /// Claim the right to schedule a flush of adapted references.
    ///
    /// Returns `true` to exactly one caller until the next
    /// [`take_unsaved_adaptations`](Self::take_unsaved_adaptations); that
    /// caller should wait out its debounce interval and then persist what
    /// `take_unsaved_adaptations` returns. Adaptation happens on almost
    /// every utterance of an enrolled speaker, so writing each one through
    /// would mean a database write per sentence.
    pub fn claim_flush(&self) -> bool {
        !self.flush_scheduled.swap(true, Ordering::AcqRel)
    }

    /// Take the adapted references that have not been persisted yet, as
    /// `(speaker, sample_count)` snapshots of the current registry state.
    /// Speakers forgotten in the meantime are dropped.
    pub fn take_unsaved_adaptations(&self) -> Vec<(Speaker, u32)> {
        // Reopen scheduling first: an adaptation that lands after the drain
        // below must be able to schedule its own flush.
        self.flush_scheduled.store(false, Ordering::Release);
        let ids: Vec<String> = lock(&self.dirty).drain().collect();
        let registry = lock(&self.registry);
        ids.iter()
            .filter_map(|id| {
                let speaker = registry.get_speaker(id)?.clone();
                Some((speaker, registry.sample_count(id).unwrap_or(1)))
            })
            .collect()
    }
}

/// Builder for DiarizationPipeline.
pub struct DiarizationPipelineBuilder {
    embedding: Option<SpeakerEmbedding>,
    registry: Option<SpeakerRegistry>,
    config: DiarizationConfig,
    reserved_speaker_numbers: u32,
}

impl DiarizationPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            embedding: None,
            registry: None,
            config: DiarizationConfig::default(),
            reserved_speaker_numbers: 0,
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

    /// Never hand out a `speaker_N` ID with `N <= floor`
    /// (see [`DiarizationPipeline::reserve_speaker_numbers`]).
    pub fn reserve_speaker_numbers(mut self, floor: u32) -> Self {
        self.reserved_speaker_numbers = floor;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> Result<DiarizationPipeline, VoxError> {
        let embedding = self
            .embedding
            .ok_or_else(|| VoxError::Diarization("speaker embedding extractor not set".into()))?;

        let registry = self.registry.unwrap_or_default();

        let pipeline = DiarizationPipeline::new(embedding, registry, self.config);
        pipeline.reserve_speaker_numbers(self.reserved_speaker_numbers);
        Ok(pipeline)
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

    fn pipeline_with(speakers: &[(&str, &[f32])]) -> DiarizationPipeline {
        let mut registry = SpeakerRegistry::new();
        for (id, embedding) in speakers {
            registry.enroll(*id, *id, embedding.to_vec()).unwrap();
        }
        DiarizationPipeline::without_encoder(registry, DiarizationConfig::default())
    }

    /// Unit vector with cosine similarity `score` to `[1, 0, 0]`.
    fn at_score(score: f32) -> Vec<f32> {
        vec![score, (1.0 - score * score).sqrt(), 0.0]
    }

    #[test]
    fn test_default_config() {
        let config = DiarizationConfig::default();
        assert!(!config.auto_enroll);
        // One constant for "too short", shared with the extractor.
        assert_eq!(config.min_audio_ms, MIN_SPEAKER_AUDIO_MS);
        assert!(config.skip_short_utterances);
    }

    #[test]
    fn test_builder_missing_embedding() {
        let builder = DiarizationPipelineBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_identify_only_does_not_enroll_unknown_voice() {
        let pipeline = pipeline_with(&[("speaker_1", &[1.0, 0.0, 0.0])]);

        let result = pipeline
            .resolve_embedding(vec![0.0, 1.0, 0.0], IdentifyMode::IdentifyOnly)
            .unwrap();
        assert_eq!(result.speaker_id, None);
        assert!(!result.enrolled);
        assert!(result.confidence.abs() < 1e-6);
        assert_eq!(pipeline.with_registry(|r| r.speaker_count()), 1);

        // With nobody enrolled at all it still creates nothing.
        let empty = pipeline_with(&[]);
        let result = empty
            .resolve_embedding(vec![0.0, 1.0, 0.0], IdentifyMode::IdentifyOnly)
            .unwrap();
        assert_eq!(result, Identification::unknown(0.0));
        assert_eq!(empty.with_registry(|r| r.speaker_count()), 0);
    }

    #[test]
    fn test_identify_only_does_not_adapt_known_voice() {
        let reference = vec![1.0, 0.0, 0.0];
        let pipeline = pipeline_with(&[("speaker_1", &reference)]);

        let result = pipeline
            .resolve_embedding(at_score(0.9), IdentifyMode::IdentifyOnly)
            .unwrap();
        assert_eq!(result.speaker_id.as_deref(), Some("speaker_1"));
        assert!((result.confidence - 0.9).abs() < 1e-4);
        assert!(!result.adapted);
        assert_eq!(
            pipeline.voiceprint("speaker_1").unwrap().0.embedding,
            reference
        );
        assert!(!pipeline.has_unsaved_adaptations());
    }

    #[test]
    fn test_adapt_mode_respects_both_thresholds() {
        let reference = vec![1.0, 0.0, 0.0];
        let pipeline = pipeline_with(&[("speaker_1", &reference)]);

        // Accepted (>= 0.35) but too weak to learn from (< 0.55).
        let weak = pipeline
            .resolve_embedding(at_score(0.36), IdentifyMode::Adapt)
            .unwrap();
        assert_eq!(weak.speaker_id.as_deref(), Some("speaker_1"));
        assert!(!weak.adapted);
        assert_eq!(
            pipeline.voiceprint("speaker_1").unwrap().0.embedding,
            reference
        );
        assert!(!pipeline.has_unsaved_adaptations());

        let strong = pipeline
            .resolve_embedding(at_score(0.6), IdentifyMode::Adapt)
            .unwrap();
        assert!(strong.adapted);
        assert!((strong.confidence - 0.6).abs() < 1e-4);
        let (speaker, count) = pipeline.voiceprint("speaker_1").unwrap();
        assert_ne!(speaker.embedding, reference);
        assert_eq!(count, 2);

        // Adapt mode never enrolls.
        let stranger = pipeline
            .resolve_embedding(vec![0.0, 0.0, 1.0], IdentifyMode::Adapt)
            .unwrap();
        assert_eq!(stranger.speaker_id, None);
        assert_eq!(pipeline.with_registry(|r| r.speaker_count()), 1);
    }

    #[test]
    fn test_unsaved_adaptations_are_claimed_once_and_drained() {
        let pipeline = pipeline_with(&[("speaker_1", &[1.0, 0.0, 0.0])]);
        pipeline
            .resolve_embedding(at_score(0.7), IdentifyMode::Adapt)
            .unwrap();
        pipeline
            .resolve_embedding(at_score(0.7), IdentifyMode::Adapt)
            .unwrap();
        assert!(pipeline.has_unsaved_adaptations());

        // Two adaptations inside one debounce window: one flush.
        assert!(pipeline.claim_flush());
        assert!(!pipeline.claim_flush());

        let pending = pipeline.take_unsaved_adaptations();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0.id, "speaker_1");
        assert_eq!(pending[0].1, 3);
        assert!(!pipeline.has_unsaved_adaptations());
        assert!(pipeline.take_unsaved_adaptations().is_empty());

        // A later adaptation can schedule again.
        assert!(pipeline.claim_flush());
    }

    #[test]
    fn test_forgotten_speaker_is_not_flushed() {
        let pipeline = pipeline_with(&[("speaker_1", &[1.0, 0.0, 0.0])]);
        pipeline
            .resolve_embedding(at_score(0.7), IdentifyMode::Adapt)
            .unwrap();
        pipeline.with_registry(|r| r.forget("speaker_1")).unwrap();
        assert!(pipeline.take_unsaved_adaptations().is_empty());
    }

    #[test]
    fn test_enroll_mode_numbers_new_speakers() {
        let pipeline = pipeline_with(&[("speaker_2", &[1.0, 0.0, 0.0])]);

        let result = pipeline
            .resolve_embedding(vec![0.0, 1.0, 0.0], IdentifyMode::AdaptAndEnroll)
            .unwrap();
        assert_eq!(result.speaker_id.as_deref(), Some("speaker_3"));
        assert!(result.enrolled);
        let name = pipeline.with_registry(|r| r.get_speaker("speaker_3").map(|s| s.name.clone()));
        assert_eq!(name.as_deref(), Some("Speaker 3"));

        // The same voice again is recognized, not enrolled twice.
        let again = pipeline
            .resolve_embedding(vec![0.0, 1.0, 0.0], IdentifyMode::AdaptAndEnroll)
            .unwrap();
        assert_eq!(again.speaker_id.as_deref(), Some("speaker_3"));
        assert!(!again.enrolled);
    }

    #[test]
    fn test_reserved_numbers_are_not_reused() {
        // speaker_3 existed, was deleted, and the process restarted: the
        // registry only holds speaker_1, the database remembers 3.
        let pipeline = pipeline_with(&[("speaker_1", &[1.0, 0.0, 0.0])]);
        pipeline.reserve_speaker_numbers(3);

        let result = pipeline
            .resolve_embedding(vec![0.0, 1.0, 0.0], IdentifyMode::AdaptAndEnroll)
            .unwrap();
        assert_eq!(result.speaker_id.as_deref(), Some("speaker_4"));

        // A lower floor never moves the counter backwards.
        pipeline.reserve_speaker_numbers(1);
        let result = pipeline
            .resolve_embedding(vec![0.0, 0.0, 1.0], IdentifyMode::AdaptAndEnroll)
            .unwrap();
        assert_eq!(result.speaker_id.as_deref(), Some("speaker_5"));
    }

    #[tokio::test]
    async fn test_audio_paths_without_encoder() {
        let pipeline = pipeline_with(&[]);
        let short = AudioChunk {
            samples: vec![0.0; 1600], // 100 ms
            sample_rate: 16_000,
            channels: 1,
        };
        // Too short: unknown, no error, and the encoder is never needed.
        assert_eq!(
            pipeline.identify_only(&short).await.unwrap(),
            Identification::unknown(0.0)
        );

        let long = AudioChunk {
            samples: vec![0.0; 32_000], // 2 s
            sample_rate: 16_000,
            channels: 1,
        };
        assert!(pipeline.identify_only(&long).await.is_err());
        // process_utterance degrades to "unknown" instead of failing.
        let utterance = pipeline
            .process_utterance(Utterance {
                audio: long,
                duration_ms: 2000,
                speaker_id: None,
            })
            .await
            .unwrap();
        assert_eq!(utterance.speaker_id.as_deref(), Some("unknown"));
    }
}
