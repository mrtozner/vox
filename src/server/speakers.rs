//! Speaker operations shared by the WebSocket endpoints and the REST API:
//! identifying the speaker of an utterance, the public speaker list,
//! rename / forget / merge / export, the enrollment consent gate, and the
//! deterministic per-speaker color.
//!
//! Two rules hold for every mutation here:
//!
//! - **Database first, registry second.** The SQLite file is the source of
//!   truth. If the write fails the in-memory registry is left as it was, so
//!   the two can never disagree in the direction of "the server remembers a
//!   voice the user was told is gone".
//! - **No voiceprint without consent.** New enrollments and adaptation of
//!   stored references only happen while enrollment consent is recorded in
//!   the database. Without it, diarization is read-only.
//!
//! The `server` feature always enables `diarization`, so this module uses
//! the diarization types unconditionally.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::FromRef;
use serde::Serialize;

use crate::diarization::{
    IdentifyMode, SpeakerExport, SpeakerRecord, merge_embeddings, validate_speaker_name,
};

use super::ServerState;

/// How long adapted reference embeddings are held back before being written
/// to the database. A known speaker adapts on most utterances; this turns a
/// write per sentence into at most one write per interval, at the cost of
/// losing up to that much adaptation (never an enrollment) on a crash.
const ADAPTATION_FLUSH_DELAY: Duration = Duration::from_secs(5);

/// The speaker subsystem: the in-memory pipeline and the persistent store.
///
/// Both are `None` when the speaker encoder model is missing. Handlers take
/// this instead of the whole [`ServerState`] so they depend on nothing else.
#[derive(Clone, Default)]
pub(super) struct SpeakerCtx {
    pub(super) pipeline: Option<Arc<crate::DiarizationPipeline>>,
    pub(super) db: Option<Arc<crate::SpeakerDatabase>>,
}

impl SpeakerCtx {
    pub(super) fn from_state(state: &ServerState) -> Self {
        Self {
            pipeline: state.diarization.clone(),
            db: state.speaker_db.clone(),
        }
    }

    fn db(&self) -> Result<&Arc<crate::SpeakerDatabase>, SpeakerError> {
        self.db.as_ref().ok_or(SpeakerError::Unavailable)
    }
}

impl FromRef<Arc<ServerState>> for SpeakerCtx {
    fn from_ref(state: &Arc<ServerState>) -> Self {
        Self::from_state(state)
    }
}

/// Why a speaker operation failed.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SpeakerError {
    /// Diarization is not loaded on this server.
    Unavailable,
    /// No speaker with this ID.
    NotFound(String),
    /// The request itself is wrong (bad name, merging a speaker into itself).
    Invalid(String),
    /// The database or registry failed.
    Internal(String),
}

impl fmt::Display for SpeakerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "speaker diarization is not available on this server"),
            Self::NotFound(id) => write!(f, "speaker '{id}' not found"),
            Self::Invalid(msg) | Self::Internal(msg) => f.write_str(msg),
        }
    }
}

impl From<crate::VoxError> for SpeakerError {
    fn from(err: crate::VoxError) -> Self {
        Self::Internal(err.to_string())
    }
}

/// Client-facing view of an enrolled speaker.
///
/// The reference embedding is never serialized over the wire — it's a
/// biometric identifier and the client has no use for it.
#[derive(Serialize, Clone, Debug)]
pub(super) struct SpeakerPublic {
    pub(super) id: String,
    pub(super) name: String,
    /// Deterministic hex color derived from `id` via DJB2.
    pub(super) color: String,
}

/// A stored speaker as the REST API reports it.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub(super) struct SpeakerDetail {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) color: String,
    /// Unix seconds.
    pub(super) created_at: u64,
    /// Unix seconds.
    pub(super) last_seen: u64,
    /// Observations folded into the reference embedding.
    pub(super) sample_count: u32,
    /// `false` when the voiceprint was invalidated by a pipeline change; the
    /// speaker is kept but cannot be recognized until enrolled again.
    pub(super) has_voiceprint: bool,
}

impl From<SpeakerRecord> for SpeakerDetail {
    fn from(record: SpeakerRecord) -> Self {
        Self {
            color: color_for(&record.id),
            id: record.id,
            name: record.name,
            created_at: record.created_at,
            last_seen: record.last_seen,
            sample_count: record.sample_count,
            has_voiceprint: record.has_voiceprint,
        }
    }
}

/// Who spoke an utterance, and how sure we are.
#[derive(Clone, Debug)]
pub(super) struct SpeakerMatch {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) color: String,
    /// Cosine similarity to the matched speaker's reference (for a new
    /// enrollment: to the nearest previously known speaker).
    pub(super) confidence: f32,
    /// Set when this call enrolled a brand-new speaker.
    pub(super) new_speaker: Option<SpeakerPublic>,
}

/// State of the enrollment consent gate.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
pub(super) struct ConsentStatus {
    pub(super) granted: bool,
    /// Unix seconds at which consent was granted.
    pub(super) granted_at: Option<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EnrollmentPolicy {
    ReadOnly,
    Automatic,
    Explicit,
}

#[derive(Debug)]
pub(super) struct EnrollmentConsentRequired;

/// Colorblind-friendly palette for visually distinct speaker colors.
const SPEAKER_PALETTE: &[&str] = &[
    "#4e79a7", // steel blue
    "#f28e2c", // orange
    "#e15759", // coral red
    "#76b7b2", // teal
    "#59a14f", // green
    "#edc949", // gold
    "#af7aa1", // purple
    "#ff9da7", // pink
    "#9c755f", // brown
    "#bab0ab", // warm gray
];

pub(super) fn color_for(id: &str) -> String {
    // For auto-enrolled speaker_N IDs, use index for maximum distinction
    if let Some(n) = id
        .strip_prefix("speaker_")
        .and_then(|n| n.parse::<usize>().ok())
    {
        let idx = n.saturating_sub(1) % SPEAKER_PALETTE.len();
        return SPEAKER_PALETTE[idx].to_string();
    }
    // For custom/renamed IDs, DJB2 hash into the palette
    let mut hash: u32 = 5381;
    for b in id.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(*b as u32);
    }
    SPEAKER_PALETTE[(hash as usize) % SPEAKER_PALETTE.len()].to_string()
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

/// Identify who spoke `audio`.
///
/// Automatic enrollment falls back to read-only recognition without
/// consent. Explicit enrollment requires uninterrupted consent.
///
/// Returns `Ok(None)` when diarization is unavailable, the audio is too short,
/// nobody matched, or extraction failed (logged; never fatal to the turn).
pub(super) async fn identify_speaker(
    ctx: &SpeakerCtx,
    audio: &crate::AudioChunk,
    policy: EnrollmentPolicy,
) -> Result<Option<SpeakerMatch>, EnrollmentConsentRequired> {
    let revision = if policy == EnrollmentPolicy::ReadOnly {
        None
    } else {
        enrollment_revision(ctx).await
    };
    if policy == EnrollmentPolicy::Explicit && revision.is_none() {
        return Err(EnrollmentConsentRequired);
    }

    let (Some(pipeline), Some(_)) = (&ctx.pipeline, &ctx.db) else {
        return Ok(None);
    };
    match pipeline.identification_embedding(audio).await {
        Ok(Some(embedding)) => identify_embedding(ctx, embedding, policy, revision).await,
        Ok(None) => Ok(None),
        Err(e) => {
            tracing::warn!(error = %e, "speaker identification failed, continuing without");
            Ok(None)
        }
    }
}

/// Check the consent captured before extraction while holding the same
/// lock as withdrawal, then finish any writes before releasing the lock.
async fn identify_embedding(
    ctx: &SpeakerCtx,
    embedding: Vec<f32>,
    policy: EnrollmentPolicy,
    revision: Option<u64>,
) -> Result<Option<SpeakerMatch>, EnrollmentConsentRequired> {
    let (Some(revision), Some(db)) = (revision, ctx.db.as_ref()) else {
        return Ok(resolve_identification(ctx, embedding, IdentifyMode::IdentifyOnly).await);
    };

    let db = Arc::clone(db);
    let ctx = ctx.clone();
    // A disconnected request must not release the lock during a SQLite write.
    tokio::spawn(async move {
        let current = db.voiceprint_write.lock().await;
        if *current == revision {
            return Ok(resolve_identification(&ctx, embedding, IdentifyMode::AdaptAndEnroll).await);
        }
        drop(current);
        if policy == EnrollmentPolicy::Explicit {
            return Err(EnrollmentConsentRequired);
        }
        Ok(resolve_identification(&ctx, embedding, IdentifyMode::IdentifyOnly).await)
    })
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "speaker identification task failed");
        Ok(None)
    })
}

async fn resolve_identification(
    ctx: &SpeakerCtx,
    embedding: Vec<f32>,
    mode: IdentifyMode,
) -> Option<SpeakerMatch> {
    let pipeline = ctx.pipeline.as_ref()?;
    let db = ctx.db.as_ref()?;
    let identification = match pipeline.resolve_embedding(embedding, mode) {
        Ok(identification) => identification,
        Err(e) => {
            tracing::warn!(error = %e, "speaker identification failed, continuing without");
            return None;
        }
    };
    let id = identification.speaker_id?;

    let mut new_speaker = None;
    if identification.enrolled {
        // Persist the enrollment before anyone is told about it. If the
        // write fails, take the speaker back out of the registry: a
        // voiceprint must not live on in memory with no way to delete it.
        let (speaker, count) = pipeline.voiceprint(&id)?;
        if let Err(e) = db.store_speaker_with_count(&speaker, count).await {
            tracing::warn!(error = %e, speaker_id = %id, "failed to persist new speaker, discarding it");
            let _ = pipeline.with_registry(|registry| registry.forget(&id));
            return None;
        }
        new_speaker = Some(SpeakerPublic {
            id: speaker.id,
            name: speaker.name,
            color: color_for(&id),
        });
    } else if mode != IdentifyMode::IdentifyOnly {
        if identification.adapted {
            schedule_adaptation_flush(pipeline, db);
        }
        if let Err(e) = db.update_last_seen(&id).await {
            tracing::debug!(error = %e, "update_last_seen failed");
        }
    }

    let name = pipeline
        .with_registry(|registry| registry.get_speaker(&id).map(|s| s.name.clone()))
        .unwrap_or_else(|| id.clone());

    Some(SpeakerMatch {
        color: color_for(&id),
        id,
        name,
        confidence: identification.confidence,
        new_speaker,
    })
}

/// Run speaker diarization on a completed utterance and return the speaker
/// metadata to attach to the next `Transcript` event.
///
/// Returns `(speaker_id, speaker_name, speaker_color, optional SpeakerPublic
/// describing a newly enrolled speaker)`. The fourth element is `Some` only
/// when a brand-new speaker was enrolled during this call so the caller can
/// emit a `SpeakerUpdate`.
///
/// This is where the consent gate sits: unless enrollment consent is
/// recorded in the database, the utterance is matched against known
/// speakers read-only and nothing biometric is created or updated.
pub(super) async fn diarize_utterance(
    state: &Arc<ServerState>,
    diarization_on: bool,
    utterance: &crate::Utterance,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<SpeakerPublic>,
) {
    if !diarization_on {
        return (None, None, None, None);
    }
    let ctx = SpeakerCtx::from_state(state);
    match identify_speaker(&ctx, &utterance.audio, EnrollmentPolicy::Automatic).await {
        Ok(Some(m)) => {
            tracing::debug!(
                speaker_id = %m.id,
                confidence = m.confidence,
                enrolled = m.new_speaker.is_some(),
                "utterance attributed to speaker"
            );
            (Some(m.id), Some(m.name), Some(m.color), m.new_speaker)
        }
        _ => (None, None, None, None),
    }
}

/// Persist adapted references after [`ADAPTATION_FLUSH_DELAY`], unless a
/// flush is already on its way.
fn schedule_adaptation_flush(
    pipeline: &Arc<crate::DiarizationPipeline>,
    db: &Arc<crate::SpeakerDatabase>,
) {
    if !pipeline.claim_flush() {
        return;
    }
    let pipeline = Arc::clone(pipeline);
    let db = Arc::clone(db);
    tokio::spawn(async move {
        tokio::time::sleep(ADAPTATION_FLUSH_DELAY).await;
        let _write = db.voiceprint_write.lock().await;
        flush_adaptations(&pipeline, &db).await;
    });
}

/// Write every adapted-but-unsaved reference embedding to the database now.
async fn flush_adaptations(pipeline: &crate::DiarizationPipeline, db: &crate::SpeakerDatabase) {
    for (speaker, count) in pipeline.take_unsaved_adaptations() {
        if let Err(e) = db
            .update_speaker_embedding(&speaker.id, &speaker.embedding, count)
            .await
        {
            tracing::warn!(error = %e, speaker_id = %speaker.id, "failed to persist adapted speaker embedding");
        }
    }
}

/// Settle pending adaptation writes before an operation that reads or
/// rewrites stored embeddings.
async fn settle(ctx: &SpeakerCtx) {
    if let (Some(pipeline), Some(db)) = (ctx.pipeline.clone(), ctx.db.clone()) {
        let task = tokio::spawn(async move {
            let _write = db.voiceprint_write.lock().await;
            flush_adaptations(&pipeline, &db).await;
        });
        if let Err(e) = task.await {
            tracing::warn!(error = %e, "speaker adaptation flush task failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Consent
// ---------------------------------------------------------------------------

/// Current state of the enrollment consent gate. Anything that goes wrong
/// reading it counts as "not granted".
pub(super) async fn enrollment_consent(ctx: &SpeakerCtx) -> ConsentStatus {
    let granted_at = match ctx.db.as_ref() {
        Some(db) => db.enrollment_consent().await.unwrap_or_else(|e| {
            tracing::warn!(error = %e, "failed to read enrollment consent, treating as not granted");
            None
        }),
        None => None,
    };
    ConsentStatus {
        granted: granted_at.is_some(),
        granted_at,
    }
}

/// Snapshot the permission governing an utterance before extracting audio.
async fn enrollment_revision(ctx: &SpeakerCtx) -> Option<u64> {
    let db = ctx.db.as_ref()?;
    let revision = db.voiceprint_write.lock().await;
    enrollment_consent(ctx).await.granted.then_some(*revision)
}

/// Grant or withdraw consent to store voiceprints.
///
/// Withdrawing stops new enrollments and adaptation immediately; speakers
/// already stored stay until they are forgotten.
pub(super) async fn set_enrollment_consent(
    ctx: &SpeakerCtx,
    granted: bool,
) -> Result<ConsentStatus, SpeakerError> {
    let db = Arc::clone(ctx.db()?);
    let ctx = ctx.clone();
    tokio::spawn(async move {
        let mut revision = db.voiceprint_write.lock().await;
        if !granted && let Some(pipeline) = &ctx.pipeline {
            // Write any references adapted while consent was still in force.
            flush_adaptations(pipeline, &db).await;
        }
        db.set_enrollment_consent(granted).await?;
        if !granted {
            *revision = revision.wrapping_add(1);
        }
        tracing::info!(granted, "speaker enrollment consent changed");
        Ok(enrollment_consent(&ctx).await)
    })
    .await
    .map_err(|e| SpeakerError::Internal(format!("speaker consent task failed: {e}")))?
}

/// A message for the client when diarization was switched on but the server
/// will not enroll anyone because consent has not been recorded.
pub(super) async fn enrollment_blocked_notice(state: &Arc<ServerState>) -> Option<String> {
    let ctx = SpeakerCtx::from_state(state);
    ctx.db.as_ref()?;
    if enrollment_consent(&ctx).await.granted {
        return None;
    }
    Some(
        "Speaker enrollment consent is not recorded on the server: known speakers are still \
         recognized, but no new voices will be stored. Grant it with PUT /v1/speakers/consent \
         {\"granted\": true}."
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Listing
// ---------------------------------------------------------------------------

/// Snapshot the speakers that can currently be recognized, for `SpeakerList`
/// events.
pub(super) async fn current_speaker_list(state: &Arc<ServerState>) -> Option<Vec<SpeakerPublic>> {
    let pipeline = state.diarization.as_ref()?;
    let mut list: Vec<SpeakerPublic> = pipeline.with_registry(|registry| {
        registry
            .list_speakers()
            .into_iter()
            .map(|s| SpeakerPublic {
                id: s.id.clone(),
                name: s.name.clone(),
                color: color_for(&s.id),
            })
            .collect()
    });
    // The registry is a hash map; give clients a stable order.
    list.sort_by(|a, b| a.id.cmp(&b.id));
    Some(list)
}

/// Every stored speaker, most recently heard first.
pub(super) async fn list_speakers(ctx: &SpeakerCtx) -> Result<Vec<SpeakerDetail>, SpeakerError> {
    let records = ctx.db()?.list_records().await?;
    Ok(records.into_iter().map(SpeakerDetail::from).collect())
}

async fn require_speaker(
    db: &crate::SpeakerDatabase,
    id: &str,
) -> Result<SpeakerRecord, SpeakerError> {
    db.get_record(id)
        .await?
        .ok_or_else(|| SpeakerError::NotFound(id.to_string()))
}

// ---------------------------------------------------------------------------
// Mutations (database first, registry second)
// ---------------------------------------------------------------------------

/// Rename a speaker. The name is validated and trimmed first.
pub(super) async fn rename(
    ctx: &SpeakerCtx,
    id: &str,
    name: &str,
) -> Result<SpeakerDetail, SpeakerError> {
    let db = ctx.db()?;
    let name = validate_speaker_name(name).map_err(|e| SpeakerError::Invalid(e.to_string()))?;
    require_speaker(db, id).await?;

    db.update_speaker_name(id, &name).await?;
    if let Some(pipeline) = ctx.pipeline.as_ref() {
        // A speaker whose voiceprint was invalidated is in the database but
        // not in the registry; that is not an error.
        let _ = pipeline.with_registry(|registry| registry.rename(id, name.as_str()));
    }
    Ok(require_speaker(db, id).await?.into())
}

/// Forget a speaker: the voiceprint, the name, and everything recorded
/// about them.
pub(super) async fn forget(ctx: &SpeakerCtx, id: &str) -> Result<(), SpeakerError> {
    let db = ctx.db()?;
    require_speaker(db, id).await?;

    db.delete_speaker(id).await?;
    if let Some(pipeline) = ctx.pipeline.as_ref() {
        let _ = pipeline.with_registry(|registry| registry.forget(id));
    }
    tracing::info!(speaker_id = %id, "speaker forgotten");
    Ok(())
}

/// Forget every speaker. Returns how many were deleted.
pub(super) async fn forget_all(ctx: &SpeakerCtx) -> Result<u64, SpeakerError> {
    let db = ctx.db()?;
    let deleted = db.clear_speakers().await?;
    if let Some(pipeline) = ctx.pipeline.as_ref() {
        pipeline.with_registry(|registry| registry.clear());
    }
    tracing::info!(deleted, "all speakers forgotten");
    Ok(deleted)
}

/// Merge `source` into `target`. The target keeps its ID and name; the two
/// voiceprints are averaged by observation count, the source's history
/// moves over, and the source is deleted.
pub(super) async fn merge(
    ctx: &SpeakerCtx,
    source: &str,
    target: &str,
) -> Result<SpeakerDetail, SpeakerError> {
    let db = ctx.db()?;
    if source == target {
        return Err(SpeakerError::Invalid(
            "cannot merge a speaker into itself".into(),
        ));
    }
    settle(ctx).await;

    let load = |id: String| async move {
        db.load_voiceprint(&id)
            .await?
            .ok_or(SpeakerError::NotFound(id))
    };
    let (src, src_count) = load(source.to_string()).await?;
    let (dst, dst_count) = load(target.to_string()).await?;

    // Either side may have lost its voiceprint to a pipeline change.
    let (embedding, count) = match (dst.embedding.is_empty(), src.embedding.is_empty()) {
        (false, false) => {
            let merged = merge_embeddings(&dst.embedding, dst_count, &src.embedding, src_count)
                .ok_or_else(|| {
                    SpeakerError::Invalid(
                        "cannot merge speakers: their voiceprints have different dimensions".into(),
                    )
                })?;
            let count = src_count
                .saturating_add(dst_count)
                .min(crate::diarization::MAX_SAMPLE_COUNT);
            (merged, count)
        }
        (false, true) => (dst.embedding.clone(), dst_count),
        (true, false) => (src.embedding.clone(), src_count),
        (true, true) => (Vec::new(), 1),
    };

    db.merge_speakers(source, target, &embedding, count).await?;
    if let Some(pipeline) = ctx.pipeline.as_ref() {
        pipeline.with_registry(|registry| {
            let _ = registry.forget(source);
            if embedding.is_empty() {
                return;
            }
            if registry.get_speaker(target).is_some() {
                let _ = registry.set_embedding(target, embedding.clone(), count);
            } else if let Err(e) =
                registry.enroll_with_count(target, dst.name.as_str(), embedding.clone(), count)
            {
                tracing::warn!(error = %e, speaker_id = %target, "failed to load merged speaker into registry");
            }
        });
    }
    tracing::info!(source = %source, target = %target, "speakers merged");
    Ok(require_speaker(db, target).await?.into())
}

/// Everything stored about one speaker.
pub(super) async fn export(
    ctx: &SpeakerCtx,
    id: &str,
    include_embedding: bool,
) -> Result<SpeakerExport, SpeakerError> {
    let db = ctx.db()?;
    settle(ctx).await;
    db.export_speaker(id, include_embedding)
        .await?
        .ok_or_else(|| SpeakerError::NotFound(id.to_string()))
}

// ---------------------------------------------------------------------------
// WebSocket entry points (`/v1/listen`)
// ---------------------------------------------------------------------------

/// Rename a speaker in the database and the in-memory registry.
pub(super) async fn rename_speaker(
    state: &Arc<ServerState>,
    id: &str,
    name: &str,
) -> Result<(), SpeakerError> {
    rename(&SpeakerCtx::from_state(state), id, name)
        .await
        .map(|_| ())
}

/// Forget a speaker from the database and the in-memory registry.
pub(super) async fn forget_speaker(state: &Arc<ServerState>, id: &str) -> Result<(), SpeakerError> {
    forget(&SpeakerCtx::from_state(state), id).await
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::diarization::{DiarizationConfig, DiarizationPipeline, SpeakerRegistry};

    /// A speaker subsystem backed by an in-memory database and a pipeline
    /// with no encoder model.
    pub(crate) async fn test_ctx() -> SpeakerCtx {
        let db = crate::SpeakerDatabase::open(":memory:").await.unwrap();
        let pipeline = DiarizationPipeline::without_encoder(
            SpeakerRegistry::new(),
            DiarizationConfig::default(),
        );
        SpeakerCtx {
            pipeline: Some(Arc::new(pipeline)),
            db: Some(Arc::new(db)),
        }
    }

    /// Enroll a speaker in both the database and the registry.
    pub(crate) async fn enroll(ctx: &SpeakerCtx, id: &str, name: &str, embedding: &[f32]) {
        let speaker = crate::Speaker {
            id: id.to_string(),
            name: name.to_string(),
            embedding: embedding.to_vec(),
        };
        ctx.db
            .as_ref()
            .unwrap()
            .store_speaker(&speaker)
            .await
            .unwrap();
        ctx.pipeline
            .as_ref()
            .unwrap()
            .with_registry(|r| r.enroll(id, name, embedding.to_vec()))
            .unwrap();
    }

    fn registry_has(ctx: &SpeakerCtx, id: &str) -> bool {
        ctx.pipeline
            .as_ref()
            .unwrap()
            .with_registry(|r| r.get_speaker(id).is_some())
    }

    #[test]
    fn test_color_is_deterministic() {
        assert_eq!(color_for("speaker_1"), SPEAKER_PALETTE[0]);
        assert_eq!(color_for("speaker_11"), SPEAKER_PALETTE[0]);
        assert_eq!(color_for("alice"), color_for("alice"));
    }

    #[tokio::test]
    async fn test_unavailable_without_database() {
        let ctx = SpeakerCtx::default();
        assert_eq!(list_speakers(&ctx).await, Err(SpeakerError::Unavailable));
        assert_eq!(forget(&ctx, "x").await, Err(SpeakerError::Unavailable));
        assert!(!enrollment_consent(&ctx).await.granted);
        assert_eq!(
            set_enrollment_consent(&ctx, true).await,
            Err(SpeakerError::Unavailable)
        );
    }

    #[tokio::test]
    async fn test_rename_validates_then_updates_db_and_registry() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Speaker 1", &[1.0, 0.0]).await;

        assert!(matches!(
            rename(&ctx, "speaker_1", "<b>x</b>").await,
            Err(SpeakerError::Invalid(_))
        ));
        assert!(matches!(
            rename(&ctx, "speaker_1", &"x".repeat(200)).await,
            Err(SpeakerError::Invalid(_))
        ));
        assert_eq!(
            rename(&ctx, "ghost", "Alice").await,
            Err(SpeakerError::NotFound("ghost".into()))
        );

        let detail = rename(&ctx, "speaker_1", "  Alice ").await.unwrap();
        assert_eq!(detail.name, "Alice");
        let registry_name = ctx
            .pipeline
            .as_ref()
            .unwrap()
            .with_registry(|r| r.get_speaker("speaker_1").map(|s| s.name.clone()));
        assert_eq!(registry_name.as_deref(), Some("Alice"));
    }

    #[tokio::test]
    async fn test_forget_removes_from_db_and_registry() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        enroll(&ctx, "speaker_2", "Bob", &[0.0, 1.0]).await;

        forget(&ctx, "speaker_1").await.unwrap();
        assert!(!registry_has(&ctx, "speaker_1"));
        assert!(registry_has(&ctx, "speaker_2"));
        let ids: Vec<String> = list_speakers(&ctx)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(ids, vec!["speaker_2"]);
        assert_eq!(
            forget(&ctx, "speaker_1").await,
            Err(SpeakerError::NotFound("speaker_1".into()))
        );

        assert_eq!(forget_all(&ctx).await.unwrap(), 1);
        assert!(!registry_has(&ctx, "speaker_2"));
        assert!(list_speakers(&ctx).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_merge_combines_two_identities() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        enroll(&ctx, "speaker_2", "Speaker 2", &[0.0, 1.0]).await;
        let db = ctx.db.as_ref().unwrap();
        db.record_utterance("speaker_2", 900, Some("hi"))
            .await
            .unwrap();

        assert!(matches!(
            merge(&ctx, "speaker_1", "speaker_1").await,
            Err(SpeakerError::Invalid(_))
        ));
        assert_eq!(
            merge(&ctx, "ghost", "speaker_1").await,
            Err(SpeakerError::NotFound("ghost".into()))
        );

        let merged = merge(&ctx, "speaker_2", "speaker_1").await.unwrap();
        assert_eq!(merged.id, "speaker_1");
        assert_eq!(merged.name, "Alice");
        assert_eq!(merged.sample_count, 2);

        assert!(!registry_has(&ctx, "speaker_2"));
        let (speaker, count) = ctx
            .pipeline
            .as_ref()
            .unwrap()
            .voiceprint("speaker_1")
            .unwrap();
        assert_eq!(count, 2);
        // Halfway between the two references, and the same in the database.
        assert!((speaker.embedding[0] - speaker.embedding[1]).abs() < 1e-6);
        let (stored, _) = db.load_voiceprint("speaker_1").await.unwrap().unwrap();
        assert_eq!(stored.embedding, speaker.embedding);
        assert_eq!(
            db.get_conversation_history("speaker_1", 10)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_adaptation_is_flushed_before_export() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0, 0.0]).await;
        let pipeline = ctx.pipeline.as_ref().unwrap();

        let result = pipeline
            .resolve_embedding(vec![0.6, 0.8, 0.0], IdentifyMode::Adapt)
            .unwrap();
        assert!(result.adapted);
        assert!(pipeline.has_unsaved_adaptations());

        let exported = export(&ctx, "speaker_1", true).await.unwrap();
        assert!(!pipeline.has_unsaved_adaptations());
        assert_eq!(
            exported.embedding.unwrap(),
            pipeline.voiceprint("speaker_1").unwrap().0.embedding
        );
        assert_eq!(exported.speaker.sample_count, 2);

        assert!(
            export(&ctx, "speaker_1", false)
                .await
                .unwrap()
                .embedding
                .is_none()
        );
        assert_eq!(
            export(&ctx, "ghost", false).await.map(|_| ()),
            Err(SpeakerError::NotFound("ghost".into()))
        );
    }

    #[tokio::test]
    async fn test_no_voiceprint_is_stored_without_consent() {
        let ctx = test_ctx().await;
        let pipeline = ctx.pipeline.as_ref().unwrap();
        let db = ctx.db.as_ref().unwrap();
        let voice = vec![0.0, 1.0, 0.0];

        // Consent not recorded: an unknown voice stays unknown and leaves
        // no trace in the registry or the database.
        let revision = enrollment_revision(&ctx).await;
        assert!(revision.is_none());
        assert!(
            identify_embedding(&ctx, voice.clone(), EnrollmentPolicy::Automatic, revision)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(pipeline.with_registry(|r| r.speaker_count()), 0);
        assert!(db.list_records().await.unwrap().is_empty());

        // Consent recorded: the same voice is enrolled and persisted.
        set_enrollment_consent(&ctx, true).await.unwrap();
        let revision = enrollment_revision(&ctx).await;
        assert!(revision.is_some());
        let matched =
            identify_embedding(&ctx, voice.clone(), EnrollmentPolicy::Automatic, revision)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(matched.id, "speaker_1");
        assert_eq!(matched.name, "Speaker 1");
        assert_eq!(matched.new_speaker.as_ref().unwrap().id, "speaker_1");
        let (stored, _) = db.load_voiceprint("speaker_1").await.unwrap().unwrap();
        assert_eq!(stored.embedding, voice);

        // Heard again: recognized with its score, not enrolled twice.
        let matched =
            identify_embedding(&ctx, voice.clone(), EnrollmentPolicy::Automatic, revision)
                .await
                .unwrap()
                .unwrap();
        assert!(matched.new_speaker.is_none());
        assert!((matched.confidence - 1.0).abs() < 1e-6);

        // Consent withdrawn: recognition still works, enrollment stops.
        set_enrollment_consent(&ctx, false).await.unwrap();
        let stranger = identify_embedding(
            &ctx,
            vec![0.0, 0.0, 1.0],
            EnrollmentPolicy::Automatic,
            revision,
        )
        .await
        .unwrap();
        assert!(stranger.is_none());
        assert_eq!(
            identify_embedding(&ctx, voice, EnrollmentPolicy::Automatic, revision)
                .await
                .unwrap()
                .unwrap()
                .id,
            "speaker_1"
        );
        assert_eq!(db.list_records().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_revocation_invalidates_pending_audio_even_after_regrant() {
        let ctx = test_ctx().await;
        let pipeline = ctx.pipeline.as_ref().unwrap();
        let db = ctx.db.as_ref().unwrap();
        let reference = vec![1.0, 0.0, 0.0];
        enroll(&ctx, "alice", "Alice", &reference).await;
        set_enrollment_consent(&ctx, true).await.unwrap();

        // Suspend automatic recognition and explicit enrollment at the
        // extraction boundary. A fresh grant cannot authorize this audio.
        let revision = enrollment_revision(&ctx).await;
        assert!(revision.is_some());
        let consent_ctx = ctx.clone();
        set_enrollment_consent(&consent_ctx, false).await.unwrap();
        set_enrollment_consent(&consent_ctx, true).await.unwrap();
        let matched = identify_embedding(
            &ctx,
            vec![0.6, 0.8, 0.0],
            EnrollmentPolicy::Automatic,
            revision,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(matched.id, "alice");
        assert!(!pipeline.has_unsaved_adaptations());
        let (in_memory, count) = pipeline.voiceprint("alice").unwrap();
        assert_eq!((in_memory.embedding, count), (reference.clone(), 1));
        let (stored, count) = db.load_voiceprint("alice").await.unwrap().unwrap();
        assert_eq!((stored.embedding, count), (reference, 1));

        let stranger = vec![0.0, 0.0, 1.0];
        assert!(matches!(
            identify_embedding(&ctx, stranger.clone(), EnrollmentPolicy::Explicit, revision).await,
            Err(EnrollmentConsentRequired)
        ));
        assert!(!registry_has(&ctx, "speaker_1"));
        assert!(db.load_voiceprint("speaker_1").await.unwrap().is_none());

        let fresh_revision = enrollment_revision(&ctx).await;
        let fresh = identify_embedding(&ctx, stranger, EnrollmentPolicy::Explicit, fresh_revision)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fresh.new_speaker.unwrap().id, "speaker_1");
    }

    #[tokio::test]
    async fn test_consent_round_trip() {
        let ctx = test_ctx().await;
        assert_eq!(
            enrollment_consent(&ctx).await,
            ConsentStatus {
                granted: false,
                granted_at: None
            }
        );
        let status = set_enrollment_consent(&ctx, true).await.unwrap();
        assert!(status.granted && status.granted_at.is_some());
        let status = set_enrollment_consent(&ctx, false).await.unwrap();
        assert!(!status.granted);
    }
}
