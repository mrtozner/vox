//! Speaker database location and diarization pipeline startup.

use std::sync::Arc;

use tracing::info;

use vox::diarization::{EMBEDDING_VERSION, EmbeddingVersionChange};

/// Resolve the path for the persistent speaker database.
///
/// Priority: `VOX_SPEAKER_DB` env var → OS data dir (`dirs::data_local_dir()/vox/speakers.db`)
/// → repo-relative fallback. The speaker DB is user data, not a model
/// artifact, so it lives outside `models/` by design.
#[cfg(feature = "diarization")]
fn speaker_db_path() -> std::path::PathBuf {
    if let Ok(custom) = std::env::var("VOX_SPEAKER_DB") {
        return std::path::PathBuf::from(custom);
    }
    if let Some(data_dir) = dirs::data_local_dir() {
        return data_dir.join("vox").join("speakers.db");
    }
    std::path::PathBuf::from("speakers.db")
}

/// Attempt to load the speaker encoder and hydrate the registry from SQLite.
/// Returns `(None, None)` if the model is missing so the server keeps
/// working with diarization disabled.
#[cfg(feature = "diarization")]
pub(super) async fn load_diarization(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<vox::DiarizationPipeline>>,
    Option<Arc<vox::SpeakerDatabase>>,
) {
    let encoder_path = models_dir.join("speaker_encoder.onnx");
    if !encoder_path.exists() {
        tracing::warn!(
            "speaker_encoder.onnx not found in {}, diarization disabled",
            models_dir.display()
        );
        return (None, None);
    }

    let embedding = match vox::SpeakerEmbedding::new(&encoder_path) {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load speaker encoder, diarization disabled");
            return (None, None);
        }
    };

    let db_path = speaker_db_path();
    match open_and_hydrate(&db_path, embedding).await {
        Ok((pipeline, db, speakers)) => {
            info!(
                model = %encoder_path.display(),
                db = %db_path.display(),
                speakers,
                "loaded diarization pipeline"
            );
            (Some(Arc::new(pipeline)), Some(Arc::new(db)))
        }
        Err(err) => {
            // Running with a registry that cannot be persisted would enroll
            // voices that are gone (or worse, un-deletable) after a restart.
            tracing::error!(
                path = %db_path.display(),
                error = %err,
                "speaker database is unusable, diarization disabled"
            );
            (None, None)
        }
    }
}

/// Open the speaker database (created owner-only if missing), reconcile the
/// embedding version, and build a pipeline hydrated with the stored
/// voiceprints. Returns the number of speakers loaded into the registry.
///
/// Any database failure is returned, not swallowed: in particular a failed
/// version write must not silently repeat the invalidation on every start.
#[cfg(feature = "diarization")]
async fn open_and_hydrate(
    db_path: &std::path::Path,
    embedding: vox::SpeakerEmbedding,
) -> Result<(vox::DiarizationPipeline, vox::SpeakerDatabase, usize), vox::VoxError> {
    let db = vox::SpeakerDatabase::open(db_path.to_string_lossy()).await?;

    // Embeddings from a different front end (CMVN, feature layout) are not
    // comparable. Only the voiceprints are invalidated; names, history and
    // preferences stay.
    if let EmbeddingVersionChange::Updated {
        previous,
        invalidated,
    } = db.apply_embedding_version(EMBEDDING_VERSION).await?
    {
        info!(
            old = previous.as_deref().unwrap_or("none"),
            new = EMBEDDING_VERSION,
            invalidated,
            "embedding version recorded; voiceprints from other versions invalidated"
        );
    }

    // Hydrate the in-memory registry from the persisted voiceprints.
    let model_dim = embedding.embedding_dim();
    let mut registry = vox::SpeakerRegistry::new();
    for (speaker, sample_count) in db.list_voiceprints().await? {
        let id = speaker.id.clone();
        // A voiceprint from a different encoder model cannot be compared
        // and would make every identification fail on a dimension mismatch.
        if model_dim != 0 && speaker.embedding.len() != model_dim {
            tracing::warn!(
                speaker_id = %id,
                stored_dim = speaker.embedding.len(),
                model_dim,
                "stored voiceprint does not match the speaker encoder, not loading it"
            );
            continue;
        }
        if let Err(err) =
            registry.enroll_with_count(speaker.id, speaker.name, speaker.embedding, sample_count)
        {
            tracing::warn!(
                speaker_id = %id,
                error = %err,
                "failed to hydrate speaker into registry"
            );
        }
    }
    let speakers = registry.speaker_count();

    // Enrollment is decided per utterance by the server (consent gate), so
    // the pipeline's own auto-enroll stays off. Reserving the database's
    // high-water mark keeps the ID of a deleted speaker from being reissued.
    let pipeline = vox::DiarizationPipelineBuilder::new()
        .embedding(embedding)
        .registry(registry)
        .auto_enroll(false)
        .reserve_speaker_numbers(db.speaker_sequence().await?)
        .build()?;

    Ok((pipeline, db, speakers))
}
