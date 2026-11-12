//! Speaker database location and diarization pipeline startup.

use std::sync::Arc;

use tracing::info;

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
        let vox_dir = data_dir.join("vox");
        let _ = std::fs::create_dir_all(&vox_dir);
        return vox_dir.join("speakers.db");
    }
    std::path::PathBuf::from("speakers.db")
}

/// Attempt to load the ECAPA-TDNN speaker encoder and hydrate the registry
/// from SQLite. Returns `(None, None)` if the model is missing so the server
/// keeps working with diarization disabled.
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
    let db_str = db_path.to_string_lossy().into_owned();
    // Ensure the database file exists before sqlx tries to open it.
    if !db_path.exists() {
        if let Err(err) = std::fs::File::create(&db_path) {
            tracing::warn!(
                path = %db_path.display(),
                error = %err,
                "failed to create speaker database file, diarization disabled"
            );
            return (None, None);
        }
    }
    let db = match vox::SpeakerDatabase::open(db_str).await {
        Ok(db) => db,
        Err(err) => {
            tracing::warn!(
                path = %db_path.display(),
                error = %err,
                "failed to open speaker database, diarization disabled"
            );
            return (None, None);
        }
    };

    // Clear stale speakers if embedding pipeline version changed.
    // Different preprocessing (CMVN, thresholds) produces incompatible embeddings.
    const EMBEDDING_VERSION: &str = "v2-cmvn";
    let stored_version = db
        .get_system_preference("embedding_version")
        .await
        .unwrap_or_default()
        .unwrap_or_default();
    if stored_version != EMBEDDING_VERSION {
        tracing::info!(
            old = %stored_version,
            new = EMBEDDING_VERSION,
            "embedding version changed, clearing stale speakers"
        );
        if let Err(e) = db.clear_speakers().await {
            tracing::warn!(error = %e, "failed to clear stale speakers");
        }
        let _ = db
            .set_system_preference("embedding_version", EMBEDDING_VERSION)
            .await;
    }

    // Hydrate in-memory registry from persisted speakers.
    let mut registry = vox::SpeakerRegistry::new();
    match db.list_speakers().await {
        Ok(speakers) => {
            for speaker in speakers {
                let id = speaker.id.clone();
                let name = speaker.name.clone();
                if let Err(err) = registry.enroll(id.clone(), name, speaker.embedding) {
                    tracing::warn!(
                        speaker_id = %id,
                        error = %err,
                        "failed to hydrate speaker into registry"
                    );
                }
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "failed to list speakers from database");
        }
    }

    let pipeline = match vox::DiarizationPipelineBuilder::new()
        .embedding(embedding)
        .registry(registry)
        .auto_enroll(true)
        .build()
    {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!(error = %err, "failed to build diarization pipeline");
            return (None, None);
        }
    };

    info!(
        model = %encoder_path.display(),
        db = %db_path.display(),
        "loaded diarization pipeline"
    );
    (Some(Arc::new(pipeline)), Some(Arc::new(db)))
}
