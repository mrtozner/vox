//! Speaker helpers shared by the WebSocket endpoints: diarization of a
//! completed utterance, the public speaker list, rename/forget, and the
//! deterministic per-speaker color.

use std::sync::Arc;

use serde::Serialize;

use super::ServerState;

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

/// Run speaker diarization on a completed utterance and return the speaker
/// metadata to attach to the next `Transcript` event.
///
/// Returns `(speaker_id, speaker_name, speaker_color, optional SpeakerPublic
/// describing a newly enrolled speaker)`. The fourth element is `Some` only
/// when the pipeline auto-enrolled a brand-new speaker during this call so
/// the caller can emit a `SpeakerUpdate`.
pub(super) async fn diarize_utterance(
    state: &Arc<ServerState>,
    diarization_on: bool,
    utterance: &vox::Utterance,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<SpeakerPublic>,
) {
    if !diarization_on {
        return (None, None, None, None);
    }
    #[cfg(feature = "diarization")]
    {
        let Some(pipeline) = state.diarization.as_ref() else {
            return (None, None, None, None);
        };
        let Some(db) = state.speaker_db.as_ref() else {
            return (None, None, None, None);
        };

        // Count speakers before processing to detect auto-enrollment below.
        let before_count = {
            let reg = pipeline.registry();
            let guard = reg.lock().unwrap();
            guard.speaker_count()
        };

        // process_utterance consumes the utterance; clone to avoid mutating
        // the caller's copy.
        let processed = match pipeline.process_utterance(utterance.clone()).await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(error = %e, "WS: diarization failed, continuing without");
                return (None, None, None, None);
            }
        };

        let Some(id) = processed.speaker_id.clone() else {
            return (None, None, None, None);
        };
        // "unknown" is the sentinel returned for too-short utterances or
        // extraction failures — don't surface it as a real speaker.
        if id == "unknown" {
            return (None, None, None, None);
        }
        let color = color_for(&id);

        // Snapshot the registry to look up the human-readable name and check
        // for new enrollments.
        let (name, new_speaker, new_embedding) = {
            let reg = pipeline.registry();
            let guard = reg.lock().unwrap();
            let name = guard
                .get_speaker(&id)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| id.clone());
            let new_speaker = if guard.speaker_count() > before_count {
                // New speaker enrolled during this call.
                guard.get_speaker(&id).map(|s| SpeakerPublic {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    color: color_for(&s.id),
                })
            } else {
                None
            };
            let new_embedding = if new_speaker.is_some() {
                guard.get_speaker(&id).map(|s| s.embedding.clone())
            } else {
                None
            };
            (name, new_speaker, new_embedding)
        };

        // Persist new enrollments to SQLite so they survive a restart.
        if let (Some(speaker_public), Some(emb)) = (&new_speaker, new_embedding) {
            let persisted = vox::Speaker {
                id: speaker_public.id.clone(),
                name: speaker_public.name.clone(),
                embedding: emb,
            };
            if let Err(e) = db.store_speaker(&persisted).await {
                tracing::warn!(error = %e, "WS: failed to persist new speaker");
            }
        } else if let Err(e) = db.update_last_seen(&id).await {
            tracing::debug!(error = %e, "WS: update_last_seen failed");
        }

        (Some(id), Some(name), Some(color), new_speaker)
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, utterance);
        (None, None, None, None)
    }
}

/// Snapshot the current list of enrolled speakers for `SpeakerList` events.
pub(super) async fn current_speaker_list(state: &Arc<ServerState>) -> Option<Vec<SpeakerPublic>> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state.diarization.as_ref()?;
        let reg = pipeline.registry();
        let guard = reg.lock().unwrap();
        let list: Vec<SpeakerPublic> = guard
            .list_speakers()
            .into_iter()
            .map(|s| SpeakerPublic {
                id: s.id.clone(),
                name: s.name.clone(),
                color: color_for(&s.id),
            })
            .collect();
        Some(list)
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = state;
        None
    }
}

/// Rename a speaker in both the in-memory registry and the SQLite database.
pub(super) async fn rename_speaker(
    state: &Arc<ServerState>,
    id: &str,
    name: &str,
) -> Result<(), vox::VoxError> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state
            .diarization
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("diarization disabled".into()))?;
        let db = state
            .speaker_db
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("speaker db disabled".into()))?;

        {
            let reg = pipeline.registry();
            let mut guard = reg.lock().unwrap();
            guard.rename(id, name)?;
        }
        db.update_speaker_name(id, name).await?;
        Ok(())
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, id, name);
        Err(vox::VoxError::Diarization("diarization disabled".into()))
    }
}

/// Forget a speaker from both the in-memory registry and the SQLite database.
pub(super) async fn forget_speaker(state: &Arc<ServerState>, id: &str) -> Result<(), vox::VoxError> {
    #[cfg(feature = "diarization")]
    {
        let pipeline = state
            .diarization
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("diarization disabled".into()))?;
        let db = state
            .speaker_db
            .as_ref()
            .ok_or_else(|| vox::VoxError::Diarization("speaker db disabled".into()))?;

        {
            let reg = pipeline.registry();
            let mut guard = reg.lock().unwrap();
            guard.forget(id)?;
        }
        db.delete_speaker(id).await?;
        Ok(())
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (state, id);
        Err(vox::VoxError::Diarization("diarization disabled".into()))
    }
}
