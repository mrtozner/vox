//! Persistent speaker database using SQLite.
//!
//! This module provides a SQLite-based speaker database for storing:
//! - Speaker profiles (ID, name, reference embedding, observation count)
//! - Conversation history
//! - Speaker preferences
//! - Database-level metadata (embedding version, enrollment consent, the
//!   speaker ID high-water mark) in a `meta` table with no foreign keys
//!
//! All data is stored locally. Voiceprints are biometric data, so the file
//! is created owner-only (`0600`), `secure_delete` is on so deleted rows are
//! overwritten rather than left in free pages, and every delete is followed
//! by a `VACUUM`. The journal stays in the default rollback mode: a WAL file
//! would keep deleted voiceprints readable until the next checkpoint.

use std::path::Path;
use std::str::FromStr;

use serde::Serialize;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{Pool, Row, Sqlite};

use crate::error::VoxError;

use super::recognition::Speaker;

/// `meta` key holding the embedding pipeline version the stored voiceprints
/// were produced with.
const META_EMBEDDING_VERSION: &str = "embedding_version";
/// `meta` key holding the highest `speaker_N` number ever issued.
const META_SPEAKER_SEQ: &str = "speaker_seq";
/// `meta` key holding the Unix time enrollment consent was granted.
const META_ENROLLMENT_CONSENT: &str = "enrollment_consent_at";

/// SQLite-based speaker database.
///
/// Stores speaker embeddings, conversation history, and preferences.
/// All data is stored locally in a SQLite file.
pub struct SpeakerDatabase {
    pool: Pool<Sqlite>,
}

/// A stored speaker without the biometric payload.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerRecord {
    /// Stable speaker identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Unix time (seconds) the speaker was first stored.
    pub created_at: u64,
    /// Unix time (seconds) the speaker was last heard.
    pub last_seen: u64,
    /// Number of observations folded into the reference embedding.
    pub sample_count: u32,
    /// `false` after the embedding pipeline changed and the stored
    /// voiceprint was invalidated; the speaker can no longer be recognized
    /// until it is enrolled again, but its name and history are kept.
    pub has_voiceprint: bool,
}

/// Everything stored about one speaker, for data-portability export.
#[derive(Debug, Clone, Serialize)]
pub struct SpeakerExport {
    /// The speaker row.
    pub speaker: SpeakerRecord,
    /// The reference embedding. Only present when explicitly requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Per-speaker preferences.
    pub preferences: Vec<PreferenceEntry>,
    /// Recorded utterances, most recent first.
    pub conversations: Vec<ConversationEntry>,
}

/// One per-speaker preference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PreferenceEntry {
    /// Preference key.
    pub key: String,
    /// Preference value.
    pub value: String,
}

/// Outcome of [`SpeakerDatabase::apply_embedding_version`].
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingVersionChange {
    /// The stored version already matches.
    Unchanged,
    /// The version was written; voiceprints produced by another pipeline
    /// version were invalidated.
    Updated {
        /// Version that was stored before, if any.
        previous: Option<String>,
        /// Number of voiceprints invalidated.
        invalidated: u64,
    },
}

impl SpeakerDatabase {
    /// The connection pool, for modules that keep their own tables in this
    /// file (see [`crate::memory`]).
    pub(crate) fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    /// Open or create a speaker database at the given path.
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file (created if missing,
    ///   owner-only), or `":memory:"` for a private in-memory database.
    ///
    /// # Example
    /// ```ignore
    /// let db = SpeakerDatabase::open("speakers.db").await?;
    /// ```
    pub async fn open(db_path: impl AsRef<str>) -> Result<Self, VoxError> {
        let db_path = db_path.as_ref();
        let open_err =
            |e: sqlx::Error| VoxError::Diarization(format!("failed to open database: {e}"));

        let pool = if db_path == ":memory:" {
            // Every connection to `:memory:` is its own empty database, so
            // the pool must be exactly one connection that is never recycled.
            let options = SqliteConnectOptions::from_str("sqlite::memory:")
                .map_err(open_err)?
                .foreign_keys(true)
                .pragma("secure_delete", "ON");
            SqlitePoolOptions::new()
                .max_connections(1)
                .min_connections(1)
                .idle_timeout(None)
                .max_lifetime(None)
                .connect_with(options)
                .await
                .map_err(open_err)?
        } else {
            restrict_db_file(Path::new(db_path))?;
            let options = SqliteConnectOptions::new()
                .filename(db_path)
                .create_if_missing(true)
                .foreign_keys(true)
                .journal_mode(SqliteJournalMode::Delete)
                .pragma("secure_delete", "ON");
            SqlitePoolOptions::new()
                .connect_with(options)
                .await
                .map_err(open_err)?
        };

        let db = Self { pool };
        db.create_tables().await?;

        Ok(db)
    }

    /// Close the database, waiting for in-flight queries to finish.
    pub async fn close(&self) {
        self.pool.close().await;
    }

    /// Create database tables if they don't exist and bring older files up
    /// to the current schema.
    async fn create_tables(&self) -> Result<(), VoxError> {
        // Speakers table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 1
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to create speakers table: {e}")))?;

        // Files created before observation counts existed.
        let has_sample_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('speakers') WHERE name = 'sample_count'",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to inspect speakers table: {e}")))?;
        if has_sample_count == 0 {
            sqlx::query("ALTER TABLE speakers ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1")
                .execute(&self.pool)
                .await
                .map_err(|e| {
                    VoxError::Diarization(format!("failed to migrate speakers table: {e}"))
                })?;
        }

        // Conversations table (for tracking speaker interactions)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                utterance_text TEXT,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to create conversations table: {e}")))?;

        // Preferences table (for speaker-specific settings)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS preferences (
                speaker_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (speaker_id, key),
                FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to create preferences table: {e}")))?;

        // Database-level key/value settings. Deliberately no foreign key:
        // these used to live in `preferences` under a `_system` speaker that
        // does not exist, so with foreign keys enforced every write failed.
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to create meta table: {e}")))?;

        // Carry over `_system` rows from files written with foreign keys off.
        sqlx::query(
            "INSERT OR IGNORE INTO meta (key, value) \
             SELECT key, value FROM preferences WHERE speaker_id = '_system'",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to migrate system settings: {e}")))?;
        sqlx::query("DELETE FROM preferences WHERE speaker_id = '_system'")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                VoxError::Diarization(format!("failed to migrate system settings: {e}"))
            })?;

        // Seed the ID high-water mark from speakers that predate it.
        let ids: Vec<String> = sqlx::query_scalar("SELECT id FROM speakers")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VoxError::Diarization(format!("failed to list speaker ids: {e}")))?;
        if let Some(max) = ids.iter().filter_map(|id| auto_speaker_number(id)).max() {
            raise_speaker_sequence(&self.pool, max).await?;
        }

        // Per-person memory lives in the same file and hangs off `speakers`.
        crate::memory::create_schema(&self.pool).await?;

        Ok(())
    }

    /// Store a speaker in the database.
    ///
    /// # Arguments
    /// * `speaker` - Speaker to store
    ///
    /// # Returns
    /// `Ok(())` if storage succeeded, `Err` if speaker already exists.
    pub async fn store_speaker(&self, speaker: &Speaker) -> Result<(), VoxError> {
        self.store_speaker_with_count(speaker, 1).await
    }

    /// Store a speaker whose reference aggregates `sample_count` observations.
    ///
    /// Also advances the `speaker_N` high-water mark, in the same
    /// transaction, so an auto-assigned ID is never handed out twice even
    /// after the speaker is deleted.
    pub async fn store_speaker_with_count(
        &self,
        speaker: &Speaker,
        sample_count: u32,
    ) -> Result<(), VoxError> {
        let embedding_bytes = embedding_to_bytes(&speaker.embedding);
        let now = timestamp_now();
        let err = |e: sqlx::Error| VoxError::Diarization(format!("failed to store speaker: {e}"));

        let mut tx = self.pool.begin().await.map_err(err)?;
        sqlx::query(
            r#"
            INSERT INTO speakers (id, name, embedding, created_at, last_seen, sample_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
        )
        .bind(&speaker.id)
        .bind(&speaker.name)
        .bind(&embedding_bytes)
        .bind(now)
        .bind(now)
        .bind(i64::from(sample_count.max(1)))
        .execute(&mut *tx)
        .await
        .map_err(err)?;
        if let Some(n) = auto_speaker_number(&speaker.id) {
            raise_speaker_sequence(&mut *tx, n).await?;
        }
        tx.commit().await.map_err(err)?;

        Ok(())
    }

    /// Load a speaker from the database by ID.
    pub async fn load_speaker(&self, id: &str) -> Result<Option<Speaker>, VoxError> {
        Ok(self.load_voiceprint(id).await?.map(|(speaker, _)| speaker))
    }

    /// Load a speaker and the observation count behind its reference.
    pub async fn load_voiceprint(&self, id: &str) -> Result<Option<(Speaker, u32)>, VoxError> {
        let row = sqlx::query(
            r#"
            SELECT id, name, embedding, sample_count FROM speakers WHERE id = ?1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to load speaker: {e}")))?;

        row.map(|row| voiceprint_from_row(&row)).transpose()
    }

    /// List all speakers in the database, most recently heard first.
    ///
    /// Includes speakers whose voiceprint was invalidated (empty embedding);
    /// use [`SpeakerDatabase::list_voiceprints`] to hydrate a registry.
    pub async fn list_speakers(&self) -> Result<Vec<Speaker>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, embedding, sample_count FROM speakers ORDER BY last_seen DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to list speakers: {e}")))?;

        rows.iter()
            .map(|row| voiceprint_from_row(row).map(|(speaker, _)| speaker))
            .collect()
    }

    /// List the speakers that have a usable voiceprint, with the observation
    /// count behind each reference.
    pub async fn list_voiceprints(&self) -> Result<Vec<(Speaker, u32)>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, embedding, sample_count FROM speakers
            WHERE length(embedding) > 0
            ORDER BY last_seen DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to list voiceprints: {e}")))?;

        rows.iter().map(voiceprint_from_row).collect()
    }

    /// List all speakers without their embeddings, most recently heard first.
    pub async fn list_records(&self) -> Result<Vec<SpeakerRecord>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, created_at, last_seen, sample_count,
                   length(embedding) AS embedding_len
            FROM speakers ORDER BY last_seen DESC, id ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to list speakers: {e}")))?;

        Ok(rows.iter().map(record_from_row).collect())
    }

    /// Load one speaker without its embedding.
    pub async fn get_record(&self, id: &str) -> Result<Option<SpeakerRecord>, VoxError> {
        let row = sqlx::query(
            r#"
            SELECT id, name, created_at, last_seen, sample_count,
                   length(embedding) AS embedding_len
            FROM speakers WHERE id = ?1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to load speaker: {e}")))?;

        Ok(row.as_ref().map(record_from_row))
    }

    /// Delete a speaker and everything stored about them.
    ///
    /// Conversations and preferences are removed in the same transaction
    /// (explicitly, not only through `ON DELETE CASCADE`), then the file is
    /// vacuumed so the freed pages do not linger on disk.
    pub async fn delete_speaker(&self, id: &str) -> Result<(), VoxError> {
        let err = |e: sqlx::Error| VoxError::Diarization(format!("failed to delete speaker: {e}"));

        let mut tx = self.pool.begin().await.map_err(err)?;
        sqlx::query("DELETE FROM conversations WHERE speaker_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        sqlx::query("DELETE FROM preferences WHERE speaker_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        let result = sqlx::query("DELETE FROM speakers WHERE id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(err)?;

        if result.rows_affected() == 0 {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' not found"
            )));
        }
        tx.commit().await.map_err(err)?;

        self.vacuum().await;
        Ok(())
    }

    /// Update the last_seen timestamp for a speaker.
    pub async fn update_last_seen(&self, id: &str) -> Result<(), VoxError> {
        let now = timestamp_now();

        sqlx::query(
            r#"
            UPDATE speakers SET last_seen = ?1 WHERE id = ?2
            "#,
        )
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to update last_seen: {e}")))?;

        Ok(())
    }

    /// Rename an enrolled speaker in place.
    ///
    /// Persists label changes from the `/v1/listen` WebSocket (e.g.
    /// `Speaker 1` → `Alice`) without touching the stored embedding.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if the speaker ID is unknown.
    pub async fn update_speaker_name(&self, id: &str, name: &str) -> Result<(), VoxError> {
        let result = sqlx::query(
            r#"
            UPDATE speakers SET name = ?1 WHERE id = ?2
            "#,
        )
        .bind(name)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to update speaker name: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' not found"
            )));
        }

        Ok(())
    }

    /// Persist an adapted reference embedding and its observation count.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if the speaker ID is unknown.
    pub async fn update_speaker_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        sample_count: u32,
    ) -> Result<(), VoxError> {
        let result = sqlx::query(
            r#"
            UPDATE speakers SET embedding = ?1, sample_count = ?2, last_seen = ?3 WHERE id = ?4
            "#,
        )
        .bind(embedding_to_bytes(embedding))
        .bind(i64::from(sample_count.max(1)))
        .bind(timestamp_now())
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to update speaker embedding: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' not found"
            )));
        }

        Ok(())
    }

    /// Merge `source` into `target` and delete `source`.
    ///
    /// The target keeps its ID and name and receives `embedding` /
    /// `sample_count` (the caller computes the merged reference, see
    /// [`merge_embeddings`](super::recognition::merge_embeddings)). The
    /// source's conversations move to the target; its preferences are copied
    /// where the target has none. All in one transaction.
    pub async fn merge_speakers(
        &self,
        source: &str,
        target: &str,
        embedding: &[f32],
        sample_count: u32,
    ) -> Result<(), VoxError> {
        if source == target {
            return Err(VoxError::Diarization(
                "cannot merge a speaker into itself".into(),
            ));
        }
        let err = |e: sqlx::Error| VoxError::Diarization(format!("failed to merge speakers: {e}"));

        let mut tx = self.pool.begin().await.map_err(err)?;
        let source_last_seen: Option<i64> =
            sqlx::query_scalar("SELECT last_seen FROM speakers WHERE id = ?1")
                .bind(source)
                .fetch_optional(&mut *tx)
                .await
                .map_err(err)?;
        let Some(source_last_seen) = source_last_seen else {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{source}' not found"
            )));
        };

        let updated = sqlx::query(
            r#"
            UPDATE speakers
            SET embedding = ?1, sample_count = ?2, last_seen = MAX(last_seen, ?3)
            WHERE id = ?4
            "#,
        )
        .bind(embedding_to_bytes(embedding))
        .bind(i64::from(sample_count.max(1)))
        .bind(source_last_seen)
        .bind(target)
        .execute(&mut *tx)
        .await
        .map_err(err)?;
        if updated.rows_affected() == 0 {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{target}' not found"
            )));
        }

        sqlx::query("UPDATE conversations SET speaker_id = ?1 WHERE speaker_id = ?2")
            .bind(target)
            .bind(source)
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        // Long-term memory moves too, before the source row (and with it,
        // by cascade, whatever still hangs off it) is deleted.
        crate::memory::reassign_person(&mut tx, source, target)
            .await
            .map_err(err)?;
        sqlx::query(
            "INSERT OR IGNORE INTO preferences (speaker_id, key, value) \
             SELECT ?1, key, value FROM preferences WHERE speaker_id = ?2",
        )
        .bind(target)
        .bind(source)
        .execute(&mut *tx)
        .await
        .map_err(err)?;
        sqlx::query("DELETE FROM preferences WHERE speaker_id = ?1")
            .bind(source)
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        sqlx::query("DELETE FROM speakers WHERE id = ?1")
            .bind(source)
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        tx.commit().await.map_err(err)?;

        self.vacuum().await;
        Ok(())
    }

    /// Collect everything stored about one speaker.
    ///
    /// The embedding is a biometric identifier and is only included when
    /// `include_embedding` is set.
    pub async fn export_speaker(
        &self,
        id: &str,
        include_embedding: bool,
    ) -> Result<Option<SpeakerExport>, VoxError> {
        let Some(speaker) = self.get_record(id).await? else {
            return Ok(None);
        };
        let embedding = if include_embedding {
            self.load_speaker(id).await?.map(|s| s.embedding)
        } else {
            None
        };
        let preferences = self
            .get_all_preferences(id)
            .await?
            .into_iter()
            .map(|(key, value)| PreferenceEntry { key, value })
            .collect();
        let conversations = self.get_conversation_history(id, u32::MAX).await?;

        Ok(Some(SpeakerExport {
            speaker,
            embedding,
            preferences,
            conversations,
        }))
    }

    /// Record a conversation utterance.
    ///
    /// # Arguments
    /// * `speaker_id` - Speaker who spoke
    /// * `duration_ms` - Duration of utterance in milliseconds
    /// * `text` - Optional transcribed text
    pub async fn record_utterance(
        &self,
        speaker_id: &str,
        duration_ms: u64,
        text: Option<&str>,
    ) -> Result<(), VoxError> {
        let now = timestamp_now();

        sqlx::query(
            r#"
            INSERT INTO conversations (speaker_id, timestamp, duration_ms, utterance_text)
            VALUES (?1, ?2, ?3, ?4)
            "#,
        )
        .bind(speaker_id)
        .bind(now)
        .bind(duration_ms as i64)
        .bind(text)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to record utterance: {e}")))?;

        self.update_last_seen(speaker_id).await?;

        Ok(())
    }

    /// Get conversation history for a speaker.
    ///
    /// # Arguments
    /// * `speaker_id` - Speaker to query
    /// * `limit` - Maximum number of utterances to return (most recent first)
    pub async fn get_conversation_history(
        &self,
        speaker_id: &str,
        limit: u32,
    ) -> Result<Vec<ConversationEntry>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT timestamp, duration_ms, utterance_text
            FROM conversations
            WHERE speaker_id = ?1
            ORDER BY timestamp DESC, id DESC
            LIMIT ?2
            "#,
        )
        .bind(speaker_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to get conversation history: {e}")))?;

        let mut entries = Vec::new();
        for row in rows {
            let timestamp: i64 = row.get("timestamp");
            let duration_ms: i64 = row.get("duration_ms");
            let text: Option<String> = row.get("utterance_text");

            entries.push(ConversationEntry {
                timestamp: timestamp as u64,
                duration_ms: duration_ms as u64,
                text,
            });
        }

        Ok(entries)
    }

    /// Set a preference for a speaker.
    ///
    /// # Arguments
    /// * `speaker_id` - Speaker ID
    /// * `key` - Preference key (e.g., "voice", "language")
    /// * `value` - Preference value
    pub async fn set_preference(
        &self,
        speaker_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), VoxError> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO preferences (speaker_id, key, value)
            VALUES (?1, ?2, ?3)
            "#,
        )
        .bind(speaker_id)
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to set preference: {e}")))?;

        Ok(())
    }

    /// Get a preference for a speaker.
    pub async fn get_preference(
        &self,
        speaker_id: &str,
        key: &str,
    ) -> Result<Option<String>, VoxError> {
        let row = sqlx::query(
            r#"
            SELECT value FROM preferences WHERE speaker_id = ?1 AND key = ?2
            "#,
        )
        .bind(speaker_id)
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to get preference: {e}")))?;

        Ok(row.map(|r| r.get("value")))
    }

    /// Delete all speakers and everything stored about them.
    ///
    /// Database-level settings (embedding version, consent, the ID
    /// high-water mark) are kept, so IDs are still never reused.
    ///
    /// Returns the number of speakers deleted.
    pub async fn clear_speakers(&self) -> Result<u64, VoxError> {
        let err = |e: sqlx::Error| VoxError::Diarization(format!("failed to clear speakers: {e}"));

        let mut tx = self.pool.begin().await.map_err(err)?;
        sqlx::query("DELETE FROM conversations")
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        sqlx::query("DELETE FROM preferences")
            .execute(&mut *tx)
            .await
            .map_err(err)?;
        let deleted = sqlx::query("DELETE FROM speakers")
            .execute(&mut *tx)
            .await
            .map_err(err)?
            .rows_affected();
        tx.commit().await.map_err(err)?;

        self.vacuum().await;
        Ok(deleted)
    }

    /// Get a database-level setting from the `meta` table.
    pub async fn get_meta(&self, key: &str) -> Result<Option<String>, VoxError> {
        sqlx::query_scalar("SELECT value FROM meta WHERE key = ?1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| VoxError::Diarization(format!("failed to read setting '{key}': {e}")))
    }

    /// Set a database-level setting in the `meta` table.
    pub async fn set_meta(&self, key: &str, value: &str) -> Result<(), VoxError> {
        sqlx::query(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to write setting '{key}': {e}")))?;
        Ok(())
    }

    /// Get a system-level preference (not tied to a speaker).
    ///
    /// Stored in the `meta` table.
    pub async fn get_system_preference(&self, key: &str) -> Result<Option<String>, VoxError> {
        self.get_meta(key).await
    }

    /// Set a system-level preference (not tied to a speaker).
    ///
    /// Stored in the `meta` table.
    pub async fn set_system_preference(&self, key: &str, value: &str) -> Result<(), VoxError> {
        self.set_meta(key, value).await
    }

    /// Record the embedding pipeline version the voiceprints are produced
    /// with, invalidating voiceprints from any other version.
    ///
    /// Only the voiceprints go: the embedding is emptied and the observation
    /// count reset, while names, conversations and preferences stay. The
    /// version write and the invalidation commit together, so a failed write
    /// can never leave the database clearing itself on every start.
    pub async fn apply_embedding_version(
        &self,
        version: &str,
    ) -> Result<EmbeddingVersionChange, VoxError> {
        let err = |e: sqlx::Error| {
            VoxError::Diarization(format!("failed to apply embedding version: {e}"))
        };

        let mut tx = self.pool.begin().await.map_err(err)?;
        let previous: Option<String> = sqlx::query_scalar("SELECT value FROM meta WHERE key = ?1")
            .bind(META_EMBEDDING_VERSION)
            .fetch_optional(&mut *tx)
            .await
            .map_err(err)?;
        if previous.as_deref() == Some(version) {
            return Ok(EmbeddingVersionChange::Unchanged);
        }

        let invalidated = sqlx::query(
            "UPDATE speakers SET embedding = x'', sample_count = 1 WHERE length(embedding) > 0",
        )
        .execute(&mut *tx)
        .await
        .map_err(err)?
        .rows_affected();
        sqlx::query(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(META_EMBEDDING_VERSION)
        .bind(version)
        .execute(&mut *tx)
        .await
        .map_err(err)?;
        tx.commit().await.map_err(err)?;

        if invalidated > 0 {
            self.vacuum().await;
        }
        Ok(EmbeddingVersionChange::Updated {
            previous,
            invalidated,
        })
    }

    /// The embedding pipeline version recorded in this database, if any.
    pub async fn embedding_version(&self) -> Result<Option<String>, VoxError> {
        self.get_meta(META_EMBEDDING_VERSION).await
    }

    /// Highest `speaker_N` number ever stored in this database (0 if none).
    ///
    /// Survives deletion of the speaker, so a new auto-enrolled speaker never
    /// inherits the ID (and with it the history) of a forgotten one.
    pub async fn speaker_sequence(&self) -> Result<u32, VoxError> {
        Ok(self
            .get_meta(META_SPEAKER_SEQ)
            .await?
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0))
    }

    /// Unix time (seconds) at which consent to store voiceprints was
    /// granted, or `None` if it was never granted or has been withdrawn.
    pub async fn enrollment_consent(&self) -> Result<Option<u64>, VoxError> {
        Ok(self
            .get_meta(META_ENROLLMENT_CONSENT)
            .await?
            .and_then(|v| v.parse::<u64>().ok()))
    }

    /// Grant or withdraw consent to store new voiceprints.
    ///
    /// Withdrawing consent does not delete existing speakers; use
    /// [`SpeakerDatabase::clear_speakers`] for that.
    pub async fn set_enrollment_consent(&self, granted: bool) -> Result<(), VoxError> {
        if granted {
            if self.enrollment_consent().await?.is_none() {
                self.set_meta(META_ENROLLMENT_CONSENT, &timestamp_now().to_string())
                    .await?;
            }
            Ok(())
        } else {
            sqlx::query("DELETE FROM meta WHERE key = ?1")
                .bind(META_ENROLLMENT_CONSENT)
                .execute(&self.pool)
                .await
                .map_err(|e| VoxError::Diarization(format!("failed to withdraw consent: {e}")))?;
            Ok(())
        }
    }

    /// Get all preferences for a speaker.
    pub async fn get_all_preferences(
        &self,
        speaker_id: &str,
    ) -> Result<Vec<(String, String)>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT key, value FROM preferences WHERE speaker_id = ?1 ORDER BY key
            "#,
        )
        .bind(speaker_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to get preferences: {e}")))?;

        let mut prefs = Vec::new();
        for row in rows {
            let key: String = row.get("key");
            let value: String = row.get("value");
            prefs.push((key, value));
        }

        Ok(prefs)
    }

    /// Rebuild the file so pages freed by a delete are not left on disk.
    ///
    /// The delete has already committed (and `secure_delete` has already
    /// overwritten the rows), so a failed vacuum is logged, not returned.
    async fn vacuum(&self) {
        if let Err(e) = sqlx::query("VACUUM").execute(&self.pool).await {
            tracing::warn!(error = %e, "speaker database vacuum failed");
        }
    }
}

/// A conversation history entry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConversationEntry {
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Optional transcribed text.
    pub text: Option<String>,
}

/// Create the database file owner-only, or tighten an existing one.
///
/// SQLite gives its rollback journal the permissions of the main file, so
/// restricting the database restricts the journal too.
fn restrict_db_file(path: &Path) -> Result<(), VoxError> {
    let err = |e: std::io::Error| {
        VoxError::Diarization(format!(
            "failed to prepare database file {}: {e}",
            path.display()
        ))
    };

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(err)?;
    }

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(err)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(err)?;
    }

    Ok(())
}

/// The `N` of an auto-assigned `speaker_N` ID.
fn auto_speaker_number(id: &str) -> Option<u32> {
    id.strip_prefix("speaker_")?.parse::<u32>().ok()
}

/// Raise the `speaker_N` high-water mark to at least `n`.
async fn raise_speaker_sequence<'e, E>(executor: E, n: u32) -> Result<(), VoxError>
where
    E: sqlx::Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        "INSERT INTO meta (key, value) VALUES (?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value \
         WHERE CAST(meta.value AS INTEGER) < CAST(excluded.value AS INTEGER)",
    )
    .bind(META_SPEAKER_SEQ)
    .bind(n.to_string())
    .execute(executor)
    .await
    .map_err(|e| VoxError::Diarization(format!("failed to record speaker sequence: {e}")))?;
    Ok(())
}

fn voiceprint_from_row(row: &sqlx::sqlite::SqliteRow) -> Result<(Speaker, u32), VoxError> {
    let embedding_bytes: Vec<u8> = row.get("embedding");
    let sample_count: i64 = row.get("sample_count");
    Ok((
        Speaker {
            id: row.get("id"),
            name: row.get("name"),
            embedding: bytes_to_embedding(&embedding_bytes)?,
        },
        u32::try_from(sample_count).unwrap_or(1).max(1),
    ))
}

fn record_from_row(row: &sqlx::sqlite::SqliteRow) -> SpeakerRecord {
    let created_at: i64 = row.get("created_at");
    let last_seen: i64 = row.get("last_seen");
    let sample_count: i64 = row.get("sample_count");
    let embedding_len: i64 = row.get("embedding_len");
    SpeakerRecord {
        id: row.get("id"),
        name: row.get("name"),
        created_at: created_at.max(0) as u64,
        last_seen: last_seen.max(0) as u64,
        sample_count: u32::try_from(sample_count).unwrap_or(1).max(1),
        has_voiceprint: embedding_len > 0,
    }
}

/// Convert an embedding vector to bytes for storage.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Convert bytes back to an embedding vector.
fn bytes_to_embedding(bytes: &[u8]) -> Result<Vec<f32>, VoxError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VoxError::Diarization(
            "invalid embedding bytes: length not divisible by 4".into(),
        ));
    }

    let mut embedding = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        embedding.push(value);
    }

    Ok(embedding)
}

/// Get current Unix timestamp in seconds.
fn timestamp_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_serialization() {
        let embedding = vec![0.1, -0.5, 0.9, 1.0];
        let bytes = embedding_to_bytes(&embedding);
        let restored = bytes_to_embedding(&bytes).unwrap();

        assert_eq!(embedding.len(), restored.len());
        for (a, b) in embedding.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_invalid_embedding_bytes() {
        let bytes = vec![0u8, 1, 2]; // Not divisible by 4
        let result = bytes_to_embedding(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_embedding() {
        let embedding: Vec<f32> = vec![];
        let bytes = embedding_to_bytes(&embedding);
        assert_eq!(bytes.len(), 0);

        let restored = bytes_to_embedding(&bytes).unwrap();
        assert_eq!(restored.len(), 0);
    }

    #[tokio::test]
    async fn test_database_lifecycle() {
        // Use in-memory database for testing
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        // Store a speaker
        let speaker = Speaker {
            id: "alice".to_string(),
            name: "Alice".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Load the speaker
        let loaded = db.load_speaker("alice").await.unwrap().unwrap();
        assert_eq!(loaded.id, "alice");
        assert_eq!(loaded.name, "Alice");
        assert_eq!(loaded.embedding.len(), 3);

        // List speakers
        let speakers = db.list_speakers().await.unwrap();
        assert_eq!(speakers.len(), 1);

        // Delete speaker
        db.delete_speaker("alice").await.unwrap();
        let loaded = db.load_speaker("alice").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_conversation_recording() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "bob".to_string(),
            name: "Bob".to_string(),
            embedding: vec![0.5, 0.5],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Record utterances
        db.record_utterance("bob", 1000, Some("Hello"))
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        db.record_utterance("bob", 1500, Some("How are you?"))
            .await
            .unwrap();

        // Get history
        let history = db.get_conversation_history("bob", 10).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text, Some("How are you?".to_string())); // Most recent first
    }

    #[tokio::test]
    async fn test_preferences() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "charlie".to_string(),
            name: "Charlie".to_string(),
            embedding: vec![0.1],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Set preferences
        db.set_preference("charlie", "voice", "af_heart")
            .await
            .unwrap();
        db.set_preference("charlie", "language", "en")
            .await
            .unwrap();

        // Get preference
        let voice = db
            .get_preference("charlie", "voice")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(voice, "af_heart");

        // Get all preferences
        let all_prefs = db.get_all_preferences("charlie").await.unwrap();
        assert_eq!(all_prefs.len(), 2);
    }

    use crate::diarization::recognition::SpeakerRegistry;

    /// A unique database path under the system temp dir, removed on drop.
    struct TempDb(std::path::PathBuf);

    impl TempDb {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!(
                "vox-speakerdb-{tag}-{}-{nanos}",
                std::process::id()
            ));
            Self(dir.join("speakers.db"))
        }

        fn path(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            if let Some(dir) = self.0.parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    fn speaker(id: &str, name: &str, embedding: &[f32]) -> Speaker {
        Speaker {
            id: id.to_string(),
            name: name.to_string(),
            embedding: embedding.to_vec(),
        }
    }

    /// Unit vector with cosine similarity `score` to `[1, 0, 0]`.
    fn at_score(score: f32) -> Vec<f32> {
        vec![score, (1.0 - score * score).sqrt(), 0.0]
    }

    #[tokio::test]
    async fn test_speakers_survive_reopen_once_version_is_written() {
        let tmp = TempDb::new("reopen");

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        // A fresh database has no version: writing it invalidates nothing.
        assert_eq!(
            db.apply_embedding_version("v2-cmvn").await.unwrap(),
            EmbeddingVersionChange::Updated {
                previous: None,
                invalidated: 0
            }
        );
        db.store_speaker(&speaker("speaker_1", "Alice", &[1.0, 0.0, 0.0]))
            .await
            .unwrap();
        db.close().await;

        // Restart: the version is still there, so nothing is cleared.
        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        assert_eq!(
            db.embedding_version().await.unwrap().as_deref(),
            Some("v2-cmvn")
        );
        assert_eq!(
            db.apply_embedding_version("v2-cmvn").await.unwrap(),
            EmbeddingVersionChange::Unchanged
        );
        let voiceprints = db.list_voiceprints().await.unwrap();
        assert_eq!(voiceprints.len(), 1);
        assert_eq!(voiceprints[0].0.name, "Alice");
        assert_eq!(voiceprints[0].0.embedding, vec![1.0, 0.0, 0.0]);
        db.close().await;
    }

    #[tokio::test]
    async fn test_system_preference_is_writable_with_foreign_keys_on() {
        // The old implementation stored this under a `_system` speaker that
        // did not exist, which the foreign key rejected.
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        db.set_system_preference("embedding_version", "v2-cmvn")
            .await
            .unwrap();
        assert_eq!(
            db.get_system_preference("embedding_version")
                .await
                .unwrap()
                .as_deref(),
            Some("v2-cmvn")
        );
        // Foreign keys really are enforced for per-speaker data.
        assert!(db.set_preference("nobody", "voice", "x").await.is_err());
    }

    #[tokio::test]
    async fn test_version_change_invalidates_only_voiceprints() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        db.apply_embedding_version("v1").await.unwrap();
        db.store_speaker_with_count(&speaker("speaker_1", "Alice", &[1.0, 0.0]), 7)
            .await
            .unwrap();
        db.set_preference("speaker_1", "voice", "af_heart")
            .await
            .unwrap();
        db.record_utterance("speaker_1", 900, Some("hello"))
            .await
            .unwrap();

        assert_eq!(
            db.apply_embedding_version("v2").await.unwrap(),
            EmbeddingVersionChange::Updated {
                previous: Some("v1".to_string()),
                invalidated: 1
            }
        );

        // The person is still known; only the voiceprint is gone.
        let record = db.get_record("speaker_1").await.unwrap().unwrap();
        assert_eq!(record.name, "Alice");
        assert!(!record.has_voiceprint);
        assert!(db.list_voiceprints().await.unwrap().is_empty());
        assert_eq!(
            db.get_preference("speaker_1", "voice").await.unwrap(),
            Some("af_heart".to_string())
        );
        assert_eq!(
            db.get_conversation_history("speaker_1", 10)
                .await
                .unwrap()
                .len(),
            1
        );

        // Idempotent from here on.
        assert_eq!(
            db.apply_embedding_version("v2").await.unwrap(),
            EmbeddingVersionChange::Unchanged
        );
    }

    #[tokio::test]
    async fn test_adaptation_thresholds_and_persistence() {
        let tmp = TempDb::new("adapt");
        let reference = vec![1.0, 0.0, 0.0];

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        db.store_speaker(&speaker("speaker_1", "Alice", &reference))
            .await
            .unwrap();
        let mut registry = SpeakerRegistry::new();
        for (s, count) in db.list_voiceprints().await.unwrap() {
            registry
                .enroll_with_count(s.id, s.name, s.embedding, count)
                .unwrap();
        }

        // A 0.36 match is accepted but must not move the reference.
        let weak = at_score(0.36);
        let scores = registry.score(&weak).unwrap();
        let (id, score) = scores.best.clone().unwrap();
        assert!((score - 0.36).abs() < 1e-4);
        assert!(!registry.adapt(&id, &weak, score, scores.runner_up));
        assert_eq!(
            registry.get_speaker("speaker_1").unwrap().embedding,
            reference
        );

        // A 0.6 match does, and the change is written through.
        let strong = at_score(0.6);
        let scores = registry.score(&strong).unwrap();
        let (id, score) = scores.best.clone().unwrap();
        assert!(registry.adapt(&id, &strong, score, scores.runner_up));
        let adapted = registry.get_speaker("speaker_1").unwrap().embedding.clone();
        assert_ne!(adapted, reference);
        db.update_speaker_embedding(
            "speaker_1",
            &adapted,
            registry.sample_count("speaker_1").unwrap(),
        )
        .await
        .unwrap();
        db.close().await;

        // Reopen: the adapted reference and its count are what we get back.
        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        let (stored, count) = db.load_voiceprint("speaker_1").await.unwrap().unwrap();
        assert_eq!(stored.embedding, adapted);
        assert_eq!(count, 2);
        db.close().await;

        assert!(
            db_update_missing_speaker_fails().await,
            "updating an unknown speaker must be an error"
        );
    }

    async fn db_update_missing_speaker_fails() -> bool {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        db.update_speaker_embedding("ghost", &[1.0], 1)
            .await
            .is_err()
    }

    #[tokio::test]
    async fn test_forget_removes_everything_and_file_is_private() {
        let tmp = TempDb::new("forget");
        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&tmp.0).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "database file must be owner-only");
        }

        db.store_speaker(&speaker("speaker_1", "Xanthippe Quillfeather", &[0.5; 64]))
            .await
            .unwrap();
        db.store_speaker(&speaker("speaker_2", "Bob", &[0.25; 64]))
            .await
            .unwrap();
        db.set_preference("speaker_1", "voice", "af_heart")
            .await
            .unwrap();
        db.record_utterance("speaker_1", 1200, Some("the zebra crossing is haunted"))
            .await
            .unwrap();
        db.record_utterance("speaker_2", 800, Some("kept"))
            .await
            .unwrap();

        db.delete_speaker("speaker_1").await.unwrap();

        assert!(db.load_speaker("speaker_1").await.unwrap().is_none());
        assert!(
            db.get_conversation_history("speaker_1", 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            db.get_all_preferences("speaker_1")
                .await
                .unwrap()
                .is_empty()
        );
        // The other speaker is untouched.
        assert_eq!(
            db.get_conversation_history("speaker_2", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(db.delete_speaker("speaker_1").await.is_err());
        db.close().await;

        // secure_delete + VACUUM: nothing of the forgotten speaker is left
        // in the file, not even in free pages.
        let raw = std::fs::read(&tmp.0).unwrap();
        let contains = |needle: &[u8]| raw.windows(needle.len()).any(|w| w == needle);
        assert!(!contains(b"Xanthippe Quillfeather"));
        assert!(!contains(b"zebra crossing"));
        assert!(contains(b"Bob"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&tmp.0).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "vacuum must not loosen permissions");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_open_tightens_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = TempDb::new("perms");
        std::fs::create_dir_all(tmp.0.parent().unwrap()).unwrap();
        std::fs::write(&tmp.0, b"").unwrap();
        std::fs::set_permissions(&tmp.0, std::fs::Permissions::from_mode(0o644)).unwrap();

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        let mode = std::fs::metadata(&tmp.0).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        db.close().await;
    }

    #[tokio::test]
    async fn test_merge_combines_two_identities() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        db.store_speaker_with_count(&speaker("speaker_1", "Alice", &[1.0, 0.0]), 3)
            .await
            .unwrap();
        db.store_speaker(&speaker("speaker_2", "Speaker 2", &[0.0, 1.0]))
            .await
            .unwrap();
        db.record_utterance("speaker_1", 500, Some("one"))
            .await
            .unwrap();
        db.record_utterance("speaker_2", 600, Some("two"))
            .await
            .unwrap();
        db.set_preference("speaker_1", "voice", "target-voice")
            .await
            .unwrap();
        db.set_preference("speaker_2", "voice", "source-voice")
            .await
            .unwrap();
        db.set_preference("speaker_2", "language", "tr")
            .await
            .unwrap();

        let merged =
            crate::diarization::recognition::merge_embeddings(&[1.0, 0.0], 3, &[0.0, 1.0], 1)
                .unwrap();
        db.merge_speakers("speaker_2", "speaker_1", &merged, 4)
            .await
            .unwrap();

        assert!(db.get_record("speaker_2").await.unwrap().is_none());
        let (target, count) = db.load_voiceprint("speaker_1").await.unwrap().unwrap();
        assert_eq!(target.name, "Alice");
        assert_eq!(target.embedding, merged);
        assert_eq!(count, 4);

        // History follows the merged identity.
        let history = db.get_conversation_history("speaker_1", 10).await.unwrap();
        assert_eq!(history.len(), 2);
        assert!(
            db.get_conversation_history("speaker_2", 10)
                .await
                .unwrap()
                .is_empty()
        );
        // The target's own preference wins; missing ones are inherited.
        assert_eq!(
            db.get_all_preferences("speaker_1").await.unwrap(),
            vec![
                ("language".to_string(), "tr".to_string()),
                ("voice".to_string(), "target-voice".to_string()),
            ]
        );

        assert!(
            db.merge_speakers("speaker_1", "speaker_1", &merged, 1)
                .await
                .is_err()
        );
        assert!(
            db.merge_speakers("ghost", "speaker_1", &merged, 1)
                .await
                .is_err()
        );
        assert!(
            db.merge_speakers("speaker_1", "ghost", &merged, 1)
                .await
                .is_err()
        );
        // A failed merge changes nothing.
        assert!(db.get_record("speaker_1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_speaker_ids_are_not_reused_after_delete_and_restart() {
        let tmp = TempDb::new("seq");

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        assert_eq!(db.speaker_sequence().await.unwrap(), 0);
        for n in 1..=3 {
            db.store_speaker(&speaker(&format!("speaker_{n}"), "S", &[1.0]))
                .await
                .unwrap();
        }
        // Custom IDs do not take part in the sequence.
        db.store_speaker(&speaker("alice", "Alice", &[1.0]))
            .await
            .unwrap();
        assert_eq!(db.speaker_sequence().await.unwrap(), 3);

        // Forget the highest-numbered speaker, then restart.
        db.delete_speaker("speaker_3").await.unwrap();
        db.close().await;

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        assert_eq!(
            db.speaker_sequence().await.unwrap(),
            3,
            "speaker_3 must not be issued again"
        );
        // Deleting everyone keeps the high-water mark too.
        assert_eq!(db.clear_speakers().await.unwrap(), 3);
        assert_eq!(db.speaker_sequence().await.unwrap(), 3);
        db.close().await;
    }

    #[tokio::test]
    async fn test_sequence_is_seeded_from_existing_speakers() {
        let tmp = TempDb::new("seed");

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        db.store_speaker(&speaker("speaker_7", "S", &[1.0]))
            .await
            .unwrap();
        // Simulate a file written before the high-water mark existed.
        sqlx::query("DELETE FROM meta")
            .execute(&db.pool)
            .await
            .unwrap();
        db.close().await;

        let db = SpeakerDatabase::open(tmp.path()).await.unwrap();
        assert_eq!(db.speaker_sequence().await.unwrap(), 7);
        db.close().await;
    }

    #[tokio::test]
    async fn test_in_memory_database_is_usable_concurrently() {
        // With a multi-connection pool each connection would see its own
        // empty `:memory:` database and these would fail with "no such table".
        let db = std::sync::Arc::new(SpeakerDatabase::open(":memory:").await.unwrap());
        let mut tasks = Vec::new();
        for n in 0..16 {
            let db = std::sync::Arc::clone(&db);
            tasks.push(tokio::spawn(async move {
                db.store_speaker(&speaker(&format!("s{n}"), "S", &[1.0]))
                    .await
                    .unwrap();
                db.list_records().await.unwrap().len()
            }));
        }
        for task in tasks {
            assert!(task.await.unwrap() >= 1);
        }
        assert_eq!(db.list_records().await.unwrap().len(), 16);
    }

    #[tokio::test]
    async fn test_enrollment_consent_round_trip() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        assert_eq!(db.enrollment_consent().await.unwrap(), None);

        db.set_enrollment_consent(true).await.unwrap();
        let granted_at = db.enrollment_consent().await.unwrap().unwrap();
        assert!(granted_at > 0);
        // Granting again keeps the original timestamp.
        db.set_enrollment_consent(true).await.unwrap();
        assert_eq!(db.enrollment_consent().await.unwrap(), Some(granted_at));

        db.set_enrollment_consent(false).await.unwrap();
        assert_eq!(db.enrollment_consent().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_export_contains_all_data_and_gates_the_embedding() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        db.store_speaker(&speaker("speaker_1", "Alice", &[0.6, 0.8]))
            .await
            .unwrap();
        db.set_preference("speaker_1", "voice", "af_heart")
            .await
            .unwrap();
        db.record_utterance("speaker_1", 700, Some("hello"))
            .await
            .unwrap();

        let export = db
            .export_speaker("speaker_1", false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(export.speaker.name, "Alice");
        assert!(export.speaker.has_voiceprint);
        assert!(export.embedding.is_none());
        assert_eq!(export.preferences.len(), 1);
        assert_eq!(export.conversations.len(), 1);
        let json = serde_json::to_value(&export).unwrap();
        assert!(json.get("embedding").is_none());

        let export = db.export_speaker("speaker_1", true).await.unwrap().unwrap();
        assert_eq!(export.embedding, Some(vec![0.6, 0.8]));

        assert!(db.export_speaker("ghost", true).await.unwrap().is_none());
    }
}
