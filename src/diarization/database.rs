//! Persistent speaker database using SQLite.
//!
//! This module provides a SQLite-based speaker database for storing:
//! - Speaker profiles (ID, name, reference embedding)
//! - Conversation history
//! - Speaker preferences
//!
//! All data is stored locally with privacy-first design.

use sqlx::{Pool, Row, Sqlite, SqlitePool};

use crate::error::VoxError;

use super::recognition::Speaker;

/// SQLite-based speaker database.
///
/// Stores speaker embeddings, conversation history, and preferences.
/// All data is stored locally in a SQLite file.
pub struct SpeakerDatabase {
    pool: Pool<Sqlite>,
}

impl SpeakerDatabase {
    /// Open or create a speaker database at the given path.
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file (will be created if it doesn't exist)
    ///
    /// # Example
    /// ```ignore
    /// let db = SpeakerDatabase::open("speakers.db").await?;
    /// ```
    pub async fn open(db_path: impl AsRef<str>) -> Result<Self, VoxError> {
        let url = format!("sqlite:{}", db_path.as_ref());
        let pool = SqlitePool::connect(&url)
            .await
            .map_err(|e| VoxError::Diarization(format!("failed to open database: {e}")))?;

        let db = Self { pool };
        db.create_tables().await?;

        Ok(db)
    }

    /// Create database tables if they don't exist.
    async fn create_tables(&self) -> Result<(), VoxError> {
        // Speakers table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen INTEGER NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to create speakers table: {e}")))?;

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
        let embedding_bytes = embedding_to_bytes(&speaker.embedding);
        let now = timestamp_now();

        sqlx::query(
            r#"
            INSERT INTO speakers (id, name, embedding, created_at, last_seen)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
        )
        .bind(&speaker.id)
        .bind(&speaker.name)
        .bind(&embedding_bytes)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to store speaker: {e}")))?;

        Ok(())
    }

    /// Load a speaker from the database by ID.
    pub async fn load_speaker(&self, id: &str) -> Result<Option<Speaker>, VoxError> {
        let row = sqlx::query(
            r#"
            SELECT id, name, embedding FROM speakers WHERE id = ?1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to load speaker: {e}")))?;

        match row {
            Some(row) => {
                let id: String = row.get("id");
                let name: String = row.get("name");
                let embedding_bytes: Vec<u8> = row.get("embedding");
                let embedding = bytes_to_embedding(&embedding_bytes)?;

                Ok(Some(Speaker {
                    id,
                    name,
                    embedding,
                }))
            }
            None => Ok(None),
        }
    }

    /// List all speakers in the database.
    pub async fn list_speakers(&self) -> Result<Vec<Speaker>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, embedding FROM speakers ORDER BY last_seen DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to list speakers: {e}")))?;

        let mut speakers = Vec::new();
        for row in rows {
            let id: String = row.get("id");
            let name: String = row.get("name");
            let embedding_bytes: Vec<u8> = row.get("embedding");
            let embedding = bytes_to_embedding(&embedding_bytes)?;

            speakers.push(Speaker {
                id,
                name,
                embedding,
            });
        }

        Ok(speakers)
    }

    /// Delete a speaker from the database.
    ///
    /// This also deletes all associated conversations and preferences (CASCADE).
    pub async fn delete_speaker(&self, id: &str) -> Result<(), VoxError> {
        let result = sqlx::query(
            r#"
            DELETE FROM speakers WHERE id = ?1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| VoxError::Diarization(format!("failed to delete speaker: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(VoxError::Diarization(format!(
                "speaker with ID '{id}' not found"
            )));
        }

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
            ORDER BY timestamp DESC
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

    /// Delete all speakers from the database.
    ///
    /// Used when embedding pipeline version changes to clear stale speakers
    /// whose embeddings are incompatible with the new preprocessing.
    pub async fn clear_speakers(&self) -> Result<(), VoxError> {
        sqlx::query("DELETE FROM speakers")
            .execute(&self.pool)
            .await
            .map_err(|e| VoxError::Diarization(format!("failed to clear speakers: {e}")))?;
        Ok(())
    }

    /// Get a system-level preference (not tied to a speaker).
    ///
    /// Uses `"_system"` as the speaker_id in the preferences table.
    pub async fn get_system_preference(&self, key: &str) -> Result<Option<String>, VoxError> {
        self.get_preference("_system", key).await
    }

    /// Set a system-level preference (not tied to a speaker).
    ///
    /// Uses `"_system"` as the speaker_id in the preferences table.
    pub async fn set_system_preference(&self, key: &str, value: &str) -> Result<(), VoxError> {
        self.set_preference("_system", key, value).await
    }

    /// Get all preferences for a speaker.
    pub async fn get_all_preferences(
        &self,
        speaker_id: &str,
    ) -> Result<Vec<(String, String)>, VoxError> {
        let rows = sqlx::query(
            r#"
            SELECT key, value FROM preferences WHERE speaker_id = ?1
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
}

/// A conversation history entry.
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Optional transcribed text.
    pub text: Option<String>,
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
    if bytes.len() % 4 != 0 {
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
        .unwrap()
        .as_secs() as i64
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
}
