//! Per-person long-term memory.
//!
//! What a recognized person and the assistant said to each other is kept
//! across sessions, so the next conversation can start with a short block of
//! remembered context. Everything stays in the local speaker database file
//! (the same SQLite file as the voiceprints; see
//! [`SpeakerDatabase`](crate::diarization::SpeakerDatabase)).
//!
//! Rules:
//!
//! - **Opt-in per person.** Nothing is written or recalled for a person
//!   until memory is switched on for them ([`MemoryStore::set_enabled`]).
//!   Switching it off stops both; what is stored stays until it is deleted
//!   ([`MemoryStore::forget`]) or the speaker is forgotten.
//! - **Known people only.** A memory row belongs to a stored speaker. There
//!   is no row for an unidentified voice, and the schema makes one
//!   impossible (foreign key to `speakers`).
//! - **Forgetting a speaker forgets their memory.** `persons` and `memories`
//!   are deleted with the speaker row (`ON DELETE CASCADE`), and a trigger
//!   removes the full-text index entries of every deleted memory. The index
//!   runs with FTS5 `secure-delete`, so removed terms are not left behind in
//!   its segments.
//!
//! Schema (created by [`create_schema`], which the speaker database calls
//! when it opens):
//!
//! ```text
//! persons(speaker_id → speakers.id, memory_opt_in, opted_in_at, updated_at)
//! memories(id, person_id → persons.speaker_id, role, content, interrupted,
//!          reply_to, created_at)        role: 'user' | 'assistant' | 'fact'
//! memories_fts                          FTS5 index over memories.content
//! ```
//!
//! The voiceprint itself stays in `speakers.embedding`; a person here is the
//! memory side of one stored speaker.

use std::collections::BTreeMap;

use serde::Serialize;
use sqlx::{Pool, Row, Sqlite, SqliteConnection};

use crate::diarization::SpeakerDatabase;
use crate::error::VoxError;

/// Longest text kept for one memory row; longer text is cut.
pub const MAX_MEMORY_CHARS: usize = 4_000;

/// Conversation rows (user and assistant) kept per person. The oldest go
/// first. Facts are not counted.
pub const MAX_CONVERSATION_ROWS: i64 = 2_000;

/// The id that marks an unidentified speaker in other parts of vox.
const UNKNOWN_SPEAKER: &str = "unknown";

/// Who a memory row is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRole {
    /// Something the person said.
    User,
    /// What the assistant answered (only what was actually heard).
    Assistant,
    /// A standalone fact about the person.
    Fact,
}

impl MemoryRole {
    /// The value of the `role` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Fact => "fact",
        }
    }

    fn from_column(value: &str) -> Self {
        match value {
            "assistant" => Self::Assistant,
            "fact" => Self::Fact,
            _ => Self::User,
        }
    }
}

/// One stored memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryItem {
    pub id: i64,
    pub role: MemoryRole,
    pub content: String,
    /// For an assistant row: the reply was cut short by the user.
    pub interrupted: bool,
    /// Unix seconds.
    pub created_at: u64,
}

/// Whether memory is on for a person, and how much is stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryStatus {
    pub speaker_id: String,
    pub enabled: bool,
    /// Unix seconds at which memory was switched on, while it is on.
    pub enabled_at: Option<u64>,
    /// Number of stored memory rows.
    pub count: u64,
}

/// Everything stored in memory about one person, for data export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PersonMemoryExport {
    pub enabled: bool,
    pub enabled_at: Option<u64>,
    /// Oldest first.
    pub memories: Vec<MemoryItem>,
}

/// The rows of one stored exchange, for correcting it afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangeRef {
    pub user_id: i64,
    pub assistant_id: Option<i64>,
}

/// How much [`MemoryStore::recall`] returns and how large a recall block
/// may get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallLimits {
    /// Newest facts.
    pub facts: u32,
    /// Newest exchanges (a user row and its reply).
    pub recent_exchanges: u32,
    /// Older rows that match the query text.
    pub related: u32,
    /// Longest single item in a block, in characters.
    pub max_item_chars: usize,
    /// Longest block, in characters.
    pub max_block_chars: usize,
}

impl Default for RecallLimits {
    fn default() -> Self {
        Self {
            facts: 8,
            recent_exchanges: 4,
            related: 4,
            max_item_chars: 240,
            max_block_chars: 1_800,
        }
    }
}

fn db_err(what: &'static str) -> impl Fn(sqlx::Error) -> VoxError {
    move |e| VoxError::Pipeline(format!("memory store: failed to {what}: {e}"))
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

/// Whether `id` can own memory at all (the speaker row is checked by the
/// database).
fn is_person_id(id: &str) -> bool {
    let id = id.trim();
    !id.is_empty() && !id.eq_ignore_ascii_case(UNKNOWN_SPEAKER)
}

/// Text as it is stored: trimmed and cut to [`MAX_MEMORY_CHARS`].
fn stored_text(text: &str) -> String {
    text.trim().chars().take(MAX_MEMORY_CHARS).collect()
}

/// Create the memory tables, the full-text index and its triggers if they do
/// not exist. Requires the `speakers` table.
pub(crate) async fn create_schema(pool: &Pool<Sqlite>) -> Result<(), VoxError> {
    const STATEMENTS: &[&str] = &[
        r#"
        CREATE TABLE IF NOT EXISTS persons (
            speaker_id TEXT PRIMARY KEY REFERENCES speakers(id) ON DELETE CASCADE,
            memory_opt_in INTEGER NOT NULL DEFAULT 0,
            opted_in_at INTEGER,
            updated_at INTEGER NOT NULL
        )
        "#,
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL REFERENCES persons(speaker_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'fact')),
            content TEXT NOT NULL,
            interrupted INTEGER NOT NULL DEFAULT 0,
            reply_to INTEGER,
            created_at INTEGER NOT NULL
        )
        "#,
        "CREATE INDEX IF NOT EXISTS memories_by_person ON memories(person_id, id)",
        r#"
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            content='memories',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
        "#,
        r#"
        CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END
        "#,
        r#"
        CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END
        "#,
        r#"
        CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END
        "#,
    ];
    for statement in STATEMENTS {
        sqlx::query(statement)
            .execute(pool)
            .await
            .map_err(db_err("create the memory schema"))?;
    }
    // Deleted text must not stay readable in the index. Needs SQLite 3.42;
    // an older library keeps working, with ordinary FTS5 deletes.
    if let Err(e) =
        sqlx::query("INSERT INTO memories_fts(memories_fts, rank) VALUES ('secure-delete', 1)")
            .execute(pool)
            .await
    {
        tracing::warn!(error = %e, "memory store: FTS5 secure-delete is not available");
    }
    Ok(())
}

/// Move the memory of `source` to `target`, inside the transaction that
/// merges the two speakers. `target` must exist in `speakers`.
///
/// The target keeps its own memory setting; if it has none yet it takes the
/// source's.
pub(crate) async fn reassign_person(
    conn: &mut SqliteConnection,
    source: &str,
    target: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT OR IGNORE INTO persons (speaker_id, memory_opt_in, opted_in_at, updated_at) \
         SELECT ?1, memory_opt_in, opted_in_at, ?3 FROM persons WHERE speaker_id = ?2",
    )
    .bind(target)
    .bind(source)
    .bind(now_secs())
    .execute(&mut *conn)
    .await?;
    sqlx::query("UPDATE memories SET person_id = ?1 WHERE person_id = ?2")
        .bind(target)
        .bind(source)
        .execute(&mut *conn)
        .await?;
    sqlx::query("DELETE FROM persons WHERE speaker_id = ?1")
        .bind(source)
        .execute(&mut *conn)
        .await?;
    Ok(())
}

fn item_from_row(row: &sqlx::sqlite::SqliteRow) -> MemoryItem {
    MemoryItem {
        id: row.get("id"),
        role: MemoryRole::from_column(row.get::<String, _>("role").as_str()),
        content: row.get("content"),
        interrupted: row.get::<i64, _>("interrupted") != 0,
        created_at: to_u64(row.get("created_at")),
    }
}

const ITEM_COLUMNS: &str = "m.id, m.role, m.content, m.interrupted, m.created_at";

/// Turn free text into an FTS5 query that matches any of its words.
/// `None` when the text has no word worth searching for.
fn fts_query(text: &str) -> Option<String> {
    const MAX_TERMS: usize = 8;
    let mut terms: Vec<String> = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.chars().count() < 3 {
            continue;
        }
        let word = word.to_lowercase();
        if !terms.contains(&word) {
            terms.push(word);
        }
        if terms.len() == MAX_TERMS {
            break;
        }
    }
    if terms.is_empty() {
        return None;
    }
    // Quoted, so a word is never read as FTS5 syntax.
    Some(
        terms
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// One line of a recall block: whitespace collapsed, cut to `max_chars`.
fn one_line(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let mut cut: String = flat.chars().take(max_chars.saturating_sub(1)).collect();
    cut.push('…');
    cut
}

/// Render remembered items as the block that is put in front of a user
/// message. `None` when there is nothing to say.
///
/// The output depends only on its arguments, so a block built once can be
/// frozen for a session. It starts with `[Memory`, which is what the
/// speaker-aware system prompt (`prompts::with_speaker_awareness`) tells the
/// model to look for.
pub fn format_recall_block(
    name: &str,
    items: &[MemoryItem],
    limits: &RecallLimits,
) -> Option<String> {
    let name = one_line(name, 64);
    let name = if name.is_empty() {
        "this person".to_string()
    } else {
        name
    };
    let facts: Vec<String> = items
        .iter()
        .filter(|i| i.role == MemoryRole::Fact)
        .map(|i| format!("- {}", one_line(&i.content, limits.max_item_chars)))
        .collect();
    let mut earlier: Vec<String> = items
        .iter()
        .filter(|i| i.role != MemoryRole::Fact)
        .map(|i| {
            let who = if i.role == MemoryRole::User {
                name.as_str()
            } else {
                "You"
            };
            format!("- {who}: {}", one_line(&i.content, limits.max_item_chars))
        })
        .collect();
    if facts.is_empty() && earlier.is_empty() {
        return None;
    }

    let render = |earlier: &[String]| {
        let mut block = format!("[Memory: notes from earlier conversations with {name}.");
        if !facts.is_empty() {
            block.push_str("\nKnown about them:\n");
            block.push_str(&facts.join("\n"));
        }
        if !earlier.is_empty() {
            block.push_str("\nSaid before, oldest first:\n");
            block.push_str(&earlier.join("\n"));
        }
        block.push_str("\nEnd of memory.]");
        block
    };
    let mut block = render(&earlier);
    // Too long: the oldest conversation lines go first.
    while block.chars().count() > limits.max_block_chars && !earlier.is_empty() {
        earlier.remove(0);
        block = render(&earlier);
    }
    Some(block)
}

/// Long-term memory in the speaker database file.
///
/// Cheap to create and to clone: it shares the database's connection pool.
#[derive(Clone)]
pub struct MemoryStore {
    pool: Pool<Sqlite>,
}

impl MemoryStore {
    /// The memory side of an open speaker database.
    pub fn new(db: &SpeakerDatabase) -> Self {
        Self {
            pool: db.pool().clone(),
        }
    }

    async fn speaker_exists(&self, speaker_id: &str) -> Result<bool, VoxError> {
        let found: Option<i64> = sqlx::query_scalar("SELECT 1 FROM speakers WHERE id = ?1")
            .bind(speaker_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(db_err("look up the speaker"))?;
        Ok(found.is_some())
    }

    /// Whether memory is switched on for this person. `false` for anyone
    /// unknown.
    pub async fn is_enabled(&self, speaker_id: &str) -> Result<bool, VoxError> {
        if !is_person_id(speaker_id) {
            return Ok(false);
        }
        let opt_in: Option<i64> =
            sqlx::query_scalar("SELECT memory_opt_in FROM persons WHERE speaker_id = ?1")
                .bind(speaker_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(db_err("read the memory setting"))?;
        Ok(opt_in == Some(1))
    }

    /// Switch memory on or off for a stored speaker. Returns `None` when
    /// there is no such speaker.
    ///
    /// Switching off keeps what is stored; use [`forget`](Self::forget) to
    /// delete it.
    pub async fn set_enabled(
        &self,
        speaker_id: &str,
        enabled: bool,
    ) -> Result<Option<MemoryStatus>, VoxError> {
        if !is_person_id(speaker_id) || !self.speaker_exists(speaker_id).await? {
            return Ok(None);
        }
        let now = now_secs();
        sqlx::query(
            r#"
            INSERT INTO persons (speaker_id, memory_opt_in, opted_in_at, updated_at)
            VALUES (?1, ?2, CASE WHEN ?2 = 1 THEN ?3 END, ?3)
            ON CONFLICT(speaker_id) DO UPDATE SET
                opted_in_at = CASE
                    WHEN excluded.memory_opt_in = 0 THEN NULL
                    WHEN persons.memory_opt_in = 1 THEN persons.opted_in_at
                    ELSE excluded.opted_in_at
                END,
                memory_opt_in = excluded.memory_opt_in,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(speaker_id)
        .bind(i64::from(enabled))
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(db_err("change the memory setting"))?;
        self.status(speaker_id).await
    }

    /// Memory setting and row count of a stored speaker; `None` when there
    /// is no such speaker.
    pub async fn status(&self, speaker_id: &str) -> Result<Option<MemoryStatus>, VoxError> {
        if !self.speaker_exists(speaker_id).await? {
            return Ok(None);
        }
        let row =
            sqlx::query("SELECT memory_opt_in, opted_in_at FROM persons WHERE speaker_id = ?1")
                .bind(speaker_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(db_err("read the memory setting"))?;
        let (enabled, enabled_at) = match row {
            Some(row) => (
                row.get::<i64, _>("memory_opt_in") == 1,
                row.get::<Option<i64>, _>("opted_in_at").map(to_u64),
            ),
            None => (false, None),
        };
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE person_id = ?1")
            .bind(speaker_id)
            .fetch_one(&self.pool)
            .await
            .map_err(db_err("count memories"))?;
        Ok(Some(MemoryStatus {
            speaker_id: speaker_id.to_string(),
            enabled,
            enabled_at: enabled_at.filter(|_| enabled),
            count: to_u64(count),
        }))
    }

    /// Store one exchange: what the person said and, if anything of it was
    /// heard, the assistant's reply.
    ///
    /// Returns `None` and writes nothing unless `speaker_id` is a stored
    /// speaker with memory switched on.
    pub async fn record_exchange(
        &self,
        speaker_id: &str,
        user_text: &str,
        assistant_text: Option<&str>,
        interrupted: bool,
    ) -> Result<Option<ExchangeRef>, VoxError> {
        let user_text = stored_text(user_text);
        if !is_person_id(speaker_id) || user_text.is_empty() {
            return Ok(None);
        }
        let assistant_text = assistant_text.map(stored_text).filter(|t| !t.is_empty());
        let err = db_err("store an exchange");
        let now = now_secs();

        let mut tx = self.pool.begin().await.map_err(&err)?;
        // Checked inside the transaction, so a setting that was just
        // switched off is honoured.
        let opt_in: Option<i64> =
            sqlx::query_scalar("SELECT memory_opt_in FROM persons WHERE speaker_id = ?1")
                .bind(speaker_id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(&err)?;
        if opt_in != Some(1) {
            return Ok(None);
        }

        let user_id = sqlx::query(
            "INSERT INTO memories (person_id, role, content, interrupted, created_at) \
             VALUES (?1, 'user', ?2, 0, ?3)",
        )
        .bind(speaker_id)
        .bind(&user_text)
        .bind(now)
        .execute(&mut *tx)
        .await
        .map_err(&err)?
        .last_insert_rowid();

        let mut assistant_id = None;
        if let Some(text) = &assistant_text {
            assistant_id = Some(
                insert_reply(&mut tx, speaker_id, text, interrupted, user_id, now)
                    .await
                    .map_err(&err)?,
            );
        }

        // Keep the newest rows only.
        sqlx::query(
            r#"
            DELETE FROM memories
            WHERE person_id = ?1 AND role != 'fact' AND id <= (
                SELECT id FROM memories
                WHERE person_id = ?1 AND role != 'fact'
                ORDER BY id DESC LIMIT 1 OFFSET ?2
            )
            "#,
        )
        .bind(speaker_id)
        .bind(MAX_CONVERSATION_ROWS)
        .execute(&mut *tx)
        .await
        .map_err(&err)?;

        tx.commit().await.map_err(&err)?;
        Ok(Some(ExchangeRef {
            user_id,
            assistant_id,
        }))
    }

    /// Correct the reply of a stored exchange after the fact: the listener
    /// heard only `heard_text` (nothing when `None` or empty) before
    /// interrupting. Returns the exchange as it is stored now.
    pub async fn amend_exchange(
        &self,
        exchange: ExchangeRef,
        heard_text: Option<&str>,
    ) -> Result<ExchangeRef, VoxError> {
        let err = db_err("correct an exchange");
        let heard = heard_text.map(stored_text).filter(|t| !t.is_empty());
        let mut tx = self.pool.begin().await.map_err(&err)?;
        // The exchange may be gone (forgotten, pruned); then there is
        // nothing to correct and nothing may be re-created.
        let owner: Option<String> =
            sqlx::query_scalar("SELECT person_id FROM memories WHERE id = ?1 AND role = 'user'")
                .bind(exchange.user_id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(&err)?;
        let Some(owner) = owner else {
            return Ok(ExchangeRef {
                user_id: exchange.user_id,
                assistant_id: None,
            });
        };

        let assistant_id = match (exchange.assistant_id, heard) {
            (Some(id), Some(text)) => {
                sqlx::query("UPDATE memories SET content = ?1, interrupted = 1 WHERE id = ?2")
                    .bind(&text)
                    .bind(id)
                    .execute(&mut *tx)
                    .await
                    .map_err(&err)?;
                Some(id)
            }
            (Some(id), None) => {
                sqlx::query("DELETE FROM memories WHERE id = ?1")
                    .bind(id)
                    .execute(&mut *tx)
                    .await
                    .map_err(&err)?;
                None
            }
            (None, Some(text)) => Some(
                insert_reply(&mut tx, &owner, &text, true, exchange.user_id, now_secs())
                    .await
                    .map_err(&err)?,
            ),
            (None, None) => None,
        };
        tx.commit().await.map_err(&err)?;
        Ok(ExchangeRef {
            user_id: exchange.user_id,
            assistant_id,
        })
    }

    /// Store a standalone fact about a person. `None` (nothing written)
    /// unless memory is switched on for them.
    pub async fn add_fact(&self, speaker_id: &str, fact: &str) -> Result<Option<i64>, VoxError> {
        let fact = stored_text(fact);
        if fact.is_empty() || !self.is_enabled(speaker_id).await? {
            return Ok(None);
        }
        let id = sqlx::query(
            "INSERT INTO memories (person_id, role, content, interrupted, created_at) \
             SELECT ?1, 'fact', ?2, 0, ?3 \
             WHERE EXISTS (SELECT 1 FROM persons WHERE speaker_id = ?1 AND memory_opt_in = 1)",
        )
        .bind(speaker_id)
        .bind(&fact)
        .bind(now_secs())
        .execute(&self.pool)
        .await
        .map_err(db_err("store a fact"))?;
        Ok((id.rows_affected() == 1).then(|| id.last_insert_rowid()))
    }

    /// What to remind the model of when this person speaks: their newest
    /// facts, the newest exchanges, and older rows that match `query` (the
    /// text they just said). Facts first, then oldest to newest.
    ///
    /// Empty unless memory is switched on for the person.
    pub async fn recall(
        &self,
        speaker_id: &str,
        query: Option<&str>,
        limits: &RecallLimits,
    ) -> Result<Vec<MemoryItem>, VoxError> {
        if !self.is_enabled(speaker_id).await? {
            return Ok(Vec::new());
        }
        let err = db_err("recall memories");

        let facts = sqlx::query(&format!(
            "SELECT {ITEM_COLUMNS} FROM memories m \
             WHERE m.person_id = ?1 AND m.role = 'fact' ORDER BY m.id DESC LIMIT ?2"
        ))
        .bind(speaker_id)
        .bind(i64::from(limits.facts))
        .fetch_all(&self.pool)
        .await
        .map_err(&err)?;

        let recent = sqlx::query(&format!(
            "SELECT {ITEM_COLUMNS} FROM memories m \
             WHERE m.person_id = ?1 AND m.role != 'fact' ORDER BY m.id DESC LIMIT ?2"
        ))
        .bind(speaker_id)
        .bind(i64::from(limits.recent_exchanges) * 2)
        .fetch_all(&self.pool)
        .await
        .map_err(&err)?;

        let related = match query.and_then(fts_query).filter(|_| limits.related > 0) {
            Some(fts) => sqlx::query(&format!(
                "SELECT {ITEM_COLUMNS} FROM memories_fts \
                 JOIN memories m ON m.id = memories_fts.rowid \
                 WHERE memories_fts MATCH ?1 AND m.person_id = ?2 AND m.role != 'fact' \
                 ORDER BY rank LIMIT ?3"
            ))
            .bind(fts)
            .bind(speaker_id)
            .bind(i64::from(limits.related))
            .fetch_all(&self.pool)
            .await
            .map_err(&err)?,
            None => Vec::new(),
        };

        let mut facts: Vec<MemoryItem> = facts.iter().map(item_from_row).collect();
        facts.reverse();
        // By id: no duplicates, oldest first.
        let conversation: BTreeMap<i64, MemoryItem> = recent
            .iter()
            .chain(related.iter())
            .map(item_from_row)
            .map(|item| (item.id, item))
            .collect();
        facts.extend(conversation.into_values());
        Ok(facts)
    }

    /// [`recall`](Self::recall) rendered with [`format_recall_block`].
    /// `None` when memory is off for the person or nothing is stored.
    pub async fn recall_block(
        &self,
        speaker_id: &str,
        name: &str,
        query: Option<&str>,
    ) -> Result<Option<String>, VoxError> {
        let limits = RecallLimits::default();
        let items = self.recall(speaker_id, query, &limits).await?;
        Ok(format_recall_block(name, &items, &limits))
    }

    /// Stored memories of a person, newest first.
    pub async fn list(&self, speaker_id: &str, limit: u32) -> Result<Vec<MemoryItem>, VoxError> {
        let rows = sqlx::query(&format!(
            "SELECT {ITEM_COLUMNS} FROM memories m \
             WHERE m.person_id = ?1 ORDER BY m.id DESC LIMIT ?2"
        ))
        .bind(speaker_id)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await
        .map_err(db_err("list memories"))?;
        Ok(rows.iter().map(item_from_row).collect())
    }

    /// Delete everything remembered about a person. The speaker, their
    /// voiceprint and their memory setting stay. Returns the number of rows
    /// deleted.
    pub async fn forget(&self, speaker_id: &str) -> Result<u64, VoxError> {
        let deleted = sqlx::query("DELETE FROM memories WHERE person_id = ?1")
            .bind(speaker_id)
            .execute(&self.pool)
            .await
            .map_err(db_err("delete memories"))?
            .rows_affected();
        if deleted > 0 {
            // As for a forgotten speaker: do not leave the pages on disk.
            if let Err(e) = sqlx::query("VACUUM").execute(&self.pool).await {
                tracing::warn!(error = %e, "memory store: vacuum failed");
            }
        }
        Ok(deleted)
    }

    /// Everything stored in memory about a person; `None` when there is no
    /// such speaker.
    pub async fn export(&self, speaker_id: &str) -> Result<Option<PersonMemoryExport>, VoxError> {
        let Some(status) = self.status(speaker_id).await? else {
            return Ok(None);
        };
        let mut memories = self.list(speaker_id, u32::MAX).await?;
        memories.reverse();
        Ok(Some(PersonMemoryExport {
            enabled: status.enabled,
            enabled_at: status.enabled_at,
            memories,
        }))
    }
}

async fn insert_reply(
    conn: &mut SqliteConnection,
    speaker_id: &str,
    text: &str,
    interrupted: bool,
    reply_to: i64,
    now: i64,
) -> Result<i64, sqlx::Error> {
    Ok(sqlx::query(
        "INSERT INTO memories (person_id, role, content, interrupted, reply_to, created_at) \
         VALUES (?1, 'assistant', ?2, ?3, ?4, ?5)",
    )
    .bind(speaker_id)
    .bind(text)
    .bind(i64::from(interrupted))
    .bind(reply_to)
    .bind(now)
    .execute(conn)
    .await?
    .last_insert_rowid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarization::Speaker;

    async fn db_with(speakers: &[(&str, &str)]) -> (SpeakerDatabase, MemoryStore) {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();
        for (id, name) in speakers {
            add_speaker(&db, id, name).await;
        }
        let memory = MemoryStore::new(&db);
        (db, memory)
    }

    async fn add_speaker(db: &SpeakerDatabase, id: &str, name: &str) {
        db.store_speaker(&Speaker {
            id: id.to_string(),
            name: name.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        })
        .await
        .unwrap();
    }

    async fn count(memory: &MemoryStore, sql: &str) -> i64 {
        sqlx::query_scalar(sql)
            .fetch_one(&memory.pool)
            .await
            .unwrap()
    }

    /// Number of distinct terms left in the full-text index.
    async fn indexed_terms(memory: &MemoryStore) -> i64 {
        sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memories_vocab \
             USING fts5vocab('main', 'memories_fts', 'row')",
        )
        .execute(&memory.pool)
        .await
        .unwrap();
        count(memory, "SELECT COUNT(*) FROM temp.memories_vocab").await
    }

    fn temp_db_path(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vox-memory-{tag}-{}-{}",
            std::process::id(),
            now_secs()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("speakers.db")
    }

    #[tokio::test]
    async fn nothing_is_written_while_memory_is_off() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;

        // Never switched on.
        let stored = memory
            .record_exchange("speaker_1", "I like tea", Some("Noted."), false)
            .await
            .unwrap();
        assert_eq!(stored, None);
        assert_eq!(
            memory.add_fact("speaker_1", "likes tea").await.unwrap(),
            None
        );
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 0);
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM persons").await, 0);

        // Switched on, then off again.
        memory
            .set_enabled("speaker_1", true)
            .await
            .unwrap()
            .unwrap();
        memory
            .set_enabled("speaker_1", false)
            .await
            .unwrap()
            .unwrap();
        let stored = memory
            .record_exchange("speaker_1", "I like tea", Some("Noted."), false)
            .await
            .unwrap();
        assert_eq!(stored, None);
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 0);
        assert_eq!(
            memory
                .recall_block("speaker_1", "Mert", None)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn nothing_is_written_for_an_unknown_speaker() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        for id in ["unknown", "Unknown", "", "speaker_99"] {
            assert_eq!(memory.set_enabled(id, true).await.unwrap(), None, "{id:?}");
            let stored = memory
                .record_exchange(id, "hello", Some("Hi."), false)
                .await
                .unwrap();
            assert_eq!(stored, None, "{id:?}");
        }
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 0);
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM persons").await, 0);
        // The schema itself refuses a memory without a stored speaker.
        let forced = sqlx::query(
            "INSERT INTO persons (speaker_id, memory_opt_in, updated_at) VALUES ('unknown', 1, 0)",
        )
        .execute(&memory.pool)
        .await;
        assert!(forced.is_err());
    }

    #[tokio::test]
    async fn status_follows_the_setting() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        assert_eq!(memory.status("nobody").await.unwrap(), None);
        let status = memory.status("speaker_1").await.unwrap().unwrap();
        assert!(!status.enabled);
        assert_eq!(status.enabled_at, None);
        assert_eq!(status.count, 0);

        let status = memory
            .set_enabled("speaker_1", true)
            .await
            .unwrap()
            .unwrap();
        assert!(status.enabled);
        let since = status.enabled_at.expect("time of opt-in");
        // Switching on again keeps the original time.
        let again = memory
            .set_enabled("speaker_1", true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(again.enabled_at, Some(since));

        memory
            .record_exchange("speaker_1", "hello", Some("Hi."), false)
            .await
            .unwrap()
            .unwrap();
        let off = memory
            .set_enabled("speaker_1", false)
            .await
            .unwrap()
            .unwrap();
        assert!(!off.enabled);
        assert_eq!(off.enabled_at, None);
        // Switching off does not delete.
        assert_eq!(off.count, 2);
    }

    #[tokio::test]
    async fn recall_is_per_person() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert"), ("speaker_2", "Ada")]).await;
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory.set_enabled("speaker_2", true).await.unwrap();
        memory
            .record_exchange(
                "speaker_1",
                "My dog is called Pamuk",
                Some("Nice name."),
                false,
            )
            .await
            .unwrap()
            .unwrap();
        memory
            .add_fact("speaker_1", "Has a dog")
            .await
            .unwrap()
            .unwrap();

        let block = memory
            .recall_block("speaker_1", "Mert", None)
            .await
            .unwrap()
            .expect("a block for Mert");
        assert!(block.starts_with("[Memory: notes from earlier conversations with Mert."));
        assert!(block.ends_with("End of memory.]"));
        assert!(block.contains("- Has a dog"));
        assert!(block.contains("- Mert: My dog is called Pamuk"));
        assert!(block.contains("- You: Nice name."));
        // Deterministic: the same store gives the same bytes.
        assert_eq!(
            memory
                .recall_block("speaker_1", "Mert", None)
                .await
                .unwrap(),
            Some(block)
        );

        assert_eq!(
            memory
                .recall_block("speaker_2", "Ada", Some("dog"))
                .await
                .unwrap(),
            None
        );
        assert!(memory.list("speaker_2", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn recall_survives_a_restart_for_the_same_person_only() {
        let path = temp_db_path("restart");
        let path_str = path.to_string_lossy().to_string();
        {
            let db = SpeakerDatabase::open(&path_str).await.unwrap();
            add_speaker(&db, "speaker_1", "Mert").await;
            add_speaker(&db, "speaker_2", "Ada").await;
            let memory = MemoryStore::new(&db);
            memory.set_enabled("speaker_1", true).await.unwrap();
            memory.set_enabled("speaker_2", true).await.unwrap();
            memory
                .record_exchange(
                    "speaker_1",
                    "I moved to Izmir",
                    Some("Good to know."),
                    false,
                )
                .await
                .unwrap()
                .unwrap();
            db.close().await;
        }

        let db = SpeakerDatabase::open(&path_str).await.unwrap();
        let memory = MemoryStore::new(&db);
        let block = memory
            .recall_block("speaker_1", "Mert", Some("where do I live"))
            .await
            .unwrap()
            .expect("the block is there after reopening");
        assert!(block.contains("I moved to Izmir"));
        assert!(block.contains("Good to know."));
        assert_eq!(
            memory
                .recall_block("speaker_2", "Ada", Some("Izmir"))
                .await
                .unwrap(),
            None
        );
        db.close().await;
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn recall_finds_an_older_exchange_by_its_words() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory
            .record_exchange(
                "speaker_1",
                "My sister Elif plays the cello",
                Some("Lovely."),
                false,
            )
            .await
            .unwrap();
        for n in 0..10 {
            memory
                .record_exchange("speaker_1", &format!("filler {n}"), Some("Okay."), false)
                .await
                .unwrap();
        }
        let limits = RecallLimits::default();

        let plain = memory.recall("speaker_1", None, &limits).await.unwrap();
        assert_eq!(plain.len(), limits.recent_exchanges as usize * 2);
        assert!(plain.iter().all(|i| !i.content.contains("cello")));
        assert!(plain.windows(2).all(|w| w[0].id < w[1].id));

        let related = memory
            .recall("speaker_1", Some("Does Elif still play?"), &limits)
            .await
            .unwrap();
        assert!(related[0].content.contains("cello"));
        // Query text is never read as FTS syntax.
        memory
            .recall("speaker_1", Some("\"NEAR( -x* ) AND OR"), &limits)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn forgetting_the_speaker_removes_memories_and_index_entries() {
        let (db, memory) = db_with(&[("speaker_1", "Mert"), ("speaker_2", "Ada")]).await;
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory.set_enabled("speaker_2", true).await.unwrap();
        memory
            .record_exchange(
                "speaker_1",
                "My passport number is zebra",
                Some("Okay."),
                false,
            )
            .await
            .unwrap();
        memory
            .add_fact("speaker_1", "Owns a giraffe")
            .await
            .unwrap();
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 3);
        assert_eq!(
            count(&memory, "SELECT COUNT(*) FROM memories_fts_docsize").await,
            3
        );
        assert!(indexed_terms(&memory).await > 0);

        db.delete_speaker("speaker_1").await.unwrap();

        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 0);
        assert_eq!(
            count(&memory, "SELECT COUNT(*) FROM memories_fts_docsize").await,
            0
        );
        assert_eq!(indexed_terms(&memory).await, 0);
        assert_eq!(
            count(
                &memory,
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'zebra OR giraffe'"
            )
            .await,
            0
        );
        assert_eq!(
            count(
                &memory,
                "SELECT COUNT(*) FROM persons WHERE speaker_id = 'speaker_1'"
            )
            .await,
            0
        );
        // The other person is untouched.
        assert!(memory.is_enabled("speaker_2").await.unwrap());

        // Forget-all takes the rest.
        memory
            .record_exchange("speaker_2", "hello there", Some("Hi."), false)
            .await
            .unwrap();
        db.clear_speakers().await.unwrap();
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM memories").await, 0);
        assert_eq!(count(&memory, "SELECT COUNT(*) FROM persons").await, 0);
        assert_eq!(indexed_terms(&memory).await, 0);
    }

    #[tokio::test]
    async fn forgetting_only_the_memory_keeps_the_speaker() {
        let (db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory
            .record_exchange("speaker_1", "remember the walrus", Some("Okay."), false)
            .await
            .unwrap();
        assert_eq!(memory.forget("speaker_1").await.unwrap(), 2);
        assert_eq!(memory.forget("speaker_1").await.unwrap(), 0);
        assert_eq!(indexed_terms(&memory).await, 0);
        assert!(db.get_record("speaker_1").await.unwrap().is_some());
        assert!(memory.is_enabled("speaker_1").await.unwrap());
    }

    #[tokio::test]
    async fn an_exchange_can_be_cut_down_to_what_was_heard() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        memory.set_enabled("speaker_1", true).await.unwrap();
        let exchange = memory
            .record_exchange(
                "speaker_1",
                "tell me a story",
                Some("Once. Upon. A time."),
                false,
            )
            .await
            .unwrap()
            .unwrap();

        let exchange = memory
            .amend_exchange(exchange, Some("Once."))
            .await
            .unwrap();
        let items = memory.list("speaker_1", 10).await.unwrap();
        assert_eq!(items[0].content, "Once.");
        assert!(items[0].interrupted);
        // The index follows the correction.
        assert_eq!(
            count(
                &memory,
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'upon'"
            )
            .await,
            0
        );

        // Nothing was heard: only what the user said stays.
        let exchange = memory.amend_exchange(exchange, None).await.unwrap();
        assert_eq!(exchange.assistant_id, None);
        let items = memory.list("speaker_1", 10).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].role, MemoryRole::User);

        // A correction after the memory was deleted re-creates nothing.
        memory.forget("speaker_1").await.unwrap();
        let exchange = memory
            .amend_exchange(exchange, Some("Once."))
            .await
            .unwrap();
        assert_eq!(exchange.assistant_id, None);
        assert!(memory.list("speaker_1", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn merging_speakers_moves_the_memory() {
        let (db, memory) = db_with(&[("speaker_1", "Mert"), ("speaker_2", "Mert again")]).await;
        memory.set_enabled("speaker_2", true).await.unwrap();
        memory
            .record_exchange("speaker_2", "I like pistachio", Some("Noted."), false)
            .await
            .unwrap();

        db.merge_speakers("speaker_2", "speaker_1", &[0.1, 0.2, 0.3], 2)
            .await
            .unwrap();

        // The target had no setting of its own and takes the source's.
        let status = memory.status("speaker_1").await.unwrap().unwrap();
        assert!(status.enabled);
        assert_eq!(status.count, 2);
        let block = memory
            .recall_block("speaker_1", "Mert", Some("pistachio"))
            .await
            .unwrap()
            .unwrap();
        assert!(block.contains("I like pistachio"));
        assert_eq!(
            count(
                &memory,
                "SELECT COUNT(*) FROM persons WHERE speaker_id = 'speaker_2'"
            )
            .await,
            0
        );
    }

    #[tokio::test]
    async fn export_lists_everything_oldest_first() {
        let (_db, memory) = db_with(&[("speaker_1", "Mert")]).await;
        assert_eq!(memory.export("nobody").await.unwrap(), None);
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory
            .record_exchange("speaker_1", "first", Some("One."), false)
            .await
            .unwrap();
        memory
            .record_exchange("speaker_1", "second", None, true)
            .await
            .unwrap();
        let export = memory.export("speaker_1").await.unwrap().unwrap();
        assert!(export.enabled);
        let texts: Vec<_> = export.memories.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(texts, ["first", "One.", "second"]);
        let json = serde_json::to_value(&export).unwrap();
        assert_eq!(json["memories"][1]["role"], "assistant");
    }

    #[test]
    fn recall_block_is_bounded_and_drops_the_oldest_lines_first() {
        let item = |id: i64, role, content: &str| MemoryItem {
            id,
            role,
            content: content.to_string(),
            interrupted: false,
            created_at: 0,
        };
        let limits = RecallLimits {
            max_item_chars: 20,
            max_block_chars: 200,
            ..RecallLimits::default()
        };
        let mut items = vec![item(
            1,
            MemoryRole::Fact,
            "likes\ntea   a lot, really a lot",
        )];
        for n in 0..20 {
            items.push(item(10 + n, MemoryRole::User, &format!("line number {n}")));
        }
        let block = format_recall_block("Mert", &items, &limits).unwrap();
        assert!(block.chars().count() <= 200);
        // One line per item, cut with an ellipsis.
        assert!(block.contains("- likes tea a lot, re…"));
        assert!(block.contains("line number 19"));
        assert!(!block.contains("line number 0\n"));
        assert_eq!(format_recall_block("Mert", &[], &limits), None);
    }

    #[test]
    fn fts_query_quotes_words_and_skips_noise() {
        assert_eq!(fts_query("a  an !!"), None);
        assert_eq!(
            fts_query("Does Elif play? DOES"),
            Some("\"does\" OR \"elif\" OR \"play\"".to_string())
        );
    }
}
