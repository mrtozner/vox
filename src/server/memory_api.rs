//! REST API for per-person long-term memory (see [`crate::memory`]).
//!
//! | Method | Path | |
//! |---|---|---|
//! | `GET` | `/v1/speakers/{id}/memory` | setting, count and stored memories (newest first, `?limit=`) |
//! | `PUT` | `/v1/speakers/{id}/memory` | switch memory on or off (`{"enabled": true}`) |
//! | `DELETE` | `/v1/speakers/{id}/memory` | delete what is remembered; the speaker stays |
//!
//! Memory is off for every speaker until it is switched on here. Switching
//! it off stops writing and recalling but deletes nothing. Forgetting the
//! speaker (`DELETE /v1/speakers/{id}`) deletes their memory with them, and
//! `GET /v1/speakers/{id}/export` includes it.
//!
//! Errors have the same shape and codes as the speaker API.

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRef, Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::memory::{MemoryItem, MemoryStatus, MemoryStore};

use super::speakers::{SpeakerCtx, SpeakerError};

/// Memories returned by `GET` when `limit` is not given.
const DEFAULT_LIST_LIMIT: u32 = 100;
/// Largest accepted `limit`.
const MAX_LIST_LIMIT: u32 = 1_000;

/// The memory routes, for merging into the server's router:
/// `.merge(memory_api::routes())`.
pub(super) fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    SpeakerCtx: FromRef<S>,
{
    Router::new().route(
        "/v1/speakers/{id}/memory",
        get(read).put(update).delete(forget),
    )
}

fn store(ctx: &SpeakerCtx) -> Result<MemoryStore, SpeakerError> {
    ctx.db
        .as_deref()
        .map(MemoryStore::new)
        .ok_or(SpeakerError::Unavailable)
}

fn not_found(id: &str) -> SpeakerError {
    SpeakerError::NotFound(id.to_string())
}

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<u32>,
}

#[derive(Serialize)]
struct MemoryResponse {
    #[serde(flatten)]
    status: MemoryStatus,
    /// Newest first; at most `limit` of the `count` stored rows.
    memories: Vec<MemoryItem>,
}

async fn read(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Response, SpeakerError> {
    let store = store(&ctx)?;
    let status = store.status(&id).await?.ok_or_else(|| not_found(&id))?;
    let limit = query
        .limit
        .unwrap_or(DEFAULT_LIST_LIMIT)
        .min(MAX_LIST_LIMIT);
    let memories = store.list(&id, limit).await?;
    // What people said: not for caches.
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(MemoryResponse { status, memories }),
    )
        .into_response())
}

#[derive(Deserialize)]
struct UpdateRequest {
    enabled: bool,
}

async fn update(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
    payload: Result<Json<UpdateRequest>, JsonRejection>,
) -> Result<Json<MemoryStatus>, SpeakerError> {
    let Json(request) =
        payload.map_err(|rejection| SpeakerError::Invalid(rejection.body_text()))?;
    let status = store(&ctx)?
        .set_enabled(&id, request.enabled)
        .await?
        .ok_or_else(|| not_found(&id))?;
    tracing::info!(speaker_id = %id, enabled = status.enabled, "long-term memory setting changed");
    Ok(Json(status))
}

async fn forget(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, SpeakerError> {
    let store = store(&ctx)?;
    store.status(&id).await?.ok_or_else(|| not_found(&id))?;
    let deleted = store.forget(&id).await?;
    tracing::info!(speaker_id = %id, deleted, "long-term memory deleted");
    Ok(Json(json!({ "speaker_id": id, "deleted": deleted })))
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::super::speakers::tests::{enroll, test_ctx};

    async fn call(
        ctx: &SpeakerCtx,
        method: Method,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let mut request = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(value) => {
                request = request.header(header::CONTENT_TYPE, "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        let app = routes::<SpeakerCtx>().with_state(ctx.clone());
        let response = app.oneshot(request.body(body).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    const URI: &str = "/v1/speakers/speaker_1/memory";

    #[tokio::test]
    async fn memory_is_off_until_switched_on() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;

        let (status, json) = call(&ctx, Method::GET, URI, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json,
            json!({
                "speaker_id": "speaker_1",
                "enabled": false,
                "enabled_at": null,
                "count": 0,
                "memories": [],
            })
        );

        let (status, json) = call(&ctx, Method::PUT, URI, Some(json!({"enabled": true}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["enabled"], true);
        assert!(json["enabled_at"].is_u64());
        assert_eq!(json["count"], 0);
    }

    #[tokio::test]
    async fn stored_memories_are_listed_and_deleted() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        let memory = store(&ctx).unwrap();
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory
            .record_exchange("speaker_1", "I like tea", Some("Noted."), false)
            .await
            .unwrap();

        let (status, json) = call(&ctx, Method::GET, URI, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["count"], 2);
        let memories = json["memories"].as_array().unwrap();
        assert_eq!(memories[0]["role"], "assistant");
        assert_eq!(memories[0]["content"], "Noted.");
        assert_eq!(memories[1]["role"], "user");
        assert_eq!(memories[1]["content"], "I like tea");

        let (_, json) = call(&ctx, Method::GET, &format!("{URI}?limit=1"), None).await;
        assert_eq!(json["count"], 2);
        assert_eq!(json["memories"].as_array().unwrap().len(), 1);

        let (status, json) = call(&ctx, Method::DELETE, URI, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({ "speaker_id": "speaker_1", "deleted": 2 }));
        // The speaker and the setting stay.
        let (_, json) = call(&ctx, Method::GET, URI, None).await;
        assert_eq!(json["enabled"], true);
        assert_eq!(json["count"], 0);
    }

    #[tokio::test]
    async fn unknown_speaker_bad_body_and_missing_diarization() {
        let ctx = test_ctx().await;
        for method in [Method::GET, Method::DELETE] {
            let (status, json) = call(&ctx, method, URI, None).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(json["error"]["code"], "speaker_not_found");
        }
        let (status, _) = call(&ctx, Method::PUT, URI, Some(json!({"enabled": true}))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        let (status, json) = call(&ctx, Method::PUT, URI, Some(json!({"on": 1}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "invalid_request");

        let (status, json) = call(&SpeakerCtx::default(), Method::GET, URI, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["error"]["code"], "diarization_unavailable");
    }
}
