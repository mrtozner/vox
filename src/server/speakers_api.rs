//! REST API for the speaker (voiceprint) store.
//!
//! | Method | Path | |
//! |---|---|---|
//! | `GET` | `/v1/speakers` | list speakers, consent state, availability |
//! | `DELETE` | `/v1/speakers?confirm=all` | forget every speaker |
//! | `PATCH` | `/v1/speakers/{id}` | rename (`{"name": "..."}`) |
//! | `DELETE` | `/v1/speakers/{id}` | forget one speaker and their data |
//! | `POST` | `/v1/speakers/merge` | merge `{"source": "...", "target": "..."}` |
//! | `GET` | `/v1/speakers/{id}/export` | everything stored about a speaker, long-term memory included |
//! | `GET` / `PUT` | `/v1/speakers/consent` | read / set enrollment consent |
//!
//! Embeddings never leave the server except through `export` with
//! `?include_embedding=true`. Errors use the server-wide body
//! (`{"error": {message, type, code, param}, "message": "..."}`) with 400
//! (invalid request), 404 (unknown speaker), 503 (diarization not loaded)
//! or 500.

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRef, Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::speakers::{self, ConsentStatus, SpeakerCtx, SpeakerDetail, SpeakerError};

/// Format tag written into every export, so a future importer can tell
/// what it is reading.
const EXPORT_FORMAT: &str = "vox-speaker-export/1";

/// The speaker routes, for merging into the server's router:
/// `.merge(speakers_api::routes())`.
pub(super) fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    SpeakerCtx: FromRef<S>,
{
    Router::new()
        .route("/v1/speakers", get(list).delete(forget_all))
        .route("/v1/speakers/merge", post(merge))
        .route("/v1/speakers/consent", get(get_consent).put(put_consent))
        .route("/v1/speakers/{id}", patch(rename).delete(forget))
        .route("/v1/speakers/{id}/export", get(export))
}

impl IntoResponse for SpeakerError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Invalid(_) => StatusCode::BAD_REQUEST,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status.is_server_error() {
            tracing::warn!(error = %self, "speaker API request failed");
        }
        let code = match &self {
            Self::Unavailable => "diarization_unavailable",
            Self::NotFound(_) => "speaker_not_found",
            Self::Invalid(_) => "invalid_request",
            Self::Internal(_) => "internal_error",
        };
        super::error::ServerError::new(status, self.to_string())
            .with_code(code)
            .into_response()
    }
}

/// Turn a body that is not the expected JSON into the API's own 400 shape
/// instead of axum's plain-text rejection.
fn body<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, SpeakerError> {
    payload
        .map(|Json(value)| value)
        .map_err(|rejection| SpeakerError::Invalid(rejection.body_text()))
}

#[derive(Serialize)]
struct SpeakerListResponse {
    /// `false` when the speaker encoder is not loaded; the list is then
    /// empty and every mutation returns 503.
    available: bool,
    enrollment_consent: ConsentStatus,
    /// Utterances shorter than this are never attributed to a speaker.
    min_audio_ms: u64,
    speakers: Vec<SpeakerDetail>,
}

async fn list(State(ctx): State<SpeakerCtx>) -> Result<Json<SpeakerListResponse>, SpeakerError> {
    let available = ctx.db.is_some();
    let speakers = if available {
        speakers::list_speakers(&ctx).await?
    } else {
        Vec::new()
    };
    Ok(Json(SpeakerListResponse {
        available,
        enrollment_consent: speakers::enrollment_consent(&ctx).await,
        min_audio_ms: crate::diarization::MIN_SPEAKER_AUDIO_MS,
        speakers,
    }))
}

#[derive(Deserialize)]
struct ForgetAllQuery {
    confirm: Option<String>,
}

async fn forget_all(
    State(ctx): State<SpeakerCtx>,
    Query(query): Query<ForgetAllQuery>,
) -> Result<Json<serde_json::Value>, SpeakerError> {
    // Deleting everyone is one request away from deleting one speaker with
    // a mistyped URL; make the caller say it.
    if query.confirm.as_deref() != Some("all") {
        return Err(SpeakerError::Invalid(
            "deleting every speaker requires ?confirm=all".into(),
        ));
    }
    let deleted = speakers::forget_all(&ctx).await?;
    Ok(Json(json!({ "deleted": deleted })))
}

#[derive(Deserialize)]
struct RenameRequest {
    name: String,
}

async fn rename(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
    payload: Result<Json<RenameRequest>, JsonRejection>,
) -> Result<Json<SpeakerDetail>, SpeakerError> {
    let request = body(payload)?;
    Ok(Json(speakers::rename(&ctx, &id, &request.name).await?))
}

async fn forget(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, SpeakerError> {
    speakers::forget(&ctx, &id).await?;
    Ok(Json(json!({ "deleted": 1, "id": id })))
}

#[derive(Deserialize)]
struct MergeRequest {
    /// The speaker that disappears.
    source: String,
    /// The speaker that remains and absorbs the source.
    target: String,
}

#[derive(Serialize)]
struct MergeResponse {
    /// The surviving speaker after the merge.
    speaker: SpeakerDetail,
    /// ID of the speaker that was merged away.
    merged: String,
}

async fn merge(
    State(ctx): State<SpeakerCtx>,
    payload: Result<Json<MergeRequest>, JsonRejection>,
) -> Result<Json<MergeResponse>, SpeakerError> {
    let request = body(payload)?;
    let speaker = speakers::merge(&ctx, &request.source, &request.target).await?;
    Ok(Json(MergeResponse {
        speaker,
        merged: request.source,
    }))
}

#[derive(Deserialize)]
struct ExportQuery {
    #[serde(default)]
    include_embedding: bool,
}

async fn export(
    State(ctx): State<SpeakerCtx>,
    Path(id): Path<String>,
    Query(query): Query<ExportQuery>,
) -> Result<Response, SpeakerError> {
    let export = speakers::export(&ctx, &id, query.include_embedding).await?;
    let mut value = serde_json::to_value(&export)
        .map_err(|e| SpeakerError::Internal(format!("failed to serialize export: {e}")))?;
    if let Some(object) = value.as_object_mut() {
        object.insert("format".into(), json!(EXPORT_FORMAT));
        object.insert(
            "embedding_version".into(),
            json!(crate::diarization::EMBEDDING_VERSION),
        );
        // Long-term memory is part of what is stored about a person.
        if let Some(db) = ctx.db.as_deref() {
            let memory = crate::memory::MemoryStore::new(db).export(&id).await?;
            object.insert("memory".into(), json!(memory));
        }
    }

    // Offer it as a download. IDs are free-form, so the file name is built
    // from the safe characters only.
    let safe_id: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(64)
        .collect();
    let disposition = format!("attachment; filename=\"vox-speaker-{safe_id}.json\"");
    Ok((
        [
            (header::CONTENT_DISPOSITION, disposition),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        Json(value),
    )
        .into_response())
}

async fn get_consent(State(ctx): State<SpeakerCtx>) -> Json<ConsentStatus> {
    Json(speakers::enrollment_consent(&ctx).await)
}

#[derive(Deserialize)]
struct ConsentRequest {
    granted: bool,
}

async fn put_consent(
    State(ctx): State<SpeakerCtx>,
    payload: Result<Json<ConsentRequest>, JsonRejection>,
) -> Result<Json<ConsentStatus>, SpeakerError> {
    let request = body(payload)?;
    Ok(Json(
        speakers::set_enrollment_consent(&ctx, request.granted).await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Method, Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::super::speakers::tests::{enroll, test_ctx};

    fn app(ctx: &SpeakerCtx) -> Router {
        routes::<SpeakerCtx>().with_state(ctx.clone())
    }

    /// Send one request; returns the status, headers and JSON body.
    async fn call(
        ctx: &SpeakerCtx,
        method: Method,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
        let mut request = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(value) => {
                request = request.header(header::CONTENT_TYPE, "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        let response = app(ctx).oneshot(request.body(body).unwrap()).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, headers, json)
    }

    fn keys(value: &serde_json::Value) -> Vec<String> {
        let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    #[tokio::test]
    async fn test_list_shape() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;

        let (status, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            keys(&json),
            [
                "available",
                "enrollment_consent",
                "min_audio_ms",
                "speakers"
            ]
        );
        assert_eq!(json["available"], true);
        assert_eq!(json["min_audio_ms"], 800);
        assert_eq!(
            json["enrollment_consent"],
            json!({ "granted": false, "granted_at": null })
        );

        let speakers = json["speakers"].as_array().unwrap();
        assert_eq!(speakers.len(), 1);
        assert_eq!(
            keys(&speakers[0]),
            [
                "color",
                "created_at",
                "has_voiceprint",
                "id",
                "last_seen",
                "name",
                "sample_count"
            ]
        );
        assert_eq!(speakers[0]["id"], "speaker_1");
        assert_eq!(speakers[0]["name"], "Alice");
        assert_eq!(speakers[0]["sample_count"], 1);
        assert_eq!(speakers[0]["has_voiceprint"], true);
        assert!(speakers[0]["color"].as_str().unwrap().starts_with('#'));
        // The biometric payload is never part of a listing.
        assert!(!json.to_string().contains("embedding"));
    }

    #[tokio::test]
    async fn test_unavailable_shapes() {
        let ctx = SpeakerCtx::default();

        // Reads degrade to an empty, clearly-marked answer...
        let (status, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["available"], false);
        assert_eq!(json["speakers"], json!([]));
        let (status, _, json) = call(&ctx, Method::GET, "/v1/speakers/consent", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({ "granted": false, "granted_at": null }));

        // ...mutations say 503.
        for (method, uri, body) in [
            (Method::DELETE, "/v1/speakers/speaker_1", None),
            (Method::DELETE, "/v1/speakers?confirm=all", None),
            (
                Method::PATCH,
                "/v1/speakers/speaker_1",
                Some(json!({ "name": "A" })),
            ),
            (
                Method::POST,
                "/v1/speakers/merge",
                Some(json!({ "source": "a", "target": "b" })),
            ),
            (
                Method::PUT,
                "/v1/speakers/consent",
                Some(json!({ "granted": true })),
            ),
            (Method::GET, "/v1/speakers/speaker_1/export", None),
        ] {
            let (status, _, json) = call(&ctx, method.clone(), uri, body).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{method} {uri}");
            assert_eq!(keys(&json), ["error", "message"], "{method} {uri}");
            assert_eq!(
                json["error"]["code"], "diarization_unavailable",
                "{method} {uri}"
            );
        }
    }

    #[tokio::test]
    async fn test_rename_shape_and_validation() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Speaker 1", &[1.0, 0.0]).await;

        let (status, _, json) = call(
            &ctx,
            Method::PATCH,
            "/v1/speakers/speaker_1",
            Some(json!({ "name": "  Alice " })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["id"], "speaker_1");
        assert_eq!(json["name"], "Alice");

        for bad in [
            json!({ "name": "" }),
            json!({ "name": "<img src=x onerror=alert(1)>" }),
            json!({ "name": "x".repeat(65) }),
            json!({ "name": "line\nbreak" }),
            json!({ "nom": "Alice" }),
            json!({ "name": 7 }),
        ] {
            let (status, _, json) = call(
                &ctx,
                Method::PATCH,
                "/v1/speakers/speaker_1",
                Some(bad.clone()),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
            assert!(json["error"]["message"].is_string(), "{bad}");
            assert_eq!(json["error"]["code"], "invalid_request", "{bad}");
        }

        let (status, _, json) = call(
            &ctx,
            Method::PATCH,
            "/v1/speakers/ghost",
            Some(json!({ "name": "Alice" })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["message"], "speaker 'ghost' not found");
        assert_eq!(json["error"]["message"], "speaker 'ghost' not found");
        assert_eq!(json["error"]["code"], "speaker_not_found");

        // The rejected names changed nothing.
        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(json["speakers"][0]["name"], "Alice");
    }

    #[tokio::test]
    async fn test_delete_shapes() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        enroll(&ctx, "speaker_2", "Bob", &[0.0, 1.0]).await;
        enroll(&ctx, "speaker_3", "Cem", &[0.6, 0.8]).await;

        let (status, _, json) = call(&ctx, Method::DELETE, "/v1/speakers/speaker_1", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({ "deleted": 1, "id": "speaker_1" }));

        let (status, _, _) = call(&ctx, Method::DELETE, "/v1/speakers/speaker_1", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Deleting everyone needs the explicit confirmation.
        let (status, _, json) = call(&ctx, Method::DELETE, "/v1/speakers", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(json["message"].as_str().unwrap().contains("confirm=all"));
        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(json["speakers"].as_array().unwrap().len(), 2);

        let (status, _, json) = call(&ctx, Method::DELETE, "/v1/speakers?confirm=all", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({ "deleted": 2 }));
        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(json["speakers"], json!([]));
    }

    #[tokio::test]
    async fn test_merge_shape() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[1.0, 0.0]).await;
        enroll(&ctx, "speaker_2", "Speaker 2", &[0.0, 1.0]).await;

        let (status, _, json) = call(
            &ctx,
            Method::POST,
            "/v1/speakers/merge",
            Some(json!({ "source": "speaker_2", "target": "speaker_1" })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(keys(&json), ["merged", "speaker"]);
        assert_eq!(json["merged"], "speaker_2");
        assert_eq!(json["speaker"]["id"], "speaker_1");
        assert_eq!(json["speaker"]["name"], "Alice");
        assert_eq!(json["speaker"]["sample_count"], 2);

        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(json["speakers"].as_array().unwrap().len(), 1);

        let (status, _, _) = call(
            &ctx,
            Method::POST,
            "/v1/speakers/merge",
            Some(json!({ "source": "speaker_1", "target": "speaker_1" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _, _) = call(
            &ctx,
            Method::POST,
            "/v1/speakers/merge",
            Some(json!({ "source": "speaker_2", "target": "speaker_1" })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = call(
            &ctx,
            Method::POST,
            "/v1/speakers/merge",
            Some(json!({ "source": "speaker_1" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_export_shape() {
        let ctx = test_ctx().await;
        enroll(&ctx, "speaker_1", "Alice", &[0.6, 0.8]).await;
        let db = ctx.db.as_ref().unwrap();
        db.set_preference("speaker_1", "voice", "af_heart")
            .await
            .unwrap();
        db.record_utterance("speaker_1", 1200, Some("hello there"))
            .await
            .unwrap();

        let (status, headers, json) =
            call(&ctx, Method::GET, "/v1/speakers/speaker_1/export", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            headers[header::CONTENT_DISPOSITION],
            "attachment; filename=\"vox-speaker-speaker_1.json\""
        );
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        assert_eq!(
            keys(&json),
            [
                "conversations",
                "embedding_version",
                "format",
                "memory",
                "preferences",
                "speaker"
            ]
        );
        assert_eq!(
            json["memory"],
            json!({ "enabled": false, "enabled_at": null, "memories": [] })
        );
        assert_eq!(json["format"], EXPORT_FORMAT);
        assert_eq!(json["speaker"]["name"], "Alice");
        assert_eq!(
            json["preferences"],
            json!([{ "key": "voice", "value": "af_heart" }])
        );
        assert_eq!(json["conversations"][0]["text"], "hello there");
        assert_eq!(json["conversations"][0]["duration_ms"], 1200);

        // The voiceprint itself only on request.
        let (_, _, json) = call(
            &ctx,
            Method::GET,
            "/v1/speakers/speaker_1/export?include_embedding=true",
            None,
        )
        .await;
        let embedding = json["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 2);

        // What is remembered about the speaker is part of the export.
        let memory = crate::memory::MemoryStore::new(db);
        memory.set_enabled("speaker_1", true).await.unwrap();
        memory
            .record_exchange("speaker_1", "I like tea", Some("Noted."), false)
            .await
            .unwrap();
        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers/speaker_1/export", None).await;
        assert_eq!(json["memory"]["enabled"], true);
        assert_eq!(json["memory"]["memories"][0]["role"], "user");
        assert_eq!(json["memory"]["memories"][0]["content"], "I like tea");
        assert_eq!(json["memory"]["memories"][1]["content"], "Noted.");

        let (status, _, _) = call(&ctx, Method::GET, "/v1/speakers/ghost/export", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_consent_shape() {
        let ctx = test_ctx().await;

        let (status, _, json) = call(
            &ctx,
            Method::PUT,
            "/v1/speakers/consent",
            Some(json!({ "granted": true })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(keys(&json), ["granted", "granted_at"]);
        assert_eq!(json["granted"], true);
        assert!(json["granted_at"].as_u64().unwrap() > 0);

        let (_, _, json) = call(&ctx, Method::GET, "/v1/speakers", None).await;
        assert_eq!(json["enrollment_consent"]["granted"], true);

        let (status, _, json) = call(
            &ctx,
            Method::PUT,
            "/v1/speakers/consent",
            Some(json!({ "granted": false })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({ "granted": false, "granted_at": null }));

        let (status, _, _) = call(
            &ctx,
            Method::PUT,
            "/v1/speakers/consent",
            Some(json!({ "granted": "yes" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}
