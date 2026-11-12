//! Cross-cutting HTTP middleware: request counters and uniform error bodies.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;

use super::ServerState;
use super::error::{default_code, error_body_json};

/// Paths that are not counted in the request statistics. `/v1/stats` would
/// otherwise inflate the number it reports every time a dashboard polls it.
const UNCOUNTED_PATHS: &[&str] = &["/v1/stats"];

/// Largest non-JSON error body we rewrite; anything bigger is replaced by the
/// status text.
const MAX_ERROR_BODY: usize = 16 * 1024;

/// Middleware: count the request and make sure an error response carries the
/// standard JSON error body.
///
/// Handlers return [`ServerError`](super::error::ServerError), which already
/// has that shape. Everything else that can fail — axum's extractors (bad
/// JSON, oversized body), the router (404, 405), the timeout layer — answers
/// in plain text or with an empty body; those are rewritten here so clients
/// only ever have to parse one error format.
pub async fn track(
    State(state): State<Arc<ServerState>>,
    request: Request,
    next: Next,
) -> Response {
    let counted = !UNCOUNTED_PATHS.contains(&request.uri().path());

    let response = next.run(request).await;

    let status = response.status();
    if counted {
        let stats = &state.stats;
        stats.requests.fetch_add(1, Ordering::Relaxed);
        if status.is_client_error() {
            stats.client_errors.fetch_add(1, Ordering::Relaxed);
        } else if status.is_server_error() {
            stats.server_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    normalize_error(response).await
}

fn is_json(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| {
            let ct = ct.to_ascii_lowercase();
            ct.starts_with("application/json") || ct.contains("+json")
        })
}

/// Rewrite a non-JSON error response into the standard error body, keeping
/// its status and headers.
pub async fn normalize_error(response: Response) -> Response {
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) || is_json(&response) {
        return response;
    }

    let (mut parts, body) = response.into_parts();
    let text = match axum::body::to_bytes(body, MAX_ERROR_BODY).await {
        Ok(bytes) => String::from_utf8(bytes.to_vec()).unwrap_or_default(),
        Err(_) => String::new(),
    };
    let text = text.trim();
    let message = if !text.is_empty() {
        text.to_string()
    } else if matches!(
        status,
        StatusCode::GATEWAY_TIMEOUT | StatusCode::REQUEST_TIMEOUT
    ) {
        "the request took too long and was cancelled".to_string()
    } else {
        status
            .canonical_reason()
            .unwrap_or("request failed")
            .to_string()
    };

    let json = error_body_json(status, &message, default_code(status), None);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from(json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    async fn json_of(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn plain_text_errors_become_json() {
        let response = (StatusCode::UNPROCESSABLE_ENTITY, "missing field `text`").into_response();
        let response = normalize_error(response).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let json = json_of(response).await;
        assert_eq!(json["error"]["message"], "missing field `text`");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "unprocessable_entity");
        assert_eq!(json["message"], "missing field `text`");
    }

    #[tokio::test]
    async fn empty_errors_get_the_status_text() {
        let json = json_of(normalize_error(StatusCode::NOT_FOUND.into_response()).await).await;
        assert_eq!(json["error"]["message"], "Not Found");
        assert_eq!(json["error"]["type"], "not_found_error");
        assert_eq!(json["error"]["code"], "not_found");

        let json =
            json_of(normalize_error(StatusCode::GATEWAY_TIMEOUT.into_response()).await).await;
        assert_eq!(json["error"]["type"], "timeout_error");
        assert_eq!(json["error"]["code"], "timeout");
    }

    #[tokio::test]
    async fn json_errors_and_successes_pass_through() {
        let original = r#"{"status":"not_ready"}"#;
        let response = (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "application/json")],
            original,
        )
            .into_response();
        let bytes = axum::body::to_bytes(normalize_error(response).await.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], original.as_bytes());

        let ok = normalize_error((StatusCode::OK, "plain").into_response()).await;
        let bytes = axum::body::to_bytes(ok.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"plain");
    }
}
