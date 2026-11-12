//! HTTP error type and the JSON error body shared by every endpoint.
//!
//! Every error response has the same shape, which is also the shape OpenAI
//! clients expect:
//!
//! ```json
//! {
//!   "error": { "message": "...", "type": "invalid_request_error", "code": "invalid_wav", "param": null },
//!   "message": "..."
//! }
//! ```
//!
//! `error` used to be a bare string. The top-level `message` string carries
//! the same text for clients that only want something to display.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// An error returned by an HTTP handler.
#[derive(Debug)]
pub struct ServerError {
    status: StatusCode,
    message: String,
    /// Machine-readable code, e.g. `invalid_wav`. Defaults per status.
    code: &'static str,
    /// Request parameter the error refers to, if any.
    param: Option<&'static str>,
    /// Seconds for a `Retry-After` header.
    retry_after: Option<u64>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
    message: &'a str,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    code: &'a str,
    param: Option<&'a str>,
}

/// The `error.type` reported for a status code.
pub fn error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::BAD_GATEWAY => "upstream_error",
        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable_error",
        StatusCode::GATEWAY_TIMEOUT | StatusCode::REQUEST_TIMEOUT => "timeout_error",
        s if s.is_client_error() => "invalid_request_error",
        _ => "server_error",
    }
}

/// The default `error.code` for a status code.
pub fn default_code(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "bad_request",
        StatusCode::UNAUTHORIZED => "unauthorized",
        StatusCode::FORBIDDEN => "forbidden",
        StatusCode::NOT_FOUND => "not_found",
        StatusCode::METHOD_NOT_ALLOWED => "method_not_allowed",
        StatusCode::REQUEST_TIMEOUT => "request_timeout",
        StatusCode::PAYLOAD_TOO_LARGE => "payload_too_large",
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "unsupported_media_type",
        StatusCode::UNPROCESSABLE_ENTITY => "unprocessable_entity",
        StatusCode::TOO_MANY_REQUESTS => "rate_limited",
        StatusCode::BAD_GATEWAY => "upstream_error",
        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable",
        StatusCode::GATEWAY_TIMEOUT => "timeout",
        s if s.is_client_error() => "invalid_request",
        _ => "internal_error",
    }
}

/// Serialize the standard error body.
pub fn error_body_json(
    status: StatusCode,
    message: &str,
    code: &str,
    param: Option<&str>,
) -> String {
    serde_json::to_string(&ErrorBody {
        error: ErrorDetail {
            message,
            kind: error_type(status),
            code,
            param,
        },
        message,
    })
    .unwrap_or_else(|_| {
        r#"{"error":{"message":"internal server error","type":"server_error","code":"internal_error","param":null},"message":"internal server error"}"#
            .to_string()
    })
}

impl ServerError {
    /// An error with the default code for `status`.
    pub fn new(status: StatusCode, msg: impl Into<String>) -> Self {
        Self {
            status,
            message: msg.into(),
            code: default_code(status),
            param: None,
            retry_after: None,
        }
    }

    /// 400 — the request is malformed or has invalid content.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, msg)
    }

    /// 403 — the request is not allowed (e.g. cross-origin).
    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, msg)
    }

    /// 404 — no such resource.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, msg)
    }

    /// 413 — the request body or text is too large.
    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, msg)
    }

    /// 500 — a failure inside vox (inference, encoding).
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, msg)
    }

    /// 502 — an upstream service (Ollama) answered with an error or garbage.
    pub fn bad_gateway(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, msg)
    }

    /// 503 — a backend is not loaded or an upstream service is unreachable.
    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, msg)
    }

    /// 503 with `Retry-After` — the server is at its inference limit.
    pub fn busy(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, msg)
            .with_code("server_busy")
            .with_retry_after(5)
    }

    /// 504 — vox or an upstream service took too long.
    pub fn gateway_timeout(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, msg)
    }

    /// Override the machine-readable `error.code`.
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = code;
        self
    }

    /// Name the request parameter the error refers to (`error.param`).
    pub fn with_param(mut self, param: &'static str) -> Self {
        self.param = Some(param);
        self
    }

    /// Like [`with_param`](Self::with_param), unless a parameter is already named.
    pub fn with_param_if_unset(mut self, param: &'static str) -> Self {
        if self.status.is_client_error() {
            self.param.get_or_insert(param);
        }
        self
    }

    /// Add a `Retry-After` header (seconds).
    pub fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }

    /// The HTTP status of this error.
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// The human-readable message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The machine-readable code.
    pub fn code(&self) -> &str {
        self.code
    }
}

impl From<crate::VoxError> for ServerError {
    fn from(err: crate::VoxError) -> Self {
        use crate::VoxError;
        let message = err.to_string();
        match &err {
            VoxError::NoStt => Self::service_unavailable(message).with_code("stt_unavailable"),
            VoxError::NoVad => Self::service_unavailable(message).with_code("vad_unavailable"),
            VoxError::ModelNotFound(_) | VoxError::IntegrityCheckFailed(_) => {
                Self::service_unavailable(message).with_code("model_unavailable")
            }
            VoxError::Stt(_) => Self::internal(message).with_code("stt_failed"),
            VoxError::Tts(_) => Self::internal(message).with_code("tts_failed"),
            _ => Self::internal(message),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let body = error_body_json(self.status, &self.message, self.code, self.param);
        let mut response = (
            self.status,
            [(header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response();
        if let Some(secs) = self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}
