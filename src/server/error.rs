//! HTTP error mapping for VoxError.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Wrapper around [`vox::VoxError`] that implements [`IntoResponse`].
pub struct ServerError {
    status: StatusCode,
    message: String,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

impl ServerError {
    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
}

impl From<vox::VoxError> for ServerError {
    fn from(err: vox::VoxError) -> Self {
        use vox::VoxError;
        match &err {
            VoxError::Stt(_) | VoxError::Tts(_) | VoxError::Pipeline(_) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: err.to_string(),
            },
            VoxError::NoStt | VoxError::NoVad => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: err.to_string(),
            },
            _ => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: err.to_string(),
            },
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let body = serde_json::to_string(&ErrorBody {
            error: self.message,
        })
        .unwrap_or_else(|_| r#"{"error":"internal server error"}"#.to_string());

        (
            self.status,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response()
    }
}
