//! Operator-configured LLM endpoints for the CLI and server.

use std::net::Ipv6Addr;
use std::time::Duration;

use reqwest::{Client, Method, RequestBuilder, Response, Url};
use serde::Deserialize;

use crate::conversation::ChatMessage;
use crate::error::VoxError;
use crate::privacy::{self, Connection};

const DEFAULT_OLLAMA_MODEL: &str = "llama3.2";
const DEFAULT_COMPATIBLE_HOST: &str = "127.0.0.1:8080";
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

/// HTTP protocol used by the local LLM runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackend {
    /// Ollama's native API, with newline-delimited streaming responses.
    Ollama,
    /// OpenAI-style chat and model APIs, with server-sent events for streaming.
    OpenAiCompatible,
}

impl LlmBackend {
    /// Backend name used in operator configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::OpenAiCompatible => "openai-compatible",
        }
    }
}

/// One runtime endpoint chosen by the operator at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmEndpoint {
    /// Protocol used to talk to the runtime.
    pub backend: LlmBackend,
    /// HTTP authority (`host:port`), with brackets around IPv6 addresses.
    pub host: String,
    /// Model to use when the request does not name one.
    pub default_model: Option<String>,
}

impl LlmEndpoint {
    /// Preserve the behavior of callers using Ollama's native API.
    pub fn ollama(host: impl Into<String>) -> Self {
        Self {
            backend: LlmBackend::Ollama,
            host: host.into(),
            default_model: None,
        }
    }

    /// Select `VOX_LLM_BACKEND`, using the existing Ollama address resolution
    /// for Ollama or `VOX_LLM_HOST` for the OpenAI-compatible runtime. Both
    /// backends accept `VOX_LLM_MODEL` as an explicit model default.
    pub fn from_env(ollama_host: &str) -> Result<Self, VoxError> {
        Self::from_values(
            ollama_host,
            std::env::var("VOX_LLM_BACKEND").ok().as_deref(),
            std::env::var("VOX_LLM_HOST").ok().as_deref(),
            std::env::var("VOX_LLM_MODEL").ok().as_deref(),
        )
    }

    fn from_values(
        ollama_host: &str,
        backend: Option<&str>,
        host: Option<&str>,
        default_model: Option<&str>,
    ) -> Result<Self, VoxError> {
        let mut endpoint = match nonempty(backend) {
            None | Some("ollama") => Self::ollama(ollama_host),
            Some("openai-compatible") => Self {
                backend: LlmBackend::OpenAiCompatible,
                host: compatible_authority(nonempty(host).unwrap_or(DEFAULT_COMPATIBLE_HOST))?,
                default_model: None,
            },
            Some(_) => {
                return Err(VoxError::Pipeline(
                    "VOX_LLM_BACKEND must be 'ollama' or 'openai-compatible'".into(),
                ));
            }
        };
        endpoint.default_model = nonempty(default_model).map(str::to_owned);
        Ok(endpoint)
    }

    /// Runtime label for status messages.
    pub fn label(&self) -> &'static str {
        match self.backend {
            LlmBackend::Ollama => "Ollama",
            LlmBackend::OpenAiCompatible => "OpenAI-compatible LLM",
        }
    }

    /// Choose the explicit model, the operator's default, or the native Ollama default.
    pub fn preferred_model(&self, requested: Option<&str>) -> Option<String> {
        nonempty(requested)
            .or_else(|| nonempty(self.default_model.as_deref()))
            .or_else(|| (self.backend == LlmBackend::Ollama).then_some(DEFAULT_OLLAMA_MODEL))
            .map(str::to_owned)
    }

    /// Choose a preferred model or discover one on the configured runtime.
    pub async fn resolve_model(
        &self,
        client: &Client,
        requested: Option<&str>,
    ) -> Result<String, VoxError> {
        if let Some(model) = self.preferred_model(requested) {
            return Ok(model);
        }
        self.list_openai_models(client, DISCOVERY_TIMEOUT)
            .await
            .map_err(|error| VoxError::Pipeline(format!("Set VOX_LLM_MODEL if model discovery is unavailable. {error}")))?
            .into_iter()
            .next()
            .ok_or_else(|| VoxError::Pipeline("The LLM runtime listed no models. Load a model in the runtime and set VOX_LLM_MODEL to its ID.".into()))
    }

    /// List nonempty IDs advertised by an OpenAI-compatible runtime.
    pub async fn list_openai_models(
        &self,
        client: &Client,
        timeout: Duration,
    ) -> Result<Vec<String>, VoxError> {
        let (request, egress) =
            self.request(client, Method::GET, "/v1/models", "llm model list", 0)?;
        let response = request
            .timeout(timeout)
            .send()
            .await
            .map_err(request_error)?;
        let body = checked_response(response, &egress).await?;
        let listing: ModelList = serde_json::from_slice(&body)
            .map_err(|error| VoxError::Pipeline(format!("Invalid LLM model list: {error}")))?;
        Ok(listing
            .data
            .into_iter()
            .filter_map(|model| nonempty(Some(&model.id)).map(str::to_owned))
            .collect())
    }

    /// Request one complete text reply from an OpenAI-compatible runtime.
    pub async fn complete_openai(
        &self,
        client: &Client,
        model: &str,
        messages: &[ChatMessage],
        timeout: Duration,
    ) -> Result<String, VoxError> {
        let body = serde_json::json!({"model": model, "messages": messages, "stream": false});
        let (request, egress) = self.request(
            client,
            Method::POST,
            "/v1/chat/completions",
            "llm chat",
            privacy::json_len(&body),
        )?;
        let response = request
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(request_error)?;
        let body = checked_response(response, &egress).await?;
        let completion: ChatCompletion = serde_json::from_slice(&body)
            .map_err(|error| VoxError::Pipeline(format!("Invalid LLM chat completion: {error}")))?;
        completion
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| VoxError::Pipeline("The LLM response has no text choice".into()))
    }

    pub(crate) fn request(
        &self,
        client: &Client,
        method: Method,
        path: &str,
        purpose: &str,
        bytes_sent: u64,
    ) -> Result<(RequestBuilder, Connection<'static>), VoxError> {
        let compatible_host;
        let host = match self.backend {
            LlmBackend::Ollama => &self.host,
            LlmBackend::OpenAiCompatible => {
                // Public endpoint fields may also come from a Rust application.
                compatible_host = compatible_authority(&self.host)?;
                &compatible_host
            }
        };
        privacy::llm_http_request(client, method, host, path, purpose, bytes_sent)
            .map_err(|error| VoxError::Pipeline(error.to_string()))
    }
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn compatible_authority(input: &str) -> Result<String, VoxError> {
    let input = input.trim();
    let input = input.strip_prefix("http://").unwrap_or(input);
    let authority = input.strip_suffix('/').unwrap_or(input);
    let error = || {
        VoxError::Pipeline("VOX_LLM_HOST must be an HTTP host:port, such as 127.0.0.1:8080".into())
    };
    if authority.is_empty()
        || authority.contains(['/', '\\', '?', '#', '@'])
        || authority
            .chars()
            .any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(error());
    }
    let authority = if authority.parse::<Ipv6Addr>().is_ok() {
        format!("[{authority}]")
    } else {
        authority.to_owned()
    };
    if authority.ends_with(':') {
        return Err(error());
    }
    Url::parse(&format!("http://{authority}")).map_err(|_| error())?;
    Ok(authority)
}

fn request_error(error: reqwest::Error) -> VoxError {
    VoxError::Pipeline(format!("LLM request failed: {error}"))
}

pub(crate) async fn read_response(
    mut response: Response,
    egress: &Connection<'_>,
) -> Result<Vec<u8>, VoxError> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(request_error)? {
        egress.add_received(chunk.len() as u64);
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(VoxError::Pipeline("LLM response body is too large".into()));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn checked_response(
    response: Response,
    egress: &Connection<'_>,
) -> Result<Vec<u8>, VoxError> {
    let status = response.status();
    if !status.is_success() {
        let _ = read_response(response, egress).await;
        return Err(VoxError::Pipeline(format!("LLM returned HTTP {status}")));
    }
    read_response(response, egress).await
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct ChatCompletion {
    choices: Vec<CompletedChoice>,
}

#[derive(Deserialize)]
struct CompletedChoice {
    message: CompletedMessage,
}

#[derive(Deserialize)]
struct CompletedMessage {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn legacy_endpoint_keeps_ollama_defaults() {
        let endpoint = LlmEndpoint::from_values("localhost:11434", None, None, None).unwrap();
        assert_eq!(endpoint, LlmEndpoint::ollama("localhost:11434"));
        assert_eq!(
            endpoint.resolve_model(&Client::new(), None).await.unwrap(),
            "llama3.2"
        );
    }

    #[tokio::test]
    async fn compatible_endpoint_honors_request_then_operator_default() {
        let endpoint = LlmEndpoint::from_values(
            "localhost:11434",
            Some("openai-compatible"),
            None,
            Some(" Chat-Default "),
        )
        .unwrap();
        assert_eq!(endpoint.host, "127.0.0.1:8080");
        let client = Client::new();
        assert_eq!(
            endpoint
                .resolve_model(&client, Some("named"))
                .await
                .unwrap(),
            "named"
        );
        assert_eq!(
            endpoint.resolve_model(&client, Some(" ")).await.unwrap(),
            "Chat-Default"
        );
    }

    #[test]
    fn compatible_host_is_an_http_authority() {
        assert_eq!(
            compatible_authority("http://127.0.0.1:8080/").unwrap(),
            "127.0.0.1:8080"
        );
        assert_eq!(compatible_authority("::1").unwrap(), "[::1]");
        for invalid in [
            "https://localhost:8080",
            "localhost:8080/v1",
            "localhost:8080?x",
            "localhost:",
            "user@localhost:8080",
        ] {
            assert!(compatible_authority(invalid).is_err(), "allowed {invalid}");
        }
    }
}
