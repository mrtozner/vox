//! Ollama model probe for the capability registry.
//!
//! Reaches out to the Ollama HTTP API (`/api/tags`) and summarizes the
//! installed models. Embedding-only families are filtered out so the
//! LLM prompt only lists models that can act as chat backends. This
//! function is a degraded-read: any failure (network, parse, timeout)
//! returns `Ok(empty vec)` so registry build never aborts.

use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct OllamaModelSummary {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
}

/// Probe `/api/tags` and return a filtered summary of chat models.
/// Embedding-only families are dropped. Never returns an error that
/// would break registry build — returns Ok(empty) on any failure.
pub async fn probe(
    http: &reqwest::Client,
    ollama_host: &str,
) -> Result<Vec<OllamaModelSummary>, reqwest::Error> {
    // Offline mode refuses a non-loopback host here; like every other
    // failure of this probe that reads as "no models".
    let Ok((request, egress)) = crate::privacy::ollama_http_request(
        http,
        reqwest::Method::GET,
        ollama_host,
        "/api/tags",
        "llm model list",
        0,
    ) else {
        return Ok(Vec::new());
    };
    let resp = match tokio::time::timeout(Duration::from_millis(1200), request.send()).await {
        Ok(Ok(r)) => r,
        _ => return Ok(Vec::new()),
    };

    if !resp.status().is_success() {
        return Ok(Vec::new());
    }

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(_) => return Ok(Vec::new()),
    };
    egress.add_received(bytes.len() as u64);
    let body: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return Ok(Vec::new()),
    };

    let embedding_families = [
        "bert",
        "nomic-bert",
        "nomic-embed-text",
        "mxbai-embed-large",
        "jina-bert",
        "snowflake-arctic-embed",
        "stella",
    ];

    let mut out = Vec::new();
    if let Some(arr) = body["models"].as_array() {
        for m in arr {
            let name = match m["name"].as_str() {
                Some(s) => s.to_string(),
                None => continue,
            };
            let family = m["details"]["family"].as_str().map(|s| s.to_lowercase());
            if let Some(f) = &family
                && embedding_families.iter().any(|e| f == e)
            {
                continue;
            }
            let size_mb = m["size"].as_u64().map(|b| b / (1024 * 1024));
            out.push(OllamaModelSummary {
                name,
                size_mb,
                family,
            });
        }
    }

    Ok(out)
}
