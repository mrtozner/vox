//! `GET /v1/privacy/ledger` — what this server process has connected to.

use std::sync::Arc;

use axum::extract::State;
use axum::{Extension, Json};
use serde::Serialize;

use crate::llm::LlmEndpoint;

use super::ServerState;

/// Body of `GET /v1/privacy/ledger`.
///
/// `ledger` covers every outbound connection made by this process since it
/// started: requests to the configured LLM runtime and model downloads. Byte counts are HTTP
/// body bytes. See [`crate::privacy`] for what is and is not covered.
#[derive(Serialize)]
pub struct PrivacyLedgerResponse {
    /// The configured Ollama address, retained for older clients.
    pub ollama_host: String,
    /// Whether that address is this machine.
    pub ollama_host_loopback: bool,
    /// The selected LLM runtime protocol and address.
    pub llm_provider: &'static str,
    pub llm_host: String,
    pub llm_host_loopback: bool,
    /// `true` when no recorded connection left this machine.
    pub fully_local: bool,
    /// `offline`, `totals`, `destinations`, `recent`, `dropped_entries`.
    #[serde(flatten)]
    pub ledger: crate::privacy::LedgerSnapshot,
}

/// GET /v1/privacy/ledger
pub async fn ledger_with_llm(
    State(state): State<Arc<ServerState>>,
    Extension(endpoint): Extension<Arc<LlmEndpoint>>,
) -> Json<PrivacyLedgerResponse> {
    let ledger = crate::privacy::ledger().snapshot();
    let ollama_host = if matches!(endpoint.backend, crate::llm::LlmBackend::Ollama) {
        &endpoint.host
    } else {
        &state.ollama_host
    };
    Json(PrivacyLedgerResponse {
        ollama_host: ollama_host.clone(),
        ollama_host_loopback: crate::privacy::is_loopback_host(ollama_host),
        llm_provider: endpoint.backend.as_str(),
        llm_host: endpoint.host.clone(),
        llm_host_loopback: crate::privacy::is_loopback_host(&endpoint.host),
        fully_local: ledger.totals.non_loopback_connections == 0,
        ledger,
    })
}
