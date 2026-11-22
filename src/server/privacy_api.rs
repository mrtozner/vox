//! `GET /v1/privacy/ledger` — what this server process has connected to.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use super::ServerState;

/// Body of `GET /v1/privacy/ledger`.
///
/// `ledger` covers every outbound connection made by this process since it
/// started: requests to Ollama and model downloads. Byte counts are HTTP
/// body bytes. See [`crate::privacy`] for what is and is not covered.
#[derive(Serialize)]
pub struct PrivacyLedgerResponse {
    /// The Ollama address this server sends LLM requests to.
    pub ollama_host: String,
    /// Whether that address is this machine.
    pub ollama_host_loopback: bool,
    /// `true` when no recorded connection left this machine.
    pub fully_local: bool,
    /// `offline`, `totals`, `destinations`, `recent`, `dropped_entries`.
    #[serde(flatten)]
    pub ledger: crate::privacy::LedgerSnapshot,
}

/// GET /v1/privacy/ledger
pub async fn ledger(State(state): State<Arc<ServerState>>) -> Json<PrivacyLedgerResponse> {
    let ledger = crate::privacy::ledger().snapshot();
    Json(PrivacyLedgerResponse {
        ollama_host: state.ollama_host.clone(),
        ollama_host_loopback: crate::privacy::is_loopback_host(&state.ollama_host),
        fully_local: ledger.totals.non_loopback_connections == 0,
        ledger,
    })
}
