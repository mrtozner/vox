//! Handler for `vox serve` — HTTP API server.

#[cfg(feature = "server")]
pub async fn run(host: &str, port: u16, cache_models: bool) -> anyhow::Result<()> {
    crate::server::run(host, port, cache_models).await
}
