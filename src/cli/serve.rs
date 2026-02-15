//! Handler for `vox serve` — HTTP API server.

#[cfg(feature = "server")]
pub async fn run(host: &str, port: u16) -> anyhow::Result<()> {
    crate::server::run(host, port).await
}
