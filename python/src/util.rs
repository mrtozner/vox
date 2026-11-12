//! Shared utility functions for model path resolution.

use std::path::PathBuf;

/// Return the default model directory.
///
/// Delegates to [`vox::paths::models_dir`], so the Python bindings, the
/// `vox` CLI and `vox serve` always agree on where models live:
///
/// 1. `VOX_MODELS_DIR` environment variable.
/// 2. `models/` next to vox's `Cargo.toml` when running from a source
///    checkout.
/// 3. The per-user data directory:
///    - macOS: `~/Library/Application Support/vox/models`
///    - Linux: `$XDG_DATA_HOME/vox/models` or `~/.local/share/vox/models`
///    - Windows: `%LOCALAPPDATA%\vox\models`
pub(crate) fn default_model_dir() -> PathBuf {
    vox::paths::models_dir()
}
