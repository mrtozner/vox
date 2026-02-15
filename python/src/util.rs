//! Shared utility functions for model path resolution.

use std::path::PathBuf;

/// Return the default model directory, matching the Rust CLI's `models_dir()`.
///
/// Uses `dirs::data_dir()` on each platform:
/// - macOS: `~/Library/Application Support/vox/models`
/// - Linux: `~/.local/share/vox/models`
/// - Windows: `{FOLDERPATH}/vox/models`
///
/// Falls back to `~/.vox/models` if the platform data dir is unavailable.
pub(crate) fn default_model_dir() -> PathBuf {
    dirs::data_dir()
        .map(|d| d.join("vox").join("models"))
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".vox")
                .join("models")
        })
}
