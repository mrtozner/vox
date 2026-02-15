//! Shared utility functions for model path resolution.

use std::path::PathBuf;

/// Return the default model directory: `~/.vox/models/`.
pub(crate) fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".vox")
        .join("models")
}
