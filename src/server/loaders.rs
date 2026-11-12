//! Model directory resolution and TTS backend loading for the server.

use std::sync::Arc;

#[allow(unused_imports)]
use tracing::info;

/// Resolve the models directory for the server.
///
/// Mirrors `crate::cli::models::models_dir` so this module can be pulled
/// into test crates via `#[path = ...]` without dragging in the CLI. The
/// resolution order is: `VOX_MODELS_DIR` env → `$CARGO_MANIFEST_DIR/models`
/// → walk upward from the running binary for a sibling `models/` dir.
pub(super) fn resolve_models_dir() -> std::path::PathBuf {
    if let Ok(explicit) = std::env::var("VOX_MODELS_DIR") {
        let p = std::path::PathBuf::from(explicit);
        if !p.exists() {
            let _ = std::fs::create_dir_all(&p);
        }
        return p;
    }

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_models = manifest_dir.join("models");
    if repo_models.exists() {
        return repo_models;
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut cursor = exe.parent().map(|p| p.to_path_buf());
        while let Some(dir) = cursor {
            let candidate = dir.join("models");
            if candidate.exists() {
                return candidate;
            }
            cursor = dir.parent().map(|p| p.to_path_buf());
        }
    }

    let _ = std::fs::create_dir_all(&repo_models);
    repo_models
}

#[cfg(feature = "qwen3")]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let _ = models_dir;

    // Try Qwen3 first (if enabled)
    match vox::tts::Qwen3Backend::new().await {
        Ok(backend) => {
            info!("loaded qwen3 TTS backend");
            return (
                Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                Some("qwen3".to_string()),
                None, // Model size not applicable (auto-downloaded to HF cache)
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load qwen3 TTS backend, trying fallback");
        }
    }

    // Fallback: try Kokoro if Qwen3 is unavailable
    #[cfg(feature = "kokoro")]
    {
        let model_path = models_dir.join("kokoro-v1.0.onnx");
        let voices_path = models_dir.join("voices.bin");
        if model_path.exists() && voices_path.exists() {
            match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
                Ok(backend) => {
                    info!("loaded kokoro TTS backend");
                    let model_name = model_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.replace('-', " "))
                        .unwrap_or_else(|| "kokoro".to_string());
                    let model_size = std::fs::metadata(&model_path)
                        .ok()
                        .map(|m| m.len() / (1024 * 1024));
                    return (
                        Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                        Some(model_name),
                        model_size,
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load kokoro TTS backend");
                }
            }
        }
    }

    // Fallback: try Piper if Kokoro is unavailable
    #[cfg(feature = "piper")]
    {
        if let Some(result) = try_load_piper(models_dir) {
            return result;
        }
    }

    (None, None, None)
}

#[cfg(all(feature = "kokoro", not(feature = "qwen3")))]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let model_path = models_dir.join("kokoro-v1.0.onnx");
    let voices_path = models_dir.join("voices.bin");
    if model_path.exists() && voices_path.exists() {
        match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
            Ok(backend) => {
                info!("loaded kokoro TTS backend");
                let model_name = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.replace('-', " "))
                    .unwrap_or_else(|| "kokoro".to_string());
                let model_size = std::fs::metadata(&model_path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                return (
                    Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                    Some(model_name),
                    model_size,
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load kokoro TTS backend");
            }
        }
    } else {
        tracing::warn!(
            "kokoro model files not found in {}, trying piper fallback",
            models_dir.display()
        );
    }

    // Fallback: try Piper if Kokoro is unavailable
    #[cfg(feature = "piper")]
    {
        if let Some(result) = try_load_piper(models_dir) {
            return result;
        }
    }

    (None, None, None)
}

#[cfg(all(feature = "piper", not(any(feature = "kokoro", feature = "qwen3"))))]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    if let Some(result) = try_load_piper(models_dir) {
        return result;
    }
    tracing::warn!(
        "no piper model found in {}, TTS disabled",
        models_dir.display()
    );
    (None, None, None)
}

#[cfg(not(any(feature = "kokoro", feature = "piper", feature = "qwen3")))]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro', 'piper', or 'qwen3' feature); models dir: {}",
        models_dir.display()
    );
    (None, None, None)
}

/// Scan models_dir/piper/ for any *.onnx.json config files and try to load the first valid one.
#[cfg(feature = "piper")]
pub(super) type PiperLoadResult = Option<(
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
)>;

#[cfg(feature = "piper")]
pub(super) fn try_load_piper(models_dir: &std::path::Path) -> PiperLoadResult {
    let piper_dir = models_dir.join("piper");
    if !piper_dir.exists() {
        return None;
    }

    let entries = std::fs::read_dir(&piper_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json")
            && path.to_str().is_some_and(|s| s.ends_with(".onnx.json"))
        {
            match vox::tts::PiperBackend::new(&path) {
                Ok(backend) => {
                    info!(model = %path.display(), "loaded piper TTS backend");
                    let model_name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| {
                            // "en_US-lessac-medium.onnx" -> "piper en_US-lessac-medium"
                            format!("piper {}", s.strip_suffix(".onnx").unwrap_or(s))
                        })
                        .unwrap_or_else(|| "piper".to_string());
                    // Size of the .onnx model file (not the json config)
                    let onnx_path = path.with_extension(""); // strips .json -> .onnx
                    let model_size = std::fs::metadata(&onnx_path)
                        .ok()
                        .map(|m| m.len() / (1024 * 1024));
                    return Some((
                        Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                        Some(model_name),
                        model_size,
                    ));
                }
                Err(e) => {
                    tracing::warn!(model = %path.display(), error = %e, "failed to load piper model");
                }
            }
        }
    }
    None
}
