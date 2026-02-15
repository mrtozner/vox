//! Handler for `vox listen` — real-time microphone transcription.

use std::path::PathBuf;

use super::models::models_dir;
use vox::{SileroVad, Vox, WhisperBackend};

/// Map a user-facing model name to the GGML filename.
fn model_filename(model: &str) -> String {
    match model {
        "tiny" => "ggml-tiny.bin".into(),
        "tiny.en" => "ggml-tiny.en.bin".into(),
        "base" => "ggml-base.bin".into(),
        "base.en" => "ggml-base.en.bin".into(),
        "small" => "ggml-small.bin".into(),
        "small.en" => "ggml-small.en.bin".into(),
        "medium" => "ggml-medium.bin".into(),
        "medium.en" => "ggml-medium.en.bin".into(),
        other => {
            // Allow passing a raw filename or full path
            if other.ends_with(".bin") {
                other.to_string()
            } else {
                format!("ggml-{other}.bin")
            }
        }
    }
}

/// Resolve a model file path: check `~/.vox/models/` first, then the
/// current directory as a fallback.
fn resolve_model(filename: &str) -> Option<PathBuf> {
    let models = models_dir();
    let candidate = models.join(filename);
    if candidate.exists() {
        return Some(candidate);
    }
    let local = PathBuf::from(filename);
    if local.exists() {
        return Some(local);
    }
    None
}

/// Run the listen command.
pub async fn run(model: &str) -> anyhow::Result<()> {
    let whisper_file = model_filename(model);
    let vad_file = "silero_vad.onnx";

    // Resolve VAD model
    let vad_path = resolve_model(vad_file).ok_or_else(|| {
        anyhow::anyhow!(
            "VAD model not found: {vad_file}\n\n\
             Download it with:\n\
             \n  vox models download silero-vad\n"
        )
    })?;

    // Resolve Whisper model
    let whisper_path = resolve_model(&whisper_file).ok_or_else(|| {
        let model_name = match model {
            "tiny.en" => "whisper-tiny.en",
            "base.en" => "whisper-base.en",
            "small.en" => "whisper-small.en",
            other => other,
        };
        anyhow::anyhow!(
            "Whisper model not found: {whisper_file}\n\n\
             Download it with:\n\
             \n  vox models download {model_name}\n"
        )
    })?;

    // Initialize backends
    println!("Loading VAD model...");
    let vad = SileroVad::new(&vad_path)?;

    println!("Loading Whisper model ({model})...");
    let stt = WhisperBackend::from_model(&whisper_path)?;

    // Build pipeline
    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;

    Ok(())
}
