//! Handler for `vox speak` — text-to-speech synthesis and playback.

#[cfg(feature = "kokoro")]
use std::path::PathBuf;

#[cfg(feature = "kokoro")]
use super::models::models_dir;

/// Resolve a model file from `~/.vox/models/` or the current directory.
#[cfg(feature = "kokoro")]
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

/// Run the speak command.
#[cfg(feature = "kokoro")]
pub async fn run(text: &str, voice: &str) -> anyhow::Result<()> {
    use vox::traits::TtsBackend;
    use vox::types::TtsRequest;

    let model_file = "kokoro-v1.0.onnx";
    let voices_file = "voices.bin";

    // Resolve Kokoro model
    let model_path = resolve_model(model_file).ok_or_else(|| {
        anyhow::anyhow!(
            "Kokoro model not found: {model_file}\n\n\
             Download it with:\n\
             \n  vox models download kokoro\n"
        )
    })?;

    // Resolve voices file
    let voices_path = resolve_model(voices_file).ok_or_else(|| {
        anyhow::anyhow!(
            "Kokoro voices not found: {voices_file}\n\n\
             Download it with:\n\
             \n  vox models download kokoro-voices\n"
        )
    })?;

    println!("Loading Kokoro TTS...");
    let tts = vox::KokoroBackend::new(&model_path, &voices_path).await?;

    println!("Synthesizing with voice '{voice}'...");
    let output = tts
        .synthesize(&TtsRequest {
            text: text.to_string(),
            voice: Some(voice.to_string()),
        })
        .await?;

    let duration_secs = output.duration_ms as f64 / 1000.0;
    println!(
        "Generated {:.1}s of audio ({} samples at {} Hz)",
        duration_secs,
        output.audio.samples.len(),
        output.audio.sample_rate,
    );

    println!("Playing audio...");
    let player = vox::AudioPlayer::new()?;
    player.play_blocking(&output.audio)?;
    println!("Done.");

    Ok(())
}

/// Run the speak command (stub when kokoro feature is disabled).
#[cfg(not(feature = "kokoro"))]
pub async fn run(_text: &str, _voice: &str) -> anyhow::Result<()> {
    anyhow::bail!(
        "TTS requires the 'kokoro' feature.\n\n\
         Rebuild with:\n\
         \n  cargo build --features cli,kokoro --release\n"
    );
}
