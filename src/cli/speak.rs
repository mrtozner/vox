//! Handler for `vox speak` — text-to-speech synthesis and playback.

/// Run the speak command.
#[cfg(feature = "kokoro")]
pub async fn run(text: &str, voice: &str, yes: bool) -> anyhow::Result<()> {
    use super::models::ensure_model;
    use vox::traits::TtsBackend;
    use vox::types::TtsRequest;

    let model_path = ensure_model("kokoro", "kokoro-v1.0.onnx", yes).await?;
    let voices_path = ensure_model("kokoro-voices", "voices.bin", yes).await?;

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
pub async fn run(_text: &str, _voice: &str, _yes: bool) -> anyhow::Result<()> {
    anyhow::bail!(
        "TTS requires the 'kokoro' feature.\n\n\
         Rebuild with:\n\
         \n  cargo build --features cli,kokoro --release\n"
    );
}
