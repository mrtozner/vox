//! Handler for `vox speak` — text-to-speech synthesis and playback.

/// Run the speak command, dispatching to the appropriate backend.
pub async fn run(text: &str, voice: &str, backend: &str, yes: bool) -> anyhow::Result<()> {
    match backend {
        "kokoro" => run_kokoro(text, voice, yes).await,
        #[cfg(feature = "piper")]
        "piper" => run_piper(text, voice, yes).await,
        #[cfg(not(feature = "piper"))]
        "piper" => anyhow::bail!(
            "Piper TTS requires the 'piper' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,piper --release\n"
        ),
        other => anyhow::bail!(
            "Unknown TTS backend: '{other}'.\n\n\
             Available backends: kokoro, piper"
        ),
    }
}

/// Run TTS with the Kokoro backend.
#[cfg(feature = "kokoro")]
async fn run_kokoro(text: &str, voice: &str, yes: bool) -> anyhow::Result<()> {
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

/// Stub when kokoro feature is disabled.
#[cfg(not(feature = "kokoro"))]
async fn run_kokoro(_text: &str, _voice: &str, _yes: bool) -> anyhow::Result<()> {
    anyhow::bail!(
        "Kokoro TTS requires the 'kokoro' feature.\n\n\
         Rebuild with:\n\
         \n  cargo build --features cli,kokoro --release\n"
    );
}

/// Run TTS with the Piper backend.
#[cfg(feature = "piper")]
async fn run_piper(text: &str, voice: &str, yes: bool) -> anyhow::Result<()> {
    use super::models::{ensure_piper_voice, piper_voice_alias};
    use vox::traits::TtsBackend;
    use vox::types::TtsRequest;

    // Map short alias to full model name
    let model_name = piper_voice_alias(voice);

    println!("Downloading Piper voice '{model_name}' if needed...");
    let config_path = ensure_piper_voice(&model_name, yes).await?;

    println!("Loading Piper TTS...");
    let tts = vox::PiperBackend::new(&config_path)?;

    println!("Synthesizing with Piper ({model_name})...");
    let output = tts
        .synthesize(&TtsRequest {
            text: text.to_string(),
            voice: None, // single-speaker models use the default
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
