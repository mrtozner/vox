//! Example: Piper TTS speech synthesis.
//!
//! ```bash
//! cargo run --example piper_speak --features piper
//! ```

#[cfg(feature = "piper")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use vox::traits::TtsBackend;
    use vox::types::TtsRequest;

    // Point to your downloaded Piper model config (.onnx.json)
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "en_US-lessac-medium.onnx.json".to_string());

    println!("Loading Piper TTS from {config_path}...");
    let tts = vox::PiperBackend::new(&config_path)?;

    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Hello from Piper TTS!".to_string());

    println!("Synthesizing: \"{text}\"");
    let output = tts.synthesize(&TtsRequest { text, voice: None }).await?;

    println!(
        "Generated {:.1}s of audio ({} samples at {} Hz)",
        output.duration_ms as f64 / 1000.0,
        output.audio.samples.len(),
        output.audio.sample_rate,
    );

    let player = vox::AudioPlayer::new()?;
    player.play_blocking(&output.audio)?;
    println!("Done.");

    Ok(())
}

#[cfg(not(feature = "piper"))]
fn main() {
    eprintln!("This example requires the 'piper' feature.");
    eprintln!("Run with: cargo run --example piper_speak --features piper");
    std::process::exit(1);
}
