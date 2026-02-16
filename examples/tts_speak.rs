//! Text-to-speech example using Kokoro backend.
//!
//! Usage:
//!   cargo run --example tts_speak --features kokoro
//!
//! Requires models in current directory or `models/`:
//!   - kokoro-v1.0.onnx
//!   - voices.bin
//!
//! Download: bash scripts/download_models.sh

use vox::{AudioPlayer, KokoroBackend, TtsBackend, TtsRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Try models/ directory first, then current directory
    let (model, voices) = if std::path::Path::new("models/kokoro-v1.0.onnx").exists() {
        ("models/kokoro-v1.0.onnx", "models/voices.bin")
    } else {
        ("kokoro-v1.0.onnx", "voices.bin")
    };

    println!("Loading Kokoro TTS...");
    let tts = KokoroBackend::new(model, voices).await?;

    let text = std::env::args().nth(1).unwrap_or_else(|| {
        "Hello! I am Vox, your local voice assistant. Everything runs on your device.".into()
    });

    println!("Synthesizing: \"{text}\"");
    let output = tts
        .synthesize(&TtsRequest {
            text: text.clone(),
            voice: Some("af_heart".into()),
            seed: None,
        })
        .await?;

    println!(
        "Synthesized {} samples ({} ms) at {} Hz",
        output.audio.samples.len(),
        output.duration_ms,
        output.audio.sample_rate,
    );

    // Play through speakers
    let player = AudioPlayer::new()?;
    println!("Playing audio...");
    player.play_blocking(&output.audio)?;
    println!("Done.");

    // Also save to WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: output.audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for sample in &output.audio.samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!("Also saved to output.wav");

    Ok(())
}
