//! Text-to-speech example using Pocket TTS backend.
//!
//! Usage:
//!   HF_TOKEN=hf_xxx cargo run --example pocket_speak --features pocket
//!
//! The model (~236MB) auto-downloads from HuggingFace on first run.

use vox::{AudioPlayer, PocketTtsBackend, TtsBackend, TtsRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("Loading Pocket TTS (first run downloads ~236MB model)...");
    let voices_dir = if std::path::Path::new("models/pocket-voices").exists() {
        "models/pocket-voices"
    } else {
        "pocket-voices"
    };
    let tts = PocketTtsBackend::with_voice("alba", voices_dir)?;

    let text = std::env::args().nth(1).unwrap_or_else(|| {
        "Hello! I am Vox running Pocket TTS. Everything runs locally on your device.".into()
    });

    println!("Synthesizing: \"{text}\"");
    let output = tts
        .synthesize(&TtsRequest {
            text: text.clone(),
            voice: Some("alba".into()),
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
    let mut writer = hound::WavWriter::create("output_pocket.wav", spec)?;
    for sample in &output.audio.samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!("Also saved to output_pocket.wav");

    Ok(())
}
