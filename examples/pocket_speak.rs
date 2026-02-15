//! Text-to-speech example using Pocket TTS backend.
//!
//! Usage:
//!   HF_TOKEN=hf_xxx cargo run --example pocket_speak --features pocket [voice] [text]
//!
//! Examples:
//!   cargo run --example pocket_speak --features pocket
//!   cargo run --example pocket_speak --features pocket alba "Hello world"
//!   cargo run --example pocket_speak --features pocket marius "Testing voice"
//!
//! Voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
//! The model (~236MB) auto-downloads from HuggingFace on first run.

use vox::{AudioPlayer, PocketTtsBackend, TtsBackend, TtsRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let voice = std::env::args().nth(1).unwrap_or_else(|| "alba".into());
    let text = std::env::args().nth(2).unwrap_or_else(|| {
        "Hello! I am Vox running Pocket TTS. Everything runs locally on your device.".into()
    });

    println!("Loading Pocket TTS (first run downloads ~236MB model)...");
    let voices_dir = if std::path::Path::new("models/pocket-voices").exists() {
        "models/pocket-voices"
    } else {
        "pocket-voices"
    };
    let tts = PocketTtsBackend::with_voice(&voice, voices_dir)?;

    println!("Voice: {voice}");
    println!("Synthesizing: \"{text}\"");
    let start = std::time::Instant::now();
    let output = tts
        .synthesize(&TtsRequest {
            text: text.clone(),
            voice: Some(voice.clone()),
        })
        .await?;
    let synth_ms = start.elapsed().as_millis();

    let rtf = synth_ms as f64 / output.duration_ms as f64;
    println!(
        "Synthesized {} samples ({} ms) at {} Hz — took {}ms (RTF: {:.2})",
        output.audio.samples.len(),
        output.duration_ms,
        output.audio.sample_rate,
        synth_ms,
        rtf,
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
