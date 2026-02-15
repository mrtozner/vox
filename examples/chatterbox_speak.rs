//! Text-to-speech with voice cloning using Chatterbox Turbo.
//!
//! Usage:
//!   cargo run --example chatterbox_speak --features chatterbox -- <reference.wav> [text]
//!   cargo run --example chatterbox_speak --features chatterbox -- --model-dir models/chatterbox <reference.wav> [text]
//!
//! The reference WAV should be 5-20 seconds of clean speech from the target
//! voice. Use a real human recording — TTS-generated audio won't clone well.
//!
//! Default model is fp16 (~1.66GB), auto-downloads from HuggingFace on first run.
//! Voices: English only.

use vox::{AudioPlayer, ChatterboxBackend, TtsBackend, TtsRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let (model_dir, reference_wav, text) = parse_args(&args);

    println!("Loading Chatterbox Turbo (first run downloads ~720MB model)...");
    let tts = if let Some(dir) = &model_dir {
        println!("Using local models: {dir}");
        ChatterboxBackend::from_model_dir(dir, &reference_wav)?
    } else {
        ChatterboxBackend::new(&reference_wav)?
    };

    println!("Reference: {reference_wav}");
    println!("Synthesizing: \"{text}\"");
    let start = std::time::Instant::now();
    let output = tts
        .synthesize(&TtsRequest {
            text: text.clone(),
            voice: None,
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

    let player = AudioPlayer::new()?;
    println!("Playing audio...");
    player.play_blocking(&output.audio)?;
    println!("Done.");

    // Save to WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: output.audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output_chatterbox.wav", spec)?;
    for sample in &output.audio.samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!("Also saved to output_chatterbox.wav");

    Ok(())
}

fn parse_args(args: &[String]) -> (Option<String>, String, String) {
    let mut model_dir = None;
    let mut positional = Vec::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model-dir" && i + 1 < args.len() {
            model_dir = Some(args[i + 1].clone());
            i += 2;
        } else {
            positional.push(args[i].clone());
            i += 1;
        }
    }

    if positional.is_empty() {
        eprintln!("Usage: chatterbox_speak [--model-dir DIR] <reference.wav> [text]");
        eprintln!("  reference.wav: 5-20s WAV of target voice for cloning");
        std::process::exit(1);
    }

    let reference_wav = positional[0].clone();
    let text = if positional.len() > 1 {
        positional[1].clone()
    } else {
        "Hello! I am Vox running Chatterbox Turbo. My voice was cloned from your reference audio."
            .into()
    };

    (model_dir, reference_wav, text)
}
