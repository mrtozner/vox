//! Test speaker identification with REAL microphone audio.
//!
//! This example:
//! 1. Asks you to speak (5 seconds) to enroll YOUR voice
//! 2. Asks your friend to speak (5 seconds) to enroll THEIR voice
//! 3. Then you both talk and it identifies who's speaking!
//!
//! Run with:
//! ```bash
//! cargo run --example test_real_voices --features diarization,cli
//! ```

use std::sync::{Arc, Mutex};
use vox::AudioChunk;
use vox::diarization::{RecognitionConfig, SpeakerEmbedding, SpeakerRegistry};

/// Display a speaker's message with colored UI.
fn display_speaker_message(speaker_name: &str, text: &str, confidence: f32) {
    // Confidence bar
    let filled = (confidence * 10.0).min(10.0) as usize;
    let empty = 10 - filled;
    let confidence_bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    // Color based on confidence
    let (color, reset) = if confidence > 0.85 {
        ("\x1b[36m", "\x1b[0m") // Cyan for high confidence
    } else if confidence > 0.70 {
        ("\x1b[33m", "\x1b[0m") // Yellow for medium
    } else {
        ("\x1b[90m", "\x1b[0m") // Gray for low
    };

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!(
        "│ 🎤 Speaker: {}{:20}{}                       │",
        color, speaker_name, reset
    );
    println!(
        "│ 📊 Confidence: {} {:.0}%                         │",
        confidence_bar,
        confidence * 100.0
    );
    println!("└─────────────────────────────────────────────────────────┘");
    println!("💬 {}{}{}: \"{}\"", color, speaker_name, reset, text);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   🎤 REAL VOICE SPEAKER IDENTIFICATION TEST 🎤          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("This will:");
    println!("  1. Record YOUR voice for 5 seconds (enrollment)");
    println!("  2. Record YOUR FRIEND's voice for 5 seconds (enrollment)");
    println!("  3. Then identify who's speaking in real-time!\n");

    // Check if speaker encoder model exists
    let model_path = "models/speaker_encoder.onnx";
    if !std::path::Path::new(model_path).exists() {
        println!("❌ Speaker encoder model not found at: {}", model_path);
        println!("\nThe model should have been downloaded. Let me check...");

        // List models directory
        if let Ok(entries) = std::fs::read_dir("models") {
            println!("\nModels directory contents:");
            for entry in entries.flatten() {
                println!("  - {}", entry.file_name().to_string_lossy());
            }
        }

        anyhow::bail!("Please download the speaker encoder model first");
    }

    println!("✅ Found speaker encoder model: {}\n", model_path);

    // Initialize speaker embedding extractor
    println!("📦 Loading speaker encoder model...");
    let mut embedding = SpeakerEmbedding::new(model_path)?;
    println!("✅ Model loaded!\n");

    // Initialize speaker registry
    let recognition_config = RecognitionConfig {
        threshold: 0.6, // Lower threshold for real voices
        require_threshold: false,
    };
    let mut registry = SpeakerRegistry::with_config(recognition_config);

    // Initialize audio capture
    use cpal::traits::{DeviceTrait, HostTrait};
    println!("🎙️  Initializing microphone...");
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    println!("✅ Using device: {}\n", device.name()?);

    // STEP 1: Enroll first speaker (you)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 ENROLLMENT: First Speaker");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nPress Enter, then speak for 5 seconds...");
    println!("(Say something like: 'Hello, my name is [your name]')");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    println!("🔴 RECORDING... (5 seconds)");
    let you_audio = record_audio(&device, 5).await?;
    println!("✅ Recording complete!");

    println!("🧠 Extracting voice features...");
    let you_embedding = embedding.extract(&you_audio)?;
    registry.enroll("speaker1", "You", you_embedding)?;
    println!("✅ YOUR voice enrolled!\n");

    // STEP 2: Enroll second speaker (friend)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 ENROLLMENT: Second Speaker");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nNow ask your FRIEND to speak.");
    println!("Press Enter, then your friend should speak for 5 seconds...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    println!("🔴 RECORDING... (5 seconds)");
    let friend_audio = record_audio(&device, 5).await?;
    println!("✅ Recording complete!");

    println!("🧠 Extracting voice features...");
    let friend_embedding = embedding.extract(&friend_audio)?;
    registry.enroll("speaker2", "Friend", friend_embedding)?;
    println!("✅ FRIEND's voice enrolled!\n");

    // STEP 3: Test identification
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎬 LIVE IDENTIFICATION TEST");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nNow I'll identify who's speaking!");
    println!("Take turns speaking. Press Enter before each person speaks.\n");

    for round in 1..=4 {
        println!(
            "Round {}/4 - Press Enter, then someone speak (3 seconds)...",
            round
        );
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        println!("🔴 RECORDING...");
        let test_audio = record_audio(&device, 3).await?;

        println!("🧠 Identifying speaker...");
        let test_embedding = embedding.extract(&test_audio)?;
        let recognition = registry.identify(&test_embedding)?;

        match recognition {
            vox::diarization::Recognition::Identified {
                speaker_id,
                confidence,
            } => {
                let speaker = registry
                    .list_speakers()
                    .into_iter()
                    .find(|s| s.id == speaker_id)
                    .unwrap();

                display_speaker_message(&speaker.name, "(recorded audio)", confidence);
            }
            vox::diarization::Recognition::Unknown { best_score } => {
                display_speaker_message("Unknown", "(recorded audio)", best_score);
            }
        }
        println!();
    }

    println!("\n✨ Test complete!");
    println!("\n💡 If identification wasn't accurate:");
    println!("  - Make sure you're in a quiet room");
    println!("  - Speak clearly and at normal volume");
    println!("  - Use a good microphone");
    println!("  - Re-enroll with longer samples (10+ seconds)");

    Ok(())
}

/// Record audio from microphone for specified duration.
async fn record_audio(device: &cpal::Device, duration_secs: u64) -> anyhow::Result<AudioChunk> {
    use cpal::traits::{DeviceTrait, StreamTrait};

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;

    let samples = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut s = samples_clone.lock().unwrap();
            s.extend_from_slice(data);
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;

    stream.play()?;

    // Record for specified duration
    tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;

    drop(stream);

    let recorded_samples = samples.lock().unwrap().clone();

    Ok(AudioChunk {
        samples: recorded_samples,
        sample_rate,
        channels: 1,
    })
}
