//! Speaker identification UI demo - shows the visual interface.
//!
//! This demonstrates what the UI will look like when identifying speakers.
//! Run with: cargo run --example speaker_id_ui_demo --features diarization

use std::io::{self, Write};
use std::thread;
use std::time::Duration;
use vox::diarization::{RecognitionConfig, SpeakerRegistry};

fn main() -> anyhow::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     🎤 SPEAKER IDENTIFICATION UI DEMO 🎤                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("This demo shows what the speaker identification UI looks like.");
    println!("Press Enter to see a simulated conversation...\n");

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    // Set up registry with sample speakers
    let config = RecognitionConfig {
        threshold: 0.7,
        require_threshold: true,
    };
    let mut registry = SpeakerRegistry::with_config(config);

    // Enroll sample speakers
    registry.enroll("alice", "Alice", vec![1.0, 0.0, 0.0])?;
    registry.enroll("bob", "Bob", vec![0.0, 1.0, 0.0])?;
    registry.enroll("charlie", "Charlie", vec![0.0, 0.0, 1.0])?;

    println!("\n📊 Enrolled Speakers:");
    println!("─────────────────────");
    for speaker in registry.list_speakers() {
        println!("   ✓ {}", speaker.name);
    }
    println!();

    // Simulate conversation
    let conversation = vec![
        (
            "alice",
            vec![1.0, 0.0, 0.0],
            0.95,
            "Hey everyone, how's it going?",
        ),
        (
            "bob",
            vec![0.0, 1.0, 0.0],
            0.92,
            "Pretty good! Working on the new feature.",
        ),
        (
            "charlie",
            vec![0.0, 0.0, 1.0],
            0.89,
            "Same here, almost done with testing.",
        ),
        (
            "alice",
            vec![0.9, 0.1, 0.0],
            0.87,
            "That's great! Let's sync up later.",
        ),
        (
            "bob",
            vec![0.0, 0.95, 0.05],
            0.88,
            "Sounds good. I'll send a meeting invite.",
        ),
        (
            "unknown",
            vec![0.5, 0.5, 0.0],
            0.45,
            "Can I join the meeting too?",
        ),
        (
            "alice",
            vec![1.0, 0.0, 0.0],
            0.94,
            "Of course! The more the merrier.",
        ),
    ];

    println!("🎙️  LIVE CONVERSATION");
    println!("════════════════════════════════════════════════════════════\n");

    for (expected_speaker, embedding, confidence, text) in conversation {
        thread::sleep(Duration::from_millis(800));

        let recognition = registry.identify(&embedding)?;

        let (speaker_name, conf) = match recognition {
            vox::diarization::Recognition::Identified {
                speaker_id,
                confidence: c,
            } => (
                registry
                    .list_speakers()
                    .iter()
                    .find(|s| s.id == speaker_id)
                    .map(|s| s.name.as_str())
                    .unwrap_or("Unknown"),
                c,
            ),
            vox::diarization::Recognition::Unknown { best_score } => ("Unknown", best_score),
        };

        // UI Display with colors and confidence bar
        display_speaker_ui(speaker_name, conf, text);
    }

    println!("\n📊 Session Statistics");
    println!("═════════════════════");
    println!("Total speakers recognized: {}", registry.speaker_count());
    println!("Utterances processed: 7");
    println!("Average confidence: 87%");
    println!("\n✨ Demo complete!\n");

    println!("💡 This is how the real UI will look when:");
    println!("   1. You download a speaker encoder model");
    println!("   2. Enroll yourself and your friend");
    println!("   3. Talk into the microphone");
    println!("   4. The system identifies who's speaking in real-time!\n");

    Ok(())
}

fn display_speaker_ui(speaker_name: &str, confidence: f32, text: &str) {
    // Confidence bar
    let filled = (confidence * 10.0) as usize;
    let empty = 10 - filled.min(10);
    let confidence_bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    // Color based on confidence
    let (color, reset) = if confidence > 0.85 {
        ("\x1b[32m", "\x1b[0m") // Green for high confidence
    } else if confidence > 0.65 {
        ("\x1b[33m", "\x1b[0m") // Yellow for medium confidence
    } else {
        ("\x1b[90m", "\x1b[0m") // Gray for low confidence
    };

    // Speaker name color
    let speaker_color = match speaker_name {
        "Alice" => "\x1b[36m",   // Cyan
        "Bob" => "\x1b[35m",     // Magenta
        "Charlie" => "\x1b[34m", // Blue
        _ => "\x1b[90m",         // Gray for unknown
    };

    println!("┌─────────────────────────────────────────────────────────┐");
    println!(
        "│ 🎤 Speaker: {}{:20}{}                       │",
        speaker_color, speaker_name, reset
    );
    println!(
        "│ 📊 Confidence: {}{} {:.0}%{}                        │",
        color,
        confidence_bar,
        confidence * 100.0,
        reset
    );
    println!("└─────────────────────────────────────────────────────────┘");
    println!(
        "💬 {}{}{}: \"{}\"",
        speaker_color, speaker_name, reset, text
    );
    println!();
}
