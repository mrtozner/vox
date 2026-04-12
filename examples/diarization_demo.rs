//! Multi-speaker diarization demo.
//!
//! This example demonstrates:
//! - Speaker enrollment with voice profiles
//! - Real-time speaker identification
//! - Multi-speaker conversation tracking
//! - Speaker database integration
//!
//! Run with:
//! ```bash
//! cargo run --example diarization_demo --features diarization
//! ```

use vox::diarization::{
    DiarizationConfig, DiarizationPipeline, DiarizationPipelineBuilder, RecognitionConfig,
    SpeakerEmbedding, SpeakerRegistry,
};
use vox::{AudioChunk, Utterance};

/// Generate test audio (sine wave at specified frequency) to simulate different speakers.
fn generate_speaker_voice(duration_ms: u64, frequency: f32, sample_rate: u32) -> AudioChunk {
    let total_samples = (duration_ms * sample_rate as u64) / 1000;
    let samples: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * 2.0 * std::f32::consts::PI * frequency).sin() * 0.3
        })
        .collect();

    AudioChunk {
        samples,
        sample_rate,
        channels: 1,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🎭 Multi-Speaker Diarization Demo");
    println!("==================================\n");

    // Note: This demo uses synthesized audio for demonstration.
    // In production, you would use a speaker encoder ONNX model like ECAPA-TDNN.
    // Download from: https://huggingface.co/onnx-community/speaker-encoder

    println!("📦 Initializing diarization system...\n");

    // For this demo, we'll use the registry without the actual embedding model
    // In production, you would do:
    // let embedding = SpeakerEmbedding::new("models/speaker_encoder.onnx")?;
    let recognition_config = RecognitionConfig {
        threshold: 0.7,
        require_threshold: true,
    };
    let mut registry = SpeakerRegistry::with_config(recognition_config);

    println!("✅ Speaker registry ready (in-memory, no database)\n");

    // ==================================================================
    // STEP 1: Enroll speakers with voice profiles
    // ==================================================================
    println!("👥 Enrolling speakers...");
    println!("   (Using frequency-based voice profiles for demo)\n");

    // Manually enroll speakers with synthesized embeddings (for demo)
    // In production, you would extract real embeddings from audio
    let alice_embedding = vec![1.0, 0.0, 0.0]; // Simplified embedding
    registry.enroll("alice", "Alice", alice_embedding)?;
    println!("   ✓ Alice enrolled");

    let bob_embedding = vec![0.0, 1.0, 0.0];
    registry.enroll("bob", "Bob", bob_embedding)?;
    println!("   ✓ Bob enrolled");

    let charlie_embedding = vec![0.0, 0.0, 1.0];
    registry.enroll("charlie", "Charlie", charlie_embedding)?;
    println!("   ✓ Charlie enrolled");

    println!("\n   Total speakers: {}\n", registry.speaker_count());

    // ==================================================================
    // STEP 2: Simulate a multi-speaker conversation
    // ==================================================================
    println!("🎬 Simulating multi-speaker conversation...\n");

    let conversation = vec![
        ("alice", vec![1.0, 0.0, 0.0], "Hello everyone!"),
        ("bob", vec![0.0, 1.0, 0.0], "Hi Alice, how are you?"),
        ("alice", vec![1.0, 0.0, 0.0], "I'm doing great, thanks!"),
        ("charlie", vec![0.0, 0.0, 1.0], "Hey folks, what's up?"),
        ("bob", vec![0.0, 1.0, 0.0], "Just having a chat."),
        ("charlie", vec![0.0, 0.0, 1.0], "Sounds good!"),
        ("alice", vec![0.9, 0.1, 0.0], "Anyone want coffee?"), // Slightly different voice
    ];

    println!("💬 Conversation Transcript:");
    println!("----------------------------\n");

    for (expected_speaker, embedding, text) in conversation {
        // Identify speaker using their voice embedding
        let recognition = registry.identify(&embedding)?;

        let (speaker_id, confidence) = match recognition {
            vox::diarization::Recognition::Identified {
                speaker_id,
                confidence,
            } => (speaker_id, Some(confidence)),
            vox::diarization::Recognition::Unknown { best_score } => {
                (format!("unknown (score: {:.2})", best_score), None)
            }
        };

        // Get speaker name
        let speaker_name = registry
            .list_speakers()
            .into_iter()
            .find(|s| s.id == speaker_id)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        // Display conversation line with speaker info
        let confidence_str = confidence
            .map(|c| format!(" ({:.0}% confidence)", c * 100.0))
            .unwrap_or_default();

        println!("🎤 [{:8}]{} {}", speaker_name, confidence_str, text);
    }

    // ==================================================================
    // STEP 3: Demonstrate speaker management
    // ==================================================================
    println!("🔧 Speaker Management Demo");
    println!("==========================\n");

    // Enroll a new speaker mid-conversation
    let david_embedding = vec![1.0, 1.0, 0.0];
    registry.enroll("david", "David", david_embedding.clone())?;
    println!("✅ New speaker enrolled: David");

    // Test identification
    let result = registry.identify(&david_embedding)?;
    if let vox::diarization::Recognition::Identified {
        speaker_id,
        confidence,
    } = result
    {
        println!(
            "✓ David identified with {:.1}% confidence",
            confidence * 100.0
        );
    }

    // Remove a speaker
    registry.forget("david")?;
    println!("✅ Speaker removed: David");
    println!("   Remaining speakers: {}\n", registry.speaker_count());

    println!("✨ Demo complete!");
    println!("\n💡 Next Steps:");
    println!("   1. Download a speaker encoder model (ECAPA-TDNN)");
    println!("   2. Use SpeakerEmbedding::new(\"path/to/model.onnx\")");
    println!("   3. Extract real embeddings from audio with embedding.extract(&audio)");
    println!("   4. Integrate with VAD+STT pipeline for real-time conversations");

    Ok(())
}
