//! Test Qwen3-TTS with actual audio playback

use vox::AudioPlayer;
use vox::traits::TtsBackend;
use vox::tts::{Qwen3Backend, Qwen3Config};
use vox::types::TtsRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Qwen3-TTS Audio Playback Test ===\n");

    // Initialize backend with Metal
    let mut config = Qwen3Config::default();
    config.device = "metal".into();

    println!("Initializing Qwen3 backend with Metal GPU...");
    let backend = Qwen3Backend::with_config(config).await?;
    println!("Backend ready!\n");

    // Create request
    let request = TtsRequest {
        text: "API stands for Application Programming Interface. It helps different applications communicate with each other.".into(),
        voice: Some("en_us_female_1".into()),
        seed: None,
    };

    println!("Synthesizing and playing audio...\n");

    // Synthesize
    let result = backend.synthesize(&request).await?;

    println!("Generated {} samples", result.audio.samples.len());
    println!("Playing audio now...\n");

    // Play audio
    let player = AudioPlayer::new()?;
    player.play_blocking(&result.audio)?;

    println!("\nDone!");

    Ok(())
}
