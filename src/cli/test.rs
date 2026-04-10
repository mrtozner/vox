//! Handler for `vox test` — audio I/O diagnostic testing.

use cpal::traits::{DeviceTrait, HostTrait};
use vox::audio::AudioCapture;
use vox::types::AudioChunk;

#[cfg(any(
    feature = "kokoro",
    feature = "pocket",
    feature = "chatterbox",
    feature = "piper",
    feature = "tts"
))]
use vox::audio::AudioPlayer;

/// Run the audio test command.
///
/// Tests microphone and speaker by recording 3 seconds of audio and playing it back.
pub async fn run() -> anyhow::Result<()> {
    println!("=== Vox Audio Test ===\n");

    // Test 1: Microphone test
    println!("Test 1: Microphone Input");
    println!("  Recording 3 seconds of audio from your microphone...");

    let (capture, mut rx) = AudioCapture::new(16000, 1)
        .map_err(|e| anyhow::anyhow!("Failed to initialize microphone: {}", e))?;

    capture
        .start()
        .map_err(|e| anyhow::anyhow!("Failed to start audio capture: {}", e))?;

    // Collect 3 seconds of audio (16000 samples per second)
    let mut recorded_samples = Vec::new();
    let target_samples = 16000 * 3; // 3 seconds at 16kHz

    while recorded_samples.len() < target_samples {
        if let Some(chunk) = rx.recv().await {
            recorded_samples.extend_from_slice(&chunk.samples);
        } else {
            anyhow::bail!("Microphone stream ended unexpectedly");
        }
    }

    // Truncate to exactly 3 seconds
    recorded_samples.truncate(target_samples);

    capture
        .stop()
        .map_err(|e| anyhow::anyhow!("Failed to stop audio capture: {}", e))?;

    println!(
        "  ✓ Successfully recorded {} samples ({} seconds)",
        recorded_samples.len(),
        recorded_samples.len() / 16000
    );

    // Calculate RMS (root mean square) to check if we actually captured audio
    let rms: f32 = (recorded_samples.iter().map(|&s| s * s).sum::<f32>()
        / recorded_samples.len() as f32)
        .sqrt();

    if rms < 0.001 {
        println!(
            "  ⚠ Warning: Audio level very low (RMS: {:.6}). Check microphone volume.",
            rms
        );
    } else {
        println!("  ✓ Audio level OK (RMS: {:.4})", rms);
    }

    println!();

    // Test 2: Speaker test
    #[cfg(any(
        feature = "kokoro",
        feature = "pocket",
        feature = "chatterbox",
        feature = "piper",
        feature = "tts"
    ))]
    {
        println!("Test 2: Speaker Output");
        println!("  Playing back recorded audio...");

        let player = AudioPlayer::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize audio player: {}", e))?;

        let playback_chunk = AudioChunk {
            samples: recorded_samples,
            sample_rate: 16000,
            channels: 1,
        };

        player
            .play_blocking(&playback_chunk)
            .map_err(|e| anyhow::anyhow!("Failed to play audio: {}", e))?;

        println!("  ✓ Playback completed");
        println!();
    }

    #[cfg(not(any(
        feature = "kokoro",
        feature = "pocket",
        feature = "chatterbox",
        feature = "piper",
        feature = "tts"
    )))]
    {
        println!("Test 2: Speaker Output");
        println!("  Skipped (requires TTS feature: kokoro, pocket, chatterbox, piper, or tts)");
        println!();
    }

    // Test 3: Device info
    println!("Test 3: Audio Device Information");

    // Get default input device info
    let host = cpal::default_host();
    if let Some(input_device) = host.default_input_device() {
        let device_name = input_device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        println!("  Input device: {}", device_name);

        if let Ok(config) = input_device.default_input_config() {
            println!("    Sample rate: {} Hz", config.sample_rate().0);
            println!("    Channels: {}", config.channels());
            println!("    Sample format: {:?}", config.sample_format());
        }
    } else {
        println!("  ✗ No default input device found");
    }

    // Get default output device info
    if let Some(output_device) = host.default_output_device() {
        let device_name = output_device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        println!("  Output device: {}", device_name);

        if let Ok(config) = output_device.default_output_config() {
            println!("    Sample rate: {} Hz", config.sample_rate().0);
            println!("    Channels: {}", config.channels());
            println!("    Sample format: {:?}", config.sample_format());
        }
    } else {
        println!("  ✗ No default output device found");
    }

    println!();
    println!("=== All Tests Completed ===");
    println!("Status: PASS");

    Ok(())
}
