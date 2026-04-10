//! Test non-streaming synthesis for comparison with streaming.

use vox::tts::{Qwen3Backend, Qwen3Config};
use vox::types::TtsRequest;
use vox::traits::TtsBackend;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Qwen3 Non-Streaming Synthesis Test ===\n");

    // Initialize backend
    println!("Initializing Qwen3 backend...");
    let backend = Qwen3Backend::with_config(Qwen3Config::default()).await?;
    println!("Backend initialized successfully.\n");

    // Create test request (same text as streaming test)
    let request = TtsRequest {
        text: "Testing streaming synthesis with Qwen3 TTS. This should produce multiple audio chunks that arrive incrementally, allowing for low-latency playback.".into(),
        voice: Some("en_us_female_1".into()),
        seed: None,
    };

    println!("Test text: {}\n", request.text);
    println!("Starting non-streaming synthesis...\n");

    // Run non-streaming synthesis
    let start = std::time::Instant::now();
    let result = backend.synthesize(&request).await?;
    let total_time = start.elapsed();

    let audio_duration_sec = result.duration_ms as f64 / 1000.0;
    let rtf = total_time.as_secs_f64() / audio_duration_sec;

    // Print results
    println!("\n=== Results ===");
    println!("Total samples: {}", result.audio.samples.len());
    println!("Audio duration: {:.2}s", audio_duration_sec);
    println!("Total synthesis time: {:.2}s", total_time.as_secs_f64());
    println!("Real-Time Factor (RTF): {:.3}", rtf);

    // Verify expectations
    println!("\n=== Verification ===");
    let rtf_ok = rtf < 1.0;
    println!("✓ RTF < 1.0 (faster than real-time): {}", if rtf_ok { "PASS" } else { "FAIL" });

    if rtf_ok {
        println!("\n✅ Performance test PASSED");
        Ok(())
    } else {
        println!("\n❌ Performance test FAILED");
        std::process::exit(1);
    }
}
