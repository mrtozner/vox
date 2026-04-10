//! Test streaming synthesis implementation for Qwen3 TTS backend.
//!
//! Verifies that:
//! - Chunks are delivered incrementally
//! - First chunk arrives quickly (< 1s)
//! - Total synthesis time is faster than audio duration (RTF < 1.0)

use vox::tts::{Qwen3Backend, Qwen3Config};
use vox::types::TtsRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better visibility
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Qwen3 Streaming Synthesis Test ===\n");

    // Initialize backend
    println!("Initializing Qwen3 backend...");
    let backend = Qwen3Backend::with_config(Qwen3Config::default()).await?;
    println!("Backend initialized successfully.\n");

    // Create test request
    let request = TtsRequest {
        text: "Testing streaming synthesis with Qwen3 TTS. This should produce multiple audio chunks that arrive incrementally, allowing for low-latency playback.".into(),
        voice: Some("en_us_female_1".into()),
        seed: None,
    };

    println!("Test text: {}\n", request.text);
    println!("Starting streaming synthesis...\n");

    // Track metrics using Arc for thread-safe sharing
    use std::sync::{Arc, Mutex};
    let chunk_count = Arc::new(Mutex::new(0));
    let total_samples = Arc::new(Mutex::new(0));
    let first_chunk_time = Arc::new(Mutex::new(None));
    let start = std::time::Instant::now();

    // Clone Arcs for the closure
    let chunk_count_clone = Arc::clone(&chunk_count);
    let total_samples_clone = Arc::clone(&total_samples);
    let first_chunk_time_clone = Arc::clone(&first_chunk_time);

    // Run streaming synthesis
    let result = backend
        .synthesize_with_streaming(&request, move |chunk| {
            let mut count = chunk_count_clone.lock().unwrap();
            let mut samples = total_samples_clone.lock().unwrap();
            let mut first_time = first_chunk_time_clone.lock().unwrap();

            *count += 1;
            *samples += chunk.samples.len();

            // Record time to first chunk
            if first_time.is_none() {
                *first_time = Some(start.elapsed());
            }

            println!(
                "Chunk {}: {} samples @ {}Hz (elapsed: {:.2}s)",
                *count,
                chunk.samples.len(),
                chunk.sample_rate,
                start.elapsed().as_secs_f64()
            );

            Ok(())
        })
        .await?;

    let total_time = start.elapsed();
    let audio_duration_sec = result.duration_ms as f64 / 1000.0;
    let rtf = total_time.as_secs_f64() / audio_duration_sec;

    // Extract final metrics
    let final_chunk_count = *chunk_count.lock().unwrap();
    let final_total_samples = *total_samples.lock().unwrap();
    let final_first_chunk_time = *first_chunk_time.lock().unwrap();

    // Print results
    println!("\n=== Results ===");
    println!("Total chunks: {}", final_chunk_count);
    println!("Total samples: {}", final_total_samples);
    println!("Audio duration: {:.2}s", audio_duration_sec);
    println!(
        "Time to first chunk: {:.3}s",
        final_first_chunk_time.unwrap_or_default().as_secs_f64()
    );
    println!("Total synthesis time: {:.2}s", total_time.as_secs_f64());
    println!("Real-Time Factor (RTF): {:.3}", rtf);

    // Verify expectations
    println!("\n=== Verification ===");
    let first_chunk_ok = final_first_chunk_time.map_or(false, |t| t.as_millis() < 1000);
    let streaming_ok = final_chunk_count > 1;
    let rtf_ok = rtf < 1.0;

    println!(
        "✓ First chunk < 1s: {}",
        if first_chunk_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "✓ Multiple chunks: {} {}",
        if streaming_ok { "PASS" } else { "FAIL" },
        if streaming_ok {
            format!("({} chunks)", final_chunk_count)
        } else {
            String::new()
        }
    );
    println!(
        "✓ RTF < 1.0 (faster than real-time): {}",
        if rtf_ok { "PASS" } else { "FAIL" }
    );

    if first_chunk_ok && streaming_ok && rtf_ok {
        println!("\n✅ All streaming tests PASSED");
        Ok(())
    } else {
        println!("\n❌ Some streaming tests FAILED");
        std::process::exit(1);
    }
}
