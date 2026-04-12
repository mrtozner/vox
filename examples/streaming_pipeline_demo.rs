//! Streaming pipeline demo - demonstrates parallel STT processing.
//!
//! This example shows how to use the optimized streaming pipeline for
//! low-latency audio processing with parallel STT execution.
//!
//! Run with:
//! ```bash
//! cargo run --example streaming_pipeline_demo --features whisper,silero
//! ```

use std::sync::Arc;
use tokio::sync::mpsc;
use vox::streaming_pipeline::{StreamingPipeline, StreamingPipelineConfig};
use vox::{SileroVad, VadBackend, VadEvent, WhisperBackend};

/// Generate test audio (440Hz sine wave).
fn generate_test_audio(duration_secs: u32, sample_rate: u32) -> Vec<vox::AudioChunk> {
    let total_samples = duration_secs * sample_rate;
    let chunk_size = 512;
    let num_chunks = (total_samples / chunk_size) as usize;

    (0..num_chunks)
        .map(|i| {
            let samples: Vec<f32> = (0..chunk_size)
                .map(|j| {
                    let sample_idx = i * chunk_size as usize + j as usize;
                    let t = sample_idx as f32 / sample_rate as f32;
                    (t * 2.0 * std::f32::consts::PI * 440.0).sin() * 0.3
                })
                .collect();
            vox::AudioChunk {
                samples,
                sample_rate,
                channels: 1,
            }
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🎙️  Streaming Pipeline Demo");
    println!("============================\n");

    // Check models exist
    let vad_model = "models/silero_vad.onnx";
    let stt_model = "models/ggml-tiny.en.bin";

    if !std::path::Path::new(vad_model).exists() || !std::path::Path::new(stt_model).exists() {
        eprintln!("❌ Models not found!");
        eprintln!("Run: bash scripts/download_models.sh");
        return Ok(());
    }

    // Initialize VAD and STT
    println!("📦 Loading models...");
    let mut vad = SileroVad::new(vad_model)?;
    let stt = Arc::new(WhisperBackend::from_model(stt_model)?);

    // Configure streaming pipeline
    let config = StreamingPipelineConfig {
        max_parallel_stt: 3,     // Process 3 chunks in parallel
        channel_buffer_size: 16, // Buffer up to 16 chunks
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    println!("✅ Models loaded\n");

    // Generate test audio (5 seconds)
    println!("🎵 Generating test audio (5 seconds, 440Hz)...");
    let chunks = generate_test_audio(5, 16000);
    println!("   Generated {} audio chunks\n", chunks.len());

    // Create channels
    let (vad_tx, vad_rx) = mpsc::channel(16);

    // Spawn VAD processor
    println!("🔊 Processing audio through VAD...");
    tokio::spawn(async move {
        for chunk in chunks {
            let events = vad.process_frame(&chunk).await.unwrap();
            for event in events {
                if let VadEvent::SpeechEnd(_) = event {
                    println!("   📢 Speech detected!");
                }
                if vad_tx.send(event).await.is_err() {
                    break;
                }
            }
        }
        println!("   ✅ VAD processing complete");
    });

    // Process through streaming pipeline
    println!("🔄 Starting parallel STT pipeline...\n");
    let t_start = std::time::Instant::now();

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await?;

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = result_rx.recv().await {
        println!("📝 Transcription: \"{}\"", result.text);
        println!("   Latency: {}ms", result.processing_time_ms);
        results.push(result);
    }

    let total_time = t_start.elapsed();

    // Get metrics
    let metrics = pipeline.metrics().await;

    println!("\n📊 Pipeline Metrics");
    println!("===================");
    println!("Total chunks:        {}", metrics.total_chunks);
    println!("Peak parallelism:    {}", metrics.peak_parallel_stt);
    println!("Avg STT latency:     {:.0}ms", metrics.avg_stt_latency_ms);
    println!("Avg total latency:   {:.0}ms", metrics.avg_total_latency_ms);
    println!("Chunks dropped:      {}", metrics.chunks_dropped);
    println!("Total wall time:     {}ms", total_time.as_millis());

    println!("\n✨ Demo complete!");

    Ok(())
}
