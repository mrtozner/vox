use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use vox::streaming_pipeline::{StreamingPipeline, StreamingPipelineConfig};
use vox::{AudioChunk, SileroVad, SttBackend, Utterance, VadBackend, VadEvent, WhisperBackend};

const VAD_MODEL: &str = "models/silero_vad.onnx";
const STT_MODEL: &str = "models/ggml-tiny.en.bin";

/// Generate synthetic audio chunks for testing.
fn generate_audio_chunks(duration_secs: u32, sample_rate: u32) -> Vec<AudioChunk> {
    let total_samples = duration_secs * sample_rate;
    let chunk_size = 512usize;
    let num_chunks = total_samples as usize / chunk_size;

    (0..num_chunks)
        .map(|i| {
            let samples: Vec<f32> = (0..chunk_size)
                .map(|j| {
                    let t = (i * chunk_size + j) as f32 / sample_rate as f32;
                    // 440Hz sine wave
                    (t * 2.0 * std::f32::consts::PI * 440.0).sin() * 0.3
                })
                .collect();
            AudioChunk {
                samples,
                sample_rate,
                channels: 1,
            }
        })
        .collect()
}

/// Benchmark sequential pipeline (current implementation).
fn bench_sequential_pipeline(c: &mut Criterion) {
    if !std::path::Path::new(VAD_MODEL).exists() || !std::path::Path::new(STT_MODEL).exists() {
        eprintln!("Skipping streaming benchmarks: models not found");
        eprintln!("Run: bash scripts/download_models.sh");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut vad = SileroVad::new(VAD_MODEL).unwrap();
    let stt = WhisperBackend::from_model(STT_MODEL).unwrap();
    let chunks = generate_audio_chunks(3, 16000);

    let mut group = c.benchmark_group("sequential_pipeline");
    group.sample_size(10);

    group.bench_function("vad_then_stt_3s", |b| {
        b.iter(|| {
            let t_start = Instant::now();

            rt.block_on(async {
                let mut collected = Vec::new();

                for chunk in &chunks {
                    let events = vad.process_frame(black_box(chunk)).await.unwrap();
                    collected.extend_from_slice(&chunk.samples);

                    for event in events {
                        if let VadEvent::SpeechEnd(ref utt) = event {
                            let _ = stt.transcribe(utt).await.unwrap();
                        }
                    }
                }

                if !collected.is_empty() {
                    let utt = Utterance {
                        audio: AudioChunk {
                            samples: collected,
                            sample_rate: 16000,
                            channels: 1,
                        },
                        duration_ms: 3000,
                    };
                    let _ = stt.transcribe(&utt).await.unwrap();
                }
            });

            vad.reset();

            tracing::debug!(
                elapsed_ms = t_start.elapsed().as_millis(),
                "sequential pipeline completed"
            );
        });
    });

    group.finish();
}

/// Benchmark streaming pipeline with parallel STT.
fn bench_streaming_pipeline(c: &mut Criterion) {
    if !std::path::Path::new(VAD_MODEL).exists() || !std::path::Path::new(STT_MODEL).exists() {
        eprintln!("Skipping streaming benchmarks: models not found");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let chunks = generate_audio_chunks(3, 16000);

    let mut group = c.benchmark_group("streaming_pipeline");
    group.sample_size(10);

    group.bench_function("parallel_stt_3s", |b| {
        b.iter(|| {
            let t_start = Instant::now();

            rt.block_on(async {
                let mut vad = SileroVad::new(VAD_MODEL).unwrap();
                let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());

                let config = StreamingPipelineConfig {
                    max_parallel_stt: 3,
                    channel_buffer_size: 16,
                    enable_detailed_metrics: true,
                };

                let pipeline = StreamingPipeline::new(config, stt);

                let (vad_tx, vad_rx) = mpsc::channel(16);

                // Spawn VAD processor
                let chunks_clone = chunks.clone();
                tokio::spawn(async move {
                    for chunk in chunks_clone {
                        let events = vad.process_frame(&chunk).await.unwrap();
                        for event in events {
                            if vad_tx.send(event).await.is_err() {
                                break;
                            }
                        }
                    }
                    // Flush VAD state
                    vad.reset();
                });

                // Process through streaming pipeline
                let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

                let mut result_count = 0;
                while let Some(_result) = result_rx.recv().await {
                    result_count += 1;
                }

                let metrics = pipeline.metrics().await;
                tracing::debug!(
                    result_count,
                    elapsed_ms = t_start.elapsed().as_millis(),
                    ?metrics,
                    "streaming pipeline completed"
                );
            });
        });
    });

    group.finish();
}

/// Benchmark latency reduction.
fn bench_latency_comparison(_c: &mut Criterion) {
    if !std::path::Path::new(VAD_MODEL).exists() || !std::path::Path::new(STT_MODEL).exists() {
        eprintln!("Skipping latency comparison: models not found");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let chunks = generate_audio_chunks(3, 16000);

    println!("\n=== LATENCY COMPARISON ===");
    println!("Testing 3 seconds of audio...\n");

    // Sequential baseline
    let sequential_time = rt.block_on(async {
        let mut vad = SileroVad::new(VAD_MODEL).unwrap();
        let stt = WhisperBackend::from_model(STT_MODEL).unwrap();

        let t_start = Instant::now();

        let mut collected = Vec::new();
        for chunk in &chunks {
            let events = vad.process_frame(chunk).await.unwrap();
            collected.extend_from_slice(&chunk.samples);

            for event in events {
                if let VadEvent::SpeechEnd(ref utt) = event {
                    let _ = stt.transcribe(utt).await.unwrap();
                }
            }
        }

        if !collected.is_empty() {
            let utt = Utterance {
                audio: AudioChunk {
                    samples: collected,
                    sample_rate: 16000,
                    channels: 1,
                },
                duration_ms: 3000,
            };
            let _ = stt.transcribe(&utt).await.unwrap();
        }

        t_start.elapsed()
    });

    // Streaming optimized
    let streaming_time = rt.block_on(async {
        let mut vad = SileroVad::new(VAD_MODEL).unwrap();
        let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());

        let config = StreamingPipelineConfig {
            max_parallel_stt: 3,
            channel_buffer_size: 16,
            enable_detailed_metrics: true,
        };

        let pipeline = StreamingPipeline::new(config, stt);

        let t_start = Instant::now();

        let (vad_tx, vad_rx) = mpsc::channel(16);

        let chunks_clone = chunks.clone();
        tokio::spawn(async move {
            for chunk in chunks_clone {
                let events = vad.process_frame(&chunk).await.unwrap();
                for event in events {
                    if vad_tx.send(event).await.is_err() {
                        break;
                    }
                }
            }
        });

        let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();
        while let Some(_) = result_rx.recv().await {}

        let metrics = pipeline.metrics().await;
        tracing::debug!(?metrics, "pipeline metrics");

        t_start.elapsed()
    });

    let sequential_ms = sequential_time.as_millis();
    let streaming_ms = streaming_time.as_millis();
    let reduction_pct =
        ((sequential_ms.saturating_sub(streaming_ms)) as f64 / sequential_ms as f64) * 100.0;

    println!("Sequential pipeline: {}ms", sequential_ms);
    println!("Streaming pipeline:  {}ms", streaming_ms);
    println!("Latency reduction:   {:.1}%", reduction_pct);
    println!("Target: 50% reduction");

    if reduction_pct >= 50.0 {
        println!("✓ Target achieved!");
    } else {
        println!("⚠ Reduction: {:.1}%", reduction_pct);
    }
}

criterion_group!(
    benches,
    bench_sequential_pipeline,
    bench_streaming_pipeline,
    bench_latency_comparison
);
criterion_main!(benches);
