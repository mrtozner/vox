use std::sync::Arc;
use tokio::sync::mpsc;
use vox::streaming_pipeline::{StreamingPipeline, StreamingPipelineConfig};
use vox::{AudioChunk, SileroVad, Utterance, VadBackend, VadEvent, WhisperBackend};

const VAD_MODEL: &str = "models/silero_vad.onnx";
const STT_MODEL: &str = "models/ggml-tiny.en.bin";

fn skip_if_models_missing() -> bool {
    let vad_exists = std::path::Path::new(VAD_MODEL).exists();
    let stt_exists = std::path::Path::new(STT_MODEL).exists();

    if !vad_exists || !stt_exists {
        eprintln!("Skipping streaming pipeline tests: models not found");
        eprintln!("Run: bash scripts/download_models.sh");
        return true;
    }
    false
}

/// Generate test audio (sine wave).
fn generate_test_audio(duration_ms: u32, sample_rate: u32) -> AudioChunk {
    let num_samples = (duration_ms as f32 / 1000.0 * sample_rate as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * 2.0 * std::f32::consts::PI * 440.0).sin() * 0.5
        })
        .collect();

    AudioChunk {
        samples,
        sample_rate,
        channels: 1,
    }
}

#[tokio::test]
async fn test_streaming_pipeline_basic() {
    if skip_if_models_missing() {
        return;
    }

    let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());
    let config = StreamingPipelineConfig {
        max_parallel_stt: 2,
        channel_buffer_size: 8,
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    let (vad_tx, vad_rx) = mpsc::channel(8);

    // Send a speech utterance
    let audio = generate_test_audio(1000, 16000);
    let utterance = Utterance {
        audio: audio.clone(),
        duration_ms: 1000,
        #[cfg(feature = "diarization")]
        speaker_id: None,
    };

    tokio::spawn(async move {
        let _ = vad_tx.send(VadEvent::SpeechEnd(utterance)).await;
    });

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

    // Should receive at least one result
    let result = result_rx.recv().await;
    assert!(result.is_some());

    let metrics = pipeline.metrics().await;
    assert_eq!(metrics.total_chunks, 1);
}

#[tokio::test]
async fn test_streaming_pipeline_parallelism() {
    if skip_if_models_missing() {
        return;
    }

    let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());
    let config = StreamingPipelineConfig {
        max_parallel_stt: 3,
        channel_buffer_size: 16,
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    let (vad_tx, vad_rx) = mpsc::channel(16);

    // Send multiple utterances
    let num_utterances = 5;
    tokio::spawn(async move {
        for i in 0..num_utterances {
            let audio = generate_test_audio(500, 16000);
            let utterance = Utterance {
                audio,
                duration_ms: 500,
                #[cfg(feature = "diarization")]
                speaker_id: None,
            };
            if vad_tx.send(VadEvent::SpeechEnd(utterance)).await.is_err() {
                eprintln!("Failed to send utterance {}", i);
                break;
            }
        }
    });

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = result_rx.recv().await {
        results.push(result);
        if results.len() >= num_utterances {
            break;
        }
    }

    let metrics = pipeline.metrics().await;
    assert!(
        metrics.peak_parallel_stt > 1,
        "Expected parallel processing"
    );
    assert!(
        metrics.peak_parallel_stt <= 3,
        "Should not exceed max_parallel_stt"
    );

    eprintln!("Processed {} utterances", results.len());
    eprintln!("Peak parallelism: {}", metrics.peak_parallel_stt);
    eprintln!("Avg STT latency: {:.0}ms", metrics.avg_stt_latency_ms);
}

#[tokio::test]
async fn test_streaming_pipeline_with_vad() {
    if skip_if_models_missing() {
        return;
    }

    let mut vad = SileroVad::new(VAD_MODEL).unwrap();
    let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());

    let config = StreamingPipelineConfig {
        max_parallel_stt: 2,
        channel_buffer_size: 8,
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    let (vad_tx, vad_rx) = mpsc::channel(8);

    // Generate frames and process through VAD
    let frames: Vec<AudioChunk> = (0..93)
        .map(|i| {
            let samples: Vec<f32> = (0..512)
                .map(|j| {
                    let t = (i * 512 + j) as f32 / 16000.0;
                    (t * 2.0 * std::f32::consts::PI * 440.0).sin() * 0.5
                })
                .collect();
            AudioChunk {
                samples,
                sample_rate: 16000,
                channels: 1,
            }
        })
        .collect();

    tokio::spawn(async move {
        for frame in frames {
            let events = vad.process_frame(&frame).await.unwrap();
            for event in events {
                if vad_tx.send(event).await.is_err() {
                    break;
                }
            }
        }
    });

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = result_rx.recv().await {
        results.push(result);
    }

    let metrics = pipeline.metrics().await;
    eprintln!(
        "VAD+STT: {} results, avg latency: {:.0}ms",
        results.len(),
        metrics.avg_stt_latency_ms
    );
}

#[tokio::test]
async fn test_streaming_pipeline_backpressure() {
    if skip_if_models_missing() {
        return;
    }

    let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());
    let config = StreamingPipelineConfig {
        max_parallel_stt: 2,
        channel_buffer_size: 4, // Small buffer to test backpressure
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    let (vad_tx, vad_rx) = mpsc::channel(4);

    // Try to send many utterances rapidly
    let num_utterances = 10;
    tokio::spawn(async move {
        for i in 0..num_utterances {
            let audio = generate_test_audio(200, 16000);
            let utterance = Utterance {
                audio,
                duration_ms: 200,
                #[cfg(feature = "diarization")]
                speaker_id: None,
            };
            // This may block due to backpressure
            if vad_tx.send(VadEvent::SpeechEnd(utterance)).await.is_err() {
                eprintln!("Channel closed at utterance {}", i);
                break;
            }
        }
    });

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

    let mut count = 0;
    while let Some(_) = result_rx.recv().await {
        count += 1;
        if count >= num_utterances {
            break;
        }
    }

    let metrics = pipeline.metrics().await;
    assert!(
        metrics.chunks_dropped == 0,
        "No chunks should be dropped with backpressure"
    );
}

#[tokio::test]
async fn test_streaming_pipeline_metrics() {
    if skip_if_models_missing() {
        return;
    }

    let stt = Arc::new(WhisperBackend::from_model(STT_MODEL).unwrap());
    let config = StreamingPipelineConfig {
        max_parallel_stt: 3,
        channel_buffer_size: 16,
        enable_detailed_metrics: true,
    };

    let pipeline = StreamingPipeline::new(config, stt);

    let (vad_tx, vad_rx) = mpsc::channel(16);

    let num_utterances = 3;
    tokio::spawn(async move {
        for _ in 0..num_utterances {
            let audio = generate_test_audio(500, 16000);
            let utterance = Utterance {
                audio,
                duration_ms: 500,
                #[cfg(feature = "diarization")]
                speaker_id: None,
            };
            if vad_tx.send(VadEvent::SpeechEnd(utterance)).await.is_err() {
                break;
            }
        }
    });

    let mut result_rx = pipeline.process_vad_stream(vad_rx).await.unwrap();

    let mut count = 0;
    while let Some(_) = result_rx.recv().await {
        count += 1;
        if count >= num_utterances {
            break;
        }
    }

    let metrics = pipeline.metrics().await;
    assert_eq!(metrics.total_chunks, num_utterances as u64);
    assert!(metrics.avg_stt_latency_ms > 0.0);
    assert!(metrics.avg_total_latency_ms > 0.0);
    assert!(metrics.peak_parallel_stt > 0);

    eprintln!("Metrics: {:?}", metrics);
}
