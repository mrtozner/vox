use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vox::{AudioChunk, SileroVad, VadBackend, VadConfig};

const MODEL_PATH: &str = "models/silero_vad.onnx";

fn bench_vad_frame(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping VAD benchmarks: model not found at {MODEL_PATH}");
        eprintln!("Run: bash scripts/download_models.sh");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut vad = SileroVad::new(MODEL_PATH).unwrap();
    let silence_frame = AudioChunk {
        samples: vec![0.0; 512],
        sample_rate: 16000,
        channels: 1,
    };

    c.bench_function("vad_frame_silence", |b| {
        b.iter(|| {
            rt.block_on(async {
                vad.process_frame(black_box(&silence_frame)).await.unwrap()
            })
        });
    });

    // Simulate speech-like signal (sine wave)
    let speech_frame = AudioChunk {
        samples: (0..512)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin() * 0.8)
            .collect(),
        sample_rate: 16000,
        channels: 1,
    };

    c.bench_function("vad_frame_speech", |b| {
        b.iter(|| {
            rt.block_on(async {
                vad.process_frame(black_box(&speech_frame)).await.unwrap()
            })
        });
    });

    // Benchmark with custom thresholds
    vad.reset();
    let mut sensitive_vad = SileroVad::with_config(
        MODEL_PATH,
        VadConfig {
            speech_threshold: 0.3,
            silence_duration_ms: 300,
            min_speech_ms: 100,
        },
    )
    .unwrap();

    c.bench_function("vad_frame_sensitive", |b| {
        b.iter(|| {
            rt.block_on(async {
                sensitive_vad
                    .process_frame(black_box(&silence_frame))
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_vad_sustained(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() {
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut vad = SileroVad::new(MODEL_PATH).unwrap();

    // 1 second of audio = ~31 frames at 512 samples/16kHz
    let frames: Vec<AudioChunk> = (0..31)
        .map(|_| AudioChunk {
            samples: vec![0.0; 512],
            sample_rate: 16000,
            channels: 1,
        })
        .collect();

    c.bench_function("vad_1sec_sustained", |b| {
        b.iter(|| {
            rt.block_on(async {
                for frame in &frames {
                    vad.process_frame(black_box(frame)).await.unwrap();
                }
            });
            vad.reset();
        });
    });
}

criterion_group!(benches, bench_vad_frame, bench_vad_sustained);
criterion_main!(benches);
