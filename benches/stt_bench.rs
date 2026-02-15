use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vox::{AudioChunk, SttBackend, Utterance, WhisperBackend};

const MODEL_PATH: &str = "models/ggml-tiny.en.bin";

fn generate_utterance(duration_secs: f32) -> Utterance {
    let num_samples = (16000.0 * duration_secs) as usize;
    // Generate a tone to simulate speech (440Hz sine wave)
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin() * 0.5)
        .collect();

    Utterance {
        audio: AudioChunk {
            samples,
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: (duration_secs * 1000.0) as u64,
    }
}

fn bench_whisper_transcribe(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Skipping STT benchmarks: model not found at {MODEL_PATH}");
        eprintln!("Run: bash scripts/download_models.sh");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let stt = WhisperBackend::from_model(MODEL_PATH).unwrap();

    let mut group = c.benchmark_group("whisper_transcribe");
    group.sample_size(10); // STT is slow, fewer samples

    for &duration in &[1.0f32, 3.0, 5.0, 10.0] {
        let utterance = generate_utterance(duration);

        group.bench_with_input(
            BenchmarkId::new("tiny_en", format!("{duration}s")),
            &utterance,
            |b, utterance| {
                b.iter(|| {
                    rt.block_on(async { stt.transcribe(black_box(utterance)).await.unwrap() })
                });
            },
        );
    }

    group.finish();
}

fn bench_whisper_model_load(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() {
        return;
    }

    c.bench_function("whisper_model_load", |b| {
        b.iter(|| WhisperBackend::from_model(black_box(MODEL_PATH)).unwrap());
    });
}

criterion_group!(benches, bench_whisper_transcribe, bench_whisper_model_load);
criterion_main!(benches);
