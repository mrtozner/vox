use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vox::{KokoroBackend, TtsBackend, TtsRequest};

const MODEL_PATH: &str = "models/kokoro-v1.0.onnx";
const VOICES_PATH: &str = "models/voices.bin";

fn bench_kokoro_synthesize(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() || !std::path::Path::new(VOICES_PATH).exists() {
        eprintln!("Skipping TTS benchmarks: models not found");
        eprintln!("Run: bash scripts/download_models.sh");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let tts = rt.block_on(async { KokoroBackend::new(MODEL_PATH, VOICES_PATH).await.unwrap() });

    let mut group = c.benchmark_group("kokoro_synthesize");
    group.sample_size(10); // TTS is slow

    let texts = [
        ("short", "Hello world."),
        (
            "medium",
            "The quick brown fox jumps over the lazy dog near the riverbank.",
        ),
        (
            "long",
            "In a world where artificial intelligence runs entirely on local devices, \
             privacy becomes the default rather than the exception. Every word you speak \
             stays on your hardware, processed by models that fit in the palm of your hand.",
        ),
    ];

    for (label, text) in texts {
        let request = TtsRequest {
            text: text.to_string(),
            voice: Some("af_heart".into()),
        };

        group.bench_with_input(
            BenchmarkId::new("af_heart", label),
            &request,
            |b, request| {
                b.iter(|| {
                    rt.block_on(async { tts.synthesize(black_box(request)).await.unwrap() })
                });
            },
        );
    }

    // Benchmark different voices
    for voice in ["af_sky", "am_adam", "bf_alice"] {
        let request = TtsRequest {
            text: "Hello, this is a voice test.".to_string(),
            voice: Some(voice.into()),
        };

        group.bench_with_input(
            BenchmarkId::new("voice_compare", voice),
            &request,
            |b, request| {
                b.iter(|| {
                    rt.block_on(async { tts.synthesize(black_box(request)).await.unwrap() })
                });
            },
        );
    }

    group.finish();
}

fn bench_kokoro_model_load(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_PATH).exists() || !std::path::Path::new(VOICES_PATH).exists() {
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("kokoro_model_load");
    group.sample_size(10);

    group.bench_function("load_fp32", |b| {
        b.iter(|| {
            rt.block_on(async {
                KokoroBackend::new(black_box(MODEL_PATH), black_box(VOICES_PATH))
                    .await
                    .unwrap()
            })
        });
    });

    group.finish();
}

criterion_group!(benches, bench_kokoro_synthesize, bench_kokoro_model_load);
criterion_main!(benches);
