use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vox::{ChatterboxBackend, TtsBackend, TtsRequest};

const REFERENCE_WAV: &str = "reference.wav";

fn bench_chatterbox_synthesize(c: &mut Criterion) {
    if !std::path::Path::new(REFERENCE_WAV).exists() {
        eprintln!("Skipping Chatterbox benchmarks: reference.wav not found");
        eprintln!("Place a 5-20s WAV file as reference.wav in the project root");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let tts = ChatterboxBackend::new(REFERENCE_WAV).unwrap();

    let mut group = c.benchmark_group("chatterbox_synthesize");
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
            voice: None,
            seed: None,
        };

        group.bench_with_input(
            BenchmarkId::new("voice_clone", label),
            &request,
            |b, request| {
                b.iter(|| rt.block_on(async { tts.synthesize(black_box(request)).await.unwrap() }));
            },
        );
    }

    group.finish();
}

fn bench_chatterbox_model_load(c: &mut Criterion) {
    if !std::path::Path::new(REFERENCE_WAV).exists() {
        return;
    }

    let mut group = c.benchmark_group("chatterbox_model_load");
    group.sample_size(10);

    group.bench_function("load_q4", |b| {
        b.iter(|| ChatterboxBackend::new(black_box(REFERENCE_WAV)).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_chatterbox_synthesize,
    bench_chatterbox_model_load
);
criterion_main!(benches);
