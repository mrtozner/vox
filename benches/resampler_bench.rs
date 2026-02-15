use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vox::AudioChunk;
use vox::audio::AudioResampler;

fn bench_passthrough(c: &mut Criterion) {
    let mut resampler = AudioResampler::new(16000, 16000).unwrap();

    for &size in &[512, 1024, 4096, 16000] {
        let chunk = AudioChunk {
            samples: vec![0.5; size],
            sample_rate: 16000,
            channels: 1,
        };

        c.bench_with_input(BenchmarkId::new("passthrough", size), &chunk, |b, chunk| {
            b.iter(|| resampler.process(black_box(chunk)).unwrap());
        });
    }
}

fn bench_resample_44100_to_16000(c: &mut Criterion) {
    let mut resampler = AudioResampler::new(44100, 16000).unwrap();

    for &duration_ms in &[30, 100, 500, 1000] {
        let num_samples = (44100 * duration_ms) / 1000;
        let chunk = AudioChunk {
            samples: vec![0.5; num_samples],
            sample_rate: 44100,
            channels: 1,
        };

        c.bench_with_input(
            BenchmarkId::new("resample_44100_to_16000", format!("{duration_ms}ms")),
            &chunk,
            |b, chunk| {
                b.iter(|| resampler.process(black_box(chunk)).unwrap());
            },
        );
    }
}

fn bench_stereo_to_mono(c: &mut Criterion) {
    let mut resampler = AudioResampler::new(16000, 16000).unwrap();

    let chunk = AudioChunk {
        samples: vec![0.5; 32000], // 1 sec stereo
        sample_rate: 16000,
        channels: 2,
    };

    c.bench_function("stereo_to_mono_1s", |b| {
        b.iter(|| resampler.process(black_box(&chunk)).unwrap());
    });
}

criterion_group!(
    benches,
    bench_passthrough,
    bench_resample_44100_to_16000,
    bench_stereo_to_mono
);
criterion_main!(benches);
