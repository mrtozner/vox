// Suppress unused warnings when no TTS feature is enabled; this bench is meant
// to be run with at least one of: kokoro, pocket, chatterbox.
#![allow(unused_imports, unused_variables, dead_code)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use vox::{TtsBackend, TtsRequest};

const SHORT_TEXT: &str = "Hello world, this is a test.";
const LONG_TEXT: &str = "In a world where artificial intelligence runs entirely on local devices, \
    privacy becomes the default rather than the exception. Every word you speak \
    stays on your hardware, processed by models that fit in the palm of your hand. \
    This is the future of voice technology, running at the edge with no cloud dependency.";

#[cfg(feature = "kokoro")]
const MODEL_PATH: &str = "models/kokoro-v1.0.onnx";
#[cfg(feature = "kokoro")]
const VOICES_PATH: &str = "models/voices.bin";

#[cfg(feature = "chatterbox")]
const REFERENCE_WAV: &str = "reference_voice.wav";

/// Compute the real-time factor: wall-clock time / audio duration.
/// RTF < 1.0 means faster than real-time.
fn rtf(wall_clock: Duration, audio_duration_ms: u64) -> f64 {
    let wall_ms = wall_clock.as_secs_f64() * 1000.0;
    wall_ms / audio_duration_ms as f64
}

/// Benchmark synthesis across all enabled TTS backends using a shared text input.
fn bench_tts_comparison(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let texts = [("short", SHORT_TEXT), ("long", LONG_TEXT)];

    // ── Kokoro ────────────────────────────────────────────────────────
    #[cfg(feature = "kokoro")]
    {
        use vox::KokoroBackend;

        if std::path::Path::new(MODEL_PATH).exists() && std::path::Path::new(VOICES_PATH).exists() {
            let tts =
                rt.block_on(async { KokoroBackend::new(MODEL_PATH, VOICES_PATH).await.unwrap() });

            let mut group = c.benchmark_group("tts_synthesize/kokoro");
            group.sample_size(10);

            for (label, text) in &texts {
                let request = TtsRequest {
                    text: text.to_string(),
                    voice: Some("af_heart".into()),
                    seed: None,
                };

                group.bench_with_input(
                    BenchmarkId::from_parameter(label),
                    &request,
                    |b, request| {
                        b.iter(|| {
                            rt.block_on(async { tts.synthesize(black_box(request)).await.unwrap() })
                        });
                    },
                );
            }

            group.finish();
        } else {
            eprintln!(
                "Skipping Kokoro benchmarks: model files not found at {} and {}",
                MODEL_PATH, VOICES_PATH
            );
        }
    }

    // ── Pocket ────────────────────────────────────────────────────────
    #[cfg(feature = "pocket")]
    {
        use vox::PocketTtsBackend;

        match PocketTtsBackend::new() {
            Ok(tts) => {
                let mut group = c.benchmark_group("tts_synthesize/pocket");
                group.sample_size(10);

                for (label, text) in &texts {
                    let request = TtsRequest {
                        text: text.to_string(),
                        voice: None,
                        seed: None,
                    };

                    group.bench_with_input(
                        BenchmarkId::from_parameter(label),
                        &request,
                        |b, request| {
                            b.iter(|| {
                                rt.block_on(async {
                                    tts.synthesize(black_box(request)).await.unwrap()
                                })
                            });
                        },
                    );
                }

                group.finish();
            }
            Err(e) => {
                eprintln!("Skipping Pocket benchmarks: {e}");
            }
        }
    }

    // ── Chatterbox ────────────────────────────────────────────────────
    #[cfg(feature = "chatterbox")]
    {
        use vox::ChatterboxBackend;

        if std::path::Path::new(REFERENCE_WAV).exists() {
            match ChatterboxBackend::new(REFERENCE_WAV) {
                Ok(tts) => {
                    let mut group = c.benchmark_group("tts_synthesize/chatterbox");
                    group.sample_size(10);

                    for (label, text) in &texts {
                        let request = TtsRequest {
                            text: text.to_string(),
                            voice: None,
                            seed: None,
                        };

                        group.bench_with_input(
                            BenchmarkId::from_parameter(label),
                            &request,
                            |b, request| {
                                b.iter(|| {
                                    rt.block_on(async {
                                        tts.synthesize(black_box(request)).await.unwrap()
                                    })
                                });
                            },
                        );
                    }

                    group.finish();
                }
                Err(e) => {
                    eprintln!("Skipping Chatterbox benchmarks: {e}");
                }
            }
        } else {
            eprintln!(
                "Skipping Chatterbox benchmarks: reference WAV not found at {}",
                REFERENCE_WAV
            );
        }
    }
}

/// Benchmark real-time factor (RTF) for each enabled backend.
/// RTF = wall-clock time / audio duration. Values < 1.0 are faster than real-time.
fn bench_tts_rtf(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    #[cfg(feature = "kokoro")]
    {
        use vox::KokoroBackend;

        if std::path::Path::new(MODEL_PATH).exists() && std::path::Path::new(VOICES_PATH).exists() {
            let tts =
                rt.block_on(async { KokoroBackend::new(MODEL_PATH, VOICES_PATH).await.unwrap() });

            let request = TtsRequest {
                text: SHORT_TEXT.to_string(),
                voice: Some("af_heart".into()),
                seed: None,
            };

            let mut group = c.benchmark_group("tts_rtf");
            group.sample_size(10);

            group.bench_function("kokoro", |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = std::time::Instant::now();
                        let output = rt
                            .block_on(async { tts.synthesize(black_box(&request)).await.unwrap() });
                        let elapsed = start.elapsed();
                        let _rtf = rtf(elapsed, output.duration_ms);
                        total += elapsed;
                    }
                    total
                });
            });

            group.finish();
        }
    }

    #[cfg(feature = "pocket")]
    {
        use vox::PocketTtsBackend;

        if let Ok(tts) = PocketTtsBackend::new() {
            let request = TtsRequest {
                text: SHORT_TEXT.to_string(),
                voice: None,
                seed: None,
            };

            let mut group = c.benchmark_group("tts_rtf");
            group.sample_size(10);

            group.bench_function("pocket", |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = std::time::Instant::now();
                        let output = rt
                            .block_on(async { tts.synthesize(black_box(&request)).await.unwrap() });
                        let elapsed = start.elapsed();
                        let _rtf = rtf(elapsed, output.duration_ms);
                        total += elapsed;
                    }
                    total
                });
            });

            group.finish();
        }
    }

    #[cfg(feature = "chatterbox")]
    {
        use vox::ChatterboxBackend;

        if std::path::Path::new(REFERENCE_WAV).exists() {
            if let Ok(tts) = ChatterboxBackend::new(REFERENCE_WAV) {
                let request = TtsRequest {
                    text: SHORT_TEXT.to_string(),
                    voice: None,
                    seed: None,
                };

                let mut group = c.benchmark_group("tts_rtf");
                group.sample_size(10);

                group.bench_function("chatterbox", |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let start = std::time::Instant::now();
                            let output = rt.block_on(async {
                                tts.synthesize(black_box(&request)).await.unwrap()
                            });
                            let elapsed = start.elapsed();
                            let _rtf = rtf(elapsed, output.duration_ms);
                            total += elapsed;
                        }
                        total
                    });
                });

                group.finish();
            }
        }
    }
}

criterion_group!(benches, bench_tts_comparison, bench_tts_rtf);
criterion_main!(benches);
