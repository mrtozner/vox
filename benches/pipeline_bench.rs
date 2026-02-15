use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vox::{AudioChunk, SileroVad, SttBackend, Utterance, VadBackend, VadEvent, WhisperBackend};

const VAD_MODEL: &str = "models/silero_vad.onnx";
const STT_MODEL: &str = "models/ggml-tiny.en.bin";

fn bench_end_to_end(c: &mut Criterion) {
    if !std::path::Path::new(VAD_MODEL).exists() || !std::path::Path::new(STT_MODEL).exists() {
        eprintln!("Skipping pipeline benchmarks: models not found");
        eprintln!("Run: bash scripts/download_models.sh");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut vad = SileroVad::new(VAD_MODEL).unwrap();
    let stt = WhisperBackend::from_model(STT_MODEL).unwrap();

    // Simulate 3 seconds of audio (93 frames of 512 samples at 16kHz)
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

    let mut group = c.benchmark_group("pipeline");
    group.sample_size(10);

    // VAD-only pass
    group.bench_function("vad_3s_audio", |b| {
        b.iter(|| {
            rt.block_on(async {
                for frame in &frames {
                    vad.process_frame(black_box(frame)).await.unwrap();
                }
            });
            vad.reset();
        });
    });

    // STT on pre-segmented audio
    let utterance = Utterance {
        audio: AudioChunk {
            samples: frames.iter().flat_map(|f| f.samples.clone()).collect(),
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: 3000,
    };

    group.bench_function("stt_3s_utterance", |b| {
        b.iter(|| rt.block_on(async { stt.transcribe(black_box(&utterance)).await.unwrap() }));
    });

    // VAD + STT combined
    group.bench_function("vad_then_stt_3s", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut collected = Vec::new();
                for frame in &frames {
                    let events = vad.process_frame(black_box(frame)).await.unwrap();
                    for event in events {
                        if let VadEvent::SpeechEnd(ref utt) = event {
                            let _ = stt.transcribe(utt).await.unwrap();
                        }
                    }
                    collected.extend_from_slice(&frame.samples);
                }
                // Force transcribe collected audio if VAD didn't trigger
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
        });
    });

    group.finish();
}

criterion_group!(benches, bench_end_to_end);
criterion_main!(benches);
