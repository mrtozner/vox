//! Handler for `vox benchmark` — measure STT/TTS/VAD latency.

use super::models::{ensure_model, model_filename, whisper_download_name};
use std::time::Instant;
use vox::traits::{SttBackend, VadBackend};
use vox::types::{AudioChunk, Utterance};
use vox::{SileroVad, WhisperBackend};

#[cfg(feature = "kokoro")]
const TEST_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const TEST_AUDIO_DURATION_MS: u64 = 3000; // 3 seconds of silence for VAD/STT test

/// Run the benchmark command.
pub async fn run(yes: bool) -> anyhow::Result<()> {
    println!("=== Vox Performance Benchmark ===\n");
    println!("Measuring latency and real-time factor (RTFx) for each backend.");
    println!("RTFx < 1.0 means faster than real-time.\n");

    // Benchmark VAD
    benchmark_vad(yes).await?;
    println!();

    // Benchmark STT (Whisper only for now)
    benchmark_stt_whisper(yes).await?;
    println!();

    // Benchmark TTS (Kokoro if available)
    #[cfg(feature = "kokoro")]
    benchmark_tts_kokoro(yes).await?;

    #[cfg(not(feature = "kokoro"))]
    println!("TTS Benchmark: Skipped (kokoro feature not enabled)");

    println!();
    println!("=== Benchmark Complete ===");

    Ok(())
}

async fn benchmark_vad(yes: bool) -> anyhow::Result<()> {
    println!("VAD Benchmark (Silero VAD)");

    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    print!("  Loading model... ");
    let load_start = Instant::now();
    let mut vad = SileroVad::new(&vad_path)?;
    let load_time = load_start.elapsed();
    println!("{}ms", load_time.as_millis());

    // Create 512 samples frame (32ms at 16kHz - Silero VAD frame size)
    let test_chunk = AudioChunk {
        samples: vec![0.0f32; 512],
        sample_rate: 16000,
        channels: 1,
    };

    // Warmup
    for _ in 0..3 {
        let _ = vad.process_frame(&test_chunk).await;
    }

    // Benchmark 100 frames
    print!("  Processing 100 frames... ");
    let process_start = Instant::now();
    for _ in 0..100 {
        let _ = vad.process_frame(&test_chunk).await;
    }
    let process_time = process_start.elapsed();
    let avg_frame_time = process_time.as_micros() / 100;
    println!("{}μs per frame (avg)", avg_frame_time);

    // Calculate RTFx: wall-clock time / audio duration
    // Each frame is 32ms, so 100 frames = 3200ms
    let audio_duration_ms = 3200;
    let rtfx = process_time.as_secs_f64() * 1000.0 / audio_duration_ms as f64;
    println!(
        "  RTFx: {:.4} ({}x real-time speed)",
        rtfx,
        if rtfx < 1.0 {
            "faster than"
        } else {
            "slower than"
        }
    );

    Ok(())
}

async fn benchmark_stt_whisper(yes: bool) -> anyhow::Result<()> {
    println!("STT Benchmark (Whisper tiny.en)");

    let whisper_file = model_filename("tiny.en");
    let whisper_name = whisper_download_name("tiny.en");
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    print!("  Loading model... ");
    let load_start = Instant::now();
    let stt = WhisperBackend::from_model(&whisper_path)?;
    let load_time = load_start.elapsed();
    println!("{}ms", load_time.as_millis());

    // Create 3 seconds of silence for testing (48000 samples at 16kHz)
    let test_utterance = Utterance {
        audio: AudioChunk {
            samples: vec![0.0f32; 48000],
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: TEST_AUDIO_DURATION_MS,
        #[cfg(feature = "diarization")]
        speaker_id: None,
    };

    // Warmup
    print!("  Warmup... ");
    let _ = stt.transcribe(&test_utterance).await;
    println!("done");

    // Benchmark 10 iterations
    print!("  Processing 10 iterations... ");
    let process_start = Instant::now();
    for _ in 0..10 {
        let _ = stt.transcribe(&test_utterance).await;
    }
    let process_time = process_start.elapsed();
    let avg_time = process_time.as_millis() / 10;
    println!("{}ms per transcription (avg)", avg_time);

    // Calculate RTFx: wall-clock time / audio duration
    // Each audio is 3 seconds, so 10 iterations = 30 seconds
    let audio_duration_ms = TEST_AUDIO_DURATION_MS * 10;
    let rtfx = process_time.as_secs_f64() * 1000.0 / audio_duration_ms as f64;
    println!(
        "  RTFx: {:.4} ({}x real-time speed)",
        rtfx,
        if rtfx < 1.0 {
            "faster than"
        } else {
            "slower than"
        }
    );

    Ok(())
}

#[cfg(feature = "kokoro")]
async fn benchmark_tts_kokoro(yes: bool) -> anyhow::Result<()> {
    use vox::{KokoroBackend, TtsBackend};

    println!("TTS Benchmark (Kokoro)");

    let model_path = ensure_model("kokoro", "kokoro-v1.0.onnx", yes).await?;
    let voices_path = ensure_model("kokoro-voices", "voices.bin", yes).await?;

    print!("  Loading model... ");
    let load_start = Instant::now();
    let tts = KokoroBackend::new(&model_path, &voices_path).await?;
    let load_time = load_start.elapsed();
    println!("{}ms", load_time.as_millis());

    let request = vox::types::TtsRequest {
        text: TEST_TEXT.to_string(),
        voice: Some("af_heart".to_string()),
        seed: None,
    };

    // Warmup
    print!("  Warmup... ");
    let _ = tts.synthesize(&request).await;
    println!("done");

    // Benchmark 10 iterations
    print!("  Processing 10 iterations... ");
    let mut total_audio_duration_ms = 0u64;
    let process_start = Instant::now();
    for _ in 0..10 {
        let output = tts.synthesize(&request).await?;
        total_audio_duration_ms += output.duration_ms;
    }
    let process_time = process_start.elapsed();
    let avg_time = process_time.as_millis() / 10;
    println!("{}ms per synthesis (avg)", avg_time);

    // Calculate RTFx: wall-clock time / audio duration
    let rtfx = process_time.as_secs_f64() * 1000.0 / total_audio_duration_ms as f64;
    println!(
        "  RTFx: {:.4} ({}x real-time speed)",
        rtfx,
        if rtfx < 1.0 {
            "faster than"
        } else {
            "slower than"
        }
    );

    Ok(())
}
