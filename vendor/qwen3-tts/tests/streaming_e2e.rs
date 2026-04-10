//! End-to-end streaming tests.
//!
//! These tests require real model weights and a CUDA GPU.
//! Run with:
//!   cargo test --release --features cuda --test streaming_e2e -- --ignored --nocapture

use qwen3_tts::{Language, Qwen3TTS, Speaker, SynthesisOptions};

fn load_model(model_dir: &str) -> Qwen3TTS {
    let device = qwen3_tts::auto_device().expect("auto_device failed");
    Qwen3TTS::from_pretrained(model_dir, device).expect("model load failed")
}

#[test]
#[ignore = "requires model weights + GPU"]
fn test_streaming_custom_voice() {
    let model = load_model("test_data/models/1.7B-CustomVoice");

    let options = SynthesisOptions {
        seed: Some(42),
        chunk_frames: 10,
        ..Default::default()
    };

    let session = model
        .synthesize_streaming(
            "Hello, this is a streaming test.",
            Speaker::Ryan,
            Language::English,
            options,
        )
        .expect("streaming session creation failed");

    let mut total_samples = 0usize;
    let mut chunk_count = 0usize;
    for chunk_result in session {
        let audio = chunk_result.expect("chunk generation failed");
        assert!(audio.len() > 0, "chunk {chunk_count} was empty");
        assert_eq!(audio.sample_rate, 24000);
        total_samples += audio.len();
        chunk_count += 1;
        println!(
            "  CustomVoice streaming chunk {}: {} samples ({:.2}s)",
            chunk_count,
            audio.len(),
            audio.duration()
        );
    }

    println!(
        "CustomVoice streaming: {} chunks, {:.2}s total",
        chunk_count,
        total_samples as f32 / 24000.0
    );
    assert!(chunk_count > 0, "no chunks generated");
    assert!(total_samples > 0, "no audio samples generated");
}

#[test]
#[ignore = "requires model weights + GPU"]
fn test_streaming_voice_design() {
    let model = load_model("test_data/models/1.7B-VoiceDesign");

    let options = SynthesisOptions {
        seed: Some(42),
        chunk_frames: 10,
        ..Default::default()
    };

    let session = model
        .synthesize_voice_design_streaming(
            "Hello, this is a streaming voice design test.",
            "A deep male voice with a calm and steady tone",
            Language::English,
            options,
        )
        .expect("streaming session creation failed");

    let mut total_samples = 0usize;
    let mut chunk_count = 0usize;
    for chunk_result in session {
        let audio = chunk_result.expect("chunk generation failed");
        assert!(audio.len() > 0, "chunk {chunk_count} was empty");
        assert_eq!(audio.sample_rate, 24000);
        total_samples += audio.len();
        chunk_count += 1;
        println!(
            "  VoiceDesign streaming chunk {}: {} samples ({:.2}s)",
            chunk_count,
            audio.len(),
            audio.duration()
        );
    }

    println!(
        "VoiceDesign streaming: {} chunks, {:.2}s total",
        chunk_count,
        total_samples as f32 / 24000.0
    );
    assert!(chunk_count > 0, "no chunks generated");
    assert!(total_samples > 0, "no audio samples generated");
}

#[test]
#[ignore = "requires model weights + GPU"]
fn test_streaming_matches_non_streaming() {
    // Verify that streaming and non-streaming produce the same number of
    // samples for the same seed (deterministic generation).
    let model = load_model("test_data/models/1.7B-CustomVoice");

    let make_options = || SynthesisOptions {
        seed: Some(123),
        chunk_frames: 10,
        ..Default::default()
    };

    // Non-streaming
    let audio_non_streaming = model
        .synthesize_with_voice(
            "Determinism test.",
            Speaker::Ryan,
            Language::English,
            Some(make_options()),
        )
        .expect("non-streaming synthesis failed");

    // Streaming — collect all chunks
    let session = model
        .synthesize_streaming(
            "Determinism test.",
            Speaker::Ryan,
            Language::English,
            make_options(),
        )
        .expect("streaming session creation failed");

    let mut streaming_samples: Vec<f32> = Vec::new();
    for chunk_result in session {
        let audio = chunk_result.expect("chunk failed");
        streaming_samples.extend_from_slice(&audio.samples);
    }

    // The total sample count should match (same frames decoded, same model)
    println!(
        "Non-streaming: {} samples, Streaming: {} samples",
        audio_non_streaming.len(),
        streaming_samples.len()
    );

    // They won't be sample-identical because streaming decodes in chunks
    // (decoder sees fewer frames of context per chunk), but frame count
    // and total sample count should match.
    assert_eq!(
        audio_non_streaming.len(),
        streaming_samples.len(),
        "streaming and non-streaming produced different sample counts"
    );
}
