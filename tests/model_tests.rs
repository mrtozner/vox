//! Integration tests that use real model files.
//!
//! These tests are SKIPPED by default. To run them:
//!
//!   VOX_TEST_MODELS=models cargo test --features whisper,silero -- --ignored
//!
//! Download models first: bash scripts/download_models.sh

use std::path::Path;

fn models_dir() -> Option<String> {
    std::env::var("VOX_TEST_MODELS").ok()
}

fn skip_unless_models() -> Option<String> {
    let dir = models_dir()?;
    if Path::new(&dir).exists() {
        Some(dir)
    } else {
        None
    }
}

// ---------- Silero VAD real model tests ----------

#[cfg(feature = "silero")]
mod vad_real {
    use super::*;
    use vox::{AudioChunk, SileroVad, VadBackend, VadConfig, VadEvent};

    #[tokio::test]
    #[ignore]
    async fn silero_loads_model() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let vad = SileroVad::new(&model);
        assert!(vad.is_ok(), "Failed to load Silero VAD: {:?}", vad.err());
    }

    #[tokio::test]
    #[ignore]
    async fn silero_detects_silence() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let mut vad = SileroVad::new(&model).unwrap();

        let silence = AudioChunk {
            samples: vec![0.0; 512],
            sample_rate: 16000,
            channels: 1,
        };

        let events = vad.process_frame(&silence).await.unwrap();
        assert!(
            events.iter().all(|e| matches!(e, VadEvent::Silence)),
            "Expected silence on zero signal, got: {:?}",
            events
        );
    }

    #[tokio::test]
    #[ignore]
    async fn silero_custom_config() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let vad = SileroVad::with_config(
            &model,
            VadConfig {
                speech_threshold: 0.3,
                silence_duration_ms: 300,
                min_speech_ms: 100,
            },
        );
        assert!(vad.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn silero_frame_size_and_rate() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let vad = SileroVad::new(&model).unwrap();
        assert_eq!(vad.frame_size(), 512);
        assert_eq!(vad.sample_rate(), 16000);
    }

    #[tokio::test]
    #[ignore]
    async fn silero_processes_many_frames() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let mut vad = SileroVad::new(&model).unwrap();

        let frame = AudioChunk {
            samples: vec![0.0; 512],
            sample_rate: 16000,
            channels: 1,
        };

        // Process 100 frames (~3.2 seconds) without error
        for _ in 0..100 {
            let events = vad.process_frame(&frame).await;
            assert!(events.is_ok());
        }
    }

    #[tokio::test]
    #[ignore]
    async fn silero_reset_and_reprocess() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/silero_vad.onnx");
        let mut vad = SileroVad::new(&model).unwrap();

        let frame = AudioChunk {
            samples: vec![0.0; 512],
            sample_rate: 16000,
            channels: 1,
        };

        // Process, reset, process again
        vad.process_frame(&frame).await.unwrap();
        vad.reset();
        let events = vad.process_frame(&frame).await.unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    #[ignore]
    fn silero_model_not_found() {
        let result = SileroVad::new("/nonexistent/model.onnx");
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error for nonexistent model"),
        };
        assert!(err.contains("model not found"), "unexpected error: {err}");
    }
}

// ---------- Whisper STT real model tests ----------

#[cfg(feature = "whisper")]
mod stt_real {
    use super::*;
    use vox::{AudioChunk, SttBackend, Utterance, WhisperBackend, WhisperModel};

    #[test]
    #[ignore]
    fn whisper_loads_model() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/ggml-tiny.en.bin");
        let stt = WhisperBackend::from_model(&model);
        assert!(stt.is_ok(), "Failed to load Whisper: {:?}", stt.err());
    }

    #[tokio::test]
    #[ignore]
    async fn whisper_transcribes_silence() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/ggml-tiny.en.bin");
        let stt = WhisperBackend::from_model(&model).unwrap();

        let utterance = Utterance {
            audio: AudioChunk {
                samples: vec![0.0; 16000], // 1 second silence
                sample_rate: 16000,
                channels: 1,
            },
            duration_ms: 1000,
        };

        let result = stt.transcribe(&utterance).await.unwrap();
        // Silence should produce empty or near-empty text
        assert!(
            result.text.len() < 50,
            "Expected minimal text for silence, got: '{}'",
            result.text
        );
        assert!(result.processing_time_ms > 0);
    }

    #[tokio::test]
    #[ignore]
    async fn whisper_returns_processing_time() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let model = format!("{dir}/ggml-tiny.en.bin");
        let stt = WhisperBackend::from_model(&model).unwrap();

        let utterance = Utterance {
            audio: AudioChunk {
                samples: vec![0.0; 48000], // 3 seconds
                sample_rate: 16000,
                channels: 1,
            },
            duration_ms: 3000,
        };

        let result = stt.transcribe(&utterance).await.unwrap();
        assert_eq!(result.duration_ms, 3000);
        assert!(result.processing_time_ms > 0);
    }

    #[test]
    #[ignore]
    fn whisper_from_dir() {
        let dir = skip_unless_models().expect("VOX_TEST_MODELS not set");
        let stt = WhisperBackend::from_dir(&dir, WhisperModel::TinyEn);
        assert!(stt.is_ok(), "Failed to load from dir: {:?}", stt.err());
    }

    #[test]
    #[ignore]
    fn whisper_model_not_found() {
        let result = WhisperBackend::from_model("/nonexistent/model.bin");
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error for nonexistent model"),
        };
        assert!(err.contains("model not found"), "unexpected error: {err}");
    }
}
