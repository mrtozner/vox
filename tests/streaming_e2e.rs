//! End-to-end streaming STT test using real sherpa-onnx model.
//! Run with: cargo test --test streaming_e2e --features sherpa -- --nocapture

#[cfg(feature = "sherpa")]
mod tests {
    use std::path::PathBuf;
    use vox::{SherpaStreamingBackend, StreamingSttBackend};

    fn streaming_model_dir() -> PathBuf {
        // Same logic as the CLI models_dir(), but without the dirs crate dep.
        let base = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        #[cfg(target_os = "macos")]
        let models = base.join("Library/Application Support/vox/models");
        #[cfg(not(target_os = "macos"))]
        let models = base.join(".local/share/vox/models");
        models.join("sherpa-streaming")
    }

    fn load_wav_f32(path: &str) -> Vec<f32> {
        let reader = hound::WavReader::open(path).expect("failed to open WAV");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16000, "expected 16kHz WAV");
        match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max = (1i64 << (bits - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max)
                    .collect()
            }
        }
    }

    #[test]
    fn streaming_transcribe_full_utterance() {
        let model_dir = streaming_model_dir();
        if !model_dir.join("encoder.int8.onnx").exists() {
            eprintln!(
                "Skipping: streaming model not found at {}",
                model_dir.display()
            );
            return;
        }

        let wav_path = "/tmp/streaming_test.wav";
        if !std::path::Path::new(wav_path).exists() {
            eprintln!("Skipping: test WAV not found at {wav_path}");
            return;
        }

        let backend = SherpaStreamingBackend::from_transducer(&model_dir)
            .expect("failed to create streaming backend");

        let samples = load_wav_f32(wav_path);
        eprintln!(
            "Loaded {} samples ({:.1}s)",
            samples.len(),
            samples.len() as f64 / 16000.0
        );

        let mut session = backend.create_session().expect("failed to create session");

        // Push audio in 512-sample chunks (like VAD frames)
        let chunk_size = 512;
        let mut partial_count = 0;

        for chunk in samples.chunks(chunk_size) {
            match session.push_audio(chunk, 16000) {
                Ok(Some(text)) => {
                    partial_count += 1;
                    eprintln!("  partial #{partial_count}: {text}");
                }
                Ok(None) => {}
                Err(e) => panic!("push_audio failed: {e}"),
            }
        }

        let result = session.finish().expect("finish failed");
        eprintln!("  final: {}", result.text);
        eprintln!("  partials: {partial_count}");
        eprintln!("  processing_time_ms: {}", result.processing_time_ms);

        assert!(!result.text.is_empty(), "expected non-empty transcription");
        assert!(partial_count > 0, "expected at least one partial result");

        let lower = result.text.to_lowercase();
        assert!(
            lower.contains("hello") || lower.contains("world") || lower.contains("streaming"),
            "expected transcription to contain key words, got: {}",
            result.text
        );
    }

    #[test]
    fn streaming_partials_arrive_during_speech() {
        let model_dir = streaming_model_dir();
        if !model_dir.join("encoder.int8.onnx").exists() {
            eprintln!("Skipping: streaming model not found");
            return;
        }

        let wav_path = "/tmp/streaming_test2.wav";
        if !std::path::Path::new(wav_path).exists() {
            eprintln!("Skipping: test WAV not found at {wav_path}");
            return;
        }

        let backend = SherpaStreamingBackend::from_transducer(&model_dir)
            .expect("failed to create streaming backend");

        let samples = load_wav_f32(wav_path);
        eprintln!(
            "Loaded {} samples ({:.1}s)",
            samples.len(),
            samples.len() as f64 / 16000.0
        );

        let mut session = backend.create_session().expect("failed to create session");

        let mut partials: Vec<String> = Vec::new();
        for chunk in samples.chunks(512) {
            if let Ok(Some(text)) = session.push_audio(chunk, 16000) {
                partials.push(text);
            }
        }

        let result = session.finish().expect("finish failed");
        eprintln!("  partials ({}):", partials.len());
        for (i, p) in partials.iter().enumerate() {
            eprintln!("    {}: {p}", i + 1);
        }
        eprintln!("  final: {}", result.text);

        // Partials should grow monotonically (each contains previous text + more)
        for i in 1..partials.len() {
            assert!(
                partials[i].len() >= partials[i - 1].len(),
                "partial {} ({}) is shorter than partial {} ({})",
                i + 1,
                partials[i],
                i,
                partials[i - 1]
            );
        }

        let lower = result.text.to_lowercase();
        assert!(
            lower.contains("fox") || lower.contains("dog") || lower.contains("quick"),
            "expected key words from 'quick brown fox', got: {}",
            result.text
        );
    }

    #[test]
    fn streaming_batch_mode_also_works() {
        use vox::SttBackend;

        let model_dir = streaming_model_dir();
        if !model_dir.join("encoder.int8.onnx").exists() {
            eprintln!("Skipping: streaming model not found");
            return;
        }

        let wav_path = "/tmp/streaming_test.wav";
        if !std::path::Path::new(wav_path).exists() {
            eprintln!("Skipping: test WAV not found");
            return;
        }

        let backend = SherpaStreamingBackend::from_transducer(&model_dir)
            .expect("failed to create streaming backend");

        let samples = load_wav_f32(wav_path);
        let utterance = vox::Utterance {
            audio: vox::AudioChunk {
                samples,
                sample_rate: 16000,
                channels: 1,
            },
            duration_ms: 3000,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt
            .block_on(backend.transcribe(&utterance))
            .expect("batch transcribe failed");
        eprintln!("  batch result: {}", result.text);
        assert!(
            !result.text.is_empty(),
            "batch transcription should not be empty"
        );
    }
}
