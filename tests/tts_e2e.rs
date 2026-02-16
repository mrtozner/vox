//! End-to-end TTS backend tests using real models.
//!
//! Piper tests:   cargo test --test tts_e2e --features piper -- --nocapture
//! Chatterbox:    cargo test --test tts_e2e --features chatterbox -- --nocapture
//! Both:          cargo test --test tts_e2e --features piper,chatterbox -- --nocapture

// ---------------------------------------------------------------------------
// Piper TTS
// ---------------------------------------------------------------------------
#[cfg(feature = "piper")]
mod piper_tests {
    use std::path::PathBuf;
    use vox::traits::TtsBackend;
    use vox::tts::PiperBackend;
    use vox::types::TtsRequest;

    fn piper_config_path() -> Option<PathBuf> {
        let base = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        #[cfg(target_os = "macos")]
        let models = base.join("Library/Application Support/vox/models");
        #[cfg(not(target_os = "macos"))]
        let models = base.join(".local/share/vox/models");
        let config = models.join("piper/en_US-lessac-medium.onnx.json");
        if config.exists() { Some(config) } else { None }
    }

    #[test]
    fn piper_loads_and_reports_backend_name() {
        let Some(config) = piper_config_path() else {
            eprintln!("Skipping: piper model not found");
            return;
        };
        let backend = PiperBackend::new(&config).expect("failed to load piper");
        assert_eq!(backend.backend_name(), "piper");
    }

    #[test]
    fn piper_lists_voices() {
        let Some(config) = piper_config_path() else {
            eprintln!("Skipping: piper model not found");
            return;
        };
        let backend = PiperBackend::new(&config).expect("failed to load piper");
        let voices = backend.list_voices();
        assert!(!voices.is_empty(), "expected at least one voice");
        eprintln!("  voices: {}", voices.len());
        for v in &voices {
            eprintln!("    {} — {} ({})", v.id, v.name, v.language);
        }
    }

    #[tokio::test]
    async fn piper_synthesizes_short_text() {
        let Some(config) = piper_config_path() else {
            eprintln!("Skipping: piper model not found");
            return;
        };
        let backend = PiperBackend::new(&config).expect("failed to load piper");

        let output = backend
            .synthesize(&TtsRequest {
                text: "Hello world.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("synthesis failed");

        eprintln!(
            "  samples: {} | rate: {} | duration: {}ms",
            output.audio.samples.len(),
            output.audio.sample_rate,
            output.duration_ms,
        );
        assert!(
            output.audio.samples.len() > 1000,
            "expected non-trivial audio output"
        );
        assert_eq!(output.audio.channels, 1);
        assert!(output.audio.sample_rate > 0);
        assert!(output.duration_ms > 0);
    }

    /// Test that longer text produces more audio than shorter text.
    /// Uses a single backend to avoid espeak-ng thread-safety issues.
    #[tokio::test]
    async fn piper_longer_text_produces_more_audio() {
        let Some(config) = piper_config_path() else {
            eprintln!("Skipping: piper model not found");
            return;
        };
        let backend = PiperBackend::new(&config).expect("failed to load piper");

        let short = backend
            .synthesize(&TtsRequest {
                text: "Hi.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("short synthesis failed");

        let long = backend
            .synthesize(&TtsRequest {
                text: "The quick brown fox jumps over the lazy dog near the riverbank.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("long synthesis failed");

        eprintln!(
            "  short: {} samples ({}ms) | long: {} samples ({}ms)",
            short.audio.samples.len(),
            short.duration_ms,
            long.audio.samples.len(),
            long.duration_ms,
        );
        assert!(
            long.audio.samples.len() > short.audio.samples.len(),
            "longer text should produce more audio"
        );
    }

    #[tokio::test]
    async fn piper_output_is_valid_audio() {
        let Some(config) = piper_config_path() else {
            eprintln!("Skipping: piper model not found");
            return;
        };
        let backend = PiperBackend::new(&config).expect("failed to load piper");

        let output = backend
            .synthesize(&TtsRequest {
                text: "Testing audio validity.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("synthesis failed");

        // Check samples are finite and within reasonable range
        let all_finite = output.audio.samples.iter().all(|s| s.is_finite());
        assert!(all_finite, "all samples should be finite");

        let max_abs = output
            .audio
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        eprintln!("  max absolute sample value: {max_abs:.4}");
        assert!(max_abs > 0.001, "audio should not be silent");
        assert!(
            max_abs < 10.0,
            "audio samples should be reasonable amplitude"
        );
    }

    #[test]
    fn piper_rejects_missing_config() {
        let result = PiperBackend::new("/nonexistent/path.onnx.json");
        assert!(result.is_err(), "should fail with missing config");
    }
}

// ---------------------------------------------------------------------------
// Chatterbox TTS
// ---------------------------------------------------------------------------
#[cfg(feature = "chatterbox")]
mod chatterbox_tests {
    use std::path::PathBuf;
    use vox::traits::TtsBackend;
    use vox::tts::ChatterboxBackend;
    use vox::types::TtsRequest;

    fn chatterbox_model_dir() -> Option<PathBuf> {
        // Check the HF cache for the model snapshot
        let base = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        let hf_cache =
            base.join(".cache/huggingface/hub/models--ResembleAI--chatterbox-turbo-ONNX");
        if !hf_cache.exists() {
            return None;
        }
        // Find the first snapshot with a real/ dir containing copied model files
        let snapshots = hf_cache.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let real_dir = entry.path().join("real");
                if real_dir.join("tokenizer.json").exists() {
                    return Some(real_dir);
                }
            }
        }
        None
    }

    fn reference_wav() -> Option<PathBuf> {
        let path = PathBuf::from("/tmp/vox_cbx_reference.wav");
        if path.exists() { Some(path) } else { None }
    }

    #[test]
    fn chatterbox_loads_from_model_dir() {
        let Some(model_dir) = chatterbox_model_dir() else {
            eprintln!("Skipping: chatterbox model not found in HF cache");
            return;
        };
        let Some(ref_wav) = reference_wav() else {
            eprintln!("Skipping: reference WAV not found at /tmp/vox_cbx_reference.wav");
            return;
        };
        let backend =
            ChatterboxBackend::from_model_dir(&model_dir, &ref_wav).expect("failed to load");
        assert_eq!(backend.backend_name(), "chatterbox");
    }

    #[test]
    fn chatterbox_lists_voices() {
        let Some(model_dir) = chatterbox_model_dir() else {
            eprintln!("Skipping: chatterbox model not found");
            return;
        };
        let Some(ref_wav) = reference_wav() else {
            eprintln!("Skipping: reference WAV not found");
            return;
        };
        let backend =
            ChatterboxBackend::from_model_dir(&model_dir, &ref_wav).expect("failed to load");
        let voices = backend.list_voices();
        assert!(!voices.is_empty(), "expected at least one voice entry");
        for v in &voices {
            eprintln!("    {} — {} ({})", v.id, v.name, v.language);
        }
    }

    #[tokio::test]
    async fn chatterbox_synthesizes_short_text() {
        let Some(model_dir) = chatterbox_model_dir() else {
            eprintln!("Skipping: chatterbox model not found");
            return;
        };
        let Some(ref_wav) = reference_wav() else {
            eprintln!("Skipping: reference WAV not found");
            return;
        };
        let backend =
            ChatterboxBackend::from_model_dir(&model_dir, &ref_wav).expect("failed to load");

        let t = std::time::Instant::now();
        let output = backend
            .synthesize(&TtsRequest {
                text: "Hello world.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("synthesis failed");

        let elapsed = t.elapsed().as_millis();
        eprintln!(
            "  samples: {} | rate: {} | duration: {}ms | wall: {}ms",
            output.audio.samples.len(),
            output.audio.sample_rate,
            output.duration_ms,
            elapsed,
        );
        assert!(
            output.audio.samples.len() > 1000,
            "expected non-trivial audio"
        );
        assert_eq!(output.audio.channels, 1);
        assert_eq!(output.audio.sample_rate, 24000);
        assert!(output.duration_ms > 0);
    }

    #[tokio::test]
    async fn chatterbox_output_is_valid_audio() {
        let Some(model_dir) = chatterbox_model_dir() else {
            eprintln!("Skipping: chatterbox model not found");
            return;
        };
        let Some(ref_wav) = reference_wav() else {
            eprintln!("Skipping: reference WAV not found");
            return;
        };
        let backend =
            ChatterboxBackend::from_model_dir(&model_dir, &ref_wav).expect("failed to load");

        let output = backend
            .synthesize(&TtsRequest {
                text: "Testing audio output quality.".into(),
                voice: None,
                seed: None,
            })
            .await
            .expect("synthesis failed");

        let all_finite = output.audio.samples.iter().all(|s| s.is_finite());
        assert!(all_finite, "all samples should be finite");

        let max_abs = output
            .audio
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        eprintln!("  max absolute sample value: {max_abs:.4}");
        assert!(max_abs > 0.001, "audio should not be silent");
        assert!(max_abs < 10.0, "samples should be reasonable amplitude");
    }

    #[test]
    fn chatterbox_rejects_missing_reference() {
        let Some(model_dir) = chatterbox_model_dir() else {
            eprintln!("Skipping: chatterbox model not found");
            return;
        };
        let backend = ChatterboxBackend::from_model_dir(&model_dir, "/nonexistent/ref.wav");
        // Loading should succeed (reference is only needed at synthesis time)
        // but synthesis should fail
        if let Ok(backend) = backend {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(backend.synthesize(&TtsRequest {
                text: "test".into(),
                voice: None,
                seed: None,
            }));
            assert!(result.is_err(), "should fail with missing reference WAV");
        }
    }

    #[test]
    fn chatterbox_rejects_bad_dtype() {
        let config = vox::ChatterboxConfig {
            dtype: "invalid_dtype".into(),
            ..Default::default()
        };
        let result = ChatterboxBackend::with_config(config);
        assert!(result.is_err(), "should fail with unknown dtype");
    }
}
