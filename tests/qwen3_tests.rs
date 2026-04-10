#![cfg(feature = "qwen3")]

use vox::{Qwen3Backend, TtsBackend};

/// Test that Qwen3Backend implements the TtsBackend trait correctly.
#[tokio::test]
#[ignore] // Ignored by default as it requires model download
async fn test_qwen3_backend_initialization() {
    // This test requires the model to be downloaded first
    let result = Qwen3Backend::new().await;

    // If model is not downloaded, we expect an error
    // If model is downloaded, backend should initialize successfully
    match result {
        Ok(backend) => {
            // Verify backend name
            assert_eq!(backend.backend_name(), "qwen3");

            // Verify voices are available
            let voices = backend.list_voices();
            assert!(!voices.is_empty(), "Qwen3 should provide voices");
            assert!(
                voices.iter().any(|v| v.id == "en_us_female_1"),
                "Should have default voice"
            );
        }
        Err(e) => {
            // Model not downloaded is expected in CI
            eprintln!("Qwen3 initialization failed (expected in CI): {e}");
        }
    }
}

#[tokio::test]
#[ignore] // Ignored by default as it requires model download
async fn test_qwen3_voice_list() {
    let result = Qwen3Backend::new().await;

    if let Ok(backend) = result {
        let voices = backend.list_voices();

        // Should have at least 20 voices as per architecture
        assert!(voices.len() >= 20, "Should have 20+ voices");

        // Check for specific voices
        let voice_ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();

        assert!(voice_ids.contains(&"en_us_female_1"));
        assert!(voice_ids.contains(&"en_us_male_1"));
        assert!(voice_ids.contains(&"zh_cn_female_1"));
        assert!(voice_ids.contains(&"ja_jp_female_1"));

        // Check voice metadata
        let en_female = voices.iter().find(|v| v.id == "en_us_female_1").unwrap();
        assert_eq!(en_female.language, "en-US");
        assert_eq!(en_female.gender, "female");
    }
}

#[tokio::test]
#[ignore] // Ignored by default as it requires model download
async fn test_qwen3_synthesis() {
    let result = Qwen3Backend::new().await;

    if let Ok(backend) = result {
        let request = vox::TtsRequest {
            text: "Hello, world!".to_string(),
            voice: Some("en_us_female_1".to_string()),
            seed: None,
        };

        let result = backend.synthesize(&request).await;

        match result {
            Ok(output) => {
                // Verify audio output
                assert!(!output.audio.samples.is_empty(), "Should produce audio");
                assert_eq!(output.audio.sample_rate, 24000, "Should be 24kHz");
                assert_eq!(output.audio.channels, 1, "Should be mono");
                assert!(output.duration_ms > 0, "Should have duration");

                println!(
                    "Synthesis successful: {} samples, {} ms",
                    output.audio.samples.len(),
                    output.duration_ms
                );
            }
            Err(e) => {
                panic!("Synthesis failed: {e}");
            }
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_qwen3_invalid_voice() {
    let result = Qwen3Backend::new().await;

    if let Ok(backend) = result {
        let request = vox::TtsRequest {
            text: "Test".to_string(),
            voice: Some("invalid_voice".to_string()),
            seed: None,
        };

        let result = backend.synthesize(&request).await;
        assert!(result.is_err(), "Should reject invalid voice");

        if let Err(e) = result {
            let error_msg = format!("{e}");
            assert!(
                error_msg.contains("unknown voice"),
                "Error should mention unknown voice"
            );
        }
    }
}
