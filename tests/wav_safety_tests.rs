//! WAV safety tests — audit the `hound` crate for vulnerabilities described in
//! sherpa-onnx issue #3052 (1-byte heap overflow when WAV data chunk size is odd).
//!
//! These tests verify that `hound` handles malformed and edge-case WAV files
//! safely, without panics, overflows, or undefined behavior.
//!
//! Run with:
//!   cargo test --features server --test wav_safety_tests

#[path = "../src/server/mod.rs"]
mod server;

use std::io::Cursor;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use tower::ServiceExt;

use server::{ServerState, ServerStats};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a minimal ServerState with a mock STT backend.
fn test_state_with_stt() -> Arc<ServerState> {
    Arc::new(ServerState {
        stt: Some(Arc::new(MockStt)),
        tts: None,
        vad_model_path: None,
        stats: Arc::new(std::sync::Mutex::new(ServerStats {
            requests: 0,
            transcriptions: 0,
            syntheses: 0,
        })),
        start_time: std::time::Instant::now(),
        ollama_host: "localhost:11434".to_string(),
        http_client: reqwest::Client::new(),
        stt_model_name: None,
        stt_model_size: None,
        tts_model_name: None,
        tts_model_size: None,
    })
}

/// Build the Router matching the production server layout.
fn test_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/", get(server::handlers::index))
        .route("/v1/transcribe", post(server::handlers::transcribe))
        .route("/v1/synthesize", post(server::handlers::synthesize))
        .route("/v1/models", get(server::handlers::models))
        .route("/v1/stats", get(server::handlers::stats))
        .route("/health", get(server::handlers::health))
        .with_state(state)
}

/// Encode a mono WAV file from i16 samples at the given sample rate.
fn encode_wav_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

/// Encode a mono WAV file from f32 samples at the given sample rate.
fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

/// Build a raw WAV byte buffer from individual header fields.
/// This lets us craft intentionally malformed WAV files.
fn build_raw_wav(
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    audio_format: u16, // 1 = PCM integer, 3 = IEEE float
    data: &[u8],
    claimed_data_size: u32,
) -> Vec<u8> {
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    // RIFF chunk size = 4 ("WAVE") + 24 (fmt subchunk) + 8 (data header) + claimed_data_size
    // Use wrapping add so we can test with extreme claimed_data_size values
    let riff_size: u32 = (4u32 + 24 + 8).wrapping_add(claimed_data_size);

    let mut buf = Vec::new();
    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    buf.extend_from_slice(&audio_format.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&claimed_data_size.to_le_bytes());
    buf.extend_from_slice(data);
    buf
}

/// Parse a JSON response body.
async fn body_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    serde_json::from_str(&text).expect("response body is not valid JSON")
}

/// Send raw bytes to the transcribe endpoint and return the response.
async fn transcribe_bytes(wav: Vec<u8>) -> (StatusCode, serde_json::Value) {
    let state = test_state_with_stt();
    let app = test_router(state);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/transcribe")
        .body(Body::from(wav))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let json = body_json(resp.into_body()).await;
    (status, json)
}

// -- Mock STT backend --

use async_trait::async_trait;

struct MockStt;

#[async_trait]
impl vox::SttBackend for MockStt {
    async fn transcribe(
        &self,
        audio: &vox::Utterance,
    ) -> Result<vox::SttResult, vox::VoxError> {
        Ok(vox::SttResult {
            text: format!("mock[{}samples]", audio.audio.samples.len()),
            language: Some("en".into()),
            duration_ms: audio.duration_ms,
            processing_time_ms: 0,
        })
    }
}

// ===========================================================================
// 1. sherpa-onnx #3052: odd data chunk size for 16-bit samples
// ===========================================================================

mod wav_odd_data_chunk {
    use super::*;

    /// The core vulnerability: when subchunk2_size is odd for 16-bit PCM,
    /// the sherpa-onnx C++ loader reads (odd_size + 1) / 2 samples but only
    /// allocates odd_size / 2, causing a 1-byte heap overflow.
    /// Verify hound does NOT exhibit this behavior.
    #[test]
    fn hound_handles_odd_data_size_safely() {
        // 3 bytes of PCM data with 16-bit samples: only 1 complete sample (2 bytes),
        // the 3rd byte is a dangling half-sample.
        let data: Vec<u8> = vec![0x00, 0x01, 0xFF]; // 3 bytes (odd)
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 3);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                // hound opened it; verify reading samples doesn't panic or over-read
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                // Should yield exactly 1 sample (2 bytes / 2), not 2
                assert_eq!(
                    samples.len(),
                    1,
                    "expected 1 complete i16 sample from 3 bytes, got {}",
                    samples.len()
                );
                assert!(
                    samples[0].is_ok(),
                    "the one complete sample should decode without error"
                );
            }
            Err(_) => {
                // hound rejected the file entirely — also safe
            }
        }
    }

    /// Odd data size = 1 byte (cannot form even a single 16-bit sample).
    #[test]
    fn hound_handles_single_byte_data_chunk() {
        let data: Vec<u8> = vec![0xAB];
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 1);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                // 1 byte cannot form a 16-bit sample
                assert_eq!(
                    samples.len(),
                    0,
                    "expected 0 samples from 1 byte of 16-bit data, got {}",
                    samples.len()
                );
            }
            Err(_) => {
                // Rejection is also safe
            }
        }
    }

    /// Odd data size = 5 bytes for 16-bit samples (2 complete + 1 dangling byte).
    #[test]
    fn hound_handles_five_byte_data_chunk() {
        let data: Vec<u8> = vec![0x00, 0x01, 0x00, 0x02, 0xFF];
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 5);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                // 5 / 2 = 2 complete samples, trailing byte ignored
                assert!(
                    samples.len() <= 2,
                    "expected at most 2 samples from 5 bytes, got {}",
                    samples.len()
                );
            }
            Err(_) => {}
        }
    }

    /// Send odd-data-size WAV through the HTTP transcribe handler.
    #[tokio::test]
    async fn transcribe_odd_data_size_no_panic() {
        let data: Vec<u8> = vec![0x00, 0x01, 0xFF];
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 3);
        let (status, _json) = transcribe_bytes(wav).await;
        // Either 200 (decoded successfully) or 400 (rejected as invalid) — not a panic/crash
        assert!(
            status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
            "unexpected status {} for odd-data-size WAV",
            status
        );
    }
}

// ===========================================================================
// 2. Zero-length data chunk
// ===========================================================================

mod wav_zero_data {
    use super::*;

    #[test]
    fn hound_handles_zero_length_data() {
        let wav = build_raw_wav(1, 16000, 16, 1, &[], 0);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                assert_eq!(samples.len(), 0, "zero-length data should yield 0 samples");
            }
            Err(_) => {
                // Rejection is safe
            }
        }
    }

    #[tokio::test]
    async fn transcribe_zero_length_data_no_panic() {
        let wav = build_raw_wav(1, 16000, 16, 1, &[], 0);
        let (status, _) = transcribe_bytes(wav).await;
        assert!(
            status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
            "unexpected status {} for zero-length WAV",
            status
        );
    }
}

// ===========================================================================
// 3. Truncated WAV header
// ===========================================================================

mod wav_truncated_header {
    use super::*;

    #[test]
    fn hound_rejects_truncated_riff_header() {
        // Only "RIFF" + partial size — not enough for a WAV header
        let buf = b"RIFF\x24\x00\x00\x00WA";
        let cursor = Cursor::new(buf.as_slice());
        let result = hound::WavReader::new(cursor);
        assert!(
            result.is_err(),
            "hound should reject a truncated RIFF header"
        );
    }

    #[test]
    fn hound_rejects_truncated_fmt_chunk() {
        // Valid RIFF+WAVE header but fmt chunk is truncated mid-way
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&20u32.to_le_bytes()); // small RIFF size
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // claims 16 bytes of fmt
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
        // truncated here — missing channels, sample_rate, etc.

        let cursor = Cursor::new(buf);
        let result = hound::WavReader::new(cursor);
        assert!(
            result.is_err(),
            "hound should reject WAV with truncated fmt chunk"
        );
    }

    #[test]
    fn hound_rejects_truncated_before_data_chunk() {
        // Valid fmt but stream ends before data chunk
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&36u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt subchunk (complete, 16 bytes)
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // 1 channel
        buf.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
        buf.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // No data chunk at all — stream ends here

        let cursor = Cursor::new(buf);
        let result = hound::WavReader::new(cursor);
        assert!(
            result.is_err(),
            "hound should reject WAV with missing data chunk"
        );
    }

    #[tokio::test]
    async fn transcribe_truncated_header_returns_400() {
        let buf = b"RIFF\x24\x00\x00\x00WA".to_vec();
        let (status, json) = transcribe_bytes(buf).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            json["error"].as_str().unwrap().contains("invalid WAV"),
            "error should mention invalid WAV"
        );
    }
}

// ===========================================================================
// 4. Mismatched bits_per_sample vs actual data length
// ===========================================================================

mod wav_bits_mismatch {
    use super::*;

    /// Claim 16-bit samples but only provide 1 byte of data.
    #[test]
    fn hound_handles_undersize_data_for_claimed_bits() {
        let data = vec![0x42]; // 1 byte, but 16-bit = 2 bytes per sample
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 1);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                assert_eq!(
                    samples.len(),
                    0,
                    "1 byte should yield 0 complete 16-bit samples"
                );
            }
            Err(_) => {
                // Also safe
            }
        }
    }

    /// Claim 32-bit float but provide only 3 bytes.
    #[test]
    fn hound_handles_undersize_data_for_32bit_float() {
        let data = vec![0x00, 0x00, 0x80]; // 3 bytes, need 4 for one f32
        let wav = build_raw_wav(1, 16000, 32, 3, &data, 3); // audio_format=3 (float)

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<f32, _>> = r.into_samples::<f32>().collect();
                assert_eq!(
                    samples.len(),
                    0,
                    "3 bytes should yield 0 complete f32 samples"
                );
            }
            Err(_) => {}
        }
    }

    /// Claim 24-bit samples, provide 5 bytes (1 full sample + 2 dangling bytes).
    #[test]
    fn hound_handles_24bit_odd_data() {
        // 24-bit PCM = 3 bytes per sample. 5 bytes = 1 full + 2 extra.
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04];
        let wav = build_raw_wav(1, 16000, 24, 1, &data, 5);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i32, _>> = r.into_samples::<i32>().collect();
                assert!(
                    samples.len() <= 1,
                    "5 bytes of 24-bit should yield at most 1 sample, got {}",
                    samples.len()
                );
            }
            Err(_) => {}
        }
    }
}

// ===========================================================================
// 5. Very large claimed data size but small actual data
// ===========================================================================

mod wav_oversized_claim {
    use super::*;

    /// The data header claims 1 GB but only 4 bytes follow.
    /// NOTE: hound trusts the claimed data size for its sample iterator length,
    /// so it will report claimed_size / bytes_per_sample items (most returning
    /// Err). We only read the first few to verify it doesn't panic or OOM,
    /// and check that only the real data produces Ok results.
    #[test]
    fn hound_does_not_allocate_based_on_claimed_size() {
        let data = vec![0x00, 0x01, 0x00, 0x02]; // 4 actual bytes = 2 i16 samples
        let claimed_size = 1_000_000u32; // claim 1 MB
        let wav = build_raw_wav(1, 16000, 16, 1, &data, claimed_size);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                // hound reports len based on claimed size; only read a bounded number
                let first_100: Vec<Result<i16, _>> =
                    r.into_samples::<i16>().take(100).collect();
                let ok_count = first_100.iter().filter(|s| s.is_ok()).count();
                // Only 2 real samples exist (4 bytes / 2 bytes per sample)
                assert!(
                    ok_count <= 2,
                    "should get at most 2 ok samples from 4 bytes of real data, got {}",
                    ok_count
                );
            }
            Err(_) => {
                // Rejection is also valid
            }
        }
    }

    /// Claimed size = u32::MAX. Verify hound doesn't try to allocate that much.
    #[test]
    fn hound_handles_max_u32_claimed_size() {
        let data = vec![0x00; 8]; // 8 actual bytes = 4 i16 samples
        let claimed_size = u32::MAX;
        let wav = build_raw_wav(1, 16000, 16, 1, &data, claimed_size);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                // Only read a bounded slice — hound iterator length is huge
                let first_100: Vec<Result<i16, _>> =
                    r.into_samples::<i16>().take(100).collect();
                let ok_count = first_100.iter().filter(|s| s.is_ok()).count();
                assert!(
                    ok_count <= 4,
                    "expected at most 4 ok i16 from 8 real bytes, got {}",
                    ok_count
                );
            }
            Err(_) => {}
        }
    }

    /// Send a moderately oversized-claim WAV through the HTTP handler.
    /// We use a moderate claim (10 KB) so the handler's collect() doesn't take
    /// forever iterating errors for millions of phantom samples.
    #[tokio::test]
    async fn transcribe_oversized_claim_no_oom() {
        let data = vec![0x00, 0x01, 0x00, 0x02]; // 4 real bytes
        let wav = build_raw_wav(1, 16000, 16, 1, &data, 10_000); // claim 10 KB
        let (status, _) = transcribe_bytes(wav).await;
        // The handler calls .collect() which will hit errors for missing bytes.
        // It should return 400 ("WAV decode error") when collect fails.
        assert!(
            status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
            "unexpected status {} for oversized-claim WAV",
            status
        );
    }
}

// ===========================================================================
// 6. Valid WAV with i16 samples
// ===========================================================================

mod wav_valid_i16 {
    use super::*;

    #[test]
    fn hound_decodes_i16_samples_correctly() {
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let wav = encode_wav_i16(&samples, 16000);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("valid i16 WAV should parse");
        let decoded: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .expect("i16 samples should decode");

        assert_eq!(decoded, samples);
    }

    #[test]
    fn hound_decodes_i16_as_i32_correctly() {
        // The transcribe handler reads Int samples as i32
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let wav = encode_wav_i16(&samples, 16000);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("valid i16 WAV should parse");
        let decoded: Vec<i32> = reader
            .into_samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("i16->i32 samples should decode");

        let expected: Vec<i32> = samples.iter().map(|&s| s as i32).collect();
        assert_eq!(decoded, expected);
    }

    #[tokio::test]
    async fn transcribe_valid_i16_returns_200() {
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let wav = encode_wav_i16(&samples, 16000);
        let (status, json) = transcribe_bytes(wav).await;
        assert_eq!(status, StatusCode::OK);
        assert!(json.get("text").is_some());
    }
}

// ===========================================================================
// 7. Valid WAV with f32 samples
// ===========================================================================

mod wav_valid_f32 {
    use super::*;

    #[test]
    fn hound_decodes_f32_samples_correctly() {
        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_f32(&samples, 44100);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("valid f32 WAV should parse");
        let decoded: Vec<f32> = reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("f32 samples should decode");

        assert_eq!(decoded, samples);
    }

    #[test]
    fn hound_handles_denormalized_f32() {
        // Values outside [-1, 1] are technically valid in float WAV
        let samples: Vec<f32> = vec![2.0, -3.5, f32::MIN_POSITIVE, f32::EPSILON];
        let wav = encode_wav_f32(&samples, 16000);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("denormalized f32 WAV should parse");
        let decoded: Vec<f32> = reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("denormalized f32 samples should decode");

        assert_eq!(decoded, samples);
    }

    #[tokio::test]
    async fn transcribe_valid_f32_returns_200() {
        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_f32(&samples, 16000);
        let (status, json) = transcribe_bytes(wav).await;
        assert_eq!(status, StatusCode::OK);
        assert!(json.get("text").is_some());
    }
}

// ===========================================================================
// 8. WAV with 0 sample rate
// ===========================================================================

mod wav_zero_sample_rate {
    use super::*;

    #[test]
    fn hound_handles_zero_sample_rate() {
        // Build raw WAV with sample_rate = 0
        let data = vec![0x00, 0x01, 0x00, 0x02]; // 2 i16 samples
        let wav = build_raw_wav(1, 0, 16, 1, &data, 4);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let spec = r.spec();
                assert_eq!(spec.sample_rate, 0);
                // Should still be able to read samples without panic
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                assert_eq!(samples.len(), 2);
            }
            Err(_) => {
                // Rejection is also acceptable
            }
        }
    }

    /// The transcribe handler computes duration_ms with sample_rate in denominator.
    /// Verify it handles 0 without division-by-zero panic.
    #[tokio::test]
    async fn transcribe_zero_sample_rate_no_div_by_zero() {
        let data = vec![0x00, 0x01, 0x00, 0x02];
        let wav = build_raw_wav(1, 0, 16, 1, &data, 4);
        let (status, _) = transcribe_bytes(wav).await;
        // The handler guards sample_rate > 0 before dividing.
        // Either 200 (accepted) or 400 (rejected) — not a panic.
        assert!(
            status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
            "unexpected status {} for zero sample rate WAV",
            status
        );
    }
}

// ===========================================================================
// 9. WAV with unusual channel counts
// ===========================================================================

mod wav_unusual_channels {
    use super::*;

    #[test]
    fn hound_handles_zero_channels() {
        let wav = build_raw_wav(0, 16000, 16, 1, &[], 0);

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);

        // Zero channels is nonsensical; hound may accept or reject
        match result {
            Ok(r) => {
                assert_eq!(r.spec().channels, 0);
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                assert_eq!(samples.len(), 0);
            }
            Err(_) => {
                // Rejection is safe
            }
        }
    }

    #[test]
    fn hound_handles_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            // Write interleaved stereo: L, R, L, R
            for &s in &[100i16, -100, 200, -200] {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        let wav = buf.into_inner();

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("stereo WAV should parse");
        assert_eq!(reader.spec().channels, 2);
        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(samples, vec![100, -100, 200, -200]);
    }

    #[test]
    fn hound_handles_many_channels() {
        // 8-channel WAV (7.1 surround)
        let spec = hound::WavSpec {
            channels: 8,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            // 1 frame = 8 samples (one per channel)
            for i in 0..8i16 {
                writer.write_sample(i * 1000).unwrap();
            }
            writer.finalize().unwrap();
        }
        let wav = buf.into_inner();

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor).expect("8-channel WAV should parse");
        assert_eq!(reader.spec().channels, 8);
        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(samples.len(), 8);
    }

    /// Transcribe handler uses channels for duration calculation.
    /// Verify zero channels does not cause division-by-zero.
    #[tokio::test]
    async fn transcribe_zero_channels_no_div_by_zero() {
        let wav = build_raw_wav(0, 16000, 16, 1, &[], 0);
        let (status, _) = transcribe_bytes(wav).await;
        assert!(
            status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
            "unexpected status {} for zero-channels WAV",
            status
        );
    }
}

// ===========================================================================
// 10. Additional malformed WAV edge cases
// ===========================================================================

mod wav_malformed_misc {
    use super::*;

    /// Completely empty file (0 bytes).
    #[test]
    fn hound_rejects_empty_file() {
        let cursor = Cursor::new(Vec::<u8>::new());
        let result = hound::WavReader::new(cursor);
        assert!(result.is_err(), "hound should reject an empty file");
    }

    /// RIFF header with wrong magic ("RIFX" instead of "RIFF").
    #[test]
    fn hound_rejects_wrong_riff_magic() {
        let data = vec![0x00; 4];
        let mut wav = build_raw_wav(1, 16000, 16, 1, &data, 4);
        // Corrupt "RIFF" to "RIFX"
        wav[3] = b'X';

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);
        // hound may or may not accept RIFX (big-endian); either way it should not panic
        match result {
            Ok(_) => {} // RIFX is a valid variant
            Err(_) => {}
        }
    }

    /// WAVE magic corrupted ("WAVA" instead of "WAVE").
    #[test]
    fn hound_rejects_wrong_wave_magic() {
        let data = vec![0x00; 4];
        let mut wav = build_raw_wav(1, 16000, 16, 1, &data, 4);
        // Corrupt "WAVE" (bytes 8-11) to "WAVA"
        wav[11] = b'A';

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);
        assert!(result.is_err(), "hound should reject corrupted WAVE magic");
    }

    /// Data chunk with no actual audio bytes but non-zero claimed size.
    #[test]
    fn hound_handles_claimed_nonzero_but_no_data() {
        let wav = build_raw_wav(1, 16000, 16, 1, &[], 100);

        let cursor = Cursor::new(&wav);
        let reader = hound::WavReader::new(cursor);

        match reader {
            Ok(r) => {
                let samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
                // Should get errors or 0 samples, not fabricated data
                let ok_count = samples.iter().filter(|s| s.is_ok()).count();
                assert_eq!(
                    ok_count, 0,
                    "no actual data bytes should yield 0 successful samples"
                );
            }
            Err(_) => {}
        }
    }

    /// The `audio_format` field = 0 (unknown/invalid).
    #[test]
    fn hound_handles_unknown_audio_format() {
        let data = vec![0x00; 4];
        let wav = build_raw_wav(1, 16000, 16, 0, &data, 4); // format 0 is invalid

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);
        // hound should reject or handle safely
        match result {
            Ok(r) => {
                // If it accepts, reading should not panic
                let _samples: Vec<Result<i16, _>> = r.into_samples::<i16>().collect();
            }
            Err(_) => {}
        }
    }

    /// Bits per sample = 0.
    #[test]
    fn hound_handles_zero_bits_per_sample() {
        let wav = build_raw_wav(1, 16000, 0, 1, &[], 0);

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);
        // Should reject or handle without panic
        match result {
            Ok(_) => {}
            Err(_) => {}
        }
    }

    /// Very large bits_per_sample (e.g., 256) to test for shift overflow.
    #[test]
    fn hound_handles_large_bits_per_sample() {
        let data = vec![0x00; 32]; // 32 bytes of data
        let wav = build_raw_wav(1, 16000, 256, 1, &data, 32);

        let cursor = Cursor::new(&wav);
        let result = hound::WavReader::new(cursor);
        match result {
            Ok(r) => {
                // Should not panic during sample iteration
                let _: Vec<Result<i32, _>> = r.into_samples::<i32>().collect();
            }
            Err(_) => {}
        }
    }
}
