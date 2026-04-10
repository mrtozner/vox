//! Unit tests for the Vox engine, pipeline builder, types, and mock backends.
//!
#![allow(clippy::field_reassign_with_default)]
//! These tests run with default features and require no model files
//! or audio hardware. All backend dependencies are replaced with mocks.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use vox::{
    AudioChunk, PipelineStats, StreamingSttBackend, SttBackend, SttResult, SttSession, TtsBackend,
    TtsOutput, TtsRequest, Utterance, VadBackend, VadEvent, Vox, VoxConfig, VoxError,
};

// ===========================================================================
// Mock backends
// ===========================================================================

/// A mock VAD that emits SpeechEnd after `trigger_at` frames.
struct MockVad {
    frame_count: usize,
    trigger_at: usize,
}

impl MockVad {
    fn new(trigger_at: usize) -> Self {
        Self {
            frame_count: 0,
            trigger_at,
        }
    }
}

#[async_trait]
impl VadBackend for MockVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        self.frame_count += 1;
        if self.frame_count == self.trigger_at {
            let utterance = Utterance {
                audio: frame.clone(),
                duration_ms: 500,
            };
            Ok(vec![VadEvent::SpeechStart, VadEvent::SpeechEnd(utterance)])
        } else {
            Ok(vec![VadEvent::Silence])
        }
    }

    fn reset(&mut self) {
        self.frame_count = 0;
    }

    fn frame_size(&self) -> usize {
        512
    }

    fn sample_rate(&self) -> u32 {
        16000
    }
}

/// A mock VAD that always returns silence.
struct SilentVad;

#[async_trait]
impl VadBackend for SilentVad {
    async fn process_frame(&mut self, _frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        Ok(vec![VadEvent::Silence])
    }

    fn reset(&mut self) {}

    fn frame_size(&self) -> usize {
        480
    }

    fn sample_rate(&self) -> u32 {
        16000
    }
}

/// A mock VAD that returns an error on every frame.
struct FailingVad;

#[async_trait]
impl VadBackend for FailingVad {
    async fn process_frame(&mut self, _frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        Err(VoxError::Vad("simulated VAD failure".into()))
    }

    fn reset(&mut self) {}

    fn frame_size(&self) -> usize {
        512
    }

    fn sample_rate(&self) -> u32 {
        16000
    }
}

/// A mock STT that returns a fixed transcription.
struct MockStt {
    response: String,
    language: Option<String>,
}

impl MockStt {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            language: Some("en".into()),
        }
    }

    fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }
}

#[async_trait]
impl SttBackend for MockStt {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        Ok(SttResult {
            text: self.response.clone(),
            language: self.language.clone(),
            duration_ms: audio.duration_ms,
            processing_time_ms: 5,
        })
    }
}

/// A mock STT that returns empty text.
struct EmptyStt;

#[async_trait]
impl SttBackend for EmptyStt {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        Ok(SttResult {
            text: String::new(),
            language: None,
            duration_ms: audio.duration_ms,
            processing_time_ms: 0,
        })
    }
}

/// A mock STT that always fails.
struct FailingStt;

#[async_trait]
impl SttBackend for FailingStt {
    async fn transcribe(&self, _audio: &Utterance) -> Result<SttResult, VoxError> {
        Err(VoxError::Stt("simulated STT failure".into()))
    }
}

/// A mock TTS that returns fixed audio output.
struct MockTts {
    output_sample_rate: u32,
    output_channels: u16,
    output_duration_ms: u64,
}

impl MockTts {
    fn new() -> Self {
        Self {
            output_sample_rate: 24000,
            output_channels: 1,
            output_duration_ms: 1000,
        }
    }

    fn with_sample_rate(mut self, rate: u32) -> Self {
        self.output_sample_rate = rate;
        self
    }

    fn with_channels(mut self, ch: u16) -> Self {
        self.output_channels = ch;
        self
    }
}

#[async_trait]
impl TtsBackend for MockTts {
    async fn synthesize(&self, _request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        let num_samples =
            (self.output_sample_rate as u64 * self.output_duration_ms / 1000) as usize;
        Ok(TtsOutput {
            audio: AudioChunk {
                samples: vec![0.0; num_samples],
                sample_rate: self.output_sample_rate,
                channels: self.output_channels,
            },
            duration_ms: self.output_duration_ms,
        })
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

/// A mock TTS that always errors.
struct FailingTts;

#[async_trait]
impl TtsBackend for FailingTts {
    async fn synthesize(&self, _request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        Err(VoxError::Tts("simulated TTS failure".into()))
    }

    fn backend_name(&self) -> &str {
        "failing-mock"
    }
}

// ===========================================================================
// Helper: build a test AudioChunk
// ===========================================================================

fn test_chunk(num_samples: usize, sample_rate: u32, channels: u16) -> AudioChunk {
    AudioChunk {
        samples: vec![0.0; num_samples],
        sample_rate,
        channels,
    }
}

fn test_utterance(duration_ms: u64) -> Utterance {
    Utterance {
        audio: test_chunk(16000, 16000, 1),
        duration_ms,
    }
}

// ===========================================================================
// 1. VoxBuilder validation
// ===========================================================================

#[test]
fn builder_without_vad_fails_with_no_vad_error() {
    let result = Vox::builder().stt(MockStt::new("hello")).build();
    match result {
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("no VAD"),
                "expected error about missing VAD, got: {msg}"
            );
        }
        Ok(_) => panic!("expected build to fail without VAD"),
    }
}

#[test]
fn builder_without_stt_fails_with_no_stt_error() {
    let result = Vox::builder().vad(MockVad::new(1)).build();
    match result {
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("no STT"),
                "expected error about missing STT, got: {msg}"
            );
        }
        Ok(_) => panic!("expected build to fail without STT"),
    }
}

#[test]
fn builder_without_any_backends_fails() {
    let result = Vox::builder().build();
    assert!(result.is_err(), "builder with no backends should fail");
}

#[test]
fn builder_error_is_no_vad_when_both_missing() {
    // VAD is checked first, so missing both should yield NoVad.
    let result = Vox::builder().build();
    match result {
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("no VAD"),
                "expected NoVad error when both are missing, got: {msg}"
            );
        }
        Ok(_) => panic!("expected build to fail when both backends are missing"),
    }
}

// NOTE: Tests that call .build() with both VAD and STT present will attempt
// to open an audio device via cpal. On headless CI or environments without
// a microphone, this will fail with an Audio error -- not a builder logic
// error. We test the success path indirectly via the mock pipeline tests
// and via the on_utterance callback setup test below.

#[test]
#[cfg_attr(target_os = "windows", ignore)]
fn builder_with_vad_and_stt_reaches_audio_init() {
    // When both backends are present, the builder should get past the
    // NoVad/NoStt checks and attempt audio device initialization.
    // In CI without a mic this returns an Audio error, not NoVad/NoStt.
    let result = Vox::builder()
        .vad(MockVad::new(1))
        .stt(MockStt::new("test"))
        .build();
    match result {
        Ok(_) => {} // audio device available -- success
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("no VAD") && !msg.contains("no STT"),
                "should not fail with missing-backend error, got: {msg}"
            );
        }
    }
}

#[test]
#[cfg_attr(target_os = "windows", ignore)]
fn builder_with_vad_stt_and_tts_reaches_audio_init() {
    let result = Vox::builder()
        .vad(MockVad::new(1))
        .stt(MockStt::new("test"))
        .tts(MockTts::new())
        .build();
    match result {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("no VAD") && !msg.contains("no STT"),
                "should not fail with missing-backend error, got: {msg}"
            );
        }
    }
}

#[test]
#[cfg_attr(target_os = "windows", ignore)]
fn builder_with_on_utterance_callback_reaches_audio_init() {
    let result = Vox::builder()
        .vad(MockVad::new(1))
        .stt(MockStt::new("test"))
        .on_utterance(|_result, _ctx| {})
        .build();
    match result {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("no VAD") && !msg.contains("no STT"),
                "should not fail with missing-backend error, got: {msg}"
            );
        }
    }
}

#[test]
fn builder_default_is_same_as_new() {
    // VoxBuilder::default() and VoxBuilder::new() should both start empty.
    // Both should fail with NoVad when built without backends.
    let r1 = Vox::builder().build();
    let r2 = vox::VoxBuilder::default().build();
    let msg1 = match r1 {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected Vox::builder().build() to fail"),
    };
    let msg2 = match r2 {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected VoxBuilder::default().build() to fail"),
    };
    assert_eq!(msg1, msg2);
}

// ===========================================================================
// 2. VoxConfig defaults and overrides
// ===========================================================================

#[test]
fn vox_config_default_sample_rate_is_16000() {
    let config = VoxConfig::default();
    assert_eq!(config.sample_rate, 16000);
}

#[test]
fn vox_config_default_channels_is_mono() {
    let config = VoxConfig::default();
    assert_eq!(config.channels, 1);
}

#[test]
fn vox_config_default_tts_is_disabled() {
    let config = VoxConfig::default();
    assert!(!config.enable_tts);
}

#[test]
fn vox_config_custom_sample_rate() {
    let config = VoxConfig {
        sample_rate: 44100,
        ..VoxConfig::default()
    };
    assert_eq!(config.sample_rate, 44100);
    assert_eq!(config.channels, 1); // other fields retain default
}

#[test]
fn vox_config_custom_channels() {
    let config = VoxConfig {
        channels: 2,
        ..VoxConfig::default()
    };
    assert_eq!(config.channels, 2);
    assert_eq!(config.sample_rate, 16000);
}

#[test]
fn vox_config_enable_tts() {
    let config = VoxConfig {
        enable_tts: true,
        ..VoxConfig::default()
    };
    assert!(config.enable_tts);
}

#[test]
fn vox_config_full_custom() {
    let config = VoxConfig {
        sample_rate: 48000,
        channels: 2,
        enable_tts: true,
    };
    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.channels, 2);
    assert!(config.enable_tts);
}

#[test]
fn vox_config_clone_is_independent() {
    let original = VoxConfig {
        sample_rate: 22050,
        channels: 1,
        enable_tts: false,
    };
    let mut cloned = original.clone();
    cloned.sample_rate = 48000;
    cloned.enable_tts = true;

    assert_eq!(original.sample_rate, 22050);
    assert!(!original.enable_tts);
    assert_eq!(cloned.sample_rate, 48000);
    assert!(cloned.enable_tts);
}

#[test]
fn vox_config_debug_format() {
    let config = VoxConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("sample_rate"));
    assert!(debug.contains("channels"));
    assert!(debug.contains("enable_tts"));
}

// ===========================================================================
// 3. Mock-based pipeline tests (trait behavior)
// ===========================================================================

#[tokio::test]
async fn mock_vad_emits_silence_before_trigger() {
    let mut vad = MockVad::new(5);
    let frame = test_chunk(512, 16000, 1);

    for _ in 0..4 {
        let events = vad.process_frame(&frame).await.unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], VadEvent::Silence));
    }
}

#[tokio::test]
async fn mock_vad_emits_speech_at_trigger() {
    let mut vad = MockVad::new(1);
    let frame = test_chunk(512, 16000, 1);

    let events = vad.process_frame(&frame).await.unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], VadEvent::SpeechStart));
    assert!(matches!(events[1], VadEvent::SpeechEnd(_)));
}

#[tokio::test]
async fn mock_vad_speech_end_contains_utterance_with_correct_duration() {
    let mut vad = MockVad::new(1);
    let frame = test_chunk(512, 16000, 1);

    let events = vad.process_frame(&frame).await.unwrap();
    if let VadEvent::SpeechEnd(utterance) = &events[1] {
        assert_eq!(utterance.duration_ms, 500);
        assert_eq!(utterance.audio.sample_rate, 16000);
        assert_eq!(utterance.audio.channels, 1);
        assert_eq!(utterance.audio.samples.len(), 512);
    } else {
        panic!("expected SpeechEnd event");
    }
}

#[tokio::test]
async fn mock_vad_returns_silence_after_trigger() {
    let mut vad = MockVad::new(2);
    let frame = test_chunk(512, 16000, 1);

    // Frame 1: silence
    let e1 = vad.process_frame(&frame).await.unwrap();
    assert!(matches!(e1[0], VadEvent::Silence));

    // Frame 2: trigger
    let e2 = vad.process_frame(&frame).await.unwrap();
    assert_eq!(e2.len(), 2);

    // Frame 3: silence again (past trigger)
    let e3 = vad.process_frame(&frame).await.unwrap();
    assert_eq!(e3.len(), 1);
    assert!(matches!(e3[0], VadEvent::Silence));
}

#[tokio::test]
async fn mock_vad_reset_restarts_frame_counter() {
    let mut vad = MockVad::new(2);
    let frame = test_chunk(512, 16000, 1);

    // Advance one frame
    vad.process_frame(&frame).await.unwrap();
    assert_eq!(vad.frame_count, 1);

    // Reset
    vad.reset();
    assert_eq!(vad.frame_count, 0);

    // After reset, trigger should fire at frame 2 again
    let e1 = vad.process_frame(&frame).await.unwrap();
    assert!(matches!(e1[0], VadEvent::Silence));

    let e2 = vad.process_frame(&frame).await.unwrap();
    assert_eq!(e2.len(), 2);
    assert!(matches!(e2[0], VadEvent::SpeechStart));
}

#[tokio::test]
async fn mock_vad_frame_size_and_sample_rate() {
    let vad = MockVad::new(1);
    assert_eq!(vad.frame_size(), 512);
    assert_eq!(vad.sample_rate(), 16000);
}

#[tokio::test]
async fn silent_vad_never_triggers() {
    let mut vad = SilentVad;
    let frame = test_chunk(480, 16000, 1);

    for _ in 0..100 {
        let events = vad.process_frame(&frame).await.unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], VadEvent::Silence));
    }
}

#[tokio::test]
async fn failing_vad_returns_error() {
    let mut vad = FailingVad;
    let frame = test_chunk(512, 16000, 1);

    let result = vad.process_frame(&frame).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("VAD"));
}

#[tokio::test]
async fn mock_stt_returns_configured_text() {
    let stt = MockStt::new("hello world");
    let utterance = test_utterance(1000);

    let result = stt.transcribe(&utterance).await.unwrap();
    assert_eq!(result.text, "hello world");
}

#[tokio::test]
async fn mock_stt_returns_configured_language() {
    let stt = MockStt::new("bonjour").with_language("fr");
    let utterance = test_utterance(500);

    let result = stt.transcribe(&utterance).await.unwrap();
    assert_eq!(result.language, Some("fr".into()));
}

#[tokio::test]
async fn mock_stt_preserves_utterance_duration() {
    let stt = MockStt::new("test");
    let utterance = test_utterance(2500);

    let result = stt.transcribe(&utterance).await.unwrap();
    assert_eq!(result.duration_ms, 2500);
}

#[tokio::test]
async fn mock_stt_reports_processing_time() {
    let stt = MockStt::new("test");
    let utterance = test_utterance(1000);

    let result = stt.transcribe(&utterance).await.unwrap();
    assert_eq!(result.processing_time_ms, 5);
}

#[tokio::test]
async fn empty_stt_returns_empty_text() {
    let stt = EmptyStt;
    let utterance = test_utterance(1000);

    let result = stt.transcribe(&utterance).await.unwrap();
    assert!(result.text.is_empty());
    assert!(result.language.is_none());
}

#[tokio::test]
async fn failing_stt_returns_error() {
    let stt = FailingStt;
    let utterance = test_utterance(1000);

    let result = stt.transcribe(&utterance).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("STT"));
}

#[tokio::test]
async fn mock_tts_produces_audio_output() {
    let tts = MockTts::new();
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };

    let output = tts.synthesize(&request).await.unwrap();
    assert_eq!(output.duration_ms, 1000);
    assert_eq!(output.audio.sample_rate, 24000);
    assert_eq!(output.audio.channels, 1);
    // 24000 Hz * 1 second = 24000 samples
    assert_eq!(output.audio.samples.len(), 24000);
}

#[tokio::test]
async fn mock_tts_custom_sample_rate() {
    let tts = MockTts::new().with_sample_rate(44100);
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };

    let output = tts.synthesize(&request).await.unwrap();
    assert_eq!(output.audio.sample_rate, 44100);
    assert_eq!(output.audio.samples.len(), 44100); // 44100 * 1s
}

#[tokio::test]
async fn mock_tts_stereo_output() {
    let tts = MockTts::new().with_channels(2);
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };

    let output = tts.synthesize(&request).await.unwrap();
    assert_eq!(output.audio.channels, 2);
}

#[tokio::test]
async fn mock_tts_backend_name() {
    let tts = MockTts::new();
    assert_eq!(tts.backend_name(), "mock");
}

#[tokio::test]
async fn failing_tts_returns_error() {
    let tts = FailingTts;
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };

    let result = tts.synthesize(&request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("TTS"));
}

#[tokio::test]
async fn failing_tts_backend_name() {
    let tts = FailingTts;
    assert_eq!(tts.backend_name(), "failing-mock");
}

#[tokio::test]
async fn vad_speech_end_feeds_into_stt_transcription() {
    // Simulate the pipeline: VAD emits SpeechEnd -> STT transcribes.
    let mut vad = MockVad::new(1);
    let stt = MockStt::new("recognized speech");
    let frame = test_chunk(512, 16000, 1);

    let events = vad.process_frame(&frame).await.unwrap();

    let mut transcriptions = Vec::new();
    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let result = stt.transcribe(&utterance).await.unwrap();
            transcriptions.push(result);
        }
    }

    assert_eq!(transcriptions.len(), 1);
    assert_eq!(transcriptions[0].text, "recognized speech");
    assert_eq!(transcriptions[0].duration_ms, 500);
}

#[tokio::test]
async fn pipeline_skips_empty_transcriptions() {
    // Simulate the engine behavior: empty STT results are not forwarded.
    let mut vad = MockVad::new(1);
    let stt = EmptyStt;
    let frame = test_chunk(512, 16000, 1);

    let events = vad.process_frame(&frame).await.unwrap();

    let mut forwarded = Vec::new();
    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let result = stt.transcribe(&utterance).await.unwrap();
            if !result.text.is_empty() {
                forwarded.push(result);
            }
        }
    }

    assert!(
        forwarded.is_empty(),
        "empty transcriptions should be skipped"
    );
}

#[tokio::test]
async fn pipeline_callback_fires_on_transcription() {
    // Simulate the callback mechanism from the engine.
    let mut vad = MockVad::new(1);
    let stt = MockStt::new("callback test");
    let frame = test_chunk(512, 16000, 1);

    let callback_count = Arc::new(Mutex::new(0u32));
    let callback_text = Arc::new(Mutex::new(String::new()));

    let events = vad.process_frame(&frame).await.unwrap();

    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let result = stt.transcribe(&utterance).await.unwrap();
            if !result.text.is_empty() {
                let mut count = callback_count.lock().unwrap();
                *count += 1;
                let mut text = callback_text.lock().unwrap();
                *text = result.text.clone();
            }
        }
    }

    assert_eq!(*callback_count.lock().unwrap(), 1);
    assert_eq!(*callback_text.lock().unwrap(), "callback test");
}

#[tokio::test]
async fn pipeline_multiple_utterances_accumulate() {
    // Simulate multiple speech events across frames.
    let mut vad = MockVad::new(1);
    let stt = MockStt::new("word");
    let frame = test_chunk(512, 16000, 1);

    let mut total_utterances = 0u32;

    // Process frame 1: triggers speech
    let events = vad.process_frame(&frame).await.unwrap();
    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let result = stt.transcribe(&utterance).await.unwrap();
            if !result.text.is_empty() {
                total_utterances += 1;
            }
        }
    }

    // Reset and process another trigger
    vad.reset();
    let events = vad.process_frame(&frame).await.unwrap();
    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let result = stt.transcribe(&utterance).await.unwrap();
            if !result.text.is_empty() {
                total_utterances += 1;
            }
        }
    }

    assert_eq!(total_utterances, 2);
}

#[tokio::test]
async fn full_pipeline_vad_stt_tts_round_trip() {
    // End-to-end mock pipeline: VAD -> STT -> TTS.
    let mut vad = MockVad::new(1);
    let stt = MockStt::new("speak this back");
    let tts = MockTts::new();
    let frame = test_chunk(512, 16000, 1);

    let events = vad.process_frame(&frame).await.unwrap();

    for event in events {
        if let VadEvent::SpeechEnd(utterance) = event {
            let stt_result = stt.transcribe(&utterance).await.unwrap();
            assert_eq!(stt_result.text, "speak this back");

            let tts_request = TtsRequest {
                text: stt_result.text.clone(),
                voice: None,
                seed: None,
            };
            let tts_output = tts.synthesize(&tts_request).await.unwrap();
            assert_eq!(tts_output.duration_ms, 1000);
            assert!(!tts_output.audio.samples.is_empty());
        }
    }
}

// ===========================================================================
// 4. AudioChunk operations
// ===========================================================================

#[test]
fn audio_chunk_duration_calculation_mono_16khz() {
    let chunk = test_chunk(16000, 16000, 1);
    // Duration = samples / sample_rate = 16000 / 16000 = 1.0 second
    let duration_secs = chunk.samples.len() as f64 / chunk.sample_rate as f64;
    assert!((duration_secs - 1.0).abs() < f64::EPSILON);
}

#[test]
fn audio_chunk_duration_calculation_half_second() {
    let chunk = test_chunk(8000, 16000, 1);
    let duration_secs = chunk.samples.len() as f64 / chunk.sample_rate as f64;
    assert!((duration_secs - 0.5).abs() < f64::EPSILON);
}

#[test]
fn audio_chunk_duration_calculation_48khz() {
    let chunk = test_chunk(48000, 48000, 1);
    let duration_secs = chunk.samples.len() as f64 / chunk.sample_rate as f64;
    assert!((duration_secs - 1.0).abs() < f64::EPSILON);
}

#[test]
fn audio_chunk_mono_has_one_channel() {
    let chunk = test_chunk(16000, 16000, 1);
    assert_eq!(chunk.channels, 1);
}

#[test]
fn audio_chunk_stereo_has_two_channels() {
    let chunk = test_chunk(32000, 16000, 2);
    assert_eq!(chunk.channels, 2);
    // For stereo, the effective duration considers interleaved samples.
    // 32000 interleaved samples / 2 channels / 16000 Hz = 1.0 second.
    let duration_secs =
        chunk.samples.len() as f64 / (chunk.sample_rate as f64 * chunk.channels as f64);
    assert!((duration_secs - 1.0).abs() < f64::EPSILON);
}

#[test]
fn audio_chunk_empty_samples() {
    let chunk = test_chunk(0, 16000, 1);
    assert!(chunk.samples.is_empty());
    let duration_secs = chunk.samples.len() as f64 / chunk.sample_rate as f64;
    assert!((duration_secs - 0.0).abs() < f64::EPSILON);
}

#[test]
fn audio_chunk_clone_is_independent() {
    let chunk = AudioChunk {
        samples: vec![0.5, -0.5, 1.0],
        sample_rate: 16000,
        channels: 1,
    };

    let mut cloned = chunk.clone();
    cloned.samples[0] = 0.0;
    cloned.sample_rate = 44100;

    assert_eq!(chunk.samples[0], 0.5);
    assert_eq!(chunk.sample_rate, 16000);
    assert_eq!(cloned.samples[0], 0.0);
    assert_eq!(cloned.sample_rate, 44100);
}

#[test]
fn audio_chunk_debug_format() {
    let chunk = test_chunk(3, 16000, 1);
    let debug = format!("{:?}", chunk);
    assert!(debug.contains("AudioChunk"));
    assert!(debug.contains("sample_rate"));
    assert!(debug.contains("channels"));
}

#[test]
fn utterance_stores_audio_and_duration() {
    let utterance = Utterance {
        audio: AudioChunk {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: 750,
    };

    assert_eq!(utterance.duration_ms, 750);
    assert_eq!(utterance.audio.samples.len(), 3);
    assert_eq!(utterance.audio.sample_rate, 16000);
}

#[test]
fn utterance_clone_is_independent() {
    let utterance = test_utterance(1000);
    let mut cloned = utterance.clone();
    cloned.duration_ms = 2000;

    assert_eq!(utterance.duration_ms, 1000);
    assert_eq!(cloned.duration_ms, 2000);
}

// ===========================================================================
// 5. TtsRequest and TtsOutput
// ===========================================================================

#[test]
fn tts_request_with_voice() {
    let request = TtsRequest {
        text: "hello".into(),
        voice: Some("af_heart".into()),
        seed: None,
    };

    assert_eq!(request.text, "hello");
    assert_eq!(request.voice, Some("af_heart".into()));
}

#[test]
fn tts_request_without_voice() {
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };

    assert_eq!(request.text, "hello");
    assert!(request.voice.is_none());
}

#[test]
fn tts_request_clone_is_independent() {
    let request = TtsRequest {
        text: "original".into(),
        voice: Some("voice1".into()),
        seed: None,
    };

    let mut cloned = request.clone();
    cloned.text = "modified".into();
    cloned.voice = Some("voice2".into());

    assert_eq!(request.text, "original");
    assert_eq!(request.voice, Some("voice1".into()));
    assert_eq!(cloned.text, "modified");
    assert_eq!(cloned.voice, Some("voice2".into()));
}

#[test]
fn tts_request_debug_format() {
    let request = TtsRequest {
        text: "test".into(),
        voice: Some("v".into()),
        seed: None,
    };
    let debug = format!("{:?}", request);
    assert!(debug.contains("TtsRequest"));
    assert!(debug.contains("test"));
}

#[test]
fn tts_output_has_correct_sample_rate_and_channels() {
    let output = TtsOutput {
        audio: AudioChunk {
            samples: vec![0.0; 24000],
            sample_rate: 24000,
            channels: 1,
        },
        duration_ms: 1000,
    };

    assert_eq!(output.audio.sample_rate, 24000);
    assert_eq!(output.audio.channels, 1);
    assert_eq!(output.duration_ms, 1000);
    assert_eq!(output.audio.samples.len(), 24000);
}

#[test]
fn tts_output_stereo() {
    let output = TtsOutput {
        audio: AudioChunk {
            samples: vec![0.0; 48000],
            sample_rate: 24000,
            channels: 2,
        },
        duration_ms: 1000,
    };

    assert_eq!(output.audio.channels, 2);
    // 48000 interleaved samples / 2 channels / 24000 Hz = 1 second
    let duration_secs = output.audio.samples.len() as f64
        / (output.audio.sample_rate as f64 * output.audio.channels as f64);
    assert!((duration_secs - 1.0).abs() < f64::EPSILON);
}

#[test]
fn tts_output_clone_is_independent() {
    let output = TtsOutput {
        audio: AudioChunk {
            samples: vec![0.5; 100],
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: 500,
    };

    let mut cloned = output.clone();
    cloned.duration_ms = 1000;
    cloned.audio.sample_rate = 44100;

    assert_eq!(output.duration_ms, 500);
    assert_eq!(output.audio.sample_rate, 16000);
    assert_eq!(cloned.duration_ms, 1000);
    assert_eq!(cloned.audio.sample_rate, 44100);
}

// ===========================================================================
// 6. PipelineStats
// ===========================================================================

#[test]
fn pipeline_stats_default_is_zero() {
    let stats = PipelineStats::default();
    assert_eq!(stats.utterance_count, 0);
    assert_eq!(stats.avg_stt_latency_ms, 0.0);
    assert_eq!(stats.uptime_secs, 0);
}

#[test]
fn pipeline_stats_manual_update() {
    let mut stats = PipelineStats::default();
    stats.utterance_count = 5;
    stats.avg_stt_latency_ms = 42.5;
    stats.uptime_secs = 120;

    assert_eq!(stats.utterance_count, 5);
    assert!((stats.avg_stt_latency_ms - 42.5).abs() < f64::EPSILON);
    assert_eq!(stats.uptime_secs, 120);
}

#[test]
fn pipeline_stats_clone_is_independent() {
    let stats = PipelineStats {
        utterance_count: 10,
        avg_stt_latency_ms: 50.0,
        uptime_secs: 300,
    };

    let mut cloned = stats.clone();
    cloned.utterance_count = 20;
    cloned.avg_stt_latency_ms = 100.0;

    assert_eq!(stats.utterance_count, 10);
    assert!((stats.avg_stt_latency_ms - 50.0).abs() < f64::EPSILON);
    assert_eq!(cloned.utterance_count, 20);
}

#[test]
fn pipeline_stats_running_average_simulation() {
    // Simulate the running average formula from engine.rs:
    //   avg = avg * ((n-1)/n) + new_value / n
    let mut stats = PipelineStats::default();

    let latencies = [10u64, 20, 30, 40, 50];

    for &latency in &latencies {
        stats.utterance_count += 1;
        let n = stats.utterance_count as f64;
        stats.avg_stt_latency_ms = stats.avg_stt_latency_ms * ((n - 1.0) / n) + latency as f64 / n;
    }

    assert_eq!(stats.utterance_count, 5);
    // Average of [10, 20, 30, 40, 50] = 30.0
    assert!(
        (stats.avg_stt_latency_ms - 30.0).abs() < 1e-10,
        "expected avg ~30.0, got {}",
        stats.avg_stt_latency_ms
    );
}

#[test]
fn pipeline_stats_single_utterance_average() {
    let mut stats = PipelineStats::default();
    stats.utterance_count = 1;
    let n = 1.0f64;
    stats.avg_stt_latency_ms = stats.avg_stt_latency_ms * ((n - 1.0) / n) + 42.0 / n;

    assert!((stats.avg_stt_latency_ms - 42.0).abs() < f64::EPSILON);
}

#[test]
fn pipeline_stats_debug_format() {
    let stats = PipelineStats::default();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("PipelineStats"));
    assert!(debug.contains("utterance_count"));
    assert!(debug.contains("avg_stt_latency_ms"));
    assert!(debug.contains("uptime_secs"));
}

// ===========================================================================
// 7. SttResult
// ===========================================================================

#[test]
fn stt_result_with_all_fields() {
    let result = SttResult {
        text: "hello world".into(),
        language: Some("en".into()),
        duration_ms: 1500,
        processing_time_ms: 200,
    };

    assert_eq!(result.text, "hello world");
    assert_eq!(result.language, Some("en".into()));
    assert_eq!(result.duration_ms, 1500);
    assert_eq!(result.processing_time_ms, 200);
}

#[test]
fn stt_result_without_language() {
    let result = SttResult {
        text: "test".into(),
        language: None,
        duration_ms: 500,
        processing_time_ms: 50,
    };

    assert!(result.language.is_none());
}

#[test]
fn stt_result_clone_is_independent() {
    let result = SttResult {
        text: "original".into(),
        language: Some("en".into()),
        duration_ms: 500,
        processing_time_ms: 10,
    };

    let mut cloned = result.clone();
    cloned.text = "modified".into();
    cloned.language = Some("fr".into());

    assert_eq!(result.text, "original");
    assert_eq!(result.language, Some("en".into()));
    assert_eq!(cloned.text, "modified");
    assert_eq!(cloned.language, Some("fr".into()));
}

#[test]
fn stt_result_empty_text() {
    let result = SttResult {
        text: String::new(),
        language: None,
        duration_ms: 100,
        processing_time_ms: 1,
    };

    assert!(result.text.is_empty());
}

// ===========================================================================
// 8. VoiceInfo
// ===========================================================================

#[test]
fn voice_info_creation() {
    let voice = vox::VoiceInfo {
        id: "af_heart".into(),
        name: "Heart".into(),
        gender: "female".into(),
        language: "en-US".into(),
        accent: "American".into(),
    };

    assert_eq!(voice.id, "af_heart");
    assert_eq!(voice.name, "Heart");
    assert_eq!(voice.gender, "female");
    assert_eq!(voice.language, "en-US");
    assert_eq!(voice.accent, "American");
}

#[test]
fn voice_info_clone_is_independent() {
    let voice = vox::VoiceInfo {
        id: "v1".into(),
        name: "Voice One".into(),
        gender: "male".into(),
        language: "en-US".into(),
        accent: "British".into(),
    };

    let mut cloned = voice.clone();
    cloned.id = "v2".into();
    cloned.name = "Voice Two".into();

    assert_eq!(voice.id, "v1");
    assert_eq!(cloned.id, "v2");
}

// ===========================================================================
// 9. VoxError variants
// ===========================================================================

#[test]
fn vox_error_no_vad_message() {
    let err = VoxError::NoVad;
    assert!(err.to_string().contains("no VAD"));
}

#[test]
fn vox_error_no_stt_message() {
    let err = VoxError::NoStt;
    assert!(err.to_string().contains("no STT"));
}

#[test]
fn vox_error_audio_message() {
    let err = VoxError::Audio("device not found".into());
    let msg = err.to_string();
    assert!(msg.contains("audio"));
    assert!(msg.contains("device not found"));
}

#[test]
fn vox_error_vad_message() {
    let err = VoxError::Vad("inference failed".into());
    let msg = err.to_string();
    assert!(msg.contains("VAD"));
    assert!(msg.contains("inference failed"));
}

#[test]
fn vox_error_stt_message() {
    let err = VoxError::Stt("model load failed".into());
    let msg = err.to_string();
    assert!(msg.contains("STT"));
    assert!(msg.contains("model load failed"));
}

#[test]
fn vox_error_tts_message() {
    let err = VoxError::Tts("synthesis error".into());
    let msg = err.to_string();
    assert!(msg.contains("TTS"));
    assert!(msg.contains("synthesis error"));
}

#[test]
fn vox_error_model_not_found() {
    let err = VoxError::ModelNotFound("/path/to/model.bin".into());
    let msg = err.to_string();
    assert!(msg.contains("model not found"));
    assert!(msg.contains("/path/to/model.bin"));
}

#[test]
fn vox_error_pipeline_message() {
    let err = VoxError::Pipeline("unexpected state".into());
    let msg = err.to_string();
    assert!(msg.contains("pipeline"));
    assert!(msg.contains("unexpected state"));
}

#[test]
fn vox_error_io_from_std() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err: VoxError = io_err.into();
    let msg = err.to_string();
    assert!(msg.contains("file missing"));
}

// ===========================================================================
// 10. Default trait implementations
// ===========================================================================

#[tokio::test]
async fn vad_default_current_speech_buffer_is_none() {
    // The default implementation of current_speech_buffer returns None.
    let vad = SilentVad;
    assert!(vad.current_speech_buffer().is_none());
}

#[tokio::test]
async fn tts_default_list_voices_is_empty() {
    let tts = MockTts::new();
    assert!(tts.list_voices().is_empty());
}

#[tokio::test]
async fn tts_default_backend_name_for_unoverridden() {
    // Create a minimal TTS that does not override backend_name.
    struct MinimalTts;
    #[async_trait]
    impl TtsBackend for MinimalTts {
        async fn synthesize(&self, _request: &TtsRequest) -> Result<TtsOutput, VoxError> {
            Ok(TtsOutput {
                audio: AudioChunk {
                    samples: vec![],
                    sample_rate: 16000,
                    channels: 1,
                },
                duration_ms: 0,
            })
        }
    }

    let tts = MinimalTts;
    assert_eq!(tts.backend_name(), "unknown");
    assert!(tts.list_voices().is_empty());
}

// ===========================================================================
// 11. Streaming STT mock backends
// ===========================================================================

/// A mock streaming STT session that returns partial text every 3rd push.
struct MockSttSession {
    push_count: usize,
    partial_text: String,
    finished: bool,
}

impl SttSession for MockSttSession {
    fn push_audio(
        &mut self,
        _samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Option<String>, VoxError> {
        if self.finished {
            return Err(VoxError::Stt("session already finished".into()));
        }
        self.push_count += 1;
        // Return partial text every 3rd push
        if self.push_count % 3 == 0 {
            Ok(Some(format!(
                "{} {}",
                self.partial_text,
                self.push_count / 3
            )))
        } else {
            Ok(None)
        }
    }

    fn finish(&mut self) -> Result<SttResult, VoxError> {
        if self.finished {
            return Err(VoxError::Stt("session already finished".into()));
        }
        self.finished = true;
        Ok(SttResult {
            text: format!("{} final", self.partial_text),
            language: Some("en".into()),
            duration_ms: 1000,
            processing_time_ms: 50,
        })
    }
}

/// A mock streaming STT backend that creates MockSttSession instances.
struct MockStreamingStt {
    partial_text: String,
}

impl StreamingSttBackend for MockStreamingStt {
    fn create_session(&self) -> Result<Box<dyn SttSession>, VoxError> {
        Ok(Box::new(MockSttSession {
            push_count: 0,
            partial_text: self.partial_text.clone(),
            finished: false,
        }))
    }
}

/// A streaming STT backend that always fails on session creation.
struct FailingStreamingStt;

impl StreamingSttBackend for FailingStreamingStt {
    fn create_session(&self) -> Result<Box<dyn SttSession>, VoxError> {
        Err(VoxError::Stt("simulated session creation failure".into()))
    }
}

// ===========================================================================
// 12. Streaming STT tests
// ===========================================================================

#[test]
fn mock_streaming_session_returns_none_initially() {
    let mut session = MockSttSession {
        push_count: 0,
        partial_text: "partial".into(),
        finished: false,
    };

    let result = session.push_audio(&[0.0; 512], 16000).unwrap();
    assert!(
        result.is_none(),
        "first push should return None, got: {:?}",
        result
    );
}

#[test]
fn mock_streaming_session_returns_partial_every_3rd_push() {
    let mut session = MockSttSession {
        push_count: 0,
        partial_text: "partial".into(),
        finished: false,
    };

    let samples = [0.0f32; 512];

    // Push 1: None
    assert!(session.push_audio(&samples, 16000).unwrap().is_none());
    // Push 2: None
    assert!(session.push_audio(&samples, 16000).unwrap().is_none());
    // Push 3: Some partial text
    let result = session.push_audio(&samples, 16000).unwrap();
    assert_eq!(result, Some("partial 1".into()));
}

#[test]
fn mock_streaming_session_finish_returns_final_text() {
    let mut session = MockSttSession {
        push_count: 0,
        partial_text: "hello".into(),
        finished: false,
    };

    let samples = [0.0f32; 512];
    // Push a few frames before finishing
    session.push_audio(&samples, 16000).unwrap();
    session.push_audio(&samples, 16000).unwrap();

    let result = session.finish().unwrap();
    assert_eq!(result.text, "hello final");
    assert_eq!(result.language, Some("en".into()));
    assert_eq!(result.duration_ms, 1000);
    assert_eq!(result.processing_time_ms, 50);
}

#[test]
fn mock_streaming_session_double_finish_returns_error() {
    let mut session = MockSttSession {
        push_count: 0,
        partial_text: "test".into(),
        finished: false,
    };

    // First finish succeeds
    let result = session.finish();
    assert!(result.is_ok());

    // Second finish fails
    let result = session.finish();
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("already finished"),
        "error should mention 'already finished'"
    );
}

#[test]
fn mock_streaming_session_push_after_finish_returns_error() {
    let mut session = MockSttSession {
        push_count: 0,
        partial_text: "test".into(),
        finished: false,
    };

    // Finish the session
    session.finish().unwrap();

    // Push after finish should fail
    let result = session.push_audio(&[0.0; 512], 16000);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("already finished"),
        "error should mention 'already finished'"
    );
}

#[test]
fn mock_streaming_backend_creates_independent_sessions() {
    let backend = MockStreamingStt {
        partial_text: "word".into(),
    };

    let mut session_a = backend.create_session().unwrap();
    let mut session_b = backend.create_session().unwrap();

    let samples = [0.0f32; 512];

    // Push 3 frames to session A (triggers partial)
    session_a.push_audio(&samples, 16000).unwrap();
    session_a.push_audio(&samples, 16000).unwrap();
    let partial_a = session_a.push_audio(&samples, 16000).unwrap();
    assert_eq!(partial_a, Some("word 1".into()));

    // Session B has only 1 push -- should return None
    let partial_b = session_b.push_audio(&samples, 16000).unwrap();
    assert!(
        partial_b.is_none(),
        "session B should be independent of session A"
    );

    // Finish both independently
    let result_a = session_a.finish().unwrap();
    let result_b = session_b.finish().unwrap();
    assert_eq!(result_a.text, "word final");
    assert_eq!(result_b.text, "word final");
}

#[test]
fn failing_streaming_backend_returns_error() {
    let backend = FailingStreamingStt;

    let result = backend.create_session();
    match result {
        Err(e) => {
            assert!(
                e.to_string().contains("session creation failure"),
                "error should mention session creation failure, got: {}",
                e
            );
        }
        Ok(_) => panic!("expected create_session to fail"),
    }
}

#[tokio::test]
async fn streaming_session_with_vad_pipeline() {
    // Simulate: VAD triggers SpeechStart -> create session -> push frames ->
    // VAD triggers SpeechEnd -> finish session -> verify final text.
    let mut vad = MockVad::new(3);
    let streaming = MockStreamingStt {
        partial_text: "streaming".into(),
    };
    let frame = test_chunk(512, 16000, 1);

    let mut active_session: Option<Box<dyn SttSession>> = None;
    let mut final_text: Option<String> = None;

    for _ in 0..3 {
        let events = vad.process_frame(&frame).await.unwrap();

        // Push audio to active session
        if let Some(session) = &mut active_session {
            let _ = session.push_audio(&frame.samples, 16000);
        }

        for event in events {
            match event {
                VadEvent::SpeechStart => {
                    active_session = Some(streaming.create_session().unwrap());
                }
                VadEvent::SpeechEnd(_utterance) => {
                    if let Some(mut session) = active_session.take() {
                        let result = session.finish().unwrap();
                        final_text = Some(result.text);
                    }
                }
                VadEvent::Silence => {}
            }
        }
    }

    assert_eq!(
        final_text,
        Some("streaming final".into()),
        "streaming session should produce final text after VAD SpeechEnd"
    );
}

#[test]
#[cfg_attr(target_os = "windows", ignore)]
fn builder_accepts_streaming_stt() {
    // When streaming_stt is provided alongside VAD and STT, the builder
    // should get past the NoVad/NoStt checks and attempt audio init.
    let result = Vox::builder()
        .vad(MockVad::new(1))
        .stt(MockStt::new("test"))
        .streaming_stt(MockStreamingStt {
            partial_text: "partial".into(),
        })
        .build();
    match result {
        Ok(_) => {} // audio device available -- success
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("no VAD") && !msg.contains("no STT"),
                "should not fail with missing-backend error, got: {msg}"
            );
        }
    }
}

#[test]
#[cfg_attr(target_os = "windows", ignore)]
fn builder_accepts_on_partial() {
    // When on_partial is provided alongside VAD and STT, the builder
    // should get past the NoVad/NoStt checks and attempt audio init.
    let result = Vox::builder()
        .vad(MockVad::new(1))
        .stt(MockStt::new("test"))
        .on_partial(|_text| {})
        .build();
    match result {
        Ok(_) => {} // audio device available -- success
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("no VAD") && !msg.contains("no STT"),
                "should not fail with missing-backend error, got: {msg}"
            );
        }
    }
}
