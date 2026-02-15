//! Integration tests for the Vox pipeline using mock backends.
//!
//! These tests exercise the builder, trait system, and type plumbing
//! without requiring real audio hardware or model files.

use async_trait::async_trait;
use vox::{
    AudioChunk, SttBackend, SttResult, TtsBackend, TtsOutput, TtsRequest, Utterance, VadBackend,
    VadEvent, Vox, VoxError,
};

// ---------------------------------------------------------------------------
// Mock VAD backend
// ---------------------------------------------------------------------------

struct MockVad {
    frame_count: usize,
    /// Emit SpeechEnd after this many frames.
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

// ---------------------------------------------------------------------------
// Mock STT backend
// ---------------------------------------------------------------------------

struct MockStt {
    response: String,
}

impl MockStt {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl SttBackend for MockStt {
    async fn transcribe(&self, audio: &Utterance) -> Result<SttResult, VoxError> {
        Ok(SttResult {
            text: self.response.clone(),
            language: Some("en".into()),
            duration_ms: audio.duration_ms,
            processing_time_ms: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Mock TTS backend
// ---------------------------------------------------------------------------

struct MockTts;

#[async_trait]
impl TtsBackend for MockTts {
    async fn synthesize(&self, _request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        Ok(TtsOutput {
            audio: AudioChunk {
                samples: vec![0.0; 16000],
                sample_rate: 16000,
                channels: 1,
            },
            duration_ms: 1000,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builder should fail when no VAD is configured.
#[test]
fn build_fails_without_vad() {
    let result = Vox::builder().stt(MockStt::new("hello")).build();

    match result {
        Err(err) => assert!(
            err.to_string().contains("no VAD"),
            "expected NoVad error, got: {err}"
        ),
        Ok(_) => panic!("expected build to fail without VAD"),
    }
}

/// Builder should fail when no STT is configured.
#[test]
fn build_fails_without_stt() {
    let result = Vox::builder().vad(MockVad::new(1)).build();

    match result {
        Err(err) => assert!(
            err.to_string().contains("no STT"),
            "expected NoStt error, got: {err}"
        ),
        Ok(_) => panic!("expected build to fail without STT"),
    }
}

/// MockVad emits silence when not triggered.
#[tokio::test]
async fn mock_vad_emits_silence() {
    let mut vad = MockVad::new(100);
    let frame = AudioChunk {
        samples: vec![0.0; 512],
        sample_rate: 16000,
        channels: 1,
    };

    let events = vad.process_frame(&frame).await.unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], VadEvent::Silence));
}

/// MockVad emits speech events at the trigger point.
#[tokio::test]
async fn mock_vad_emits_speech_at_trigger() {
    let mut vad = MockVad::new(3);
    let frame = AudioChunk {
        samples: vec![0.0; 512],
        sample_rate: 16000,
        channels: 1,
    };

    // Frames 1 and 2: silence
    let e1 = vad.process_frame(&frame).await.unwrap();
    assert!(matches!(e1[0], VadEvent::Silence));

    let e2 = vad.process_frame(&frame).await.unwrap();
    assert!(matches!(e2[0], VadEvent::Silence));

    // Frame 3: speech start + end
    let e3 = vad.process_frame(&frame).await.unwrap();
    assert_eq!(e3.len(), 2);
    assert!(matches!(e3[0], VadEvent::SpeechStart));
    assert!(matches!(e3[1], VadEvent::SpeechEnd(_)));
}

/// MockStt returns the configured response.
#[tokio::test]
async fn mock_stt_transcribes() {
    let stt = MockStt::new("hello world");
    let utterance = Utterance {
        audio: AudioChunk {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 1,
        },
        duration_ms: 1000,
    };

    let result = stt.transcribe(&utterance).await.unwrap();
    assert_eq!(result.text, "hello world");
    assert_eq!(result.language, Some("en".into()));
    assert_eq!(result.duration_ms, 1000);
}

/// MockTts produces audio output.
#[tokio::test]
async fn mock_tts_synthesizes() {
    let tts = MockTts;
    let request = TtsRequest {
        text: "hello".into(),
        voice: None,
    };

    let output = tts.synthesize(&request).await.unwrap();
    assert_eq!(output.audio.sample_rate, 16000);
    assert_eq!(output.audio.samples.len(), 16000);
    assert_eq!(output.duration_ms, 1000);
}

/// VAD reset clears internal state.
#[tokio::test]
async fn vad_reset_clears_state() {
    let mut vad = MockVad::new(2);
    let frame = AudioChunk {
        samples: vec![0.0; 512],
        sample_rate: 16000,
        channels: 1,
    };

    // Advance to frame 1
    vad.process_frame(&frame).await.unwrap();
    assert_eq!(vad.frame_count, 1);

    // Reset
    vad.reset();
    assert_eq!(vad.frame_count, 0);

    // After reset, need 2 more frames to trigger
    let e1 = vad.process_frame(&frame).await.unwrap();
    assert!(matches!(e1[0], VadEvent::Silence));

    let e2 = vad.process_frame(&frame).await.unwrap();
    assert_eq!(e2.len(), 2);
    assert!(matches!(e2[0], VadEvent::SpeechStart));
}

/// AudioChunk clone produces independent data.
#[test]
fn audio_chunk_clone_is_independent() {
    let chunk = AudioChunk {
        samples: vec![1.0, 2.0, 3.0],
        sample_rate: 16000,
        channels: 1,
    };

    let mut cloned = chunk.clone();
    cloned.samples[0] = 99.0;

    assert_eq!(chunk.samples[0], 1.0);
    assert_eq!(cloned.samples[0], 99.0);
}

/// SttResult clone produces independent data.
#[test]
fn stt_result_clone_is_independent() {
    let result = SttResult {
        text: "hello".into(),
        language: Some("en".into()),
        duration_ms: 500,
        processing_time_ms: 10,
    };

    let mut cloned = result.clone();
    cloned.text = "goodbye".into();

    assert_eq!(result.text, "hello");
    assert_eq!(cloned.text, "goodbye");
}

/// PipelineStats defaults to zero.
#[test]
fn pipeline_stats_default() {
    let stats = vox::PipelineStats::default();
    assert_eq!(stats.utterance_count, 0);
    assert_eq!(stats.avg_stt_latency_ms, 0.0);
    assert_eq!(stats.uptime_secs, 0);
}

/// VoxConfig defaults are sensible.
#[test]
fn vox_config_defaults() {
    let config = vox::VoxConfig::default();
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.channels, 1);
    assert!(!config.enable_tts);
}
