//! Tests for core types.

use vox::{AudioChunk, PipelineStats, SttResult, TtsOutput, TtsRequest, Utterance};

#[test]
fn audio_chunk_debug() {
    let chunk = AudioChunk {
        samples: vec![0.1, 0.2],
        sample_rate: 16000,
        channels: 1,
    };
    let debug = format!("{:?}", chunk);
    assert!(debug.contains("16000"));
}

#[test]
fn utterance_duration_consistency() {
    let chunk = AudioChunk {
        samples: vec![0.0; 48000],
        sample_rate: 16000,
        channels: 1,
    };
    let utt = Utterance {
        audio: chunk,
        duration_ms: 3000, // 48000 samples / 16000 Hz = 3s
    };
    let computed_ms = (utt.audio.samples.len() as u64 * 1000) / utt.audio.sample_rate as u64;
    assert_eq!(computed_ms, utt.duration_ms);
}

#[test]
fn stt_result_empty_text() {
    let result = SttResult {
        text: String::new(),
        language: None,
        duration_ms: 0,
        processing_time_ms: 0,
    };
    assert!(result.text.is_empty());
    assert!(result.language.is_none());
}

#[test]
fn tts_request_with_voice() {
    let req = TtsRequest {
        text: "hello".into(),
        voice: Some("af_heart".into()),
        seed: None,
    };
    assert_eq!(req.voice.as_deref(), Some("af_heart"));
}

#[test]
fn tts_request_without_voice() {
    let req = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: None,
    };
    assert!(req.voice.is_none());
}

#[test]
fn tts_request_with_seed() {
    let req = TtsRequest {
        text: "hello".into(),
        voice: None,
        seed: Some(42),
    };
    assert_eq!(req.seed, Some(42));
}

#[test]
fn tts_request_without_seed() {
    let req = TtsRequest {
        text: "hello".into(),
        voice: Some("af_heart".into()),
        seed: None,
    };
    assert!(req.seed.is_none());
}

#[test]
fn tts_output_sample_rate() {
    let output = TtsOutput {
        audio: AudioChunk {
            samples: vec![0.0; 24000],
            sample_rate: 24000,
            channels: 1,
        },
        duration_ms: 1000,
    };
    assert_eq!(output.audio.sample_rate, 24000);
    assert_eq!(output.audio.samples.len(), 24000);
}

#[test]
fn pipeline_stats_update() {
    let mut stats = PipelineStats::default();
    stats.utterance_count = 5;
    stats.avg_stt_latency_ms = 200.0;
    stats.uptime_secs = 60;

    let cloned = stats.clone();
    assert_eq!(cloned.utterance_count, 5);
    assert_eq!(cloned.avg_stt_latency_ms, 200.0);
    assert_eq!(cloned.uptime_secs, 60);
}

#[test]
fn audio_chunk_mono_vs_stereo() {
    let mono = AudioChunk {
        samples: vec![0.5; 1000],
        sample_rate: 16000,
        channels: 1,
    };
    let stereo = AudioChunk {
        samples: vec![0.5; 2000],
        sample_rate: 16000,
        channels: 2,
    };
    // Stereo has twice the samples for same duration
    assert_eq!(stereo.samples.len(), mono.samples.len() * 2);
}
