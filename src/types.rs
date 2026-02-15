//! Core types shared across the Vox pipeline.

/// Raw audio samples at a specific sample rate.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Interleaved f32 samples normalized to [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 16000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
}

/// A detected speech utterance segmented by VAD.
#[derive(Debug, Clone)]
pub struct Utterance {
    /// The audio data for this utterance.
    pub audio: AudioChunk,
    /// Duration of the utterance in milliseconds.
    pub duration_ms: u64,
}

/// Result from speech-to-text transcription.
#[derive(Debug, Clone)]
pub struct SttResult {
    /// The transcribed text.
    pub text: String,
    /// Detected language code (e.g. "en"), if available.
    pub language: Option<String>,
    /// Duration of the source audio in milliseconds.
    pub duration_ms: u64,
    /// Time taken to process/transcribe in milliseconds.
    pub processing_time_ms: u64,
}

/// Request to synthesize speech from text.
#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// The text to synthesize.
    pub text: String,
    /// Optional voice identifier.
    pub voice: Option<String>,
}

/// Audio output produced by a TTS backend.
#[derive(Debug, Clone)]
pub struct TtsOutput {
    /// The synthesized audio.
    pub audio: AudioChunk,
    /// Duration of the synthesized audio in milliseconds.
    pub duration_ms: u64,
}

/// Pipeline runtime statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total number of utterances processed.
    pub utterance_count: u64,
    /// Average STT latency in milliseconds.
    pub avg_stt_latency_ms: f64,
    /// Total pipeline uptime in seconds.
    pub uptime_secs: u64,
}
