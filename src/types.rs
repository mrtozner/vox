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

impl AudioChunk {
    /// Number of sample frames (samples per channel).
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// Playback duration in milliseconds, derived from the sample count.
    ///
    /// Returns 0 when the sample rate is 0.
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        self.frames() as u64 * 1000 / u64::from(self.sample_rate)
    }
}

/// A detected speech utterance segmented by VAD.
#[derive(Debug, Clone)]
pub struct Utterance {
    /// The audio data for this utterance.
    pub audio: AudioChunk,
    /// Duration of the utterance in milliseconds.
    pub duration_ms: u64,
    /// Speaker identifier (if diarization is enabled).
    #[cfg(feature = "diarization")]
    pub speaker_id: Option<String>,
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
    /// Speaker identifier (if diarization is enabled).
    #[cfg(feature = "diarization")]
    pub speaker_id: Option<String>,
}

/// Request to synthesize speech from text.
#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// The text to synthesize.
    pub text: String,
    /// Optional voice identifier.
    pub voice: Option<String>,
    /// Optional seed for deterministic generation.
    /// Not all backends support this; unsupported backends will ignore it.
    pub seed: Option<u64>,
}

/// Audio output produced by a TTS backend.
#[derive(Debug, Clone)]
pub struct TtsOutput {
    /// The synthesized audio.
    pub audio: AudioChunk,
    /// Duration of the synthesized audio in milliseconds.
    pub duration_ms: u64,
}

impl TtsOutput {
    /// Wrap mono samples, deriving `duration_ms` from the sample count.
    ///
    /// Backends should build their output through this so the reported
    /// duration is the length of the audio, not the time inference took.
    pub fn from_mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        let audio = AudioChunk {
            samples,
            sample_rate,
            channels: 1,
        };
        let duration_ms = audio.duration_ms();
        Self { audio, duration_ms }
    }
}

/// Information about a TTS voice available from a backend.
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Voice identifier (e.g. "af_heart").
    pub id: String,
    /// Human-readable voice name (e.g. "Heart").
    pub name: String,
    /// Gender: "female" or "male".
    pub gender: String,
    /// Language code (e.g. "en-US", "ja", "zh").
    pub language: String,
    /// Accent or style label (e.g. "American", "British").
    pub accent: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_follows_sample_count() {
        let out = TtsOutput::from_mono(vec![0.0; 36_000], 24_000);
        assert_eq!(out.duration_ms, 1500);
        assert_eq!(out.audio.duration_ms(), 1500);
        assert_eq!(out.audio.channels, 1);
    }

    #[test]
    fn duration_counts_frames_not_interleaved_samples() {
        let stereo = AudioChunk {
            samples: vec![0.0; 32_000],
            sample_rate: 16_000,
            channels: 2,
        };
        assert_eq!(stereo.frames(), 16_000);
        assert_eq!(stereo.duration_ms(), 1000);
    }

    #[test]
    fn duration_of_degenerate_audio_is_zero() {
        assert_eq!(TtsOutput::from_mono(Vec::new(), 24_000).duration_ms, 0);
        assert_eq!(TtsOutput::from_mono(vec![0.0; 10], 0).duration_ms, 0);
    }
}
