//! Audio capture, buffering, resampling, and playback.
//!
//! This module provides the audio I/O layer for the Vox pipeline,
//! including microphone capture via `cpal` (feature `mic`), ring buffering
//! for frame-based processing, sample rate conversion via `rubato`,
//! and audio playback via `rodio`.

mod buffer;
#[cfg(feature = "mic")]
mod capture;
mod frame;
mod resampler;

#[cfg(any(
    feature = "kokoro",
    feature = "pocket",
    feature = "chatterbox",
    feature = "piper",
    feature = "qwen3",
    feature = "tts"
))]
mod playback;

pub use buffer::AudioBuffer;
#[cfg(feature = "mic")]
pub use capture::AudioCapture;
pub use frame::FrameAssembler;
pub use resampler::{AudioResampler, resample_once};

#[cfg(any(
    feature = "kokoro",
    feature = "pocket",
    feature = "chatterbox",
    feature = "piper",
    feature = "qwen3",
    feature = "tts"
))]
pub use playback::AudioPlayer;

use crate::error::VoxError;
use crate::types::AudioChunk;

/// Sample rate expected by the speech models (VAD, STT, speaker embeddings).
pub const SPEECH_SAMPLE_RATE: u32 = 16_000;

/// Convert a complete interleaved PCM buffer to 16 kHz mono.
///
/// `samples` holds `channels` interleaved channels at `sample_rate` Hz.
/// Channels are averaged, then the signal is resampled with the same
/// band-limited resampler the live pipeline uses. The result has
/// `round(frames * 16000 / sample_rate)` samples.
///
/// Returns an error if `sample_rate` or `channels` is zero.
pub fn to_mono_16k(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<f32>, VoxError> {
    if channels == 0 {
        return Err(VoxError::Audio("audio has zero channels".into()));
    }
    if sample_rate == 0 {
        return Err(VoxError::Audio("audio has a sample rate of 0 Hz".into()));
    }

    if sample_rate == SPEECH_SAMPLE_RATE {
        return Ok(if channels == 1 {
            samples.to_vec()
        } else {
            resampler::downmix_to_mono(samples, channels)
        });
    }

    let chunk = AudioChunk {
        samples: samples.to_vec(),
        sample_rate,
        channels,
    };
    Ok(resample_once(&chunk, SPEECH_SAMPLE_RATE)?.samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_mono_16k_44k_stereo_length() {
        // 1 second of 44.1 kHz stereo -> exactly 1 second of 16 kHz mono.
        let samples = vec![0.25f32; 44_100 * 2];
        let out = to_mono_16k(&samples, 44_100, 2).unwrap();
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn to_mono_16k_odd_length_rounds() {
        // 0.5 s + 1 frame of 44.1 kHz stereo: round(22051 * 16000 / 44100) = 8000.
        let samples = vec![0.0f32; 22_051 * 2];
        assert_eq!(to_mono_16k(&samples, 44_100, 2).unwrap().len(), 8_000);
    }

    #[test]
    fn to_mono_16k_preserves_signal() {
        // 440 Hz on both channels of a 44.1 kHz stereo buffer.
        let n = 44_100;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44_100.0).sin();
            samples.push(s);
            samples.push(s);
        }
        let out = to_mono_16k(&samples, 44_100, 2).unwrap();
        assert_eq!(out.len(), 16_000);
        for (i, s) in out.iter().enumerate().take(15_800).skip(200) {
            // The resampler output sits a constant sub-sample offset from the
            // input: sample i is at (i + 1) / 16000 - 1 / 44100 seconds.
            let t = (i as f64 + 1.0) / 16_000.0 - 1.0 / 44_100.0;
            let ideal = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32;
            assert!((s - ideal).abs() < 0.02, "sample {i}: {s} vs {ideal}");
        }
    }

    #[test]
    fn to_mono_16k_passthrough_and_downmix_at_16k() {
        assert_eq!(
            to_mono_16k(&[0.1, 0.2, 0.3], 16_000, 1).unwrap(),
            vec![0.1, 0.2, 0.3]
        );
        let out = to_mono_16k(&[1.0, 0.0, 0.0, 1.0], 16_000, 2).unwrap();
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn to_mono_16k_upsamples_8k() {
        assert_eq!(
            to_mono_16k(&vec![0.0; 8_000], 8_000, 1).unwrap().len(),
            16_000
        );
    }

    #[test]
    fn to_mono_16k_empty_and_invalid() {
        assert!(to_mono_16k(&[], 44_100, 2).unwrap().is_empty());
        assert!(to_mono_16k(&[0.0; 4], 0, 1).is_err());
        assert!(to_mono_16k(&[0.0; 4], 44_100, 0).is_err());
    }
}
