//! Sample rate conversion using `rubato`.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::error::VoxError;
use crate::types::AudioChunk;

/// Resamples audio to a target sample rate and converts stereo to mono.
///
/// If source and target rates are the same, operates in passthrough mode
/// with zero allocation overhead for resampling.
pub struct AudioResampler {
    resampler: Option<SincFixedIn<f32>>,
    #[allow(dead_code)]
    source_rate: u32,
    target_rate: u32,
}

impl AudioResampler {
    /// Create a resampler. If `source_rate == target_rate`, no resampler is allocated.
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, VoxError> {
        let resampler = if source_rate != target_rate {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                oversampling_factor: 128,
                interpolation: SincInterpolationType::Cubic,
                window: WindowFunction::BlackmanHarris2,
            };

            let ratio = target_rate as f64 / source_rate as f64;
            // Use a reasonable chunk size for the input
            let chunk_size = 1024;

            let r = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
                .map_err(|e| VoxError::Audio(format!("failed to create resampler: {e}")))?;
            Some(r)
        } else {
            None
        };

        Ok(Self {
            resampler,
            source_rate,
            target_rate,
        })
    }

    /// Resample an audio chunk. Also converts stereo to mono if needed.
    pub fn process(&mut self, chunk: &AudioChunk) -> Result<AudioChunk, VoxError> {
        // Step 1: Convert to mono if stereo
        let mono_samples = if chunk.channels > 1 {
            stereo_to_mono(&chunk.samples, chunk.channels)
        } else {
            chunk.samples.clone()
        };

        // Step 2: Resample if needed
        let resampled = match self.resampler.as_mut() {
            Some(resampler) => {
                let input_frames_max = resampler.input_frames_max();
                let mut output = Vec::new();

                // Process in chunks that match the resampler's expected input size
                let mut offset = 0;
                while offset < mono_samples.len() {
                    let end = (offset + input_frames_max).min(mono_samples.len());
                    let mut input_chunk = mono_samples[offset..end].to_vec();

                    // Pad with zeros if the last chunk is smaller than expected
                    if input_chunk.len() < input_frames_max {
                        input_chunk.resize(input_frames_max, 0.0);
                    }

                    let input = vec![input_chunk];
                    let result = resampler
                        .process(&input, None)
                        .map_err(|e| VoxError::Audio(format!("resample error: {e}")))?;

                    if let Some(channel) = result.into_iter().next() {
                        // If we padded the input, trim proportionally
                        if end - offset < input_frames_max {
                            let valid_ratio = (end - offset) as f64 / input_frames_max as f64;
                            let valid_output = (channel.len() as f64 * valid_ratio) as usize;
                            output.extend_from_slice(&channel[..valid_output]);
                        } else {
                            output.extend(channel);
                        }
                    }

                    offset += input_frames_max;
                }

                output
            }
            None => mono_samples,
        };

        Ok(AudioChunk {
            samples: resampled,
            sample_rate: self.target_rate,
            channels: 1,
        })
    }
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
fn stereo_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    let frame_count = samples.len() / ch;
    let mut mono = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += samples[i * ch + c];
        }
        mono.push(sum / ch as f32);
    }
    mono
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_same_rate() {
        let mut r = AudioResampler::new(16000, 16000).unwrap();
        let chunk = AudioChunk {
            samples: vec![0.1, 0.2, 0.3, 0.4],
            sample_rate: 16000,
            channels: 1,
        };
        let out = r.process(&chunk).unwrap();
        assert_eq!(out.samples, chunk.samples);
        assert_eq!(out.sample_rate, 16000);
        assert_eq!(out.channels, 1);
    }

    #[test]
    fn stereo_to_mono_conversion() {
        let samples = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = stereo_to_mono(&samples, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resampler_creates_for_different_rates() {
        let r = AudioResampler::new(44100, 16000);
        assert!(r.is_ok());
        assert!(r.unwrap().resampler.is_some());
    }
}
