//! Sample rate conversion using `rubato`.
//!
//! [`AudioResampler`] is a *streaming* resampler: input that does not fill a
//! whole resampler block is carried over to the next call instead of being
//! zero-padded, so consecutive chunks join without discontinuities. Call
//! [`AudioResampler::flush`] at end of stream to drain the tail, or use
//! [`resample_once`] for one-shot conversion of a complete buffer.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::error::VoxError;
use crate::types::AudioChunk;

/// Smallest resampler block, in input frames.
const MIN_BLOCK_FRAMES: usize = 64;

/// Resamples audio to a target sample rate and converts multi-channel input to mono.
///
/// The resampler is stateful: [`process`](Self::process) may return fewer
/// samples than `input * ratio` (or none at all) for a given call, because
/// input is consumed in fixed blocks and the remainder is kept for the next
/// call. Over a whole stream no sample is lost or invented, and the output
/// stays time-aligned with the input (to within one output sample; the
/// filter's look-ahead shows up as latency, not as leading silence).
///
/// If source and target rates are the same, operates in passthrough mode
/// (only the mono downmix is applied).
pub struct AudioResampler {
    resampler: Option<SincFixedIn<f32>>,
    source_rate: u32,
    target_rate: u32,
    /// Mono input samples not yet consumed by the resampler.
    pending: Vec<f32>,
    /// Mono input frames received since the last reset.
    total_in: u64,
    /// Output frames returned since the last reset.
    total_out: u64,
}

impl AudioResampler {
    /// Create a resampler. If `source_rate == target_rate`, no resampler is allocated.
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, VoxError> {
        if source_rate == 0 || target_rate == 0 {
            return Err(VoxError::Audio(format!(
                "invalid sample rate: {source_rate} Hz -> {target_rate} Hz"
            )));
        }

        let resampler = if source_rate != target_rate {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                oversampling_factor: 128,
                interpolation: SincInterpolationType::Cubic,
                window: WindowFunction::BlackmanHarris2,
            };

            let ratio = target_rate as f64 / source_rate as f64;
            // 10 ms blocks keep the added buffering latency low for live capture.
            let block = (source_rate as usize / 100).max(MIN_BLOCK_FRAMES);

            let r = SincFixedIn::<f32>::new(ratio, 2.0, params, block, 1)
                .map_err(|e| VoxError::Audio(format!("failed to create resampler: {e}")))?;
            Some(r)
        } else {
            None
        };

        Ok(Self {
            resampler,
            source_rate,
            target_rate,
            pending: Vec::new(),
            total_in: 0,
            total_out: 0,
        })
    }

    /// Sample rate this resampler expects as input.
    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Sample rate of the produced audio.
    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Feed the next chunk of a stream. Also converts multi-channel input to mono.
    ///
    /// Returns the output that is ready so far; input that does not fill a
    /// resampler block stays buffered until the next call or [`flush`](Self::flush).
    pub fn process(&mut self, chunk: &AudioChunk) -> Result<AudioChunk, VoxError> {
        let samples = if chunk.channels > 1 {
            let mono = downmix_to_mono(&chunk.samples, chunk.channels);
            self.push_mono(&mono)?
        } else {
            self.push_mono(&chunk.samples)?
        };

        Ok(AudioChunk {
            samples,
            sample_rate: self.target_rate,
            channels: 1,
        })
    }

    /// Drain everything still buffered at end of stream and reset.
    ///
    /// After this call the total output length of the stream is
    /// `round(input_frames * target_rate / source_rate)`.
    pub fn flush(&mut self) -> Result<AudioChunk, VoxError> {
        let mut out = Vec::new();

        if let Some(block) = self.resampler.as_ref().map(|r| r.input_frames_next()) {
            let expected = (self.total_in * u64::from(self.target_rate)
                + u64::from(self.source_rate) / 2)
                / u64::from(self.source_rate);
            let missing = expected.saturating_sub(self.total_out) as usize;

            while out.len() < missing {
                // Zero-pad to push the tail (held back by the filter's
                // look-ahead and by block buffering) through.
                let padded = (self.pending.len() / block + 1) * block;
                self.pending.resize(padded, 0.0);
                self.run_blocks(&mut out)?;
            }
            out.truncate(missing);
        }

        self.reset();

        Ok(AudioChunk {
            samples: out,
            sample_rate: self.target_rate,
            channels: 1,
        })
    }

    /// Discard all buffered audio and filter history (start of a new stream).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.total_in = 0;
        self.total_out = 0;
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
    }

    fn push_mono(&mut self, mono: &[f32]) -> Result<Vec<f32>, VoxError> {
        if self.resampler.is_none() {
            return Ok(mono.to_vec());
        }

        self.total_in += mono.len() as u64;
        self.pending.extend_from_slice(mono);

        let mut out = Vec::new();
        self.run_blocks(&mut out)?;
        self.total_out += out.len() as u64;
        Ok(out)
    }

    /// Run every complete block in `pending` through the resampler.
    fn run_blocks(&mut self, out: &mut Vec<f32>) -> Result<(), VoxError> {
        let Some(resampler) = self.resampler.as_mut() else {
            return Ok(());
        };

        let mut offset = 0;
        loop {
            let block = resampler.input_frames_next();
            if self.pending.len() - offset < block {
                break;
            }

            let result = resampler
                .process(&[&self.pending[offset..offset + block]], None)
                .map_err(|e| VoxError::Audio(format!("resample error: {e}")))?;
            offset += block;

            if let Some(channel) = result.first() {
                out.extend_from_slice(channel);
            }
        }

        self.pending.drain(..offset);
        Ok(())
    }
}

/// Convert a complete buffer in one call: downmix to mono and resample to `target_rate`.
///
/// Unlike a single [`AudioResampler::process`] call this includes the tail of
/// the signal; the result has `round(frames * target_rate / sample_rate)` samples.
pub fn resample_once(chunk: &AudioChunk, target_rate: u32) -> Result<AudioChunk, VoxError> {
    let mut resampler = AudioResampler::new(chunk.sample_rate, target_rate)?;
    let mut out = resampler.process(chunk)?;
    out.samples.extend(resampler.flush()?.samples);
    Ok(out)
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
///
/// A trailing partial frame is ignored.
pub(crate) fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = usize::from(channels.max(1));
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * freq / rate as f32).sin())
            .collect()
    }

    fn mono(samples: &[f32], rate: u32) -> AudioChunk {
        AudioChunk {
            samples: samples.to_vec(),
            sample_rate: rate,
            channels: 1,
        }
    }

    /// Feed `input` in `chunk`-sized pieces and flush.
    fn stream(input: &[f32], from: u32, to: u32, chunk: usize) -> Vec<f32> {
        let mut r = AudioResampler::new(from, to).unwrap();
        let mut out = Vec::new();
        for piece in input.chunks(chunk) {
            out.extend(r.process(&mono(piece, from)).unwrap().samples);
        }
        out.extend(r.flush().unwrap().samples);
        out
    }

    #[test]
    fn passthrough_when_same_rate() {
        let mut r = AudioResampler::new(16000, 16000).unwrap();
        let chunk = mono(&[0.1, 0.2, 0.3, 0.4], 16000);
        let out = r.process(&chunk).unwrap();
        assert_eq!(out.samples, chunk.samples);
        assert_eq!(out.sample_rate, 16000);
        assert_eq!(out.channels, 1);
        assert!(r.flush().unwrap().samples.is_empty());
    }

    #[test]
    fn rejects_zero_rates() {
        assert!(AudioResampler::new(0, 16000).is_err());
        assert!(AudioResampler::new(16000, 0).is_err());
    }

    #[test]
    fn downmix_averages_channels() {
        let samples = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = downmix_to_mono(&samples, 2);
        assert_eq!(mono.len(), 3);
        for s in mono {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn downmix_ignores_trailing_partial_frame() {
        assert_eq!(downmix_to_mono(&[1.0, 1.0, 0.5], 2), vec![1.0]);
    }

    #[test]
    fn resampler_creates_for_different_rates() {
        let r = AudioResampler::new(44100, 16000).unwrap();
        assert!(r.resampler.is_some());
        assert_eq!(r.source_rate(), 44100);
        assert_eq!(r.target_rate(), 16000);
    }

    #[test]
    fn downsampling_one_second_gives_exact_length() {
        let out = stream(&vec![0.5; 44100], 44100, 16000, 44100);
        assert_eq!(out.len(), 16000);
    }

    #[test]
    fn upsampling_one_second_gives_exact_length() {
        let out = stream(&vec![0.5; 8000], 8000, 16000, 8000);
        assert_eq!(out.len(), 16000);
    }

    #[test]
    fn stereo_input_converted_to_mono() {
        let mut r = AudioResampler::new(16000, 16000).unwrap();
        // Stereo: left=1.0, right=0.0 => mono should be 0.5
        let chunk = AudioChunk {
            samples: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            sample_rate: 16000,
            channels: 2,
        };
        let out = r.process(&chunk).unwrap();
        assert_eq!(out.channels, 1);
        assert_eq!(out.samples.len(), 4);
        for s in &out.samples {
            assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
        }
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let mut r = AudioResampler::new(16000, 16000).unwrap();
        assert!(r.process(&mono(&[], 16000)).unwrap().samples.is_empty());

        let mut r = AudioResampler::new(48000, 16000).unwrap();
        assert!(r.process(&mono(&[], 48000)).unwrap().samples.is_empty());
        assert!(r.flush().unwrap().samples.is_empty());
    }

    #[test]
    fn resampler_output_values_in_range() {
        let out = stream(&sine(440.0, 44100, 4410), 44100, 16000, 4410);
        for s in &out {
            assert!(s.abs() <= 1.5, "sample out of range: {s}"); // allow small overshoot from sinc
        }
    }

    /// Streaming 48 kHz -> 16 kHz in capture-sized chunks must give exactly
    /// the same signal as resampling the whole buffer in one call.
    #[test]
    fn streaming_matches_single_shot() {
        let input = sine(440.0, 48000, 48000);

        let single = resample_once(&mono(&input, 48000), 16000).unwrap().samples;
        // 1440 = the 30 ms chunk AudioCapture produces at 48 kHz; it is not a
        // multiple of the resampler block, so every call leaves a remainder.
        let streamed = stream(&input, 48000, 16000, 1440);

        assert_eq!(single.len(), 16000);
        assert_eq!(streamed.len(), single.len());
        for (i, (a, b)) in streamed.iter().zip(&single).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i} differs: {a} vs {b}");
        }
    }

    /// No clicks at chunk boundaries: the streamed output tracks the ideal
    /// 16 kHz sine everywhere (a zero-padded block would break this).
    #[test]
    fn streaming_has_no_discontinuities() {
        // Output sample k of the sinc resampler sits at (k + 1) / to - 1 / from
        // seconds: a constant offset of less than one output sample.
        let ideal: Vec<f32> = (0..16000)
            .map(|k| {
                let t = (k as f64 + 1.0) / 16000.0 - 1.0 / 48000.0;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32
            })
            .collect();

        for chunk in [1440, 1323, 100, 7] {
            let streamed = stream(&sine(440.0, 48000, 48000), 48000, 16000, chunk);
            assert_eq!(streamed.len(), ideal.len());

            // Skip the filter ramp-up/down at the very ends of the stream.
            let edge = 200;
            let mut max_err = 0.0f32;
            for i in edge..ideal.len() - edge {
                max_err = max_err.max((streamed[i] - ideal[i]).abs());
            }
            assert!(
                max_err < 0.02,
                "chunk {chunk}: max deviation from ideal sine {max_err}"
            );
        }
    }

    #[test]
    fn process_carries_partial_blocks_instead_of_padding() {
        let mut r = AudioResampler::new(48000, 16000).unwrap();
        // Less than one resampler block: nothing can be produced yet.
        let out = r.process(&mono(&[0.25; 100], 48000)).unwrap();
        assert!(out.samples.is_empty());
        // The carried samples are not lost: flush returns them.
        assert_eq!(r.flush().unwrap().samples.len(), 33);
    }

    #[test]
    fn flush_resets_for_a_new_stream() {
        let input = sine(440.0, 48000, 4800);
        let mut r = AudioResampler::new(48000, 16000).unwrap();

        let mut first = r.process(&mono(&input, 48000)).unwrap().samples;
        first.extend(r.flush().unwrap().samples);
        let mut second = r.process(&mono(&input, 48000)).unwrap().samples;
        second.extend(r.flush().unwrap().samples);

        assert_eq!(first.len(), 1600);
        assert_eq!(first, second);
    }

    #[test]
    fn resample_once_stereo_44k_length() {
        let chunk = AudioChunk {
            samples: vec![0.1; 44100 * 2],
            sample_rate: 44100,
            channels: 2,
        };
        let out = resample_once(&chunk, 16000).unwrap();
        assert_eq!(out.samples.len(), 16000);
        assert_eq!(out.channels, 1);
        assert_eq!(out.sample_rate, 16000);
    }
}
