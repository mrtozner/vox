use std::collections::VecDeque;

use async_trait::async_trait;
use ort::{inputs, session::Session, value::TensorRef};

use crate::error::VoxError;
use crate::traits::{VadBackend, VadEvent};
use crate::types::{AudioChunk, Utterance};

/// Frame size expected by Silero VAD v5 at 16 kHz.
const FRAME_SIZE: usize = 512;

/// Sample rate expected by Silero VAD.
const SAMPLE_RATE: u32 = 16_000;

/// Duration of one frame in milliseconds (512 samples / 16000 Hz = 32 ms).
const FRAME_MS: u32 = (FRAME_SIZE as u32 * 1000) / SAMPLE_RATE;

/// Number of elements in the LSTM state tensor: shape [2, 1, 128].
const STATE_LEN: usize = 256; // 2 * 1 * 128

/// Configuration for Silero VAD.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Probability threshold for speech detection (default: 0.5).
    pub speech_threshold: f32,
    /// How many ms of silence before speech is considered ended (default: 500).
    pub silence_duration_ms: u32,
    /// Minimum speech duration in ms to trigger an event (default: 250).
    pub min_speech_ms: u32,
    /// Milliseconds of audio to retain before speech onset (default: 300).
    /// This lookback buffer is prepended to the utterance when speech starts,
    /// preventing clipping of the first syllable.
    pub pre_speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_duration_ms: 500,
            min_speech_ms: 250,
            pre_speech_pad_ms: 300,
        }
    }
}

/// Silero VAD backend using ONNX Runtime.
///
/// Wraps the Silero VAD v5 ONNX model and performs frame-level speech
/// probability inference. Maintains LSTM hidden/cell state across frames
/// and implements speech segmentation logic (start/end detection with
/// configurable silence and minimum duration thresholds).
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    config: VadConfig,
    is_speaking: bool,
    silence_frames: u32,
    speech_buffer: Vec<f32>,
    speech_frames: u32,
    /// Ring buffer holding the most recent audio samples before speech onset.
    lookback_buffer: VecDeque<f32>,
    /// Maximum number of samples to keep in the lookback buffer.
    lookback_capacity: usize,
}

impl SileroVad {
    /// Create a new Silero VAD instance with default configuration.
    ///
    /// `model_path` must point to a valid `silero_vad.onnx` file.
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Result<Self, VoxError> {
        Self::with_config(model_path, VadConfig::default())
    }

    /// Create a new Silero VAD instance with custom configuration.
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        config: VadConfig,
    ) -> Result<Self, VoxError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }

        let session = Session::builder()
            .map_err(|e| VoxError::Vad(format!("failed to create session builder: {e}")))?
            .commit_from_file(path)
            .map_err(|e| VoxError::Vad(format!("failed to load model: {e}")))?;

        let lookback_capacity =
            (config.pre_speech_pad_ms as usize * SAMPLE_RATE as usize) / 1000;

        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            config,
            is_speaking: false,
            silence_frames: 0,
            speech_buffer: Vec::new(),
            speech_frames: 0,
            lookback_buffer: VecDeque::with_capacity(lookback_capacity),
            lookback_capacity,
        })
    }

    /// Run inference on a single 512-sample frame and return speech probability.
    fn infer(&mut self, frame: &[f32]) -> Result<f32, VoxError> {
        debug_assert_eq!(frame.len(), FRAME_SIZE);

        let input_audio = TensorRef::from_array_view(([1usize, FRAME_SIZE], frame))
            .map_err(|e| VoxError::Vad(format!("failed to create input tensor: {e}")))?;

        let sr_data: [i64; 1] = [SAMPLE_RATE as i64];
        let input_sr = TensorRef::from_array_view(([0usize; 0], &sr_data[..1]))
            .map_err(|e| VoxError::Vad(format!("failed to create sr tensor: {e}")))?;

        let input_state = TensorRef::from_array_view(([2usize, 1, 128], &self.state[..]))
            .map_err(|e| VoxError::Vad(format!("failed to create state tensor: {e}")))?;

        let outputs = self
            .session
            .run(inputs![
                "input" => input_audio,
                "sr" => input_sr,
                "state" => input_state
            ])
            .map_err(|e| VoxError::Vad(format!("inference failed: {e}")))?;

        let (_shape, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Vad(format!("failed to extract output: {e}")))?;
        let probability = prob_data.first().copied().unwrap_or(0.0);

        let (_shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Vad(format!("failed to extract stateN: {e}")))?;
        self.state.copy_from_slice(state_data);

        Ok(probability)
    }
}

#[async_trait]
impl VadBackend for SileroVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        let probability = self.infer(&frame.samples)?;
        let mut events = Vec::new();

        if probability >= self.config.speech_threshold {
            // Speech detected in this frame.
            if !self.is_speaking {
                self.is_speaking = true;
                // Prepend the lookback buffer to capture audio just before speech onset.
                if self.lookback_capacity > 0 {
                    self.speech_buffer
                        .extend(self.lookback_buffer.iter().copied());
                    self.lookback_buffer.clear();
                }
                events.push(VadEvent::SpeechStart);
            }
            self.speech_buffer.extend_from_slice(&frame.samples);
            self.speech_frames += 1;
            self.silence_frames = 0;
        } else if self.is_speaking {
            // Silence while we were speaking -- still buffer the audio in case
            // speech resumes within the tolerance window.
            self.speech_buffer.extend_from_slice(&frame.samples);
            self.silence_frames += 1;

            let silence_ms = self.silence_frames * FRAME_MS;
            if silence_ms >= self.config.silence_duration_ms {
                let speech_ms = self.speech_frames * FRAME_MS;
                if speech_ms >= self.config.min_speech_ms {
                    let duration_ms =
                        (self.speech_buffer.len() as u64 * 1000) / u64::from(SAMPLE_RATE);
                    let utterance = Utterance {
                        audio: AudioChunk {
                            samples: std::mem::take(&mut self.speech_buffer),
                            sample_rate: SAMPLE_RATE,
                            channels: 1,
                        },
                        duration_ms,
                    };
                    events.push(VadEvent::SpeechEnd(utterance));
                }
                // Reset segmentation state regardless of whether we emitted.
                self.is_speaking = false;
                self.silence_frames = 0;
                self.speech_buffer.clear();
                self.speech_frames = 0;
            }
        } else {
            // Not speaking -- feed the lookback ring buffer.
            if self.lookback_capacity > 0 {
                for &sample in &frame.samples {
                    if self.lookback_buffer.len() >= self.lookback_capacity {
                        self.lookback_buffer.pop_front();
                    }
                    self.lookback_buffer.push_back(sample);
                }
            }
            events.push(VadEvent::Silence);
        }

        Ok(events)
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
        self.is_speaking = false;
        self.silence_frames = 0;
        self.speech_buffer.clear();
        self.speech_frames = 0;
        self.lookback_buffer.clear();
    }

    fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn current_speech_buffer(&self) -> Option<crate::types::AudioChunk> {
        if self.is_speaking && !self.speech_buffer.is_empty() {
            Some(crate::types::AudioChunk {
                samples: self.speech_buffer.clone(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute expected lookback capacity in samples for a given pad_ms.
    fn expected_capacity(pad_ms: u32) -> usize {
        (pad_ms as usize * SAMPLE_RATE as usize) / 1000
    }

    /// Test that the lookback buffer capacity is calculated correctly for 300ms.
    #[test]
    fn pad_lookback_capacity_300ms() {
        // 300ms at 16kHz = 4800 samples
        let cap = expected_capacity(300);
        assert_eq!(cap, 4800);
    }

    /// Test that the lookback buffer capacity is 0 when pre_speech_pad_ms = 0.
    #[test]
    fn pad_lookback_capacity_zero() {
        let cap = expected_capacity(0);
        assert_eq!(cap, 0);
    }

    /// Test that the ring buffer retains only the last N samples after many pushes.
    #[test]
    fn pad_ring_buffer_wrapping() {
        let capacity = expected_capacity(300); // 4800
        let mut buf: VecDeque<f32> = VecDeque::with_capacity(capacity);

        // Push 10000 samples (more than 4800 capacity).
        for i in 0..10_000u32 {
            if buf.len() >= capacity {
                buf.pop_front();
            }
            buf.push_back(i as f32);
        }

        assert_eq!(buf.len(), capacity);
        // The buffer should contain [5200, 5201, ..., 9999].
        assert_eq!(*buf.front().unwrap(), 5200.0);
        assert_eq!(*buf.back().unwrap(), 9999.0);
    }

    /// Test that pre-padded audio appears at the start of the speech buffer
    /// when transitioning from silence to speech.
    #[test]
    fn pad_prepend_on_speech_start() {
        let capacity = 10; // small for easy testing
        let mut lookback: VecDeque<f32> = VecDeque::with_capacity(capacity);
        let mut speech_buffer: Vec<f32> = Vec::new();

        // Simulate filling lookback with samples 1..=10.
        for i in 1..=10 {
            if lookback.len() >= capacity {
                lookback.pop_front();
            }
            lookback.push_back(i as f32);
        }
        assert_eq!(lookback.len(), 10);

        // Simulate speech start: prepend lookback, then add current frame.
        speech_buffer.extend(lookback.iter().copied());
        lookback.clear();
        let frame_samples = vec![100.0, 101.0, 102.0];
        speech_buffer.extend_from_slice(&frame_samples);

        // speech_buffer should be [1,2,...,10, 100,101,102].
        assert_eq!(speech_buffer.len(), 13);
        assert_eq!(&speech_buffer[..10], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(&speech_buffer[10..], &[100.0, 101.0, 102.0]);
    }

    /// Test that with pad_ms = 0 no lookback occurs (same behavior as before).
    #[test]
    fn pad_zero_no_lookback() {
        let capacity = expected_capacity(0);
        assert_eq!(capacity, 0);

        let lookback: VecDeque<f32> = VecDeque::with_capacity(capacity);
        let mut speech_buffer: Vec<f32> = Vec::new();

        // With capacity 0, the guard `if capacity > 0` prevents any lookback ops.
        if capacity > 0 {
            speech_buffer.extend(lookback.iter().copied());
        }
        let frame_samples = vec![42.0, 43.0];
        speech_buffer.extend_from_slice(&frame_samples);

        // No padding prepended.
        assert_eq!(speech_buffer, vec![42.0, 43.0]);
    }

    /// Test that after wrapping, only the most recent samples are retained.
    #[test]
    fn pad_buffer_only_retains_last_n_ms() {
        // 100ms at 16kHz = 1600 samples
        let capacity = expected_capacity(100);
        assert_eq!(capacity, 1600);

        let mut buf: VecDeque<f32> = VecDeque::with_capacity(capacity);

        // Feed 5 frames of 512 samples each = 2560 samples total.
        for frame_idx in 0..5u32 {
            let frame: Vec<f32> = (0..512).map(|s| (frame_idx * 512 + s) as f32).collect();
            for &sample in &frame {
                if buf.len() >= capacity {
                    buf.pop_front();
                }
                buf.push_back(sample);
            }
        }

        assert_eq!(buf.len(), capacity); // 1600
        // 2560 total samples - 1600 capacity = first 960 were evicted.
        // So buffer starts at sample index 960.
        assert_eq!(*buf.front().unwrap(), 960.0);
        assert_eq!(*buf.back().unwrap(), 2559.0);
    }
}
