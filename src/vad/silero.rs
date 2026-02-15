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

/// Number of elements in the LSTM hidden/cell state: shape [2, 1, 64].
const STATE_LEN: usize = 128; // 2 * 1 * 64

/// Configuration for Silero VAD.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Probability threshold for speech detection (default: 0.5).
    pub speech_threshold: f32,
    /// How many ms of silence before speech is considered ended (default: 500).
    pub silence_duration_ms: u32,
    /// Minimum speech duration in ms to trigger an event (default: 250).
    pub min_speech_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_duration_ms: 500,
            min_speech_ms: 250,
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
    h: Vec<f32>,
    c: Vec<f32>,
    config: VadConfig,
    is_speaking: bool,
    silence_frames: u32,
    speech_buffer: Vec<f32>,
    speech_frames: u32,
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

        Ok(Self {
            session,
            h: vec![0.0; STATE_LEN],
            c: vec![0.0; STATE_LEN],
            config,
            is_speaking: false,
            silence_frames: 0,
            speech_buffer: Vec::new(),
            speech_frames: 0,
        })
    }

    /// Run inference on a single 512-sample frame and return speech probability.
    fn infer(&mut self, frame: &[f32]) -> Result<f32, VoxError> {
        debug_assert_eq!(frame.len(), FRAME_SIZE);

        let input_audio = TensorRef::from_array_view(([1usize, FRAME_SIZE], frame))
            .map_err(|e| VoxError::Vad(format!("failed to create input tensor: {e}")))?;

        let sr_data: [i64; 1] = [SAMPLE_RATE as i64];
        let input_sr = TensorRef::from_array_view(([1usize], &sr_data[..]))
            .map_err(|e| VoxError::Vad(format!("failed to create sr tensor: {e}")))?;

        let input_h = TensorRef::from_array_view(([2usize, 1, 64], &self.h[..]))
            .map_err(|e| VoxError::Vad(format!("failed to create h tensor: {e}")))?;

        let input_c = TensorRef::from_array_view(([2usize, 1, 64], &self.c[..]))
            .map_err(|e| VoxError::Vad(format!("failed to create c tensor: {e}")))?;

        let outputs = self
            .session
            .run(inputs![
                "input" => input_audio,
                "sr" => input_sr,
                "h" => input_h,
                "c" => input_c
            ])
            .map_err(|e| VoxError::Vad(format!("inference failed: {e}")))?;

        // Extract speech probability scalar.
        let (_shape, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Vad(format!("failed to extract output: {e}")))?;
        let probability = prob_data.first().copied().unwrap_or(0.0);

        // Update LSTM hidden and cell states.
        let (_shape, hn_data) = outputs["hn"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Vad(format!("failed to extract hn: {e}")))?;
        self.h.copy_from_slice(hn_data);

        let (_shape, cn_data) = outputs["cn"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Vad(format!("failed to extract cn: {e}")))?;
        self.c.copy_from_slice(cn_data);

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
            events.push(VadEvent::Silence);
        }

        Ok(events)
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.is_speaking = false;
        self.silence_frames = 0;
        self.speech_buffer.clear();
        self.speech_frames = 0;
    }

    fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}
