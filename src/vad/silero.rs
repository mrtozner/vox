use std::collections::VecDeque;

use async_trait::async_trait;
use ort::{inputs, session::Session, value::TensorRef};

use crate::error::VoxError;
use crate::traits::{VadBackend, VadEvent};
use crate::types::{AudioChunk, Utterance};

/// Silero VAD model revision this backend is written for.
///
/// The ONNX graph of this revision takes `input` (`[batch, 576]`: 64 context
/// samples + 512 new samples), `state` (`[2, batch, 128]`) and `sr`, and
/// returns `output` and `stateN`. Older revisions (v3/v4: `h`/`c` LSTM
/// inputs) are rejected at load time.
pub const SILERO_MODEL_REVISION: &str = "v5.1";

/// Number of new samples consumed per frame by Silero VAD v5 at 16 kHz.
const FRAME_SIZE: usize = 512;

/// Number of samples from the end of the previous frame that Silero VAD v5
/// expects in front of every frame at 16 kHz.
const CONTEXT_SIZE: usize = 64;

/// Length of the model's audio input: context + frame.
const INPUT_SIZE: usize = CONTEXT_SIZE + FRAME_SIZE;

/// Sample rate expected by Silero VAD.
const SAMPLE_RATE: u32 = 16_000;

/// Duration of one frame in milliseconds (512 samples / 16000 Hz = 32 ms).
const FRAME_MS: u32 = (FRAME_SIZE as u32 * 1000) / SAMPLE_RATE;

/// Number of elements in the LSTM state tensor: shape [2, 1, 128].
const STATE_LEN: usize = 256; // 2 * 1 * 128

/// Model inputs of the supported revision.
const EXPECTED_INPUTS: [&str; 3] = ["input", "state", "sr"];

/// Model outputs of the supported revision.
const EXPECTED_OUTPUTS: [&str; 2] = ["output", "stateN"];

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
    /// Maximum utterance length in ms (default: 30000, `0` = unlimited).
    ///
    /// When the buffered utterance (including the pre-speech pad) reaches
    /// this length without a pause, a `SpeechEnd` is forced and segmentation
    /// restarts, so continuous speech or steady noise cannot grow the buffer
    /// without bound. The default matches Whisper's 30 s window.
    pub max_utterance_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_duration_ms: 500,
            min_speech_ms: 250,
            pre_speech_pad_ms: 300,
            max_utterance_ms: 30_000,
        }
    }
}

/// Turns per-frame speech probabilities into `SpeechStart` / `SpeechEnd`
/// events. Independent of the model so it can be tested without one.
struct SpeechSegmenter {
    config: VadConfig,
    is_speaking: bool,
    silence_frames: u32,
    speech_buffer: Vec<f32>,
    speech_frames: u32,
    /// Ring buffer holding the most recent audio samples before speech onset.
    lookback_buffer: VecDeque<f32>,
    /// Maximum number of samples to keep in the lookback buffer.
    lookback_capacity: usize,
    /// Utterance length (in samples) at which `SpeechEnd` is forced; 0 = never.
    max_utterance_samples: usize,
}

impl SpeechSegmenter {
    fn new(config: VadConfig) -> Self {
        let lookback_capacity = (config.pre_speech_pad_ms as usize * SAMPLE_RATE as usize) / 1000;
        let max_utterance_samples =
            (config.max_utterance_ms as usize * SAMPLE_RATE as usize) / 1000;

        Self {
            config,
            is_speaking: false,
            silence_frames: 0,
            speech_buffer: Vec::new(),
            speech_frames: 0,
            lookback_buffer: VecDeque::with_capacity(lookback_capacity),
            lookback_capacity,
            max_utterance_samples,
        }
    }

    /// Feed one frame and its speech probability.
    fn push(&mut self, probability: f32, samples: &[f32]) -> Vec<VadEvent> {
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
            self.speech_buffer.extend_from_slice(samples);
            self.speech_frames += 1;
            self.silence_frames = 0;
        } else if self.is_speaking {
            // Silence while we were speaking -- still buffer the audio in case
            // speech resumes within the tolerance window.
            self.speech_buffer.extend_from_slice(samples);
            self.silence_frames += 1;

            let silence_ms = self.silence_frames * FRAME_MS;
            if silence_ms >= self.config.silence_duration_ms {
                let speech_ms = self.speech_frames * FRAME_MS;
                if speech_ms >= self.config.min_speech_ms {
                    events.push(VadEvent::SpeechEnd(self.take_utterance()));
                }
                // Reset segmentation state regardless of whether we emitted.
                self.end_segment();
            }
        } else {
            // Not speaking -- feed the lookback ring buffer.
            if self.lookback_capacity > 0 {
                for &sample in samples {
                    if self.lookback_buffer.len() >= self.lookback_capacity {
                        self.lookback_buffer.pop_front();
                    }
                    self.lookback_buffer.push_back(sample);
                }
            }
            events.push(VadEvent::Silence);
        }

        // Forced end: the utterance hit the configured maximum without a pause.
        // The next speech frame starts a new utterance (with a new SpeechStart),
        // so no audio is lost across the cut.
        if self.is_speaking
            && self.max_utterance_samples > 0
            && self.speech_buffer.len() >= self.max_utterance_samples
        {
            tracing::debug!(
                max_utterance_ms = self.config.max_utterance_ms,
                "utterance reached maximum length, forcing speech end"
            );
            events.push(VadEvent::SpeechEnd(self.take_utterance()));
            self.end_segment();
        }

        events
    }

    fn take_utterance(&mut self) -> Utterance {
        let duration_ms = (self.speech_buffer.len() as u64 * 1000) / u64::from(SAMPLE_RATE);
        Utterance {
            audio: AudioChunk {
                samples: std::mem::take(&mut self.speech_buffer),
                sample_rate: SAMPLE_RATE,
                channels: 1,
            },
            duration_ms,
            #[cfg(feature = "diarization")]
            speaker_id: None,
        }
    }

    fn end_segment(&mut self) {
        self.is_speaking = false;
        self.silence_frames = 0;
        self.speech_buffer.clear();
        self.speech_frames = 0;
    }

    fn reset(&mut self) {
        self.end_segment();
        self.lookback_buffer.clear();
    }
}

/// Silero VAD backend using ONNX Runtime.
///
/// Wraps the Silero VAD v5 ONNX model and performs frame-level speech
/// probability inference. Maintains the LSTM state and the 64-sample audio
/// context across frames and implements speech segmentation logic
/// (start/end detection with configurable silence, minimum and maximum
/// duration thresholds).
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    /// Last `CONTEXT_SIZE` samples of the previous frame.
    context: Vec<f32>,
    /// Scratch buffer for the model input (`context` + frame).
    input: Vec<f32>,
    segmenter: SpeechSegmenter,
}

impl SileroVad {
    /// Create a new Silero VAD instance with default configuration.
    ///
    /// `model_path` must point to a valid `silero_vad.onnx` file.
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Result<Self, VoxError> {
        Self::with_config(model_path, VadConfig::default())
    }

    /// Create a new Silero VAD instance with custom configuration.
    ///
    /// Fails if the file is not a Silero VAD v5 model: the input/output names
    /// are checked and one silent frame is run through the model to make sure
    /// it accepts the `[1, 576]` input this backend feeds it.
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

        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        check_model_signature(&inputs, &outputs)?;

        let mut vad = Self {
            session,
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT_SIZE],
            input: vec![0.0; INPUT_SIZE],
            segmenter: SpeechSegmenter::new(config),
        };

        // Probe the input shape once so a wrong model revision fails here
        // with a clear message instead of on the first microphone frame.
        vad.infer(&[0.0; FRAME_SIZE]).map_err(|e| {
            VoxError::Vad(format!(
                "{} is not a compatible Silero VAD {SILERO_MODEL_REVISION} model \
                 (a [1, {INPUT_SIZE}] input was rejected): {e}",
                path.display()
            ))
        })?;
        vad.state.fill(0.0);
        vad.context.fill(0.0);

        Ok(vad)
    }

    /// Run inference on a single 512-sample frame and return speech probability.
    ///
    /// Silero v5 needs the last 64 samples of the previous frame in front of
    /// the new samples; without that context its probabilities are unreliable.
    fn infer(&mut self, frame: &[f32]) -> Result<f32, VoxError> {
        if frame.len() != FRAME_SIZE {
            return Err(VoxError::Vad(format!(
                "Silero VAD needs frames of exactly {FRAME_SIZE} samples, got {}",
                frame.len()
            )));
        }

        self.input[..CONTEXT_SIZE].copy_from_slice(&self.context);
        self.input[CONTEXT_SIZE..].copy_from_slice(frame);

        let input_audio = TensorRef::from_array_view(([1usize, INPUT_SIZE], &self.input[..]))
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
        if state_data.len() != STATE_LEN {
            return Err(VoxError::Vad(format!(
                "unexpected stateN size {} (expected {STATE_LEN})",
                state_data.len()
            )));
        }
        self.state.copy_from_slice(state_data);
        self.context
            .copy_from_slice(&frame[FRAME_SIZE - CONTEXT_SIZE..]);

        Ok(probability)
    }
}

/// Check that a model's input/output names match the supported Silero revision.
fn check_model_signature(inputs: &[&str], outputs: &[&str]) -> Result<(), VoxError> {
    let missing: Vec<&str> = EXPECTED_INPUTS
        .iter()
        .filter(|name| !inputs.contains(name))
        .chain(
            EXPECTED_OUTPUTS
                .iter()
                .filter(|name| !outputs.contains(name)),
        )
        .copied()
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    Err(VoxError::Vad(format!(
        "model is not Silero VAD {SILERO_MODEL_REVISION}: missing {missing:?} \
         (model inputs: {inputs:?}, outputs: {outputs:?}). \
         Re-download the model with `vox models download`."
    )))
}

#[async_trait]
impl VadBackend for SileroVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        let probability = self.infer(&frame.samples)?;
        // Per-frame trace (only visible at RUST_LOG=trace) so we can see
        // what Silero actually emits when debugging pipeline issues.
        tracing::trace!(
            prob = probability,
            is_speaking = self.segmenter.is_speaking,
            threshold = self.segmenter.config.speech_threshold,
            "silero frame"
        );

        Ok(self.segmenter.push(probability, &frame.samples))
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
        self.segmenter.reset();
    }

    fn voiced_ms(&self) -> Option<u32> {
        Some(if self.segmenter.is_speaking {
            self.segmenter.speech_frames * FRAME_MS
        } else {
            0
        })
    }

    fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn current_speech_buffer(&self) -> Option<crate::types::AudioChunk> {
        if self.segmenter.is_speaking && !self.segmenter.speech_buffer.is_empty() {
            Some(crate::types::AudioChunk {
                samples: self.segmenter.speech_buffer.clone(),
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

    const SPEECH: f32 = 0.9;
    const QUIET: f32 = 0.1;

    fn config(pad_ms: u32, max_ms: u32) -> VadConfig {
        VadConfig {
            pre_speech_pad_ms: pad_ms,
            max_utterance_ms: max_ms,
            ..VadConfig::default()
        }
    }

    /// A frame whose samples all carry the frame index, to track audio through
    /// the segmenter.
    fn frame(index: usize) -> Vec<f32> {
        vec![index as f32; FRAME_SIZE]
    }

    fn utterances(events: Vec<VadEvent>) -> Vec<Utterance> {
        events
            .into_iter()
            .filter_map(|e| match e {
                VadEvent::SpeechEnd(u) => Some(u),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn default_max_utterance_is_30s() {
        assert_eq!(VadConfig::default().max_utterance_ms, 30_000);
    }

    #[test]
    fn lookback_capacity_follows_pad_ms() {
        // 300 ms at 16 kHz = 4800 samples.
        assert_eq!(SpeechSegmenter::new(config(300, 0)).lookback_capacity, 4800);
        assert_eq!(SpeechSegmenter::new(config(0, 0)).lookback_capacity, 0);
    }

    #[test]
    fn silence_only_emits_silence() {
        let mut seg = SpeechSegmenter::new(config(300, 0));
        for i in 0..50 {
            let events = seg.push(QUIET, &frame(i));
            assert!(matches!(events.as_slice(), [VadEvent::Silence]));
        }
        // The lookback ring keeps only the most recent 4800 samples.
        assert_eq!(seg.lookback_buffer.len(), 4800);
        assert_eq!(*seg.lookback_buffer.back().unwrap(), 49.0);
    }

    #[test]
    fn speech_then_silence_emits_one_utterance_with_lookback() {
        // 64 ms pad = 1024 samples = exactly two frames of lookback.
        let mut seg = SpeechSegmenter::new(config(64, 0));
        let mut events = Vec::new();

        for i in 0..4 {
            events.extend(seg.push(QUIET, &frame(i)));
        }
        let start = seg.push(SPEECH, &frame(4));
        assert!(matches!(start.as_slice(), [VadEvent::SpeechStart]));
        for i in 5..14 {
            assert!(seg.push(SPEECH, &frame(i)).is_empty());
        }
        // 500 ms of silence = 16 frames (16 * 32 = 512 ms).
        let mut ended = Vec::new();
        for i in 14..30 {
            ended.extend(seg.push(QUIET, &frame(i)));
        }

        let utts = utterances(ended);
        assert_eq!(utts.len(), 1);
        let samples = &utts[0].audio.samples;
        // 2 lookback frames + 10 speech frames + 16 trailing silence frames.
        assert_eq!(samples.len(), 28 * FRAME_SIZE);
        assert_eq!(samples[0], 2.0, "lookback starts two frames before onset");
        assert_eq!(samples[2 * FRAME_SIZE], 4.0);
        assert_eq!(utts[0].duration_ms, 28 * 32);
        assert!(!seg.is_speaking);
    }

    #[test]
    fn short_blip_is_dropped() {
        let mut seg = SpeechSegmenter::new(config(0, 0));
        let mut events = seg.push(SPEECH, &frame(0)); // 32 ms < min_speech_ms (250)
        for i in 1..20 {
            events.extend(seg.push(QUIET, &frame(i)));
        }
        assert!(utterances(events).is_empty());
        assert!(!seg.is_speaking);
    }

    /// Continuous speech longer than `max_utterance_ms` must produce a forced
    /// `SpeechEnd` instead of buffering forever.
    #[test]
    fn max_utterance_forces_speech_end() {
        // 320 ms = 10 frames.
        let mut seg = SpeechSegmenter::new(config(0, 320));

        for i in 0..9 {
            let events = seg.push(SPEECH, &frame(i));
            assert!(
                utterances(events).is_empty(),
                "no SpeechEnd before the limit (frame {i})"
            );
        }
        let events = seg.push(SPEECH, &frame(9));
        let utts = utterances(events);
        assert_eq!(utts.len(), 1, "SpeechEnd forced at the limit");
        assert_eq!(utts[0].audio.samples.len(), 10 * FRAME_SIZE);
        assert_eq!(utts[0].duration_ms, 320);
        assert_eq!(utts[0].audio.sample_rate, SAMPLE_RATE);
        assert!(!seg.is_speaking);
    }

    /// After a forced end, ongoing speech starts a new utterance and no
    /// sample is lost or duplicated across the cut.
    #[test]
    fn max_utterance_splits_without_losing_audio() {
        let mut seg = SpeechSegmenter::new(config(0, 320));
        let mut starts = 0;
        let mut collected = Vec::new();
        let mut expected = Vec::new();

        for i in 0..30 {
            let f = frame(i);
            expected.extend_from_slice(&f);
            for event in seg.push(SPEECH, &f) {
                match event {
                    VadEvent::SpeechStart => starts += 1,
                    VadEvent::SpeechEnd(u) => collected.extend(u.audio.samples),
                    VadEvent::Silence => panic!("unexpected silence"),
                }
            }
        }

        assert_eq!(starts, 3, "each forced segment is announced by SpeechStart");
        assert_eq!(collected, expected);
    }

    /// The limit also applies while trailing silence is being buffered, and
    /// the pre-speech pad counts towards it.
    #[test]
    fn max_utterance_counts_lookback_and_silence() {
        // pad 64 ms (2 frames), limit 160 ms (5 frames).
        let mut seg = SpeechSegmenter::new(config(64, 160));
        seg.push(QUIET, &frame(0));
        seg.push(QUIET, &frame(1));
        assert!(utterances(seg.push(SPEECH, &frame(2))).is_empty()); // 3 frames buffered
        assert!(utterances(seg.push(QUIET, &frame(3))).is_empty()); // 4
        let utts = utterances(seg.push(QUIET, &frame(4))); // 5 -> forced
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].audio.samples.len(), 5 * FRAME_SIZE);
    }

    #[test]
    fn max_utterance_zero_disables_limit() {
        let mut seg = SpeechSegmenter::new(config(0, 0));
        // 2000 frames = 64 s of uninterrupted speech.
        for i in 0..2000 {
            assert!(utterances(seg.push(SPEECH, &frame(i))).is_empty());
        }
        assert_eq!(seg.speech_buffer.len(), 2000 * FRAME_SIZE);
    }

    #[test]
    fn reset_clears_segmentation_state() {
        let mut seg = SpeechSegmenter::new(config(300, 0));
        seg.push(QUIET, &frame(0));
        seg.push(SPEECH, &frame(1));
        seg.reset();
        assert!(!seg.is_speaking);
        assert!(seg.speech_buffer.is_empty());
        assert!(seg.lookback_buffer.is_empty());
    }

    #[test]
    fn model_input_is_context_plus_frame() {
        assert_eq!(CONTEXT_SIZE, 64);
        assert_eq!(INPUT_SIZE, 576);
        assert_eq!(FRAME_MS, 32);
    }

    #[test]
    fn signature_check_accepts_v5() {
        assert!(check_model_signature(&["input", "state", "sr"], &["output", "stateN"]).is_ok());
    }

    #[test]
    fn signature_check_rejects_v4() {
        // Silero v4 exposes separate h/c LSTM tensors.
        let err = check_model_signature(&["input", "sr", "h", "c"], &["output", "hn", "cn"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("state"), "{err}");
        assert!(err.contains("stateN"), "{err}");
        assert!(err.contains(SILERO_MODEL_REVISION), "{err}");
    }

    // ---- Real model tests -------------------------------------------------
    //
    //   VOX_TEST_MODELS=<dir with silero_vad.onnx> \
    //     cargo test --lib --features silero vad::silero -- --ignored

    fn real_model() -> std::path::PathBuf {
        let dir = std::env::var("VOX_TEST_MODELS").expect("VOX_TEST_MODELS not set");
        std::path::Path::new(&dir).join("silero_vad.onnx")
    }

    /// The real v5 model loads and accepts the 576-sample (64 + 512) input.
    #[tokio::test]
    #[ignore]
    async fn real_model_accepts_576_sample_input() {
        let mut vad = SileroVad::new(real_model()).expect("model should load and pass the probe");
        assert_eq!(vad.input.len(), 576);
        // The load-time probe must not leak into the stream state.
        assert!(vad.state.iter().all(|&s| s == 0.0));
        assert!(vad.context.iter().all(|&s| s == 0.0));

        let prob = vad.infer(&[0.0; FRAME_SIZE]).unwrap();
        assert!((0.0..=1.0).contains(&prob));
        assert!(prob < 0.5, "silence must not be classified as speech");

        // Context is taken from the tail of the frame just processed.
        let ramp: Vec<f32> = (0..FRAME_SIZE).map(|i| i as f32 / 1000.0).collect();
        vad.infer(&ramp).unwrap();
        assert_eq!(vad.context, &ramp[FRAME_SIZE - CONTEXT_SIZE..]);

        vad.reset();
        assert!(vad.context.iter().all(|&s| s == 0.0));
    }

    #[tokio::test]
    #[ignore]
    async fn real_model_rejects_wrong_frame_size() {
        let mut vad = SileroVad::new(real_model()).unwrap();
        let frame = AudioChunk {
            samples: vec![0.0; 480],
            sample_rate: SAMPLE_RATE,
            channels: 1,
        };
        assert!(vad.process_frame(&frame).await.is_err());
    }

    /// An older model revision fails at load time with a clear message.
    /// Point `VOX_TEST_SILERO_V4` at a Silero v4 `.onnx` file to run this.
    #[test]
    #[ignore]
    fn real_v4_model_is_rejected_at_load() {
        let path = std::env::var("VOX_TEST_SILERO_V4").expect("VOX_TEST_SILERO_V4 not set");
        let err = match SileroVad::new(&path) {
            Ok(_) => panic!("a v4 model must not load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(SILERO_MODEL_REVISION), "{err}");
    }

    /// Real speech is segmented into an utterance. Point `VOX_TEST_SPEECH_WAV`
    /// at a 16 kHz mono 16-bit WAV with a few seconds of speech, e.g.
    /// `say -o speech.wav --data-format=LEI16@16000 "hello there"` on macOS.
    #[tokio::test]
    #[ignore]
    async fn real_model_detects_speech_in_wav() {
        let wav = std::env::var("VOX_TEST_SPEECH_WAV").expect("VOX_TEST_SPEECH_WAV not set");
        let mut reader = hound::WavReader::open(wav).unwrap();
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let mut samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect();
        let speech_len = samples.len();
        // One second of trailing silence so the utterance can end.
        samples.resize(speech_len + SAMPLE_RATE as usize, 0.0);

        let mut vad = SileroVad::new(real_model()).unwrap();
        let mut max_prob = 0.0f32;
        let mut starts = 0;
        let mut utterance_samples = 0;
        for chunk in samples.chunks_exact(FRAME_SIZE) {
            let frame = AudioChunk {
                samples: chunk.to_vec(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
            };
            for event in vad.process_frame(&frame).await.unwrap() {
                match event {
                    VadEvent::SpeechStart => starts += 1,
                    VadEvent::SpeechEnd(u) => utterance_samples += u.audio.samples.len(),
                    VadEvent::Silence => {}
                }
            }
        }

        // Probabilities on the same audio, to show the model is confident.
        vad.reset();
        for chunk in samples.chunks_exact(FRAME_SIZE) {
            max_prob = max_prob.max(vad.infer(chunk).unwrap());
        }

        println!("starts={starts} utterance_samples={utterance_samples} max_prob={max_prob}");
        assert!(starts >= 1, "speech must be detected");
        assert!(max_prob > 0.9, "model should be confident on clean speech");
        assert!(
            utterance_samples * 2 > speech_len,
            "most of the recording should end up in utterances"
        );
    }

    /// With the context fed correctly, digital silence stays silent.
    #[tokio::test]
    #[ignore]
    async fn real_model_silence_stream_stays_silent() {
        let mut vad = SileroVad::new(real_model()).unwrap();
        let frame = AudioChunk {
            samples: vec![0.0; FRAME_SIZE],
            sample_rate: SAMPLE_RATE,
            channels: 1,
        };
        for _ in 0..100 {
            let events = vad.process_frame(&frame).await.unwrap();
            assert!(matches!(events.as_slice(), [VadEvent::Silence]));
        }
        assert!(vad.current_speech_buffer().is_none());
    }
}
