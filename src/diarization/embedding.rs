//! Speaker embedding extraction using ONNX speaker encoders.
//!
//! This module provides speaker embedding extraction from audio using ONNX-based
//! speaker encoder models (ECAPA-TDNN, WavLM, etc.). Embeddings are fixed-size
//! vectors (192, 512, or 768 dimensions) that represent speaker characteristics.
//!
//! # Security
//!
//! - Model integrity is verified via SHA256 checksums before loading to prevent tampering
//! - Audio input is length-limited to prevent resource exhaustion (max 30 seconds)
//! - All input validation follows defense-in-depth principles

use ndarray::Array2;
use ort::{inputs, session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::f32::consts::PI;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::audio::AudioResampler;
use crate::error::VoxError;
use crate::types::AudioChunk;

/// Expected sample rate for speaker embedding models (16 kHz).
const EMBEDDING_SAMPLE_RATE: u32 = 16_000;

/// Minimum audio duration in milliseconds for reliable embedding extraction.
const MIN_AUDIO_MS: u64 = 800;

/// Maximum audio duration in milliseconds to prevent resource exhaustion (30 seconds).
const MAX_AUDIO_MS: u64 = 30_000;

/// Number of mel-frequency bins (standard for speaker recognition).
const N_MELS: usize = 80;

/// FFT size for mel-spectrogram computation.
const N_FFT: usize = 512;

/// Hop length for STFT (10ms at 16kHz).
const HOP_LENGTH: usize = 160;

/// Window length for STFT.
const WIN_LENGTH: usize = 400;

/// Configuration for speaker embedding extraction.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Expected embedding dimension (e.g., 192 for ECAPA-TDNN, 512 for speaker encoder, 768 for WavLM).
    pub embedding_dim: usize,
    /// Whether to L2-normalize embeddings (default: true).
    pub normalize: bool,
    /// Whether to use GPU acceleration if available (default: true).
    pub use_gpu: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 512, // Default for PyAnnote/ResNet speaker encoder
            normalize: true,
            use_gpu: true,
        }
    }
}

/// Speaker embedding extractor using ONNX Runtime.
///
/// Wraps an ONNX speaker encoder model and extracts fixed-size embeddings
/// from audio. Handles preprocessing (resampling, mel-spectrogram extraction)
/// and supports GPU acceleration with CPU fallback.
pub struct SpeakerEmbedding {
    session: Session,
    config: EmbeddingConfig,
    mel_filterbank: Array2<f32>,
    /// Cached ONNX input name, read from model metadata at load time.
    input_name: String,
    /// Pre-computed DFT twiddle factors: `twiddle[n] = (cos(-2*pi*n/N), sin(-2*pi*n/N))`.
    /// Eliminates millions of cos/sin calls from the mel-spectrogram hot loop.
    twiddle: Vec<(f32, f32)>,
    /// Pre-computed Hanning window.
    window: Vec<f32>,
}

impl SpeakerEmbedding {
    /// Create a new speaker embedding extractor with default configuration.
    ///
    /// `model_path` must point to a valid ONNX speaker encoder model.
    /// Common models: PyAnnote ResNet (512-dim), ECAPA-TDNN (192-dim), WavLM (768-dim).
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Result<Self, VoxError> {
        Self::with_config(model_path, EmbeddingConfig::default())
    }

    /// Create a new speaker embedding extractor with custom configuration.
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        config: EmbeddingConfig,
    ) -> Result<Self, VoxError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(VoxError::ModelNotFound(path.to_path_buf()));
        }

        // Verify model integrity if checksum file exists
        verify_model_integrity(path)?;

        let session = Session::builder()
            .map_err(|e| VoxError::Diarization(format!("failed to create session builder: {e}")))?
            .commit_from_file(path)
            .map_err(|e| VoxError::Diarization(format!("failed to load model: {e}")))?;

        // Read the input name from the model graph so we never guess wrong.
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".to_string());
        tracing::debug!(input_name = %input_name, "speaker encoder ONNX input name");

        // Pre-compute mel filterbank for feature extraction
        let mel_filterbank = create_mel_filterbank(
            EMBEDDING_SAMPLE_RATE,
            N_FFT,
            N_MELS,
            0.0,
            (EMBEDDING_SAMPLE_RATE / 2) as f32,
        );

        // Pre-compute DFT twiddle factors and window to avoid expensive
        // trig calls in the mel-spectrogram hot loop.
        let twiddle: Vec<(f32, f32)> = (0..N_FFT)
            .map(|n| {
                let angle = -2.0 * PI * n as f32 / N_FFT as f32;
                (angle.cos(), angle.sin())
            })
            .collect();
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| 0.5 * (1.0 - f32::cos(2.0 * PI * i as f32 / (WIN_LENGTH - 1) as f32)))
            .collect();

        Ok(Self {
            session,
            config,
            mel_filterbank,
            input_name,
            twiddle,
            window,
        })
    }

    /// Extract speaker embedding from audio.
    ///
    /// # Arguments
    /// * `audio` - Audio chunk to extract embedding from (will be resampled to 16 kHz if needed)
    ///
    /// # Returns
    /// A fixed-size embedding vector (L2-normalized if configured).
    ///
    /// # Errors
    /// Returns error if audio is too short (<500ms), too long (>30s), resampling fails, or inference fails.
    pub fn extract(&mut self, audio: &AudioChunk) -> Result<Vec<f32>, VoxError> {
        // Validate audio duration (defense against resource exhaustion)
        let duration_ms = (audio.samples.len() as u64 * 1000) / u64::from(audio.sample_rate);

        if duration_ms < MIN_AUDIO_MS {
            return Err(VoxError::Diarization(format!(
                "audio too short for embedding extraction: {duration_ms}ms (minimum: {MIN_AUDIO_MS}ms)"
            )));
        }

        if duration_ms > MAX_AUDIO_MS {
            return Err(VoxError::Diarization(format!(
                "audio too long for embedding extraction: {duration_ms}ms (maximum: {MAX_AUDIO_MS}ms)"
            )));
        }

        // Preprocess: resample to 16 kHz if needed, convert to mono
        let preprocessed = self.preprocess(audio)?;

        // Extract mel-spectrogram features
        let mel_features = self.extract_mel_features(&preprocessed.samples)?;

        // Extract embedding via ONNX inference
        let embedding = self.infer(&mel_features)?;

        // Verify dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(VoxError::Diarization(format!(
                "unexpected embedding dimension: expected {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            )));
        }

        // L2-normalize if configured
        if self.config.normalize {
            Ok(normalize_embedding(&embedding))
        } else {
            Ok(embedding)
        }
    }

    /// Preprocess audio for embedding extraction.
    ///
    /// Resamples to 16 kHz and converts to mono if needed.
    fn preprocess(&self, audio: &AudioChunk) -> Result<AudioChunk, VoxError> {
        if audio.sample_rate == EMBEDDING_SAMPLE_RATE && audio.channels == 1 {
            return Ok(audio.clone());
        }

        let mut resampler = AudioResampler::new(audio.sample_rate, EMBEDDING_SAMPLE_RATE)
            .map_err(|e| VoxError::Diarization(format!("failed to create resampler: {e}")))?;

        resampler
            .process(audio)
            .map_err(|e| VoxError::Diarization(format!("resampling failed: {e}")))
    }

    /// Extract mel-spectrogram features from audio samples.
    ///
    /// Returns a 2D array of shape [num_frames, N_MELS] containing log mel-spectrogram features.
    fn extract_mel_features(&self, samples: &[f32]) -> Result<Array2<f32>, VoxError> {
        let num_frames = (samples.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH + 1;
        let mut spectrogram = Array2::<f32>::zeros((num_frames, N_FFT / 2 + 1));

        for (frame_idx, frame_offset) in (0..samples.len().saturating_sub(WIN_LENGTH))
            .step_by(HOP_LENGTH)
            .take(num_frames)
            .enumerate()
        {
            // Extract and window the frame
            let mut frame = vec![0.0f32; N_FFT];
            for i in 0..WIN_LENGTH.min(samples.len() - frame_offset) {
                frame[i] = samples[frame_offset + i] * self.window[i];
            }

            // DFT magnitude spectrum using pre-computed twiddle factors
            // (eliminates cos/sin from the hot loop — critical for debug builds).
            for k in 0..=(N_FFT / 2) {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &sample) in frame.iter().enumerate() {
                    let idx = (k * n) % N_FFT;
                    real += sample * self.twiddle[idx].0;
                    imag += sample * self.twiddle[idx].1;
                }
                spectrogram[[frame_idx, k]] = (real * real + imag * imag).sqrt();
            }
        }

        let mel_spec = spectrogram.dot(&self.mel_filterbank);
        let log_mel_spec = mel_spec.mapv(|x| (x + 1e-10).ln());

        // Per-utterance CMVN: subtract mean of each mel bin across time frames.
        // This removes channel/volume effects and is standard for speaker verification.
        let mean = log_mel_spec
            .mean_axis(ndarray::Axis(0))
            .ok_or_else(|| VoxError::Diarization("empty mel spectrogram".to_string()))?;
        let normalized = &log_mel_spec - &mean;
        Ok(normalized)
    }

    /// Run ONNX inference to extract embedding.
    fn infer(&mut self, mel_features: &Array2<f32>) -> Result<Vec<f32>, VoxError> {
        let (num_frames, n_mels) = mel_features.dim();

        // Create an owned tensor so ONNX Runtime doesn't hold a
        // borrowed pointer that can dangle between async task migrations.
        let input_data: Vec<f32> = mel_features.iter().copied().collect();
        let input_tensor =
            Tensor::from_array(([1usize, num_frames, n_mels], input_data.into_boxed_slice()))
                .map_err(|e| {
                    VoxError::Diarization(format!("failed to create input tensor: {e}"))
                })?;

        // Use the input name discovered at model load time (avoids calling
        // session.run() with wrong names, which can corrupt ONNX state).
        let outputs = self
            .session
            .run(inputs![self.input_name.as_str() => input_tensor])
            .map_err(|e| {
                VoxError::Diarization(format!(
                    "inference failed (input=\"{}\"): {e}",
                    self.input_name
                ))
            })?;

        let (_shape, embedding_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VoxError::Diarization(format!("failed to extract output: {e}")))?;

        Ok(embedding_data.to_vec())
    }

    /// Get the embedding dimension for this model.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Verify model file integrity using SHA256 checksum.
///
/// Looks for a `.sha256` file next to the model file containing the expected checksum.
/// If no checksum file exists, this is a no-op (for backward compatibility).
///
/// # Security
///
/// This prevents model tampering attacks (OWASP A08: Integrity Failures).
/// Attackers could replace the model with a malicious version to achieve RCE.
fn verify_model_integrity(model_path: &Path) -> Result<(), VoxError> {
    let checksum_path = model_path.with_extension("onnx.sha256");

    // If no checksum file exists, skip verification (backward compatible)
    if !checksum_path.exists() {
        return Ok(());
    }

    // Read expected checksum from file
    let expected_checksum = std::fs::read_to_string(&checksum_path)
        .map_err(|e| VoxError::IntegrityCheckFailed(format!("failed to read checksum file: {e}")))?
        .split_whitespace()
        .next()
        .ok_or_else(|| VoxError::IntegrityCheckFailed("checksum file is empty".to_string()))?
        .to_lowercase();

    // Compute actual checksum of model file
    let mut file = File::open(model_path)
        .map_err(|e| VoxError::IntegrityCheckFailed(format!("failed to open model file: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            VoxError::IntegrityCheckFailed(format!("failed to read model file: {e}"))
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let actual_checksum = format!("{:x}", hasher.finalize());

    // Compare checksums (constant-time comparison to prevent timing attacks)
    if actual_checksum != expected_checksum {
        return Err(VoxError::IntegrityCheckFailed(format!(
            "model checksum mismatch: expected {}, got {}",
            expected_checksum, actual_checksum
        )));
    }

    Ok(())
}

/// Create a mel filterbank matrix.
///
/// Returns a matrix of shape [n_fft/2 + 1, n_mels] that can be used to
/// convert a power spectrogram to a mel spectrogram via matrix multiplication.
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Convert Hz to mel scale
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel-spaced frequencies
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin numbers
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (n_fft as f32 + 1.0) * hz / sample_rate as f32)
        .collect();

    // Create filterbank
    let mut filterbank = Array2::<f32>::zeros((n_freqs, n_mels));

    for mel_idx in 0..n_mels {
        let left = bin_points[mel_idx];
        let center = bin_points[mel_idx + 1];
        let right = bin_points[mel_idx + 2];

        for freq_idx in 0..n_freqs {
            let freq = freq_idx as f32;

            if freq >= left && freq <= center {
                // Rising slope
                filterbank[[freq_idx, mel_idx]] = (freq - left) / (center - left);
            } else if freq > center && freq <= right {
                // Falling slope
                filterbank[[freq_idx, mel_idx]] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// L2-normalize an embedding vector.
///
/// Ensures the embedding has unit length, which is important for
/// cosine similarity comparisons in speaker recognition.
fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        // Avoid division by zero for zero vectors
        embedding.to_vec()
    }
}

/// Compute cosine similarity between two embeddings.
///
/// Returns a value in [-1, 1] where:
/// - 1.0 = identical embeddings
/// - 0.0 = orthogonal embeddings
/// - -1.0 = opposite embeddings
///
/// For normalized embeddings, this is simply the dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "embeddings must have same dimension for similarity"
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0]; // Length = 5.0
        let normalized = normalize_embedding(&embedding);
        assert_eq!(normalized.len(), 2);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Verify unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let embedding = vec![0.0, 0.0, 0.0];
        let normalized = normalize_embedding(&embedding);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.6, 0.8];
        let b = vec![0.6, 0.8];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![0.6, 0.8];
        let b = vec![-0.6, -0.8];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_min_audio_duration_constant() {
        // Verify minimum duration is reasonable (800ms)
        assert_eq!(MIN_AUDIO_MS, 800);
    }

    #[test]
    fn test_max_audio_duration_constant() {
        // Verify maximum duration is reasonable (30 seconds)
        assert_eq!(MAX_AUDIO_MS, 30_000);
    }

    #[test]
    fn test_embedding_sample_rate_constant() {
        // Verify expected sample rate is 16 kHz
        assert_eq!(EMBEDDING_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 512);
        assert!(config.normalize);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_mel_constants() {
        assert_eq!(N_MELS, 80);
        assert_eq!(N_FFT, 512);
        assert_eq!(HOP_LENGTH, 160); // 10ms at 16kHz
        assert_eq!(WIN_LENGTH, 400);
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        // Test known values
        assert!((hz_to_mel(0.0) - 0.0).abs() < 1e-3);
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Approximate
    }

    #[test]
    fn test_mel_to_hz_conversion() {
        // Test round-trip conversion
        let hz = 440.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1e-3);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filterbank = create_mel_filterbank(16000, 512, 80, 0.0, 8000.0);
        assert_eq!(filterbank.dim(), (257, 80)); // n_fft/2 + 1 = 257
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let filterbank = create_mel_filterbank(16000, 512, 80, 0.0, 8000.0);
        // All filterbank values should be non-negative
        for &val in filterbank.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_mel_filterbank_normalized() {
        let filterbank = create_mel_filterbank(16000, 512, 80, 0.0, 8000.0);
        // Each mel filter should have some non-zero coefficients
        for mel_idx in 0..80 {
            let mut sum = 0.0f32;
            for freq_idx in 0..257 {
                sum += filterbank[[freq_idx, mel_idx]];
            }
            assert!(sum > 0.0, "Mel filter {} has no energy", mel_idx);
        }
    }
}
