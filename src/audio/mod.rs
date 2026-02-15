//! Audio capture, buffering, resampling, and playback.
//!
//! This module provides the audio I/O layer for the Vox pipeline,
//! including microphone capture via `cpal`, ring buffering for
//! frame-based processing, sample rate conversion via `rubato`,
//! and audio playback via `rodio`.

mod buffer;
mod capture;
mod resampler;

#[cfg(any(feature = "kokoro", feature = "pocket", feature = "tts"))]
mod playback;

pub use buffer::AudioBuffer;
pub use capture::AudioCapture;
pub use resampler::AudioResampler;

#[cfg(any(feature = "kokoro", feature = "pocket", feature = "tts"))]
pub use playback::AudioPlayer;
