//! Audio capture, buffering, and resampling.
//!
//! This module provides the audio I/O layer for the Vox pipeline,
//! including microphone capture via `cpal`, ring buffering for
//! frame-based processing, and sample rate conversion via `rubato`.

mod buffer;
mod capture;
mod resampler;

pub use buffer::AudioBuffer;
pub use capture::AudioCapture;
pub use resampler::AudioResampler;
