//! Audio playback through system speakers via `rodio`.

use rodio::{OutputStream, OutputStreamHandle, Sink};

use crate::error::VoxError;
use crate::types::AudioChunk;

/// Plays audio through the default output device.
pub struct AudioPlayer {
    _stream: OutputStream,
    handle: OutputStreamHandle,
}

impl AudioPlayer {
    /// Create a new player using the default output device.
    pub fn new() -> Result<Self, VoxError> {
        let (_stream, handle) = OutputStream::try_default()
            .map_err(|e| VoxError::Audio(format!("no audio output device: {e}")))?;
        Ok(Self { _stream, handle })
    }

    /// Play an AudioChunk and block until playback completes.
    pub fn play_blocking(&self, audio: &AudioChunk) -> Result<(), VoxError> {
        let sink = Sink::try_new(&self.handle)
            .map_err(|e| VoxError::Audio(format!("failed to create audio sink: {e}")))?;

        let source = rodio::buffer::SamplesBuffer::new(
            audio.channels,
            audio.sample_rate,
            audio.samples.clone(),
        );

        sink.append(source);
        sink.sleep_until_end();
        Ok(())
    }

    /// Play an AudioChunk without blocking (returns immediately).
    pub fn play(&self, audio: &AudioChunk) -> Result<Sink, VoxError> {
        let sink = Sink::try_new(&self.handle)
            .map_err(|e| VoxError::Audio(format!("failed to create audio sink: {e}")))?;

        let source = rodio::buffer::SamplesBuffer::new(
            audio.channels,
            audio.sample_rate,
            audio.samples.clone(),
        );

        sink.append(source);
        Ok(sink)
    }
}
