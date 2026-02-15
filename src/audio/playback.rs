//! Audio playback through system speakers via `rodio`.
//!
//! Uses a dedicated thread to own the audio output stream,
//! with channel-based communication for Send+Sync compatibility.

use std::sync::mpsc;
use std::thread;

use crate::error::VoxError;
use crate::types::AudioChunk;

enum PlaybackCommand {
    Play(AudioChunk),
    PlayBlocking(AudioChunk, mpsc::Sender<Result<(), String>>),
    Stop,
}

/// Plays audio through the default output device.
///
/// Internally spawns a dedicated thread that owns the rodio `OutputStream`.
/// The handle is `Send + Sync` and can be shared across threads.
pub struct AudioPlayer {
    tx: mpsc::Sender<PlaybackCommand>,
}

impl AudioPlayer {
    /// Create a new player using the default output device.
    ///
    /// Spawns a background thread for audio playback.
    pub fn new() -> Result<Self, VoxError> {
        let (tx, rx) = mpsc::channel::<PlaybackCommand>();

        // Test that we can open the output device before spawning the thread
        // by having the thread signal back success/failure.
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        thread::spawn(move || {
            let (_stream, handle) = match rodio::OutputStream::try_default() {
                Ok(pair) => {
                    let _ = init_tx.send(Ok(()));
                    pair
                }
                Err(e) => {
                    let _ = init_tx.send(Err(format!("no audio output device: {e}")));
                    return;
                }
            };

            while let Ok(cmd) = rx.recv() {
                match cmd {
                    PlaybackCommand::Play(audio) => {
                        if let Ok(sink) = rodio::Sink::try_new(&handle) {
                            let source = rodio::buffer::SamplesBuffer::new(
                                audio.channels,
                                audio.sample_rate,
                                audio.samples,
                            );
                            sink.append(source);
                            // sink is detached -- plays in background
                            sink.detach();
                        }
                    }
                    PlaybackCommand::PlayBlocking(audio, done_tx) => {
                        let result = match rodio::Sink::try_new(&handle) {
                            Ok(sink) => {
                                let source = rodio::buffer::SamplesBuffer::new(
                                    audio.channels,
                                    audio.sample_rate,
                                    audio.samples,
                                );
                                sink.append(source);
                                sink.sleep_until_end();
                                Ok(())
                            }
                            Err(e) => Err(format!("failed to create audio sink: {e}")),
                        };
                        let _ = done_tx.send(result);
                    }
                    PlaybackCommand::Stop => break,
                }
            }
        });

        // Wait for the thread to confirm device initialization
        init_rx
            .recv()
            .map_err(|_| VoxError::Audio("playback thread failed to start".into()))?
            .map_err(VoxError::Audio)?;

        Ok(Self { tx })
    }

    /// Play an AudioChunk and block until playback completes.
    pub fn play_blocking(&self, audio: &AudioChunk) -> Result<(), VoxError> {
        let (done_tx, done_rx) = mpsc::channel();
        self.tx
            .send(PlaybackCommand::PlayBlocking(audio.clone(), done_tx))
            .map_err(|_| VoxError::Audio("playback thread died".into()))?;
        done_rx
            .recv()
            .map_err(|_| VoxError::Audio("playback thread died".into()))?
            .map_err(VoxError::Audio)
    }

    /// Play an AudioChunk without blocking (fire and forget).
    pub fn play(&self, audio: &AudioChunk) -> Result<(), VoxError> {
        self.tx
            .send(PlaybackCommand::Play(audio.clone()))
            .map_err(|_| VoxError::Audio("playback thread died".into()))
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        let _ = self.tx.send(PlaybackCommand::Stop);
    }
}
