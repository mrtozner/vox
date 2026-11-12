//! Main Vox engine and pipeline orchestration.
//!
//! The [`Vox`] struct is the primary entry point. Use [`VoxBuilder`]
//! to configure backends and options, then call `build()` to create
//! a running pipeline.

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::audio::{AudioCapture, AudioResampler, FrameAssembler};
use crate::error::VoxError;
use crate::traits::{
    StreamingSttBackend, SttBackend, SttSession, TtsBackend, VadBackend, VadEvent,
};
use crate::types::{AudioChunk, PipelineStats, SttResult, TtsOutput, TtsRequest};

/// Configuration for the Vox pipeline.
#[derive(Debug, Clone)]
pub struct VoxConfig {
    /// Preferred sample rate for audio capture (default: 16000).
    ///
    /// If the input device does not support it, the device's own rate is
    /// used and the audio is resampled for the VAD.
    pub sample_rate: u32,
    /// Preferred number of audio channels (default: 1 -- mono).
    pub channels: u16,
    /// Whether to enable TTS output (default: false).
    pub enable_tts: bool,
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            enable_tts: false,
        }
    }
}

/// Context passed to the user callback alongside each transcription result.
///
/// Provides access to optional TTS synthesis and pipeline statistics.
pub struct VoxContext {
    tts: Option<Arc<dyn TtsBackend>>,
    stats: Arc<Mutex<PipelineStats>>,
    #[cfg(any(
        feature = "kokoro",
        feature = "pocket",
        feature = "chatterbox",
        feature = "piper",
        feature = "tts"
    ))]
    audio_player: Option<Arc<crate::audio::AudioPlayer>>,
}

impl VoxContext {
    /// Speak text using the TTS backend (if configured).
    pub async fn speak(&self, text: &str) -> Result<TtsOutput, VoxError> {
        let tts = self
            .tts
            .as_ref()
            .ok_or(VoxError::Tts("no TTS configured".into()))?;
        tts.synthesize(&TtsRequest {
            text: text.to_string(),
            voice: None,
            seed: None,
        })
        .await
    }

    /// Speak text and play through speakers.
    ///
    /// Synthesizes TTS then plays the audio through the default output device.
    #[cfg(any(
        feature = "kokoro",
        feature = "pocket",
        feature = "chatterbox",
        feature = "piper",
        feature = "tts"
    ))]
    pub async fn speak_and_play(&self, text: &str) -> Result<TtsOutput, VoxError> {
        let output = self.speak(text).await?;
        if let Some(player) = &self.audio_player {
            player.play_blocking(&output.audio)?;
        }
        Ok(output)
    }

    /// Get current pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Builder for constructing a [`Vox`] pipeline.
pub struct VoxBuilder {
    config: VoxConfig,
    vad: Option<Box<dyn VadBackend>>,
    stt: Option<Box<dyn SttBackend>>,
    tts: Option<Box<dyn TtsBackend>>,
    streaming_stt: Option<Box<dyn StreamingSttBackend>>,
    on_partial: Option<Box<dyn Fn(String) + Send + Sync>>,
    callback: Option<Box<dyn Fn(SttResult, VoxContext) + Send + Sync>>,
}

impl VoxBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: VoxConfig::default(),
            vad: None,
            stt: None,
            tts: None,
            streaming_stt: None,
            on_partial: None,
            callback: None,
        }
    }

    /// Set the pipeline configuration.
    pub fn config(mut self, config: VoxConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the VAD backend.
    pub fn vad(mut self, vad: impl VadBackend + 'static) -> Self {
        self.vad = Some(Box::new(vad));
        self
    }

    /// Set the STT backend.
    pub fn stt(mut self, stt: impl SttBackend + 'static) -> Self {
        self.stt = Some(Box::new(stt));
        self
    }

    /// Set the TTS backend.
    pub fn tts(mut self, tts: impl TtsBackend + 'static) -> Self {
        self.tts = Some(Box::new(tts));
        self
    }

    /// Set the optional streaming STT backend for real-time partial results.
    ///
    /// When set, audio is processed incrementally during speech,
    /// producing partial transcriptions via the [`on_partial`](Self::on_partial) callback.
    /// The batch STT backend is still used as a fallback.
    pub fn streaming_stt(mut self, stt: impl StreamingSttBackend + 'static) -> Self {
        self.streaming_stt = Some(Box::new(stt));
        self
    }

    /// Register a callback for partial transcription results.
    ///
    /// Called when the streaming STT session produces updated text
    /// during speech (before the utterance ends).
    pub fn on_partial(mut self, callback: impl Fn(String) + Send + Sync + 'static) -> Self {
        self.on_partial = Some(Box::new(callback));
        self
    }

    /// Register a callback invoked for each transcribed utterance.
    ///
    /// The callback receives the [`SttResult`] and a [`VoxContext`] that
    /// can be used to speak back or inspect pipeline statistics.
    pub fn on_utterance(
        mut self,
        callback: impl Fn(SttResult, VoxContext) + Send + Sync + 'static,
    ) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }

    /// Build the Vox pipeline.
    ///
    /// Initializes audio capture and resampler. Requires at minimum a VAD
    /// and STT backend to be configured.
    pub fn build(self) -> Result<Vox, VoxError> {
        if self.vad.is_none() {
            return Err(VoxError::NoVad);
        }
        if self.stt.is_none() {
            return Err(VoxError::NoStt);
        }

        let (capture, audio_rx) = AudioCapture::new(self.config.sample_rate, self.config.channels)?;
        // The device may have fallen back to its own format (commonly 44.1 or
        // 48 kHz); the resampler must be built from what is actually captured.
        let source_rate = capture.sample_rate();

        self.assemble(Some(capture), audio_rx, source_rate)
    }

    /// Build the Vox pipeline on top of an external audio source instead of
    /// the microphone (a file reader, a network stream, a test harness).
    ///
    /// `source_rate` is the expected sample rate of the chunks; chunks may be
    /// of any length and channel count, and a chunk carrying a different
    /// `sample_rate` reconfigures the resampler. [`Vox::listen`] returns when
    /// the sending side of `audio_rx` is dropped.
    pub fn build_with_source(
        self,
        audio_rx: mpsc::Receiver<AudioChunk>,
        source_rate: u32,
    ) -> Result<Vox, VoxError> {
        if self.vad.is_none() {
            return Err(VoxError::NoVad);
        }
        if self.stt.is_none() {
            return Err(VoxError::NoStt);
        }

        self.assemble(None, audio_rx, source_rate)
    }

    fn assemble(
        mut self,
        capture: Option<AudioCapture>,
        audio_rx: mpsc::Receiver<AudioChunk>,
        source_rate: u32,
    ) -> Result<Vox, VoxError> {
        let vad = self.vad.take().ok_or(VoxError::NoVad)?;
        let stt = self.stt.take().ok_or(VoxError::NoStt)?;
        let callback = self.callback.unwrap_or_else(|| {
            Box::new(|result: SttResult, _ctx: VoxContext| {
                tracing::info!(text = %result.text, "utterance received (no callback registered)");
            })
        });

        let resampler = AudioResampler::new(source_rate, vad.sample_rate())?;
        let framer = FrameAssembler::new(vad.frame_size());

        let tts: Option<Arc<dyn TtsBackend>> = self.tts.map(Arc::from);

        #[cfg(any(
            feature = "kokoro",
            feature = "pocket",
            feature = "chatterbox",
            feature = "piper",
            feature = "tts"
        ))]
        let audio_player = if tts.is_some() {
            Some(Arc::new(crate::audio::AudioPlayer::new()?))
        } else {
            None
        };

        Ok(Vox {
            config: self.config,
            vad,
            stt,
            tts,
            streaming_stt: self.streaming_stt,
            on_partial: self.on_partial,
            active_session: None,
            audio_rx,
            resampler,
            framer,
            capture,
            callback,
            stats: Arc::new(Mutex::new(PipelineStats::default())),
            #[cfg(any(
                feature = "kokoro",
                feature = "pocket",
                feature = "chatterbox",
                feature = "piper",
                feature = "tts"
            ))]
            audio_player,
        })
    }
}

impl Default for VoxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The main Vox pipeline.
///
/// Orchestrates audio capture, VAD, STT, and optional TTS into
/// a unified event loop. Created via [`VoxBuilder`].
pub struct Vox {
    config: VoxConfig,
    vad: Box<dyn VadBackend>,
    stt: Box<dyn SttBackend>,
    tts: Option<Arc<dyn TtsBackend>>,
    streaming_stt: Option<Box<dyn StreamingSttBackend>>,
    on_partial: Option<Box<dyn Fn(String) + Send + Sync>>,
    active_session: Option<Box<dyn SttSession>>,
    audio_rx: mpsc::Receiver<AudioChunk>,
    resampler: AudioResampler,
    /// Carries resampled samples across chunks so the VAD only ever sees
    /// complete frames and no sample is dropped at a chunk boundary.
    framer: FrameAssembler,
    /// Microphone capture; `None` when built with an external audio source.
    capture: Option<AudioCapture>,
    callback: Box<dyn Fn(SttResult, VoxContext) + Send + Sync>,
    stats: Arc<Mutex<PipelineStats>>,
    #[cfg(any(
        feature = "kokoro",
        feature = "pocket",
        feature = "chatterbox",
        feature = "piper",
        feature = "tts"
    ))]
    audio_player: Option<Arc<crate::audio::AudioPlayer>>,
}

impl Vox {
    /// Create a new [`VoxBuilder`].
    pub fn builder() -> VoxBuilder {
        VoxBuilder::new()
    }

    /// Run the voice pipeline until shutdown.
    ///
    /// Captures audio from the microphone, runs VAD to detect speech,
    /// transcribes speech segments via STT, and calls the registered
    /// callback with each result.
    ///
    /// The loop terminates when:
    /// - The audio capture channel closes (e.g. device disconnected)
    /// - Ctrl+C is received
    pub async fn listen(mut self) -> Result<(), VoxError> {
        if let Some(capture) = &self.capture {
            capture.start()?;
        }

        let start_time = std::time::Instant::now();

        tracing::info!(
            requested_sample_rate = self.config.sample_rate,
            requested_channels = self.config.channels,
            source_sample_rate = self.resampler.source_rate(),
            vad_sample_rate = self.resampler.target_rate(),
            vad_frame_size = self.framer.frame_size(),
            "vox pipeline started, listening..."
        );

        loop {
            tokio::select! {
                chunk = self.audio_rx.recv() => {
                    let chunk = match chunk {
                        Some(c) => c,
                        None => {
                            tracing::info!("audio channel closed, shutting down");
                            self.finish_stream(start_time).await?;
                            break;
                        }
                    };

                    self.process_chunk(chunk, start_time).await?;
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("ctrl+c received, shutting down");
                    break;
                }
            }
        }

        if let Some(capture) = &self.capture {
            capture.stop()?;
        }
        tracing::info!(
            uptime_secs = start_time.elapsed().as_secs(),
            "vox pipeline stopped"
        );

        Ok(())
    }

    /// Process a single audio chunk through the VAD/STT pipeline.
    ///
    /// The chunk is resampled to the VAD rate and appended to the frame
    /// assembler; every complete VAD frame is then processed. Samples that do
    /// not fill a frame wait for the next chunk.
    async fn process_chunk(
        &mut self,
        chunk: AudioChunk,
        start_time: std::time::Instant,
    ) -> Result<(), VoxError> {
        let t_chunk = std::time::Instant::now();

        if chunk.sample_rate != self.resampler.source_rate() {
            tracing::warn!(
                expected = self.resampler.source_rate(),
                actual = chunk.sample_rate,
                "audio source sample rate changed, rebuilding resampler"
            );
            self.resampler = AudioResampler::new(chunk.sample_rate, self.resampler.target_rate())?;
        }

        let resampled = self.resampler.process(&chunk)?;
        self.framer.push(&resampled.samples);
        self.drain_frames(start_time).await?;

        tracing::debug!(
            elapsed_us = t_chunk.elapsed().as_micros(),
            "process_chunk total"
        );

        Ok(())
    }

    /// Push out what is still buffered when the audio source ends.
    ///
    /// The resampler tail is flushed and the last partial frame is padded
    /// with silence, so the end of the stream reaches the VAD too.
    async fn finish_stream(&mut self, start_time: std::time::Instant) -> Result<(), VoxError> {
        let tail = self.resampler.flush()?;
        self.framer.push(&tail.samples);

        let pending = self.framer.pending() % self.framer.frame_size();
        if pending > 0 {
            let padding = vec![0.0; self.framer.frame_size() - pending];
            self.framer.push(&padding);
        }

        self.drain_frames(start_time).await
    }

    /// Run every complete frame in the assembler through VAD and STT.
    async fn drain_frames(&mut self, start_time: std::time::Instant) -> Result<(), VoxError> {
        let sample_rate = self.resampler.target_rate();

        while let Some(samples) = self.framer.pop() {
            let frame = AudioChunk {
                samples,
                sample_rate,
                channels: 1,
            };

            let t_vad = std::time::Instant::now();
            let events = self.vad.process_frame(&frame).await?;
            tracing::debug!(
                elapsed_us = t_vad.elapsed().as_micros(),
                "vad frame processed"
            );

            if let Some(session) = &mut self.active_session {
                let t_push = std::time::Instant::now();
                match session.push_audio(&frame.samples, sample_rate) {
                    Ok(Some(partial)) => {
                        if let Some(on_partial) = &self.on_partial {
                            on_partial(partial);
                        }
                    }
                    Ok(None) => {} // no new text yet
                    Err(e) => {
                        tracing::warn!("streaming push_audio failed: {e}, dropping session");
                        self.active_session = None;
                    }
                }
                tracing::debug!(
                    elapsed_us = t_push.elapsed().as_micros(),
                    "streaming push_audio"
                );
            }

            for event in events {
                match event {
                    VadEvent::SpeechStart => {
                        tracing::debug!("speech started");
                        if let Some(streaming) = &self.streaming_stt {
                            match streaming.create_session() {
                                Ok(session) => self.active_session = Some(session),
                                Err(e) => tracing::warn!("failed to create streaming session: {e}"),
                            }
                        }
                    }
                    VadEvent::SpeechEnd(utterance) => {
                        tracing::debug!(
                            duration_ms = utterance.duration_ms,
                            "speech ended, transcribing..."
                        );

                        let t_stt = std::time::Instant::now();
                        let stt_result = if let Some(mut session) = self.active_session.take() {
                            #[cfg(feature = "diarization")]
                            session.set_speaker_id(utterance.speaker_id.clone());
                            match session.finish() {
                                Ok(result) => result,
                                Err(e) => {
                                    tracing::warn!(
                                        "streaming finish failed: {e}, falling back to batch"
                                    );
                                    self.stt.transcribe(&utterance).await?
                                }
                            }
                        } else {
                            self.stt.transcribe(&utterance).await?
                        };
                        tracing::debug!(elapsed_us = t_stt.elapsed().as_micros(), "stt transcribe");

                        if !stt_result.text.is_empty() {
                            tracing::info!(
                                text = %stt_result.text,
                                latency_ms = stt_result.processing_time_ms,
                                "transcription complete"
                            );

                            {
                                let mut stats =
                                    self.stats.lock().unwrap_or_else(|e| e.into_inner());
                                stats.utterance_count += 1;
                                let n = stats.utterance_count as f64;
                                stats.avg_stt_latency_ms = stats.avg_stt_latency_ms
                                    * ((n - 1.0) / n)
                                    + stt_result.processing_time_ms as f64 / n;
                                stats.uptime_secs = start_time.elapsed().as_secs();
                            }

                            let ctx = VoxContext {
                                tts: self.tts.clone(),
                                stats: self.stats.clone(),
                                #[cfg(any(
                                    feature = "kokoro",
                                    feature = "pocket",
                                    feature = "chatterbox",
                                    feature = "piper",
                                    feature = "tts"
                                ))]
                                audio_player: self.audio_player.clone(),
                            };
                            (self.callback)(stt_result, ctx);
                        }
                    }
                    VadEvent::Silence => {}
                }
            }
        }

        Ok(())
    }
}
