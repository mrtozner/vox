//! Python wrapper for the full Vox voice pipeline.
//!
//! The `Vox` class orchestrates VAD + STT (+ optional TTS) into a single
//! `listen()` call that yields transcription results as a Python iterator.

use std::sync::{Mutex, mpsc};

use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;

use crate::stt::{TranscribeResult, WhisperStt};
use crate::tts::KokoroTts;
use crate::vad::SileroVad;

/// The Vox voice pipeline.
///
/// Assembles VAD, STT, and optional TTS into a unified event loop
/// that captures microphone audio and yields transcription results.
///
/// Args:
///     vad: A SileroVad instance for voice activity detection.
///     stt: A WhisperStt instance for speech-to-text.
///     tts: An optional KokoroTts instance for text-to-speech.
///
/// Example:
///     ```python
///     from vox_voice import Vox, SileroVad, WhisperStt
///
///     vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
///     for result in vox.listen():
///         print(result.text)
///     ```
#[pyclass]
pub struct Vox {
    vad: SileroVad,
    stt: WhisperStt,
    tts: Option<KokoroTts>,
}

#[pymethods]
impl Vox {
    #[new]
    #[pyo3(signature = (*, vad, stt, tts=None))]
    fn new(vad: SileroVad, stt: WhisperStt, tts: Option<KokoroTts>) -> PyResult<Self> {
        Ok(Self { vad, stt, tts })
    }

    /// Start listening on the microphone and return an iterator of
    /// TranscribeResult objects.
    ///
    /// The iterator blocks while waiting for audio. It terminates when the
    /// audio device disconnects or the pipeline is interrupted (Ctrl+C).
    ///
    /// Returns:
    ///     VoxListener: An iterator yielding TranscribeResult for each
    ///     detected speech segment.
    fn listen(&self) -> PyResult<VoxListener> {
        let vad_path = self.vad.model_path.clone();
        let stt_path = self.stt.model_path.clone();
        let tts_config = self.tts.clone();

        // Channel to bridge async pipeline results to the sync Python iterator.
        let (tx, rx) = mpsc::sync_channel::<vox::SttResult>(32);

        // Oneshot channel to report build errors back to the calling thread.
        let (err_tx, err_rx) = mpsc::sync_channel::<Option<String>>(1);

        // Build and run the entire pipeline on a dedicated OS thread.
        // The Vox struct contains cpal audio handles that are !Send,
        // so construction AND listening must happen on the same thread.
        // We pass only the Send-able config strings across the boundary.
        let sender = tx;

        std::thread::Builder::new()
            .name("vox-pipeline".into())
            .spawn(move || {
                // Create a thread-local tokio runtime for async ops.
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = err_tx.send(Some(format!("failed to create tokio runtime: {e}")));
                        return;
                    }
                };

                // Build the pipeline inside the dedicated thread.
                let build_result: Result<vox::Vox, vox::VoxError> = rt.block_on(async {
                    let vad = vox::SileroVad::new(&vad_path)?;
                    let stt = vox::WhisperBackend::from_model(&stt_path)?;
                    let mut builder = vox::Vox::builder().vad(vad).stt(stt);

                    if let Some(ref tts_cfg) = tts_config {
                        let tts_backend =
                            vox::KokoroBackend::new(&tts_cfg.model_path, &tts_cfg.voices_path)
                                .await?;
                        builder = builder.tts(tts_backend);
                    }

                    builder = builder.on_utterance(move |result, _ctx| {
                        let _ = sender.send(result);
                    });

                    builder.build()
                });

                let pipeline = match build_result {
                    Ok(p) => {
                        // Signal success to the calling thread.
                        let _ = err_tx.send(None);
                        p
                    }
                    Err(e) => {
                        let _ = err_tx.send(Some(e.to_string()));
                        return;
                    }
                };

                // Run the pipeline. Blocks until audio stops or Ctrl+C.
                if let Err(e) = rt.block_on(pipeline.listen()) {
                    eprintln!("vox pipeline error: {e}");
                }
            })
            .map_err(|e| PyRuntimeError::new_err(format!("failed to spawn thread: {e}")))?;

        // Wait for the pipeline thread to report build success or failure.
        match err_rx.recv() {
            Ok(None) => Ok(VoxListener { rx: Mutex::new(rx) }),
            Ok(Some(err_msg)) => Err(PyRuntimeError::new_err(err_msg)),
            Err(_) => Err(PyRuntimeError::new_err(
                "pipeline thread exited before reporting status",
            )),
        }
    }

    fn __repr__(&self) -> String {
        let tts_str = if self.tts.is_some() { "yes" } else { "no" };
        format!(
            "Vox(vad={}, stt={}, tts={})",
            self.vad.repr(),
            self.stt.repr(),
            tts_str
        )
    }
}

/// Iterator that yields TranscribeResult objects from the live pipeline.
///
/// Created by `Vox.listen()`. Blocks on each call to `__next__` waiting
/// for the next transcription result.
#[pyclass]
pub struct VoxListener {
    rx: Mutex<mpsc::Receiver<vox::SttResult>>,
}

#[pymethods]
impl VoxListener {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> PyResult<TranscribeResult> {
        let rx_guard = self
            .rx
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("lock poisoned: {e}")))?;

        // Block until the next transcription result arrives.
        // The pipeline thread produces results asynchronously;
        // this method consumes them synchronously.
        match rx_guard.recv() {
            Ok(stt_result) => Ok(TranscribeResult {
                text: stt_result.text,
                language: stt_result.language,
                duration_ms: stt_result.duration_ms,
                processing_time_ms: stt_result.processing_time_ms,
            }),
            Err(_) => {
                // Channel closed -- pipeline has stopped.
                Err(PyStopIteration::new_err("pipeline stopped"))
            }
        }
    }
}
