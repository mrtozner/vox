//! PyO3 bindings for the Vox voice AI framework.
//!
//! This crate exposes Vox's Rust pipeline to Python as `vox_voice._vox_voice`.
//! Install with `pip install vox-voice` (or `maturin develop` during
//! development).

use pyo3::prelude::*;

mod error;
mod pipeline;
mod runtime;
mod stt;
mod tts;
mod util;
mod vad;

/// Native extension module for vox-voice.
#[pymodule]
fn _vox_voice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<vad::SileroVad>()?;
    m.add_class::<stt::WhisperStt>()?;
    m.add_class::<stt::TranscribeResult>()?;
    m.add_class::<tts::KokoroTts>()?;
    m.add_class::<tts::AudioOutput>()?;
    m.add_class::<pipeline::Vox>()?;
    m.add_class::<pipeline::VoxListener>()?;
    Ok(())
}
