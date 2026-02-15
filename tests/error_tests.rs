//! Tests for error types and error handling paths.

use std::path::PathBuf;
use vox::VoxError;

#[test]
fn error_display_audio() {
    let err = VoxError::Audio("device disconnected".into());
    assert_eq!(err.to_string(), "audio device error: device disconnected");
}

#[test]
fn error_display_vad() {
    let err = VoxError::Vad("inference failed".into());
    assert_eq!(err.to_string(), "VAD error: inference failed");
}

#[test]
fn error_display_stt() {
    let err = VoxError::Stt("model corrupt".into());
    assert_eq!(err.to_string(), "STT error: model corrupt");
}

#[test]
fn error_display_tts() {
    let err = VoxError::Tts("synthesis timeout".into());
    assert_eq!(err.to_string(), "TTS error: synthesis timeout");
}

#[test]
fn error_display_no_stt() {
    let err = VoxError::NoStt;
    assert_eq!(err.to_string(), "no STT backend configured");
}

#[test]
fn error_display_no_vad() {
    let err = VoxError::NoVad;
    assert_eq!(err.to_string(), "no VAD backend configured");
}

#[test]
fn error_display_model_not_found() {
    let err = VoxError::ModelNotFound(PathBuf::from("/tmp/missing.onnx"));
    assert_eq!(err.to_string(), "model not found: /tmp/missing.onnx");
}

#[test]
fn error_display_pipeline() {
    let err = VoxError::Pipeline("unexpected state".into());
    assert_eq!(err.to_string(), "pipeline error: unexpected state");
}

#[test]
fn error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err: VoxError = io_err.into();
    assert!(err.to_string().contains("file missing"));
}

#[test]
fn error_is_debug() {
    let err = VoxError::NoStt;
    let debug = format!("{:?}", err);
    assert!(debug.contains("NoStt"));
}
