//! Handler for `vox listen` — real-time microphone transcription.

use super::models::{ensure_model, model_filename, whisper_download_name};
use vox::{SileroVad, Vox, WhisperBackend};

/// Run the listen command.
pub async fn run(model: &str, yes: bool) -> anyhow::Result<()> {
    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    let whisper_file = model_filename(model);
    let whisper_name = whisper_download_name(model);
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    // Initialize backends
    println!("Loading VAD model...");
    let vad = SileroVad::new(&vad_path)?;

    println!("Loading Whisper model ({model})...");
    let stt = WhisperBackend::from_model(&whisper_path)?;

    // Build pipeline
    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;

    Ok(())
}
