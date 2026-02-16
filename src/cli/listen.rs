//! Handler for `vox listen` — real-time microphone transcription.

use super::models::{ensure_model, model_filename, whisper_download_name};
use vox::{SileroVad, Vox};

/// Run the listen command.
pub async fn run(model: &str, stt_backend: &str, yes: bool) -> anyhow::Result<()> {
    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    println!("Loading VAD model...");
    let vad = SileroVad::new(&vad_path)?;

    match stt_backend {
        "whisper" => run_whisper(model, vad, yes).await,
        #[cfg(feature = "sherpa")]
        "sherpa" => run_sherpa(vad, yes).await,
        #[cfg(not(feature = "sherpa"))]
        "sherpa" => anyhow::bail!(
            "Sherpa STT requires the 'sherpa' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,sherpa --release\n"
        ),
        other => anyhow::bail!("Unknown STT backend: '{other}'. Use 'whisper' or 'sherpa'."),
    }
}

async fn run_whisper(model: &str, vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    let whisper_file = model_filename(model);
    let whisper_name = whisper_download_name(model);
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    println!("Loading Whisper model ({model})...");
    let stt = vox::WhisperBackend::from_model(&whisper_path)?;

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

#[cfg(feature = "sherpa")]
async fn run_sherpa(vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    let model_dir = super::models::ensure_sherpa_sensevoice(yes).await?;

    println!("Loading Sherpa SenseVoice model...");
    let stt = vox::SherpaBackend::from_sensevoice(&model_dir)?;

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
