//! Handler for `vox listen` — real-time microphone transcription.

#[cfg(feature = "distil-whisper")]
use super::models::{distil_whisper_download_name, distil_whisper_model_filename};
use super::models::{ensure_model, model_filename, whisper_download_name};
use vox::{SileroVad, Vox};

/// Run the listen command.
pub async fn run(model: &str, stt_backend: &str, yes: bool) -> anyhow::Result<()> {
    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    println!("Loading VAD model...");
    let vad = SileroVad::new(&vad_path)?;

    match stt_backend {
        "whisper" => run_whisper(model, vad, yes).await,
        #[cfg(feature = "distil-whisper")]
        "distil-whisper" => run_distil_whisper(model, vad, yes).await,
        #[cfg(not(feature = "distil-whisper"))]
        "distil-whisper" => anyhow::bail!(
            "Distil-Whisper STT requires the 'distil-whisper' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,distil-whisper --release\n"
        ),
        #[cfg(feature = "sherpa")]
        "sherpa" => run_sherpa(vad, yes).await,
        #[cfg(not(feature = "sherpa"))]
        "sherpa" => anyhow::bail!(
            "Sherpa STT requires the 'sherpa' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,sherpa --release\n"
        ),
        #[cfg(feature = "sherpa")]
        "sherpa-streaming" => run_sherpa_streaming(vad, yes).await,
        #[cfg(not(feature = "sherpa"))]
        "sherpa-streaming" => anyhow::bail!(
            "Sherpa streaming STT requires the 'sherpa' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,sherpa --release\n"
        ),
        other => anyhow::bail!(
            "Unknown STT backend: '{other}'. Use 'whisper', 'distil-whisper', 'sherpa', or 'sherpa-streaming'."
        ),
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

#[cfg(feature = "distil-whisper")]
async fn run_distil_whisper(model: &str, vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    let distil_whisper_file = distil_whisper_model_filename(model);
    let distil_whisper_name = distil_whisper_download_name(model);
    let distil_whisper_path = ensure_model(&distil_whisper_name, &distil_whisper_file, yes).await?;

    println!("Loading Distil-Whisper model ({model})...");
    let stt = vox::DistilWhisperBackend::from_model(&distil_whisper_path)?;

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

#[cfg(feature = "sherpa")]
async fn run_sherpa_streaming(vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    use std::io::Write;

    // Download streaming model.
    let model_dir = super::models::ensure_sherpa_streaming(yes).await?;

    println!("Loading Sherpa streaming zipformer model...");
    let streaming_backend = vox::SherpaStreamingBackend::from_transducer(&model_dir)?;

    // Also load a batch STT backend as fallback.
    let sensevoice_dir = super::models::ensure_sherpa_sensevoice(yes).await?;
    println!("Loading Sherpa SenseVoice model (batch fallback)...");
    let batch_stt = vox::SherpaBackend::from_sensevoice(&sensevoice_dir)?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(batch_stt)
        .streaming_stt(streaming_backend)
        .on_partial(|text| {
            print!("\r\x1b[2K[partial] {text}");
            let _ = std::io::stdout().flush();
        })
        .on_utterance(|result, _ctx| {
            println!("\r\x1b[2K[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;
    Ok(())
}
