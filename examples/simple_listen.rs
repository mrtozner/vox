//! Simple voice transcription example.
//!
//! Listens to the default microphone, detects speech, and prints
//! transcriptions to stdout.
//!
//! Requirements:
//!   - Download a Whisper model: `wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`
//!   - Download Silero VAD: `wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
//!
//! Run:
//!   cargo run --example simple_listen

use vox::{SileroVad, Vox, WhisperBackend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let vad = SileroVad::new("silero_vad.onnx")?;
    let stt = WhisperBackend::from_model("ggml-tiny.en.bin")?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("You said: {}", result.text);
        })
        .build()?;

    println!("Listening... (Ctrl+C to stop)");
    vox.listen().await?;

    Ok(())
}
