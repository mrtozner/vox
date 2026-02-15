//! End-to-end voice assistant: mic -> VAD -> STT -> LLM -> TTS -> speaker.
//!
//! Demonstrates the complete Vox pipeline. Speech is captured from the
//! microphone, segmented by VAD, transcribed by Whisper, passed through
//! a pluggable LLM function, and spoken back through Kokoro TTS.
//!
//! Replace `ask_llm()` with your own model (local llama.cpp, remote API, etc.).
//!
//! Requirements:
//!   - Whisper model: `wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`
//!   - Silero VAD:    `wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
//!   - Kokoro model:  `kokoro-v1.0.onnx` + `voices.bin` in working directory or `models/`
//!
//! Run:
//!   cargo run --example voice_assistant --features whisper,silero,kokoro

use vox::{KokoroBackend, SileroVad, Vox, WhisperBackend};

fn ask_llm(input: &str) -> String {
    format!("You said: \"{input}\". I'm a placeholder -- plug in your own model here!")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let vad = SileroVad::new("silero_vad.onnx")?;
    let stt = WhisperBackend::from_model("ggml-tiny.en.bin")?;

    let (kokoro_model, kokoro_voices) = if std::path::Path::new("models/kokoro-v1.0.onnx").exists()
    {
        ("models/kokoro-v1.0.onnx", "models/voices.bin")
    } else {
        ("kokoro-v1.0.onnx", "voices.bin")
    };

    println!("Loading Kokoro TTS...");
    let tts = KokoroBackend::new(kokoro_model, kokoro_voices).await?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .tts(tts)
        .on_utterance(|result, ctx| {
            println!("[You]       {}", result.text);

            let response = ask_llm(&result.text);
            println!("[Assistant] {response}");

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Err(e) = ctx.speak_and_play(&response).await {
                        eprintln!("TTS error: {e}");
                    }
                });
            });

            let stats = ctx.stats();
            println!(
                "  [{} utterances | avg STT latency: {:.0}ms]",
                stats.utterance_count, stats.avg_stt_latency_ms
            );
        })
        .build()?;

    println!("Voice assistant ready -- speak and I'll respond. (Ctrl+C to stop)");
    vox.listen().await?;

    Ok(())
}
