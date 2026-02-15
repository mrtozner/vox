//! Voice assistant example -- full pipeline with LLM + TTS placeholders.
//!
//! Demonstrates the complete Vox pipeline: mic input -> VAD -> STT -> your
//! logic (LLM placeholder) -> TTS response. In a real application you would
//! replace the placeholder LLM call with an actual inference call (e.g. to
//! a local llama.cpp model or a remote API).
//!
//! Requirements:
//!   - Download a Whisper model: `wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`
//!   - Download Silero VAD: `wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
//!
//! Run:
//!   cargo run --example voice_assistant

use vox::{SileroVad, Vox, WhisperBackend};

/// Placeholder for an LLM call. In a real assistant you would send the
/// transcribed text to a language model and return the response.
fn ask_llm(input: &str) -> String {
    // Placeholder -- echo back with a canned response.
    format!("You said: \"{input}\". I'm a placeholder LLM -- plug in your own model here!")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let vad = SileroVad::new("silero_vad.onnx")?;
    let stt = WhisperBackend::from_model("ggml-tiny.en.bin")?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[You]       {}", result.text);

            let response = ask_llm(&result.text);
            println!("[Assistant] {response}");

            // When a TTS backend is configured, you could speak the response:
            //
            //   let rt = tokio::runtime::Handle::current();
            //   rt.spawn(async move {
            //       if let Err(e) = ctx.speak(&response).await {
            //           eprintln!("TTS error: {e}");
            //       }
            //   });
            //
            // TTS backends (Kokoro, Piper) are coming soon.

            let stats = _ctx.stats();
            println!(
                "  [{} utterances | avg STT latency: {:.0}ms]",
                stats.utterance_count, stats.avg_stt_latency_ms
            );
        })
        .build()?;

    println!("Voice assistant ready. Speak into your microphone. (Ctrl+C to stop)");
    vox.listen().await?;

    Ok(())
}
