//! Handler for `vox chat` -- voice conversation with an LLM via Ollama.
//!
//! Each utterance becomes one turn of a [`vox::streaming_chat::ChatSession`]:
//! the reply is streamed from Ollama's `/api/chat`, printed token by token,
//! and spoken segment by segment while the rest is still being generated. The
//! session keeps the conversation, so later turns see earlier ones.
//!
//! TTS is Piper when the build has the `piper` feature, otherwise Kokoro.

#[cfg(any(feature = "piper", feature = "kokoro"))]
use super::models::{ensure_model, model_filename, whisper_download_name};

/// Turn the `--language` value into a language code, or `None` for
/// auto-detect.
#[cfg(any(feature = "piper", feature = "kokoro"))]
fn normalize_language(language: Option<&str>) -> Option<String> {
    let lang = language?.trim().to_lowercase();
    if lang.is_empty() || lang == "auto" {
        None
    } else {
        Some(lang)
    }
}

#[cfg(any(feature = "piper", feature = "kokoro"))]
fn is_english(language: &str) -> bool {
    language == "en" || language == "english" || language.starts_with("en-")
}

/// Pick the Whisper model for the requested language.
///
/// English-only models (`*.en`) cannot transcribe other languages, so a
/// non-English `--language` switches to the multilingual model of the same
/// size. Returns the model name and whether it was switched.
#[cfg(any(feature = "piper", feature = "kokoro"))]
fn whisper_model_for_language(model: &str, language: Option<&str>) -> (String, bool) {
    match (model.strip_suffix(".en"), language) {
        (Some(multilingual), Some(lang)) if !is_english(lang) => (multilingual.to_string(), true),
        _ => (model.to_string(), false),
    }
}

/// Load Piper with the voice for `language` (English when unset or unknown).
#[cfg(feature = "piper")]
async fn load_tts(language: Option<&str>, yes: bool) -> anyhow::Result<vox::PiperBackend> {
    use super::models::{ensure_piper_voice, piper_voice_alias};

    let wanted = language.unwrap_or("en-us");
    let piper_voice = piper_voice_alias(wanted);
    if piper_voice == "piper-en-us" && !is_english(wanted) {
        println!(
            "⚠ No Piper voice for language '{wanted}'; replies are spoken with the English voice."
        );
    }
    let piper_config = ensure_piper_voice(&piper_voice, yes).await?;
    println!("Loading Piper TTS ({piper_voice})...");
    Ok(vox::PiperBackend::new(&piper_config)?)
}

/// Load Kokoro: the fallback for builds without the `piper` feature.
#[cfg(all(feature = "kokoro", not(feature = "piper")))]
async fn load_tts(language: Option<&str>, yes: bool) -> anyhow::Result<vox::KokoroBackend> {
    if let Some(lang) = language.filter(|l| !is_english(l)) {
        println!(
            "⚠ This build has no Piper TTS; Kokoro speaks replies with its default English voice \
             (requested language: '{lang}')."
        );
        println!("  For other languages rebuild with: --features cli,piper");
    }
    let model_path = ensure_model("kokoro", "kokoro-v1.0.onnx", yes).await?;
    let voices_path = ensure_model("kokoro-voices", "voices.bin", yes).await?;
    println!("Loading Kokoro TTS...");
    Ok(vox::KokoroBackend::new(&model_path, &voices_path).await?)
}

/// Run the chat command.
#[cfg(any(feature = "piper", feature = "kokoro"))]
pub async fn run(
    whisper_model: &str,
    ollama_model: &str,
    ollama_host: &str,
    yes: bool,
    voice_mode: bool,
    diarize: bool,
    language: Option<&str>,
) -> anyhow::Result<()> {
    use std::io::Write;
    use std::sync::Arc;
    use vox::streaming_chat::{ChatSession, StopReason};

    if diarize {
        println!("⚠ --diarize is not supported in `vox chat` yet and is ignored.");
        println!("  Speaker identification is available in Live Talk: vox serve\n");
    }

    let language = normalize_language(language);
    let (whisper_model, switched) = whisper_model_for_language(whisper_model, language.as_deref());
    if switched {
        println!(
            "Using multilingual Whisper model '{whisper_model}' for language '{}'.",
            language.as_deref().unwrap_or_default()
        );
    }

    // Resolve all required models (auto-download if --yes)
    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    let whisper_file = model_filename(&whisper_model);
    let whisper_name = whisper_download_name(&whisper_model);
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    // Initialize backends
    println!("Loading VAD model...");
    let vad = vox::SileroVad::new(&vad_path)?;

    println!("Loading Whisper model ({whisper_model})...");
    let stt = vox::WhisperBackend::with_config(
        vox::WhisperConfig::new(whisper_path).with_language(language.clone()),
    )?;

    let tts = load_tts(language.as_deref(), yes).await?;

    // Build system prompt based on voice mode
    let mode = if voice_mode {
        println!("Using voice-optimized prompts for natural TTS output");
        vox::VoicePromptMode::Voice
    } else {
        vox::VoicePromptMode::Standard
    };
    let system_prompt = vox::prompts::build_system_prompt_for_language(mode, language.as_deref());

    // Verify Ollama is reachable
    let client = reqwest::Client::new();

    println!("Checking Ollama at {ollama_host}...");
    let (request, _egress) = vox::privacy::ollama_http_request(
        &client,
        reqwest::Method::GET,
        ollama_host,
        "/api/tags",
        "llm status check",
        0,
    )?;
    match request.send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✓ Ollama connected.");
        }
        _ => {
            println!("⚠ Warning: could not reach Ollama at {ollama_host}. Chat may fail.");
            println!("  Start Ollama with: ollama serve\n");
        }
    }

    // One session for the whole run: it carries the conversation memory.
    let session = Arc::new(tokio::sync::Mutex::new(ChatSession::new(
        client,
        ollama_host,
        ollama_model,
        Some(system_prompt),
    )));

    // Build pipeline
    let pipeline = vox::Vox::builder()
        .vad(vad)
        .stt(stt)
        .tts(tts)
        .on_utterance(move |result, ctx| {
            let user_text = result.text.trim().to_string();
            if user_text.is_empty() {
                return;
            }
            println!("\n[You] {user_text}");
            print!("\n[Assistant] ");
            std::io::stdout().flush().ok();

            let session = Arc::clone(&session);
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut session = session.lock().await;
                    let ctx = &ctx;
                    let outcome = session
                        .turn(
                            &user_text,
                            tokio_util::sync::CancellationToken::new(),
                            |token| {
                                print!("{token}");
                                std::io::stdout().flush().ok();
                            },
                            |segment: &str| {
                                let segment = segment.to_string();
                                async move {
                                    // The text is already on screen, so a
                                    // TTS failure does not end the turn.
                                    if let Err(e) = ctx.speak_and_play(&segment).await {
                                        eprintln!("\nTTS error: {e}");
                                    }
                                    Ok(())
                                }
                            },
                        )
                        .await;
                    println!("\n");
                    match outcome {
                        Ok(StopReason::Finished) | Ok(StopReason::Cancelled) => {}
                        Err(e) => {
                            eprintln!("[Error] {e}");
                            eprintln!("  Is Ollama running? Start it with: ollama serve\n");
                        }
                    }
                });
            });
        })
        .build()?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("🎙  Chat ready! Speak into your microphone");
    println!("🤖 AI: {ollama_model} via Ollama");
    match &language {
        Some(lang) => println!("🌐 Language: {lang}"),
        None => println!("🌐 Language: auto-detect"),
    }
    println!("⏹  Press Ctrl+C to stop");
    println!("═══════════════════════════════════════════════════════════\n");

    pipeline.listen().await?;

    Ok(())
}

/// Stub when no chat-capable TTS backend is compiled in.
#[cfg(not(any(feature = "piper", feature = "kokoro")))]
pub async fn run(
    _whisper_model: &str,
    _ollama_model: &str,
    _ollama_host: &str,
    _yes: bool,
    _voice_mode: bool,
    _diarize: bool,
    _language: Option<&str>,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "Chat needs a TTS backend, and this build has none.\n\n\
         Install with Piper (recommended) or Kokoro:\n\
         \n  cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli,piper\
         \n  cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli,kokoro\n"
    );
}

#[cfg(all(test, any(feature = "piper", feature = "kokoro")))]
mod tests {
    use super::*;

    #[test]
    fn language_is_normalized() {
        assert_eq!(normalize_language(None), None);
        assert_eq!(normalize_language(Some("")), None);
        assert_eq!(normalize_language(Some("auto")), None);
        assert_eq!(normalize_language(Some(" TR ")), Some("tr".to_string()));
    }

    #[test]
    fn english_only_model_is_swapped_for_other_languages() {
        assert_eq!(
            whisper_model_for_language("tiny.en", Some("tr")),
            ("tiny".to_string(), true)
        );
        assert_eq!(
            whisper_model_for_language("small.en", Some("de")),
            ("small".to_string(), true)
        );
    }

    #[test]
    fn model_is_kept_when_it_fits_the_language() {
        for (model, lang) in [
            ("tiny.en", None),
            ("tiny.en", Some("en")),
            ("tiny.en", Some("en-us")),
            ("base", Some("tr")),
            ("turbo", Some("de")),
        ] {
            assert_eq!(
                whisper_model_for_language(model, lang),
                (model.to_string(), false)
            );
        }
    }
}
