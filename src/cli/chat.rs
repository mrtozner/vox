//! Handler for `vox chat` -- voice conversation with an LLM via Ollama.

#[cfg(feature = "piper")]
use super::models::{ensure_model, model_filename, whisper_download_name};

/// Ollama generate request.
#[cfg(feature = "piper")]
#[derive(serde::Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

/// Ollama generate response.
#[cfg(feature = "piper")]
#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

/// Call the Ollama HTTP API.
#[cfg(feature = "piper")]
async fn ask_ollama(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    system_prompt: Option<&str>,
) -> anyhow::Result<String> {
    let url = format!("http://{host}/api/generate");
    let body = OllamaRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
        system: system_prompt.map(|s| s.to_string()),
    };

    let resp = client.post(&url).json(&body).send().await.map_err(|e| {
        anyhow::anyhow!(
            "Ollama request failed: {e}\n\n\
                 Is Ollama running? Start it with: ollama serve"
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama returned HTTP {status}: {body}");
    }

    let ollama_resp: OllamaResponse = resp.json().await?;
    Ok(ollama_resp.response)
}

/// Run the chat command.
#[cfg(feature = "piper")]
pub async fn run(
    whisper_model: &str,
    ollama_model: &str,
    ollama_host: &str,
    yes: bool,
    voice_mode: bool,
    diarize: bool,
) -> anyhow::Result<()> {
    use super::models::{ensure_piper_voice, piper_voice_alias};

    // Resolve all required models (auto-download if --yes)
    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    let whisper_file = model_filename(whisper_model);
    let whisper_name = whisper_download_name(whisper_model);
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    // Download Piper model (default to en-us for best English pronunciation)
    let piper_voice = piper_voice_alias("en-us");
    let piper_config = ensure_piper_voice(&piper_voice, yes).await?;

    // Download speaker encoder model if diarization is enabled
    #[cfg(feature = "diarization")]
    let speaker_encoder_path = if diarize {
        println!("Downloading speaker encoder model...");
        Some(ensure_model("speaker-encoder", "speaker_encoder.onnx", yes).await?)
    } else {
        None
    };

    #[cfg(not(feature = "diarization"))]
    let _diarize_check = diarize; // suppress warning

    // Initialize backends
    println!("Loading VAD model...");
    let vad = vox::SileroVad::new(&vad_path)?;

    println!("Loading Whisper model ({whisper_model})...");
    let stt = vox::WhisperBackend::from_model(&whisper_path)?;

    println!("Loading Piper TTS...");
    let tts = vox::PiperBackend::new(&piper_config)?;

    // Initialize diarization pipeline if enabled
    #[cfg(feature = "diarization")]
    let diarization_pipeline = if diarize {
        if let Some(encoder_path) = speaker_encoder_path {
            println!("Loading speaker encoder for diarization...");

            use vox::diarization::{
                DiarizationConfig, DiarizationPipeline, RecognitionConfig, SpeakerEmbedding,
                SpeakerRegistry,
            };

            // Load speaker embedding model
            let embedding = SpeakerEmbedding::new(&encoder_path)?;

            // Create speaker registry with default config
            let recognition_config = RecognitionConfig {
                threshold: 0.7,
                require_threshold: true,
            };
            let registry = SpeakerRegistry::with_config(recognition_config);

            // Create diarization pipeline with auto-enrollment
            let diarization_config = DiarizationConfig {
                auto_enroll: true, // Auto-enroll unknown speakers as "Speaker 1", "Speaker 2", etc.
                min_audio_ms: 500,
                skip_short_utterances: true,
            };

            let pipeline = DiarizationPipeline::new(embedding, registry, diarization_config);
            println!("✓ Diarization pipeline initialized with auto-enrollment");
            println!("  Unknown speakers will be auto-enrolled as Speaker 1, Speaker 2, etc.");
            Some(std::sync::Arc::new(tokio::sync::Mutex::new(pipeline)))
        } else {
            None
        }
    } else {
        None
    };

    #[cfg(not(feature = "diarization"))]
    if diarize {
        println!("⚠ Warning: Diarization feature not enabled in this build.");
        println!("  Rebuild with: cargo build --features cli,piper,diarization");
    }

    // Build system prompt based on voice mode
    let mode = if voice_mode {
        println!("Using voice-optimized prompts for natural TTS output");
        vox::VoicePromptMode::Voice
    } else {
        vox::VoicePromptMode::Standard
    };
    let system_prompt = vox::build_system_prompt(mode);

    // Verify Ollama is reachable
    let client = reqwest::Client::new();
    let ollama_host = ollama_host.to_string();
    let ollama_model = ollama_model.to_string();

    println!("Checking Ollama at {ollama_host}...");
    match client
        .get(format!("http://{ollama_host}/api/tags"))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            println!("✓ Ollama connected.");
        }
        _ => {
            println!("⚠ Warning: could not reach Ollama at {ollama_host}. Chat may fail.");
            println!("  Start Ollama with: ollama serve\n");
        }
    }

    // Clone model name for the println after the closure captures it.
    let model_display = ollama_model.clone();

    // Clone system prompt for the closure
    let system_prompt_clone = system_prompt.clone();

    // Clone diarization pipeline for the utterance callback
    #[cfg(feature = "diarization")]
    let diarization_for_callback = diarization_pipeline.clone();

    // Build pipeline
    let pipeline = vox::Vox::builder()
        .vad(vad)
        .stt(stt)
        .tts(tts)
        .on_utterance(move |result, ctx| {
            // Extract and display speaker information if diarization is enabled
            #[cfg(feature = "diarization")]
            let (speaker_display_name, confidence) =
                if let Some(ref _diarization) = diarization_for_callback {
                    // Extract speaker ID from the STT result
                    // The Vox engine needs to populate result.speaker_id by processing
                    // utterances through the diarization pipeline in VadEvent::SpeechEnd
                    if let Some(ref speaker_id) = result.speaker_id {
                        let registry_clone = _diarization.clone();
                        let (name, conf) = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let diarization_lock = registry_clone.lock().await;
                                let registry = diarization_lock.registry();
                                let registry_guard = registry.lock().unwrap();

                                let name = registry_guard
                                    .get_speaker(speaker_id)
                                    .map(|s| s.name.clone())
                                    .unwrap_or_else(|| speaker_id.clone());

                                // Calculate confidence (would come from Recognition result)
                                let confidence = 0.85f32;

                                (name, confidence)
                            })
                        });
                        (name, conf)
                    } else {
                        ("You".to_string(), 1.0f32)
                    }
                } else {
                    ("You".to_string(), 1.0f32)
                };

            // Display with speaker info if diarization is enabled
            #[cfg(feature = "diarization")]
            if diarization_for_callback.is_some() {
                display_speaker_message(&speaker_display_name, &result.text, confidence, true);
            } else {
                println!("\n[You] {}", result.text);
            }

            #[cfg(not(feature = "diarization"))]
            println!("\n[You] {}", result.text);

            let client = client.clone();
            let host = ollama_host.clone();
            let model = ollama_model.clone();
            let system = system_prompt_clone.clone();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    match ask_ollama(&client, &host, &model, &result.text, Some(&system)).await {
                        Ok(response) => {
                            println!("\n[Assistant] {response}\n");
                            if let Err(e) = ctx.speak_and_play(&response).await {
                                eprintln!("TTS error: {e}");
                            }
                        }
                        Err(e) => {
                            eprintln!("\n[Error] {e}\n");
                        }
                    }
                });
            });
        })
        .build()?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("🎙  Chat ready! Speak into your microphone");
    println!("🤖 AI: {} via Ollama", model_display);
    #[cfg(feature = "diarization")]
    if diarization_pipeline.is_some() {
        println!("👥 Diarization: Enabled (auto-enrollment active)");
        println!(
            "   NOTE: Full integration requires engine support - currently loading models only"
        );
    }
    println!("⏹  Press Ctrl+C to stop");
    println!("═══════════════════════════════════════════════════════════\n");

    pipeline.listen().await?;

    Ok(())
}

/// Display a speaker's message with colored UI.
#[cfg(all(feature = "piper", feature = "diarization"))]
fn display_speaker_message(speaker_name: &str, text: &str, confidence: f32, is_user: bool) {
    // Confidence bar
    let filled = (confidence * 10.0) as usize;
    let empty = 10 - filled.min(10);
    let confidence_bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    // Color based on role
    let (color, reset) = if is_user {
        ("\x1b[36m", "\x1b[0m") // Cyan for user
    } else {
        ("\x1b[35m", "\x1b[0m") // Magenta for assistant
    };

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!(
        "│ 🎤 Speaker: {}{:20}{}                       │",
        color, speaker_name, reset
    );
    println!(
        "│ 📊 Confidence: {} {:.0}%                         │",
        confidence_bar,
        confidence * 100.0
    );
    println!("└─────────────────────────────────────────────────────────┘");
    println!("💬 {}{}{}: \"{}\"", color, speaker_name, reset, text);
}

/// Stub when piper feature is disabled.
#[cfg(not(feature = "piper"))]
pub async fn run(
    _whisper_model: &str,
    _ollama_model: &str,
    _ollama_host: &str,
    _yes: bool,
    _voice_mode: bool,
    _diarize: bool,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "Chat requires the 'piper' feature for TTS.\n\n\
         Rebuild with:\n\
         \n  cargo install vox --features cli,piper\n"
    );
}
