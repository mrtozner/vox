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

    // Initialize backends
    println!("Loading VAD model...");
    let vad = vox::SileroVad::new(&vad_path)?;

    println!("Loading Whisper model ({whisper_model})...");
    let stt = vox::WhisperBackend::from_model(&whisper_path)?;

    println!("Loading Piper TTS...");
    let tts = vox::PiperBackend::new(&piper_config)?;

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
            println!("Ollama connected.");
        }
        _ => {
            println!("Warning: could not reach Ollama at {ollama_host}. Chat may fail.");
            println!("Start Ollama with: ollama serve\n");
        }
    }

    // Clone model name for the println after the closure captures it.
    let model_display = ollama_model.clone();

    // Clone system prompt for the closure
    let system_prompt_clone = system_prompt.clone();

    // Build pipeline
    let pipeline = vox::Vox::builder()
        .vad(vad)
        .stt(stt)
        .tts(tts)
        .on_utterance(move |result, ctx| {
            println!("\n[You] {}", result.text);

            let client = client.clone();
            let host = ollama_host.clone();
            let model = ollama_model.clone();
            let system = system_prompt_clone.clone();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    match ask_ollama(&client, &host, &model, &result.text, Some(&system)).await {
                        Ok(response) => {
                            println!("[Assistant] {response}");
                            if let Err(e) = ctx.speak_and_play(&response).await {
                                eprintln!("TTS error: {e}");
                            }
                        }
                        Err(e) => {
                            eprintln!("[Error] {e}");
                        }
                    }
                });
            });
        })
        .build()?;

    println!(
        "\nChat ready -- speak into your mic, I'll respond via {} (Ctrl+C to stop)\n",
        model_display
    );
    pipeline.listen().await?;

    Ok(())
}

/// Stub when piper feature is disabled.
#[cfg(not(feature = "piper"))]
pub async fn run(
    _whisper_model: &str,
    _ollama_model: &str,
    _ollama_host: &str,
    _yes: bool,
    _voice_mode: bool,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "Chat requires the 'piper' feature for TTS.\n\n\
         Rebuild with:\n\
         \n  cargo install vox --features cli,piper\n"
    );
}
