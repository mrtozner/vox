//! Handler for `vox models` — model registry, download, and management.

use std::path::PathBuf;

/// Metadata for a downloadable model.
pub struct ModelInfo {
    /// Human-readable name (used in `vox models download <name>`).
    pub name: &'static str,
    /// Filename on disk inside the models directory.
    pub filename: &'static str,
    /// Download URL.
    pub url: &'static str,
    /// Expected file size in bytes (approximate, for progress display).
    pub size_bytes: u64,
    /// Model kind label (VAD, STT, TTS).
    pub kind: &'static str,
}

/// All known models that can be managed through the CLI.
pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "silero-vad",
        filename: "silero_vad.onnx",
        url: "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
        size_bytes: 2_000_000,
        kind: "VAD",
    },
    ModelInfo {
        name: "whisper-tiny.en",
        filename: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        size_bytes: 75_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-base.en",
        filename: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        size_bytes: 142_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "kokoro",
        filename: "kokoro-v1.0.onnx",
        url: "https://github.com/hexgrad/kokoro/releases/download/v1.0/kokoro-v1.0.onnx",
        size_bytes: 310_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "kokoro-voices",
        filename: "voices.bin",
        url: "https://github.com/hexgrad/kokoro/releases/download/v1.0/voices-v1.0.bin",
        size_bytes: 27_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "sherpa-sensevoice",
        filename: "sensevoice/model.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
        size_bytes: 230_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-sensevoice-tokens",
        filename: "sensevoice/tokens.txt",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
        size_bytes: 300_000,
        kind: "STT",
    },
    // Streaming Sherpa-ONNX zipformer transducer model
    ModelInfo {
        name: "sherpa-streaming-encoder",
        filename: "sherpa-streaming/encoder.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        size_bytes: 12_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-decoder",
        filename: "sherpa-streaming/decoder.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        size_bytes: 3_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-joiner",
        filename: "sherpa-streaming/joiner.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        size_bytes: 12_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-tokens",
        filename: "sherpa-streaming/tokens.txt",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/tokens.txt",
        size_bytes: 50_000,
        kind: "STT",
    },
    // Piper TTS voices
    ModelInfo {
        name: "piper-en-us",
        filename: "piper/en_US-lessac-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        size_bytes: 63_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-en-us-config",
        filename: "piper/en_US-lessac-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        size_bytes: 5_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-de",
        filename: "piper/de_DE-thorsten-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        size_bytes: 63_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-de-config",
        filename: "piper/de_DE-thorsten-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
        size_bytes: 5_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-es",
        filename: "piper/es_ES-davefx-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        size_bytes: 63_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-es-config",
        filename: "piper/es_ES-davefx-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
        size_bytes: 5_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-fr",
        filename: "piper/fr_FR-siwis-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
        size_bytes: 63_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-fr-config",
        filename: "piper/fr_FR-siwis-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
        size_bytes: 5_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-zh",
        filename: "piper/zh_CN-huayan-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        size_bytes: 63_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-zh-config",
        filename: "piper/zh_CN-huayan-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
        size_bytes: 5_000,
        kind: "TTS",
    },
];

/// Return the models directory path (`~/.vox/models/`), creating it if needed.
///
/// Uses `dirs::data_dir()` on each platform:
/// - macOS: `~/Library/Application Support/vox/models`
/// - Linux: `~/.local/share/vox/models`
/// - Windows: `{FOLDERPATH}/vox/models`
///
/// Falls back to `~/.vox/models` if the platform data dir is unavailable.
pub fn models_dir() -> PathBuf {
    let dir = dirs::data_dir()
        .map(|d| d.join("vox").join("models"))
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".vox")
                .join("models")
        });

    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }

    dir
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// List all available models, showing which are downloaded.
pub fn list() -> anyhow::Result<()> {
    let dir = models_dir();

    println!("Models directory: {}\n", dir.display());
    println!("  {:<20} {:<6} {:<12} STATUS", "NAME", "KIND", "SIZE");
    println!("  {}", "-".repeat(55));

    for model in MODELS {
        let file_path = dir.join(model.filename);
        let status = if file_path.exists() {
            let actual_size = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
            format!("downloaded ({})", format_bytes(actual_size))
        } else {
            "not downloaded".to_string()
        };

        println!(
            "  {:<20} {:<6} {:<12} {}",
            model.name,
            model.kind,
            format_bytes(model.size_bytes),
            status,
        );
    }

    println!();
    Ok(())
}

/// Download a model by name with a progress bar.
pub async fn download(name: &str) -> anyhow::Result<()> {
    let model = MODELS.iter().find(|m| m.name == name).ok_or_else(|| {
        let available: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        anyhow::anyhow!(
            "Unknown model: '{name}'\n\nAvailable models:\n  {}",
            available.join("\n  ")
        )
    })?;

    let dir = models_dir();
    let dest = dir.join(model.filename);

    if dest.exists() {
        println!("Model '{}' is already downloaded at:", model.name);
        println!("  {}", dest.display());
        return Ok(());
    }

    println!(
        "Downloading {} ({})...",
        model.name,
        format_bytes(model.size_bytes)
    );
    println!("  URL: {}", model.url);
    println!("  Destination: {}", dest.display());
    println!();

    // Stream the download with a progress bar
    let client = reqwest::Client::new();
    let response = client
        .get(model.url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Download failed: {e}"))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download failed with HTTP {}: {}",
            response.status(),
            model.url
        );
    }

    let total_size = response.content_length().unwrap_or(model.size_bytes);

    let pb = indicatif::ProgressBar::new(total_size);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    // Ensure parent directories exist (for subdirectory filenames like sensevoice/model.int8.onnx)
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Write to a temporary file first, then rename (atomic-ish)
    let tmp_dest = dest.with_extension("part");

    let mut file = tokio::fs::File::create(&tmp_dest).await?;
    let mut stream = response.bytes_stream();

    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow::anyhow!("Download stream error: {e}"))?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);

    // Move into place
    tokio::fs::rename(&tmp_dest, &dest).await?;

    pb.finish_and_clear();
    println!("Downloaded {} to {}", model.name, dest.display());

    Ok(())
}

/// Print the models directory path.
pub fn path() -> anyhow::Result<()> {
    println!("{}", models_dir().display());
    Ok(())
}

/// Resolve a model file: check ~/.vox/models/ first, then cwd.
pub fn resolve_model(filename: &str) -> Option<PathBuf> {
    let models = models_dir();
    let candidate = models.join(filename);
    if candidate.exists() {
        return Some(candidate);
    }
    let local = PathBuf::from(filename);
    if local.exists() {
        return Some(local);
    }
    None
}

/// Map user-facing Whisper model name to GGML filename.
pub fn model_filename(model: &str) -> String {
    match model {
        "tiny" => "ggml-tiny.bin".into(),
        "tiny.en" => "ggml-tiny.en.bin".into(),
        "base" => "ggml-base.bin".into(),
        "base.en" => "ggml-base.en.bin".into(),
        "small" => "ggml-small.bin".into(),
        "small.en" => "ggml-small.en.bin".into(),
        "medium" => "ggml-medium.bin".into(),
        "medium.en" => "ggml-medium.en.bin".into(),
        other => {
            if other.ends_with(".bin") {
                other.to_string()
            } else {
                format!("ggml-{other}.bin")
            }
        }
    }
}

/// Map user model name to the download registry name.
pub fn whisper_download_name(model: &str) -> String {
    match model {
        "tiny.en" => "whisper-tiny.en".into(),
        "base.en" => "whisper-base.en".into(),
        "small.en" => "whisper-small.en".into(),
        other => format!("whisper-{other}"),
    }
}

/// Prompt user to download a missing model. Returns true if yes.
pub fn prompt_download(model_name: &str, filename: &str, auto_yes: bool) -> anyhow::Result<bool> {
    if auto_yes {
        println!("Model '{model_name}' not found. Downloading automatically...");
        return Ok(true);
    }

    use std::io::Write;
    print!("Model '{model_name}' ({filename}) not found. Download now? [Y/n] ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    Ok(input.is_empty() || input == "y" || input == "yes")
}

/// Resolve a model, offering to download if missing.
pub async fn ensure_model(
    model_name: &str,
    filename: &str,
    auto_yes: bool,
) -> anyhow::Result<PathBuf> {
    if let Some(p) = resolve_model(filename) {
        return Ok(p);
    }
    if prompt_download(model_name, filename, auto_yes)? {
        download(model_name).await?;
        resolve_model(filename).ok_or_else(|| {
            anyhow::anyhow!("Download of '{model_name}' succeeded but file not found")
        })
    } else {
        anyhow::bail!(
            "Required model '{model_name}' not found.\n\n\
             Download with:\n\
             \n  vox models download {model_name}\n"
        );
    }
}

/// Download both SenseVoice model files and return the model directory path.
#[cfg(feature = "sherpa")]
pub async fn ensure_sherpa_sensevoice(yes: bool) -> anyhow::Result<PathBuf> {
    let model_path = ensure_model("sherpa-sensevoice", "sensevoice/model.int8.onnx", yes).await?;
    ensure_model("sherpa-sensevoice-tokens", "sensevoice/tokens.txt", yes).await?;
    // Return the parent directory (sensevoice/)
    Ok(model_path.parent().unwrap().to_path_buf())
}

/// Download streaming zipformer model files and return the model directory path.
#[cfg(feature = "sherpa")]
pub async fn ensure_sherpa_streaming(yes: bool) -> anyhow::Result<PathBuf> {
    let encoder_path = ensure_model(
        "sherpa-streaming-encoder",
        "sherpa-streaming/encoder.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-decoder",
        "sherpa-streaming/decoder.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-joiner",
        "sherpa-streaming/joiner.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-tokens",
        "sherpa-streaming/tokens.txt",
        yes,
    )
    .await?;
    // Return the parent directory (sherpa-streaming/)
    Ok(encoder_path.parent().unwrap().to_path_buf())
}

/// Download both Piper voice model files (.onnx + .onnx.json) and return the config path.
///
/// `voice_name` is the registry prefix (e.g. "piper-en-us", "piper-de").
/// Returns the path to the `.onnx.json` config file.
#[cfg(feature = "piper")]
pub async fn ensure_piper_voice(voice_name: &str, yes: bool) -> anyhow::Result<PathBuf> {
    let model_registry_name = voice_name;
    let config_registry_name = format!("{voice_name}-config");

    // Look up the model entry to get the filename
    let model_info = MODELS
        .iter()
        .find(|m| m.name == model_registry_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown piper voice: '{voice_name}'"))?;
    let config_info = MODELS
        .iter()
        .find(|m| m.name == config_registry_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown piper voice config: '{config_registry_name}'"))?;

    // Download both files
    ensure_model(model_registry_name, model_info.filename, yes).await?;
    let config_path = ensure_model(&config_registry_name, config_info.filename, yes).await?;

    Ok(config_path)
}

/// Map a short voice alias to a piper model registry name.
///
/// Supports shorthand like "en", "de", "fr", etc. as well as the full
/// registry names like "piper-en-us".
#[cfg(feature = "piper")]
pub fn piper_voice_alias(voice: &str) -> String {
    let lower = voice.to_lowercase();
    match lower.as_str() {
        "en" | "en-us" | "english" => "piper-en-us".to_string(),
        "de" | "german" | "deutsch" => "piper-de".to_string(),
        "es" | "spanish" | "espanol" => "piper-es".to_string(),
        "fr" | "french" | "francais" => "piper-fr".to_string(),
        "zh" | "chinese" | "mandarin" => "piper-zh".to_string(),
        _ if lower.starts_with("piper-") => lower,
        _ => "piper-en-us".to_string(), // default to English
    }
}
