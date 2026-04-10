//! Handler for `vox models` — model registry, download, and management.

use std::fs;
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
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        size_bytes: 2_327_524,
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
        name: "whisper-tiny.en-int8",
        filename: "ggml-tiny.en-int8.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q8_0.bin",
        size_bytes: 42_000_000,
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
        name: "whisper-base.en-int8",
        filename: "ggml-base.en-int8.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q8_0.bin",
        size_bytes: 78_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-tiny.en",
        filename: "ggml-distil-tiny.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3/resolve/main/ggml-distil-tiny.en.bin",
        size_bytes: 75_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-base.en",
        filename: "ggml-distil-base.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3/resolve/main/ggml-distil-base.en.bin",
        size_bytes: 142_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-small.en",
        filename: "ggml-distil-small.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-model.bin",
        size_bytes: 466_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-medium.en",
        filename: "ggml-distil-medium.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-model.bin",
        size_bytes: 1_500_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo",
        filename: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        size_bytes: 1_550_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo-q5",
        filename: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        size_bytes: 574_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo-q8",
        filename: "ggml-large-v3-turbo-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
        size_bytes: 842_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "kokoro",
        filename: "kokoro-v1.0.onnx",
        url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        size_bytes: 325_532_387,
        kind: "TTS",
    },
    ModelInfo {
        name: "kokoro-int8",
        filename: "kokoro-v1.0.int8.onnx",
        url: "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx",
        size_bytes: 85_000_000,
        kind: "TTS",
    },
    ModelInfo {
        name: "kokoro-voices",
        filename: "voices.bin",
        url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        size_bytes: 28_214_398,
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
        filename: "sherpa-streaming/encoder.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 71_000_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-decoder",
        filename: "sherpa-streaming/decoder.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 1_300_000,
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-joiner",
        filename: "sherpa-streaming/joiner.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 260_000,
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

/// Clean up any orphaned .part files from previous incomplete downloads.
pub fn cleanup_partial_downloads() -> anyhow::Result<usize> {
    let dir = models_dir();
    let mut cleaned = 0;

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "part" && fs::remove_file(&path).is_ok() {
                    cleaned += 1;
                    println!("  Cleaned up partial download: {}", path.display());
                }
            }
        }
    }

    Ok(cleaned)
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

    // Clean up partial downloads on startup
    if let Ok(cleaned) = cleanup_partial_downloads() {
        if cleaned > 0 {
            println!("Cleaned up {cleaned} partial download(s)\n");
        }
    }

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
pub async fn download(name: &str, force: bool) -> anyhow::Result<()> {
    let model = MODELS.iter().find(|m| m.name == name).ok_or_else(|| {
        let available: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        anyhow::anyhow!(
            "Unknown model: '{name}'\n\nAvailable models:\n  {}",
            available.join("\n  ")
        )
    })?;

    let dir = models_dir();
    let dest = dir.join(model.filename);

    // Check for existing file
    if dest.exists() && !force {
        println!("Model '{}' is already downloaded at:", model.name);
        println!("  {}", dest.display());
        println!(
            "\nTo re-download, use: vox models download {} --force",
            name
        );
        return Ok(());
    }

    // Clean up any existing .part file from a previous failed download
    let tmp_dest = dest.with_extension("part");
    if tmp_dest.exists() {
        println!("Found partial download from previous attempt, cleaning up...");
        let _ = tokio::fs::remove_file(&tmp_dest).await;
    }

    // If forcing a re-download, remove the existing file
    if force && dest.exists() {
        println!("Removing existing file for forced re-download...");
        tokio::fs::remove_file(&dest).await?;
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
    let mut file = tokio::fs::File::create(&tmp_dest).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to create temporary download file: {}\n\n\
             This may indicate insufficient disk space or permission issues.\n\
             Models directory: {}",
            e,
            dir.display()
        )
    })?;
    let mut stream = response.bytes_stream();

    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            anyhow::anyhow!(
                "Download stream error: {}\n\n\
                 The download may have been interrupted due to network issues.\n\
                 Partial file will be cleaned up automatically on next run.\n\
                 To retry, run: vox models download {}",
                e,
                model.name
            )
        })?;

        file.write_all(&chunk).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to write downloaded data: {}\n\n\
                 This typically indicates insufficient disk space.\n\
                 Partial download location: {}\n\
                 Required space: {}\n\n\
                 To clean up partial downloads, run: vox models list\n\
                 To retry after freeing space, run: vox models download {}",
                e,
                tmp_dest.display(),
                format_bytes(model.size_bytes),
                model.name
            )
        })?;

        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);

    pb.finish_and_clear();

    // Validate the downloaded file size
    let actual_size = tokio::fs::metadata(&tmp_dest).await?.len();

    // Allow some tolerance (within 1% or the exact expected size)
    let size_diff = actual_size.abs_diff(model.size_bytes);
    let tolerance = model.size_bytes / 100; // 1% tolerance

    if size_diff > tolerance && actual_size != total_size {
        let _ = tokio::fs::remove_file(&tmp_dest).await;
        anyhow::bail!(
            "Downloaded file size mismatch!\n\n\
             Expected: {} ({} bytes)\n\
             Downloaded: {} ({} bytes)\n\
             Difference: {}\n\n\
             The download may have been corrupted or interrupted.\n\
             Partial file has been removed.\n\
             Models directory: {}\n\n\
             To retry the download, run: vox models download {}",
            format_bytes(model.size_bytes),
            model.size_bytes,
            format_bytes(actual_size),
            actual_size,
            format_bytes(size_diff),
            dir.display(),
            model.name
        );
    }

    // Move into place
    tokio::fs::rename(&tmp_dest, &dest).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to finalize download: {}\n\n\
             The file was downloaded successfully but could not be moved to final location.\n\
             Temporary file: {}\n\
             Final location: {}\n\
             Models directory: {}",
            e,
            tmp_dest.display(),
            dest.display(),
            dir.display()
        )
    })?;

    println!(
        "Successfully downloaded {} to {}",
        model.name,
        dest.display()
    );
    println!(
        "File size: {} ({} bytes)",
        format_bytes(actual_size),
        actual_size
    );

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
///
/// WARNING: Whisper Turbo models (large-v3-turbo*) require GPU acceleration
/// and are 24-35x SLOWER on CPU than tiny.en. Use with CoreML/GPU only.
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
        // Whisper Turbo aliases
        "turbo" => "ggml-large-v3-turbo.bin".into(),
        "turbo-q5" => "ggml-large-v3-turbo-q5_0.bin".into(),
        "turbo-q8" => "ggml-large-v3-turbo-q8_0.bin".into(),
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
        // Whisper Turbo aliases
        "turbo" => "whisper-large-v3-turbo".into(),
        "turbo-q5" => "whisper-large-v3-turbo-q5".into(),
        "turbo-q8" => "whisper-large-v3-turbo-q8".into(),
        other => format!("whisper-{other}"),
    }
}

/// Map user-facing Distil-Whisper model name to GGML filename.
#[cfg(feature = "distil-whisper")]
pub fn distil_whisper_model_filename(model: &str) -> String {
    match model {
        "tiny" => "ggml-distil-tiny.bin".into(),
        "tiny.en" => "ggml-distil-tiny.en.bin".into(),
        "base" => "ggml-distil-base.bin".into(),
        "base.en" => "ggml-distil-base.en.bin".into(),
        "small" => "ggml-distil-small.bin".into(),
        "small.en" => "ggml-distil-small.en.bin".into(),
        "medium" => "ggml-distil-medium.bin".into(),
        "medium.en" => "ggml-distil-medium.en.bin".into(),
        other => {
            if other.ends_with(".bin") {
                other.to_string()
            } else {
                format!("ggml-distil-{other}.bin")
            }
        }
    }
}

/// Map user model name to the distil-whisper download registry name.
#[cfg(feature = "distil-whisper")]
pub fn distil_whisper_download_name(model: &str) -> String {
    match model {
        "tiny.en" => "distil-whisper-tiny.en".into(),
        "base.en" => "distil-whisper-base.en".into(),
        "small.en" => "distil-whisper-small.en".into(),
        "medium.en" => "distil-whisper-medium.en".into(),
        other => format!("distil-whisper-{other}"),
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
        download(model_name, false).await?;
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
        "sherpa-streaming/encoder.int8.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-decoder",
        "sherpa-streaming/decoder.int8.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-joiner",
        "sherpa-streaming/joiner.int8.onnx",
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
