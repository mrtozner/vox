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
