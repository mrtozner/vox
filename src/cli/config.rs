//! Handler for `vox config` — interactive configuration wizard.

use std::io::{self, Write};
use std::path::PathBuf;

/// Configuration structure for vox.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VoxConfig {
    pub stt_backend: String,
    pub stt_model: String,
    pub tts_backend: String,
    pub tts_voice: String,
    pub quantization: bool,
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            stt_backend: "whisper".to_string(),
            stt_model: "tiny.en".to_string(),
            tts_backend: "kokoro".to_string(),
            tts_voice: "af_heart".to_string(),
            quantization: false,
        }
    }
}

/// Return the vox config directory path (`~/.vox/`), creating it if needed.
pub fn config_dir() -> PathBuf {
    let dir = dirs::data_dir().map(|d| d.join("vox")).unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".vox")
    });

    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }

    dir
}

/// Run the interactive configuration wizard.
pub async fn run() -> anyhow::Result<()> {
    println!("=== Vox Configuration Wizard ===\n");
    println!("This wizard will help you configure your vox defaults.\n");

    let mut config = VoxConfig::default();

    // STT Backend selection
    println!("1. Select STT (Speech-to-Text) Backend:");
    println!("   a) whisper (default, recommended)");
    println!("   b) sherpa (requires sherpa feature)");
    println!("   c) sherpa-streaming (requires sherpa feature)");
    print!("   Choice [a]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    config.stt_backend = match input.trim().to_lowercase().as_str() {
        "b" | "sherpa" => "sherpa".to_string(),
        "c" | "sherpa-streaming" => "sherpa-streaming".to_string(),
        _ => "whisper".to_string(),
    };

    println!();

    // STT Model size (only for Whisper)
    if config.stt_backend == "whisper" {
        println!("2. Select Whisper Model Size:");
        println!("   a) tiny.en (fastest, 75MB)");
        println!("   b) base.en (better accuracy, 142MB)");
        println!("   c) small.en (best accuracy, larger)");
        print!("   Choice [a]: ");
        io::stdout().flush()?;

        input.clear();
        io::stdin().read_line(&mut input)?;
        config.stt_model = match input.trim().to_lowercase().as_str() {
            "b" | "base" => "base.en".to_string(),
            "c" | "small" => "small.en".to_string(),
            _ => "tiny.en".to_string(),
        };

        println!();

        // Quantization
        println!("3. Enable INT8 Quantization?");
        println!("   Reduces model size and improves speed with minimal accuracy loss.");
        print!("   Enable quantization? [y/N]: ");
        io::stdout().flush()?;

        input.clear();
        io::stdin().read_line(&mut input)?;
        config.quantization = matches!(input.trim().to_lowercase().as_str(), "y" | "yes");

        if config.quantization {
            config.stt_model = format!("{}-int8", config.stt_model);
        }

        println!();
    }

    // TTS Backend selection
    let step = if config.stt_backend == "whisper" {
        4
    } else {
        2
    };
    println!("{}. Select TTS (Text-to-Speech) Backend:", step);
    println!("   a) kokoro (default, high quality)");
    println!("   b) piper (multi-language, requires piper feature)");
    println!("   c) chatterbox (voice cloning, requires chatterbox feature)");
    println!("   d) pocket (fast, requires pocket feature)");
    print!("   Choice [a]: ");
    io::stdout().flush()?;

    input.clear();
    io::stdin().read_line(&mut input)?;
    config.tts_backend = match input.trim().to_lowercase().as_str() {
        "b" | "piper" => "piper".to_string(),
        "c" | "chatterbox" => "chatterbox".to_string(),
        "d" | "pocket" => "pocket".to_string(),
        _ => "kokoro".to_string(),
    };

    println!();

    // TTS Voice selection (backend-specific)
    let voice_step = step + 1;
    match config.tts_backend.as_str() {
        "kokoro" => {
            println!("{}. Select Kokoro Voice:", voice_step);
            println!("   a) af_heart (default, female)");
            println!("   b) af_sky (female, alternative)");
            println!("   c) am_adam (male)");
            print!("   Choice [a]: ");
            io::stdout().flush()?;

            input.clear();
            io::stdin().read_line(&mut input)?;
            config.tts_voice = match input.trim().to_lowercase().as_str() {
                "b" | "sky" => "af_sky".to_string(),
                "c" | "adam" => "am_adam".to_string(),
                _ => "af_heart".to_string(),
            };
        }
        "piper" => {
            println!("{}. Select Piper Language:", voice_step);
            println!("   a) en (English)");
            println!("   b) de (German)");
            println!("   c) fr (French)");
            println!("   d) es (Spanish)");
            print!("   Choice [a]: ");
            io::stdout().flush()?;

            input.clear();
            io::stdin().read_line(&mut input)?;
            config.tts_voice = match input.trim().to_lowercase().as_str() {
                "b" | "de" => "de".to_string(),
                "c" | "fr" => "fr".to_string(),
                "d" | "es" => "es".to_string(),
                _ => "en".to_string(),
            };
        }
        _ => {
            config.tts_voice = "default".to_string();
        }
    }

    println!();

    // Save configuration
    let config_path = config_dir().join("config.toml");
    let toml_str = toml::to_string_pretty(&config)?;
    std::fs::write(&config_path, toml_str)?;

    println!("=== Configuration Saved ===");
    println!("Config file: {}", config_path.display());
    println!();
    println!("Your configuration:");
    println!("  STT Backend: {}", config.stt_backend);
    if config.stt_backend == "whisper" {
        println!("  STT Model: {}", config.stt_model);
    }
    println!("  TTS Backend: {}", config.tts_backend);
    println!("  TTS Voice: {}", config.tts_voice);
    println!();
    println!("You can manually edit this file or re-run 'vox config' to change settings.");

    Ok(())
}
