//! Handler for `vox config` — interactive configuration wizard — and the
//! `config.toml` file it shares with `vox tune`.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Overrides the directory holding `config.toml`.
pub const CONFIG_DIR_ENV: &str = "VOX_CONFIG_DIR";

/// Configuration structure for vox.
///
/// Every field has a default, so a file that sets only some of them loads.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VoxConfig {
    pub stt_backend: String,
    pub stt_model: String,
    pub tts_backend: String,
    pub tts_voice: String,
    pub quantization: bool,
    /// Ollama address for `vox chat` and `vox tune`. `--ollama-host` and
    /// `VOX_OLLAMA_HOST` take precedence (see `vox::privacy::resolve_ollama_host`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ollama_host: Option<String>,
    /// What `vox tune` measured on this machine and picked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tune: Option<TuneConfig>,
}

/// The `[tune]` table: the models `vox tune` picked and the measurements
/// behind the choice.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TuneConfig {
    /// When the measurement ran, in seconds since the Unix epoch.
    pub measured_unix: u64,
    /// One-line description of the machine that was measured.
    pub machine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Whisper model name as accepted by `--model` (`base.en`, `small`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_model: Option<String>,
    /// Processing time divided by audio duration; below 1 is faster than real time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_rtf: Option<f64>,
    /// Share of the reference words transcribed correctly, 0 to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_word_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts_rtf: Option<f64>,
    /// Model ID at the endpoint used for the measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_host: Option<String>,
    /// Present when the runtime's stream yields individual tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_tokens_per_sec: Option<f64>,
    /// Time to first output token or, for other protocols, first visible text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_first_token_ms: Option<u64>,
}

impl TuneConfig {
    /// A legacy tune file has no endpoint metadata and describes Ollama.
    pub fn llm_for_endpoint(&self, endpoint: &vox::llm::LlmEndpoint) -> Option<&str> {
        match (self.llm_backend.as_deref(), self.llm_host.as_deref()) {
            (None, None) if endpoint.backend == vox::llm::LlmBackend::Ollama => {
                self.llm_model.as_deref()
            }
            (Some(backend), Some(host))
                if backend == endpoint.backend.as_str() && host == endpoint.host =>
            {
                self.llm_model.as_deref()
            }
            _ => None,
        }
    }
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            stt_backend: "whisper".to_string(),
            stt_model: "tiny.en".to_string(),
            tts_backend: "kokoro".to_string(),
            tts_voice: "af_heart".to_string(),
            quantization: false,
            ollama_host: None,
            tune: None,
        }
    }
}

/// The directory holding `config.toml`: `VOX_CONFIG_DIR` if set, otherwise
/// `vox` under the per-user data directory (`~/.vox` if there is none).
/// Not created here; [`save_to`] creates it.
pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(CONFIG_DIR_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    dirs::data_dir().map(|d| d.join("vox")).unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".vox")
    })
}

/// Where `config.toml` lives. Does not create anything.
pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// Read a config file. A missing file is the default configuration; a file
/// that does not parse is an error.
pub fn load_from(path: &Path) -> anyhow::Result<VoxConfig> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(VoxConfig::default()),
        Err(e) => anyhow::bail!("cannot read {}: {e}", path.display()),
    };
    toml::from_str(&text).map_err(|e| anyhow::anyhow!("invalid config {}: {e}", path.display()))
}

/// Read the user's config. An unreadable or invalid file is reported on
/// stderr and treated as the default configuration, so a broken config file
/// never stops a command from running.
pub fn load() -> VoxConfig {
    load_from(&config_path()).unwrap_or_else(|e| {
        eprintln!("warning: {e}; using defaults");
        VoxConfig::default()
    })
}

/// Write a config file, creating its directory.
pub fn save_to(config: &VoxConfig, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", parent.display()))?;
    }
    let text = toml::to_string_pretty(config)?;
    std::fs::write(path, text).map_err(|e| anyhow::anyhow!("cannot write {}: {e}", path.display()))
}

/// Run the interactive configuration wizard.
pub async fn run() -> anyhow::Result<()> {
    println!("=== Vox Configuration Wizard ===\n");
    println!("This wizard will help you configure your vox defaults.\n");

    // Start from the existing file so settings the wizard does not ask about
    // (`ollama_host`, the `[tune]` results) survive.
    let existing = load();
    let mut config = VoxConfig {
        ollama_host: existing.ollama_host,
        tune: existing.tune,
        ..VoxConfig::default()
    };

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
    let config_path = config_path();
    save_to(&config, &config_path)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vox-config-test-{}-{name}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn missing_file_is_the_default_config() {
        let dir = scratch("missing");
        assert_eq!(
            load_from(&dir.join("config.toml")).unwrap(),
            VoxConfig::default()
        );
    }

    #[test]
    fn config_written_before_tune_existed_still_loads() {
        let dir = scratch("legacy");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "stt_backend = \"whisper\"\nstt_model = \"base.en\"\ntts_backend = \"piper\"\n\
             tts_voice = \"de\"\nquantization = false\n",
        )
        .unwrap();
        let config = load_from(&path).unwrap();
        assert_eq!(config.stt_model, "base.en");
        assert_eq!(config.tts_voice, "de");
        assert_eq!(config.ollama_host, None);
        assert_eq!(config.tune, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tune_results_round_trip() {
        let dir = scratch("roundtrip");
        let path = dir.join("nested").join("config.toml");
        let config = VoxConfig {
            ollama_host: Some("localhost:11434".into()),
            tune: Some(TuneConfig {
                measured_unix: 1_700_000_000,
                machine: "test machine".into(),
                language: Some("en".into()),
                stt_model: Some("small.en".into()),
                stt_rtf: Some(0.12),
                stt_word_accuracy: Some(1.0),
                tts_backend: Some("piper".into()),
                tts_rtf: Some(0.05),
                llm_model: Some("llama3.2:3b".into()),
                llm_backend: Some("ollama".into()),
                llm_host: Some("localhost:11434".into()),
                llm_tokens_per_sec: Some(42.5),
                llm_first_token_ms: Some(180),
            }),
            ..VoxConfig::default()
        };
        save_to(&config, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[tune]"), "{text}");
        assert!(text.contains("llm_model = \"llama3.2:3b\""), "{text}");
        assert_eq!(load_from(&path).unwrap(), config);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_tune_uses_its_model_only_with_ollama() {
        let tuned: TuneConfig = toml::from_str("llm_model = \"llama3.2:3b\"").unwrap();
        let ollama = vox::llm::LlmEndpoint::ollama("localhost:11434");
        let compatible = vox::llm::LlmEndpoint {
            backend: vox::llm::LlmBackend::OpenAiCompatible,
            host: "127.0.0.1:8080".into(),
            default_model: None,
        };
        assert_eq!(tuned.llm_for_endpoint(&ollama), Some("llama3.2:3b"));
        assert_eq!(tuned.llm_for_endpoint(&compatible), None);
    }

    #[test]
    fn unset_optional_fields_are_not_written() {
        let text = toml::to_string_pretty(&VoxConfig::default()).unwrap();
        assert!(!text.contains("ollama_host"), "{text}");
        assert!(!text.contains("tune"), "{text}");
    }

    #[test]
    fn invalid_file_is_an_error_naming_the_path() {
        let dir = scratch("invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "stt_model = [not toml").unwrap();
        let err = load_from(&path).unwrap_err().to_string();
        assert!(err.contains("config.toml"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
