//! Vox CLI — the open-source voice AI framework.
//!
//! An Ollama-like interface for voice: manage models, transcribe speech,
//! and synthesize audio, all running locally on your hardware.

#[path = "../cli/mod.rs"]
mod cli;

#[cfg(feature = "server")]
#[path = "../server/mod.rs"]
mod server;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vox",
    version,
    about = "The open-source voice AI framework",
    long_about = "Vox is an open-source voice AI framework that runs entirely on your device.\n\
                  Whisper STT + Kokoro TTS + Silero VAD — no cloud, no Python, just Rust."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Listen to microphone and transcribe speech in real time
    Listen {
        /// Whisper model (tiny.en, base.en, small.en are English-only;
        /// tiny, base, small, medium, turbo are multilingual).
        /// Default: tiny.en, or base when --language is not English
        #[arg(long)]
        model: Option<String>,
        /// STT backend to use (whisper, distil-whisper, sherpa, or sherpa-streaming)
        #[arg(long, default_value = "whisper")]
        stt_backend: String,
        /// Spoken language as a Whisper code (en, tr, de, ...) or "auto"
        /// to detect it. Needs a multilingual model
        #[arg(long)]
        language: Option<String>,
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Speak text aloud using TTS (kokoro or piper backend)
    Speak {
        /// Text to synthesize and play
        text: String,
        /// TTS voice name (e.g. af_heart for kokoro, en/de/fr for piper)
        #[arg(long, default_value = "af_heart")]
        voice: String,
        /// TTS backend to use (kokoro, piper, or chatterbox)
        #[arg(long, default_value = "kokoro")]
        backend: String,
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
        /// Stream audio sentence-by-sentence (start playing before full synthesis)
        #[arg(long, default_value_t = false)]
        stream: bool,
    },
    /// Chat with an LLM using your voice
    Chat {
        /// Whisper model size (tiny.en, base.en, small.en)
        #[arg(long, default_value = "tiny.en")]
        model: String,
        /// Ollama model name
        #[arg(long, default_value = "llama3.2")]
        llm: String,
        /// Ollama host:port
        #[arg(long, default_value = "localhost:11434")]
        ollama_host: String,
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
        /// Use voice-optimized prompts for better TTS output
        #[arg(long)]
        voice_mode: bool,
        /// Enable speaker identification (show who's talking)
        #[arg(long)]
        diarize: bool,
    },
    /// Test audio I/O (record and playback)
    Test,
    /// Measure STT/TTS/VAD latency and performance
    Benchmark {
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Interactive configuration wizard
    Config,
    /// Start the HTTP API server
    #[cfg(feature = "server")]
    Serve {
        /// Port to listen on
        #[arg(long, default_value = "3000")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Enable model caching (reduces cold start latency)
        #[arg(long)]
        cache_models: bool,
        /// Download the default models that are missing before starting
        #[arg(long, short = 'y', visible_alias = "yes")]
        download: bool,
        /// Spoken language as a Whisper code (en, tr, de, ...) or "auto".
        /// Selects a multilingual STT model and a matching Piper voice
        /// [env: VOX_LANGUAGE]
        #[arg(long)]
        language: Option<String>,
        /// Whisper model to load (base, small.en, turbo-q5, ...).
        /// Default: best installed model for this machine [env: VOX_STT_MODEL]
        #[arg(long)]
        stt_model: Option<String>,
        /// Piper voice for Live Talk/Converse (en, tr, de, or a voice file
        /// name like tr_TR-dfki-medium) [env: VOX_PIPER_VOICE]
        #[arg(long)]
        piper_voice: Option<String>,
    },
    /// Check the installation: models directory, missing or corrupt models, and why
    Doctor {
        /// Also verify the SHA-256 of every installed model (slow)
        #[arg(long)]
        verify: bool,
        /// Check the models needed for this language (en, tr, de, ..., auto)
        #[arg(long)]
        language: Option<String>,
        /// Check for this Whisper model instead of the recommended one
        #[arg(long)]
        stt_model: Option<String>,
        /// Check for this Piper voice instead of the language default
        #[arg(long)]
        piper_voice: Option<String>,
    },
    /// Manage voice models (download, list, find paths)
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// List available and downloaded models
    List,
    /// Download a model by name
    Download {
        /// Model name (e.g. silero-vad, whisper-tiny.en, kokoro, kokoro-voices)
        name: String,
        /// Force re-download even if the file already exists
        #[arg(long)]
        force: bool,
    },
    /// Show the models directory path
    Path,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tracing level: RUST_LOG=debug → DEBUG, RUST_LOG=info → INFO, else WARN.
    // Operators can opt in to WS + VAD + STT traces without touching the binary.
    // Complex EnvFilter syntax (`vox=debug,hyper=info`) requires the
    // `env-filter` feature on tracing-subscriber; keep the simple path for now.
    let tracing_level = match std::env::var("RUST_LOG")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::WARN,
    };
    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .without_time()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Listen {
            model,
            stt_backend,
            language,
            yes,
        } => {
            cli::listen::run(model.as_deref(), &stt_backend, language.as_deref(), yes).await?;
        }
        Commands::Speak {
            text,
            voice,
            backend,
            yes,
            stream,
        } => {
            cli::speak::run(&text, &voice, &backend, yes, stream).await?;
        }
        Commands::Chat {
            model,
            llm,
            ollama_host,
            yes,
            voice_mode,
            diarize,
        } => {
            cli::chat::run(&model, &llm, &ollama_host, yes, voice_mode, diarize).await?;
        }
        Commands::Test => {
            cli::test::run().await?;
        }
        Commands::Benchmark { yes } => {
            cli::benchmark::run(yes).await?;
        }
        Commands::Config => {
            cli::config::run().await?;
        }
        #[cfg(feature = "server")]
        Commands::Serve {
            host,
            port,
            cache_models,
            download,
            language,
            stt_model,
            piper_voice,
        } => {
            cli::serve::run(cli::serve::ServeArgs {
                host,
                port,
                cache_models,
                download,
                language,
                stt_model,
                piper_voice,
            })
            .await?;
        }
        Commands::Doctor {
            verify,
            language,
            stt_model,
            piper_voice,
        } => {
            cli::models::doctor(
                &cli::models::BundleOptions {
                    language,
                    stt_model,
                    piper_voice,
                },
                verify,
            )?;
        }
        Commands::Models { action } => match action {
            ModelAction::List => {
                cli::models::list()?;
            }
            ModelAction::Download { name, force } => {
                cli::models::download(&name, force).await?;
            }
            ModelAction::Path => {
                cli::models::path()?;
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Commands {
        Cli::try_parse_from(args)
            .unwrap_or_else(|e| panic!("{args:?} failed to parse: {e}"))
            .command
    }

    #[test]
    fn listen_accepts_language_flag() {
        match parse(&["vox", "listen", "--language", "tr"]) {
            Commands::Listen {
                model, language, ..
            } => {
                assert_eq!(language.as_deref(), Some("tr"));
                assert_eq!(model, None, "model default is resolved from the language");
            }
            _ => panic!("expected listen"),
        }
        match parse(&["vox", "listen"]) {
            Commands::Listen { language, .. } => assert_eq!(language, None),
            _ => panic!("expected listen"),
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serve_accepts_download_language_and_voice_flags() {
        match parse(&[
            "vox",
            "serve",
            "-y",
            "--language",
            "tr",
            "--stt-model",
            "small",
            "--piper-voice",
            "tr",
        ]) {
            Commands::Serve {
                download,
                language,
                stt_model,
                piper_voice,
                ..
            } => {
                assert!(download);
                assert_eq!(language.as_deref(), Some("tr"));
                assert_eq!(stt_model.as_deref(), Some("small"));
                assert_eq!(piper_voice.as_deref(), Some("tr"));
            }
            _ => panic!("expected serve"),
        }
        for flag in ["--download", "--yes"] {
            match parse(&["vox", "serve", flag]) {
                Commands::Serve { download, .. } => assert!(download, "{flag}"),
                _ => panic!("expected serve"),
            }
        }
        match parse(&["vox", "serve"]) {
            Commands::Serve {
                download, language, ..
            } => {
                assert!(!download);
                assert_eq!(language, None);
            }
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn doctor_parses_with_and_without_flags() {
        match parse(&["vox", "doctor"]) {
            Commands::Doctor {
                verify, language, ..
            } => {
                assert!(!verify);
                assert_eq!(language, None);
            }
            _ => panic!("expected doctor"),
        }
        match parse(&["vox", "doctor", "--verify", "--language", "tr"]) {
            Commands::Doctor {
                verify, language, ..
            } => {
                assert!(verify);
                assert_eq!(language.as_deref(), Some("tr"));
            }
            _ => panic!("expected doctor"),
        }
    }

    #[test]
    fn cli_definition_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
