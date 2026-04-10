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
        /// Whisper model size (tiny.en, base.en, small.en)
        #[arg(long, default_value = "tiny.en")]
        model: String,
        /// STT backend to use (whisper, distil-whisper, sherpa, or sherpa-streaming)
        #[arg(long, default_value = "whisper")]
        stt_backend: String,
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .without_time()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Listen {
            model,
            stt_backend,
            yes,
        } => {
            cli::listen::run(&model, &stt_backend, yes).await?;
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
        } => {
            cli::chat::run(&model, &llm, &ollama_host, yes, voice_mode).await?;
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
        } => {
            cli::serve::run(&host, port, cache_models).await?;
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
