//! Vox CLI: the offline voice AI runtime.
//!
//! An Ollama-like interface for voice: manage models, transcribe speech,
//! and synthesize audio, all running locally on your hardware.

#[path = "../cli/mod.rs"]
mod cli;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vox",
    version,
    about = "Offline voice AI runtime: speech-to-text, text-to-speech and voice chat on your machine",
    long_about = "Vox is an offline voice AI runtime in a single binary.\n\
                  Whisper STT, Kokoro/Piper TTS, Silero VAD, speaker identification and barge-in voice chat\n\
                  with a local Ollama model. No cloud APIs, no Python."
)]
struct Cli {
    /// Stay on this machine: refuse an Ollama host that is not loopback and
    /// refuse model downloads [env: VOX_OFFLINE=1]
    #[arg(long, global = true)]
    offline: bool,
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
        /// Whisper model size (tiny.en, base.en, small.en).
        /// Default: the model picked by `vox tune`, else tiny.en
        #[arg(long)]
        model: Option<String>,
        /// Ollama model name. Default: the model picked by `vox tune`, else llama3.2
        #[arg(long)]
        llm: Option<String>,
        /// Ollama host:port. Default: VOX_OLLAMA_HOST, then `ollama_host` in
        /// config.toml, then localhost:11434
        #[arg(long)]
        ollama_host: Option<String>,
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
        /// Use voice-optimized prompts for better TTS output
        #[arg(long)]
        voice_mode: bool,
        /// Not supported in chat yet (speaker identification lives in Live Talk)
        #[arg(long, hide = true)]
        diarize: bool,
        /// Spoken language code, e.g. en, de, tr (default: auto-detect)
        #[arg(long)]
        language: Option<String>,
    },
    /// Test audio I/O (record and playback)
    Test,
    /// Measure STT/TTS/VAD latency and performance
    Benchmark {
        /// Auto-download missing models without prompting
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Measure STT, TTS and LLM speed on this machine with real speech and
    /// save the best installed models to config.toml
    Tune {
        /// Recommend from the hardware profile and installed models only:
        /// loads nothing, connects to nothing, writes nothing
        #[arg(long)]
        dry_run: bool,
        /// Download the recommended models that are missing
        #[arg(long, short = 'y')]
        yes: bool,
        /// WAV file with speech to transcribe instead of a synthesized sentence
        #[arg(long)]
        sample: Option<std::path::PathBuf>,
        /// What is said in --sample, to score transcription accuracy
        #[arg(long, requires = "sample")]
        reference: Option<String>,
        /// Spoken language as a Whisper code (en, tr, de, ...) or "auto"
        #[arg(long)]
        language: Option<String>,
        /// Ollama model to measure (repeatable). Default: installed models
        /// that fit in RAM, smallest first
        #[arg(long)]
        llm: Vec<String>,
        /// Ollama host:port. Default: VOX_OLLAMA_HOST, then `ollama_host` in
        /// config.toml, then localhost:11434
        #[arg(long)]
        ollama_host: Option<String>,
        /// Timed repetitions per model
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Skip the LLM measurement
        #[arg(long)]
        no_llm: bool,
        /// Measure but do not write config.toml
        #[arg(long)]
        no_save: bool,
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
        /// Ollama address for chat, Converse and Live Talk (host:port).
        /// Clients cannot override it [env: VOX_OLLAMA_HOST] [default: localhost:11434]
        #[arg(long)]
        ollama_host: Option<String>,
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
    if cli.offline {
        vox::privacy::set_offline(true);
    }

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
            language,
        } => {
            // Flag, then VOX_OLLAMA_HOST, then config.toml, then the default.
            let config = cli::config::load();
            let host = vox::privacy::resolve_ollama_host(
                ollama_host.as_deref(),
                config.ollama_host.as_deref(),
            );
            // Offline mode: refuse a remote Ollama before loading any model.
            vox::privacy::ensure_ollama_host_allowed(&host.host)?;
            let tuned = config.tune.unwrap_or_default();
            let model = model
                .or(tuned.stt_model)
                .unwrap_or_else(|| "tiny.en".to_string());
            let llm = llm
                .or(tuned.llm_model)
                .unwrap_or_else(|| "llama3.2".to_string());
            cli::chat::run(
                &model,
                &llm,
                &host.host,
                yes,
                voice_mode,
                diarize,
                language.as_deref(),
            )
            .await?;
        }
        Commands::Test => {
            cli::test::run().await?;
        }
        Commands::Benchmark { yes } => {
            cli::benchmark::run(yes).await?;
        }
        Commands::Tune {
            dry_run,
            yes,
            sample,
            reference,
            language,
            llm,
            ollama_host,
            runs,
            no_llm,
            no_save,
        } => {
            cli::tune::run(cli::tune::TuneArgs {
                dry_run,
                yes,
                sample,
                reference,
                language,
                llm,
                ollama_host,
                runs,
                no_llm,
                no_save,
            })
            .await?;
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
            ollama_host,
        } => {
            cli::serve::run(cli::serve::ServeArgs {
                host,
                port,
                cache_models,
                download,
                language,
                stt_model,
                piper_voice,
                ollama_host,
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
        match parse(&["vox", "serve", "--ollama-host", "gpu-box:11434"]) {
            Commands::Serve { ollama_host, .. } => {
                assert_eq!(ollama_host.as_deref(), Some("gpu-box:11434"));
            }
            _ => panic!("expected serve"),
        }
        match parse(&["vox", "serve"]) {
            Commands::Serve { ollama_host, .. } => assert_eq!(ollama_host, None),
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
    fn offline_flag_is_accepted_before_and_after_the_subcommand() {
        for args in [
            &["vox", "--offline", "models", "list"][..],
            &["vox", "models", "list", "--offline"][..],
            &["vox", "models", "--offline", "download", "kokoro"][..],
        ] {
            let cli = Cli::try_parse_from(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert!(cli.offline, "{args:?}");
        }
        assert!(
            !Cli::try_parse_from(["vox", "models", "list"])
                .unwrap()
                .offline
        );
    }

    #[test]
    fn chat_defaults_are_left_to_config_resolution() {
        match parse(&["vox", "chat"]) {
            Commands::Chat {
                model,
                llm,
                ollama_host,
                ..
            } => {
                assert_eq!(model, None);
                assert_eq!(llm, None);
                assert_eq!(ollama_host, None);
            }
            _ => panic!("expected chat"),
        }
        match parse(&[
            "vox",
            "chat",
            "--llm",
            "qwen2.5:3b",
            "--ollama-host",
            "box:1",
        ]) {
            Commands::Chat {
                llm, ollama_host, ..
            } => {
                assert_eq!(llm.as_deref(), Some("qwen2.5:3b"));
                assert_eq!(ollama_host.as_deref(), Some("box:1"));
            }
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn tune_parses_its_flags() {
        match parse(&["vox", "tune", "--dry-run"]) {
            Commands::Tune {
                dry_run,
                runs,
                llm,
                no_llm,
                no_save,
                ..
            } => {
                assert!(dry_run);
                assert_eq!(runs, 3);
                assert!(llm.is_empty());
                assert!(!no_llm && !no_save);
            }
            _ => panic!("expected tune"),
        }
        match parse(&[
            "vox",
            "tune",
            "--llm",
            "a:1b",
            "--llm",
            "b:3b",
            "--language",
            "tr",
            "--runs",
            "5",
            "--sample",
            "me.wav",
            "--reference",
            "merhaba",
            "--no-save",
        ]) {
            Commands::Tune {
                llm,
                language,
                runs,
                sample,
                reference,
                no_save,
                dry_run,
                ..
            } => {
                assert_eq!(llm, ["a:1b", "b:3b"]);
                assert_eq!(language.as_deref(), Some("tr"));
                assert_eq!(runs, 5);
                assert_eq!(sample, Some(std::path::PathBuf::from("me.wav")));
                assert_eq!(reference.as_deref(), Some("merhaba"));
                assert!(no_save && !dry_run);
            }
            _ => panic!("expected tune"),
        }
        assert!(
            Cli::try_parse_from(["vox", "tune", "--reference", "text"]).is_err(),
            "--reference needs --sample"
        );
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
