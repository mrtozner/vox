mod audio;
pub mod chatterbox;
pub mod hf;
mod hf_sizes;
pub mod voice;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use hf_hub::{Cache, Repo as HfRepo, RepoType};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DType {
    Fp32,
    Fp16,
    Quantized,
    Q4,
    Q4f16,
    Q8,
    Q8f16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExecutionProviderArg {
    Auto,
    Cpu,
    Coreml,
    Directml,
    Cuda,
}

fn ep_label(ep: ExecutionProviderArg) -> &'static str {
    match ep {
        ExecutionProviderArg::Auto => "auto",
        ExecutionProviderArg::Cpu => "cpu",
        ExecutionProviderArg::Coreml => "coreml",
        ExecutionProviderArg::Directml => "directml",
        ExecutionProviderArg::Cuda => "cuda",
    }
}

fn dtype_from_command(cmd: &Command) -> Option<DType> {
    match cmd {
        Command::Download { dtype, .. } => Some(*dtype),
        Command::Speak { dtype, .. } => Some(*dtype),
        Command::Inspect { dtype, .. } => Some(*dtype),
        Command::Voice { cmd } => match cmd {
            VoiceCommand::Add { dtype, .. } => Some(*dtype),
            _ => None,
        },
        _ => None,
    }
}

fn detect_effective_ep(ep: ExecutionProviderArg) -> Option<&'static str> {
    // For explicit EPs, only report it if ORT says it is available.
    match ep {
        ExecutionProviderArg::Cpu => return Some("cpu"),
        ExecutionProviderArg::Coreml => {
            #[cfg(feature = "coreml")]
            {
                use ort::ep::ExecutionProvider as _;
                if ort::ep::coreml::CoreML::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Some("coreml");
                }
                return None;
            }
            #[cfg(not(feature = "coreml"))]
            return None;
        }
        ExecutionProviderArg::Directml => {
            #[cfg(feature = "directml")]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if ort::execution_providers::DirectMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Some("directml");
                }
                return None;
            }
            #[cfg(not(feature = "directml"))]
            return None;
        }
        ExecutionProviderArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if ort::execution_providers::CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Some("cuda");
                }
                return None;
            }
            #[cfg(not(feature = "cuda"))]
            return None;
        }
        ExecutionProviderArg::Auto => {}
    }

    // Auto: prefer best EP for this platform if it's available; else CPU.
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        use ort::ep::ExecutionProvider as _;
        if ort::ep::coreml::CoreML::default()
            .is_available()
            .unwrap_or(false)
        {
            return Some("coreml");
        }
    }
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    {
        use ort::execution_providers::ExecutionProvider as _;
        if ort::execution_providers::CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            return Some("cuda");
        }
    }
    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        use ort::execution_providers::ExecutionProvider as _;
        if ort::execution_providers::DirectMLExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            return Some("directml");
        }
    }
    Some("cpu")
}

impl From<DType> for hf::ModelVariant {
    fn from(value: DType) -> Self {
        match value {
            DType::Fp32 => hf::ModelVariant::Fp32,
            DType::Fp16 => hf::ModelVariant::Fp16,
            DType::Quantized => hf::ModelVariant::Quantized,
            DType::Q4 => hf::ModelVariant::Q4,
            DType::Q4f16 => hf::ModelVariant::Q4f16,
            DType::Q8 => hf::ModelVariant::Q8,
            DType::Q8f16 => hf::ModelVariant::Q8f16,
        }
    }
}

impl DType {
    fn label(self) -> &'static str {
        match self {
            DType::Fp32 => "fp32",
            DType::Fp16 => "fp16",
            DType::Quantized => "quantized",
            DType::Q4 => "q4",
            DType::Q4f16 => "q4f16",
            DType::Q8 => "q8",
            DType::Q8f16 => "q8f16",
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            DType::Fp32 => "",
            DType::Fp16 => "_fp16",
            DType::Quantized => "_quantized",
            DType::Q4 => "_q4",
            DType::Q4f16 => "_q4f16",
            DType::Q8 => "_q8",
            DType::Q8f16 => "_q8f16",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "cbx",
    version,
    about = "Local TTS from ONNX Runtime — Chatterbox Turbo in a single Rust binary"
)]
struct Cli {
    /// Threads used to parallelize execution *within* operators (intra-op).
    ///
    /// Defaults to all available cores.
    #[arg(long, global = true)]
    intra_threads: Option<usize>,

    /// Threads used to parallelize execution *across* operators (inter-op).
    ///
    /// Only used when `--parallel-execution` is enabled.
    #[arg(long, global = true)]
    inter_threads: Option<usize>,

    /// Enable ORT parallel execution mode (inter-op parallelism).
    #[arg(long, global = true)]
    parallel_execution: bool,

    /// Execution provider (backend).
    ///
    /// CPU is the default and fastest for this model. CoreML/DirectML are available but slower
    /// due to graph partitioning overhead. See README for benchmark details.
    #[arg(long, value_enum, default_value_t = ExecutionProviderArg::Cpu, global = true)]
    ep: ExecutionProviderArg,

    /// Optional cache directory for compiled CoreML models (when `--ep coreml`).
    #[arg(long, global = true)]
    coreml_cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Download required model/tokenizer files into the local HF cache.
    Download {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, value_enum, default_value_t = DType::Fp16)]
        dtype: DType,
    },

    /// Run TTS and write a WAV file (auto-downloads models on first run).
    Speak {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, value_enum, default_value_t = DType::Fp16)]
        dtype: DType,

        /// Use a cached voice profile by name.
        ///
        /// If omitted and no `--voice-wav` is provided, cbx will use the first cached voice (if any).
        #[arg(long)]
        voice: Option<String>,

        /// Input text to synthesize.
        #[arg(long)]
        text: String,

        /// Reference voice WAV (slow path). Will be converted to 24kHz mono f32 and encoded each run.
        #[arg(long)]
        voice_wav: Option<PathBuf>,

        /// Output WAV path.
        #[arg(long, default_value = "out.wav")]
        out_wav: PathBuf,

        /// Maximum new tokens to generate (speech tokens).
        #[arg(long, default_value_t = 8192)]
        max_new_tokens: usize,

        /// Repetition penalty, 1.0 disables.
        #[arg(long, default_value_t = 1.2)]
        repetition_penalty: f32,
    },

    /// Print model input/output names for debugging.
    Inspect {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, value_enum, default_value_t = DType::Fp16)]
        dtype: DType,
    },

    /// Delete Hugging Face cache entries for this model (useful to reclaim disk space).
    ///
    /// This command deletes either:
    /// - the snapshot for a given `--revision` (default), or
    /// - the entire repo cache folder when `--all` is used.
    ///
    /// To avoid surprises, deletion requires `--yes` unless `--dry-run` is provided.
    #[command(alias = "rm", alias = "delete")]
    Clean {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        /// Delete only a specific snapshot (commit hash) under `.../snapshots/<hash>`.
        ///
        /// If provided, this bypasses the `refs/<revision>` lookup.
        #[arg(long)]
        snapshot: Option<String>,
        /// Delete only specific model variants within the selected snapshot (repeatable).
        ///
        /// Example:
        ///   cbx clean --dtype fp32 --dtype q4 --dry-run
        ///
        /// If omitted, cbx deletes the entire snapshot (current behavior).
        #[arg(long, value_enum, action = ArgAction::Append)]
        dtype: Vec<DType>,
        /// Delete the entire repo cache folder (blobs + snapshots + refs).
        #[arg(long)]
        all: bool,
        /// Also delete cached voice profiles (cbx voice cache).
        #[arg(long)]
        voices: bool,
        /// Print what would be deleted without deleting anything.
        #[arg(long)]
        dry_run: bool,
        /// Actually perform deletion.
        #[arg(long)]
        yes: bool,
    },

    /// List which model variants are already present in the local Hugging Face cache.
    ///
    /// This inspects the HF cache folder on disk; it does not contact the network.
    #[command(alias = "models", alias = "ls")]
    List {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        /// Show per-snapshot detailed paths.
        #[arg(long)]
        long: bool,
    },

    /// Show what has actually been downloaded for this model (snapshots + variants + disk usage).
    ///
    /// This inspects the HF cache folder on disk; it does not contact the network.
    Downloads {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        /// Show per-snapshot detailed paths.
        #[arg(long)]
        long: bool,
    },

    /// Show the total download size per model variant (fp32/fp16/q4/q8/etc).
    ///
    /// This queries the Hugging Face model metadata API and does not download the files.
    Sizes {
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
    },

    /// Manage cached voices (fast path for `speak`).
    Voice {
        #[command(subcommand)]
        cmd: VoiceCommand,
    },
}

#[derive(Subcommand, Debug)]
enum VoiceCommand {
    /// Create/update a named voice profile from a reference WAV.
    Add {
        /// Name for this voice profile (used by `--voice`).
        #[arg(long)]
        name: String,
        /// Reference voice WAV (mono or stereo). Will be converted to 24kHz mono f32.
        #[arg(long)]
        voice_wav: PathBuf,
        #[arg(long, default_value = "ResembleAI/chatterbox-turbo-ONNX")]
        repo_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, value_enum, default_value_t = DType::Fp16)]
        dtype: DType,
    },
    /// List cached voice profiles.
    List,
    /// Remove a cached voice profile.
    Remove {
        #[arg(long)]
        name: String,
    },
    /// Delete all cached voice profiles.
    ///
    /// To avoid surprises, deletion requires `--yes` unless `--dry-run` is provided.
    Clean {
        /// Print what would be deleted without deleting anything.
        #[arg(long)]
        dry_run: bool,
        /// Actually perform deletion.
        #[arg(long)]
        yes: bool,
    },
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ORT uses global initialization.
    ort::init().with_name("cbx").commit();

    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let intra_threads = cli.intra_threads.unwrap_or(default_threads);
    let inter_threads = cli.inter_threads.unwrap_or(default_threads);

    // Default CoreML cache dir (if not provided): $HF_HOME/cbx/coreml-cache
    let mut coreml_cache_dir = cli.coreml_cache_dir.clone();
    if coreml_cache_dir.is_none() {
        let hub = Cache::from_env().path().clone();
        if let Some(hf_home) = hub.parent() {
            coreml_cache_dir = Some(hf_home.join("cbx").join("coreml-cache"));
        }
    }

    let session_cfg = chatterbox::SessionConfig {
        intra_threads: Some(intra_threads),
        inter_threads: Some(inter_threads),
        parallel_execution: cli.parallel_execution,
        execution_provider: match cli.ep {
            ExecutionProviderArg::Auto => chatterbox::ExecutionProvider::Auto,
            ExecutionProviderArg::Cpu => chatterbox::ExecutionProvider::Cpu,
            ExecutionProviderArg::Coreml => chatterbox::ExecutionProvider::CoreML,
            ExecutionProviderArg::Directml => chatterbox::ExecutionProvider::DirectML,
            ExecutionProviderArg::Cuda => chatterbox::ExecutionProvider::Cuda,
        },
        coreml_cache_dir,
    };

    // Startup banner: show what we are configured to use (threads, EP, dtype).
    let dtype = dtype_from_command(&cli.cmd).map(|d| d.label());
    let requested_ep = ep_label(cli.ep);
    let effective_ep = detect_effective_ep(cli.ep).unwrap_or("unavailable");
    println!("Runtime:");
    println!("  available_cores: {default_threads}");
    println!("  intra_threads:   {intra_threads}");
    println!("  parallel_exec:   {}", cli.parallel_execution);
    if cli.parallel_execution {
        println!("  inter_threads:   {inter_threads}");
    }
    println!("  ep_requested:    {requested_ep}");
    println!("  ep_effective:    {effective_ep}");
    if let Some(d) = dtype {
        println!("  dtype:           {d}");
    }
    println!();

    match cli.cmd {
        Command::Download {
            repo_id,
            revision,
            dtype,
        } => {
            println!("Model: {}@{} dtype={}", repo_id, revision, dtype.label());
            let paths = hf::download_chatterbox_assets(&repo_id, &revision, dtype.into())?;
            println!("Downloaded assets:");
            println!("  tokenizer: {}", paths.tokenizer_json.display());
            println!(
                "  conditional_decoder: {}",
                paths.conditional_decoder.display()
            );
            println!("  speech_encoder: {}", paths.speech_encoder.display());
            println!("  embed_tokens: {}", paths.embed_tokens.display());
            println!("  language_model: {}", paths.language_model.display());
            Ok(())
        }
        Command::Inspect {
            repo_id,
            revision,
            dtype,
        } => {
            println!("Model: {}@{} dtype={}", repo_id, revision, dtype.label());
            let paths = hf::download_chatterbox_assets(&repo_id, &revision, dtype.into())?;
            let model = chatterbox::Chatterbox::load_with(&paths, &session_cfg)?;
            model.print_io();
            Ok(())
        }
        Command::Speak {
            repo_id,
            revision,
            dtype,
            voice,
            text,
            voice_wav,
            out_wav,
            max_new_tokens,
            repetition_penalty,
        } => {
            let t_total = Instant::now();
            println!("Model: {}@{} dtype={}", repo_id, revision, dtype.label());
            let t_download = Instant::now();
            let paths = hf::download_chatterbox_assets(&repo_id, &revision, dtype.into())?;
            let download_ms = t_download.elapsed().as_secs_f64() * 1000.0;
            println!("Model cache paths:");
            println!("  tokenizer: {}", paths.tokenizer_json.display());
            println!(
                "  conditional_decoder: {}",
                paths.conditional_decoder.display()
            );
            println!("  speech_encoder: {}", paths.speech_encoder.display());
            println!("  embed_tokens: {}", paths.embed_tokens.display());
            println!("  language_model: {}", paths.language_model.display());
            if let Some(snapshot_root) = hf::snapshot_root_from_cache_path(&paths.tokenizer_json) {
                println!("  snapshot_root: {}", snapshot_root.display());
            }

            let t_load = Instant::now();
            let mut model = chatterbox::Chatterbox::load_with(&paths, &session_cfg)?;
            let model_load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

            let cache_root = voice::voice_cache_dir()?;
            println!("Voice cache dir: {}", cache_root.display());
            let t_voice = Instant::now();
            let profile = if let Some(wav_path) = voice_wav {
                // Slow path: encode voice every run.
                println!("Voice: --voice-wav provided (slow path, not cached)");
                model.encode_voice_profile(&wav_path, &repo_id, &revision, dtype.label())?
            } else {
                // Fast path: cached voice (default to first profile if `--voice` not provided).
                let name = if let Some(name) = voice {
                    name
                } else {
                    voice::pick_voice_for_model(&cache_root, &repo_id, &revision, dtype.label())?
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "no cached voice found for dtype={}; run `cbx voice add --dtype {} --name default --voice-wav ./voice.wav`, install the bundled default voice, or pass `--voice-wav`",
                                dtype.label(),
                                dtype.label()
                            )
                        })?
                };
                println!("Voice: {name} (cached)");
                voice::load_voice_profile(&cache_root, &name).with_context(|| {
                    format!("failed to load voice profile '{name}' (try `cbx voice list`)")
                })?
            };
            let voice_profile_ms = t_voice.elapsed().as_secs_f64() * 1000.0;

            let t_synth = Instant::now();
            let wav = model.synthesize_with_voice_profile(
                &text,
                &repo_id,
                &revision,
                dtype.label(),
                &profile,
                max_new_tokens,
                repetition_penalty,
            )?;
            let synth_ms = t_synth.elapsed().as_secs_f64() * 1000.0;
            audio::write_wav_f32_mono_24k(&out_wav, &wav)
                .with_context(|| format!("failed to write {}", out_wav.display()))?;
            println!("Wrote {}", out_wav.display());

            let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            let inner = model.take_last_timings().unwrap_or_default();
            println!(
                "Timings(ms): download={:.0} model_load={:.0} voice_profile={:.0} synth={:.0} total={:.0} | tokenize={:.0} embed={:.0} lm={:.0} decoder={:.0} inner_total={:.0}",
                download_ms,
                model_load_ms,
                voice_profile_ms,
                synth_ms,
                total_ms,
                inner.tokenize_ms,
                inner.embed_ms,
                inner.lm_ms,
                inner.decoder_ms,
                inner.total_ms
            );
            Ok(())
        }
        Command::Clean {
            repo_id,
            revision,
            snapshot,
            dtype,
            all,
            voices,
            dry_run,
            yes,
        } => {
            if !dry_run && !yes {
                anyhow::bail!("refusing to delete without --yes (or use --dry-run to preview)");
            }

            let cache = Cache::from_env();
            let repo = HfRepo::new(repo_id.clone(), RepoType::Model);
            let mut repo_dir = cache.path().clone();
            repo_dir.push(repo.folder_name());

            let voice_dir = if voices {
                Some(voice::voice_cache_dir()?)
            } else {
                None
            };

            if all {
                if !dtype.is_empty() || snapshot.is_some() {
                    anyhow::bail!("--all cannot be combined with --dtype or --snapshot");
                }
                println!("HF cache root: {}", cache.path().display());
                println!("Would delete repo cache folder: {}", repo_dir.display());
                if let Some(vd) = &voice_dir {
                    println!("Would delete voice cache dir: {}", vd.display());
                }
                if !dry_run {
                    std::fs::remove_dir_all(&repo_dir)
                        .with_context(|| format!("failed to delete {}", repo_dir.display()))?;
                    println!("Deleted {}", repo_dir.display());
                    if let Some(vd) = &voice_dir {
                        voice::remove_all_voices(vd)?;
                        println!("Deleted {}", vd.display());
                    }
                }
                return Ok(());
            }

            // Select the snapshot directory.
            // - If `--snapshot` is provided: use it directly.
            // - Else: resolve via `refs/<revision>` (existing behavior).
            let commit_hash = if let Some(s) = snapshot.as_deref() {
                s.trim().to_string()
            } else {
                let mut ref_path = repo_dir.clone();
                ref_path.push("refs");
                ref_path.push(&revision);

                if !ref_path.exists() {
                    println!("HF cache root: {}", cache.path().display());
                    println!("No ref found at {} (nothing to do).", ref_path.display());
                    return Ok(());
                }

                let hash = std::fs::read_to_string(&ref_path)
                    .with_context(|| format!("failed to read {}", ref_path.display()))?;
                let hash_trimmed = hash.trim();
                if hash_trimmed.is_empty() {
                    println!("Ref {} is empty (nothing to do).", ref_path.display());
                    return Ok(());
                }
                hash_trimmed.to_string()
            };

            let mut snapshot_dir = repo_dir.clone();
            snapshot_dir.push("snapshots");
            snapshot_dir.push(&commit_hash);

            println!("HF cache root: {}", cache.path().display());
            if dtype.is_empty() {
                println!("Would delete snapshot: {}", snapshot_dir.display());
            } else {
                println!("Snapshot dir: {}", snapshot_dir.display());
                println!(
                    "Would delete variants: {}",
                    dtype
                        .iter()
                        .map(|d| d.label())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            if let Some(vd) = &voice_dir {
                println!("Would delete voice cache dir: {}", vd.display());
            }

            if !snapshot_dir.exists() {
                println!("Snapshot does not exist (nothing to do).");
                return Ok(());
            }

            let mut removed_paths: Vec<std::path::PathBuf> = Vec::new();
            let mut ignored_snapshot_dirs: Vec<std::path::PathBuf> = Vec::new();

            if dtype.is_empty() {
                if dry_run {
                    ignored_snapshot_dirs.push(snapshot_dir.clone());
                } else {
                    std::fs::remove_dir_all(&snapshot_dir)
                        .with_context(|| format!("failed to delete {}", snapshot_dir.display()))?;
                    println!("Deleted {}", snapshot_dir.display());
                }
            } else {
                removed_paths = delete_variant_files_in_snapshot(&snapshot_dir, &dtype, dry_run)?;
                if removed_paths.is_empty() {
                    println!("No matching variant files found to delete.");
                }
            }

            // Best-effort blob pruning to actually reclaim disk (HF snapshots commonly symlink into blobs/).
            // For dry-run, approximate which blobs *would* become unreferenced after deletion.
            prune_unreferenced_hf_blobs(
                &repo_dir,
                dry_run,
                &removed_paths,
                &ignored_snapshot_dirs,
            )?;

            if !dry_run {
                if let Some(vd) = &voice_dir {
                    voice::remove_all_voices(vd)?;
                    println!("Deleted {}", vd.display());
                }
            }

            Ok(())
        }
        Command::List { repo_id, long } => print_repo_cache_listing(&repo_id, long, false),
        Command::Downloads { repo_id, long } => print_repo_cache_listing(&repo_id, long, true),
        Command::Voice { cmd } => match cmd {
            VoiceCommand::List => {
                let cache_root = voice::voice_cache_dir()?;
                println!("Voice cache dir: {}", cache_root.display());
                let names = voice::list_voice_profiles(&cache_root)?;
                if names.is_empty() {
                    println!("No cached voices found.");
                } else {
                    println!("Cached voices:");
                    for n in names {
                        println!("  {n}");
                    }
                    println!();
                    println!("Default behavior:");
                    println!(
                        "  If you run `cbx speak` without --voice-wav, cbx will pick a cached voice that matches the requested --dtype."
                    );
                }
                Ok(())
            }
            VoiceCommand::Remove { name } => {
                let cache_root = voice::voice_cache_dir()?;
                voice::remove_voice_profile(&cache_root, &name)?;
                println!("Removed voice profile: {name}");
                Ok(())
            }
            VoiceCommand::Clean { dry_run, yes } => {
                if !dry_run && !yes {
                    anyhow::bail!("refusing to delete without --yes (or use --dry-run to preview)");
                }
                let cache_root = voice::voice_cache_dir()?;
                println!("Voice cache dir: {}", cache_root.display());
                println!("Would delete voice cache dir: {}", cache_root.display());
                if !dry_run {
                    voice::remove_all_voices(&cache_root)?;
                    println!("Deleted {}", cache_root.display());
                }
                Ok(())
            }
            VoiceCommand::Add {
                name,
                voice_wav,
                repo_id,
                revision,
                dtype,
            } => {
                let model_paths =
                    hf::download_chatterbox_assets(&repo_id, &revision, dtype.into())?;
                let mut model = chatterbox::Chatterbox::load_with(&model_paths, &session_cfg)?;
                let profile =
                    model.encode_voice_profile(&voice_wav, &repo_id, &revision, dtype.label())?;
                let cache_root = voice::voice_cache_dir()?;
                voice::save_voice_profile(&cache_root, &name, &profile)?;
                println!("Saved voice profile: {name}");
                println!("Voice cache dir: {}", cache_root.display());
                println!("You can now run:");
                println!("  cbx speak --voice {name} --text \"Hello\" --out-wav out.wav");
                Ok(())
            }
        },
        Command::Sizes { repo_id, revision } => {
            println!("Repo: {repo_id}@{revision}");
            let table = hf_sizes::fetch_size_table(&repo_id, &revision)?;

            fn variant_label(v: hf::ModelVariant) -> &'static str {
                match v {
                    hf::ModelVariant::Fp32 => "fp32",
                    hf::ModelVariant::Fp16 => "fp16",
                    hf::ModelVariant::Quantized => "quantized",
                    hf::ModelVariant::Q4 => "q4",
                    hf::ModelVariant::Q4f16 => "q4f16",
                    hf::ModelVariant::Q8 => "q8",
                    hf::ModelVariant::Q8f16 => "q8f16",
                }
            }

            let rows: Vec<_> = table
                .variants
                .into_iter()
                .map(|v| {
                    let label = variant_label(v.variant);
                    let files = v.file_count.to_string();
                    let all_model_files_missing =
                        v.missing.len() == 8 && !v.missing.iter().any(|f| f == "tokenizer.json");

                    let total = if all_model_files_missing {
                        "N/A".to_string()
                    } else {
                        hf_sizes::format_bytes(v.total_bytes)
                    };

                    let notes = if all_model_files_missing {
                        "not published in this repo".to_string()
                    } else if v.missing.is_empty() {
                        "".to_string()
                    } else {
                        format!("missing {} file(s)", v.missing.len())
                    };

                    (label.to_string(), files, total, notes, v.missing)
                })
                .collect();

            let w_variant = rows
                .iter()
                .map(|r| r.0.len())
                .max()
                .unwrap_or("variant".len())
                .max("variant".len());
            let w_files = rows
                .iter()
                .map(|r| r.1.len())
                .max()
                .unwrap_or("files".len())
                .max("files".len());
            let w_total = rows
                .iter()
                .map(|r| r.2.len())
                .max()
                .unwrap_or("total".len())
                .max("total".len());

            println!();
            println!(
                "{:w_variant$}  {:w_files$}  {:w_total$}  {}",
                "variant",
                "files",
                "total",
                "notes",
                w_variant = w_variant,
                w_files = w_files,
                w_total = w_total
            );
            println!(
                "{:w_variant$}  {:w_files$}  {:w_total$}  {}",
                "-".repeat("variant".len()),
                "-".repeat("files".len()),
                "-".repeat("total".len()),
                "-".repeat("notes".len()),
                w_variant = w_variant,
                w_files = w_files,
                w_total = w_total
            );

            for (label, files, total, note, _) in &rows {
                println!(
                    "{:w_variant$}  {:w_files$}  {:w_total$}  {}",
                    label,
                    files,
                    total,
                    note,
                    w_variant = w_variant,
                    w_files = w_files,
                    w_total = w_total
                );
            }

            if let Some(sz) = table.tokenizer_bytes {
                println!();
                println!("tokenizer.json: {}", hf_sizes::format_bytes(sz));
            }

            // Print missing detail (if any).
            let mut any_missing = false;
            for (label, _files, _total, _note, missing) in &rows {
                if missing.is_empty() {
                    continue;
                }
                any_missing = true;
                println!();
                println!("{label}: missing files:");
                for f in missing {
                    println!("  - {f}");
                }
            }
            let _ = any_missing;

            Ok(())
        }
    }
}

fn delete_variant_files_in_snapshot(
    snapshot_dir: &std::path::Path,
    dtypes: &[DType],
    dry_run: bool,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let onnx_stems = [
        "conditional_decoder",
        "speech_encoder",
        "embed_tokens",
        "language_model",
    ];

    let mut removed = Vec::new();
    for dtype in dtypes {
        let suf = dtype.suffix();
        for stem in onnx_stems {
            for ext in ["onnx", "onnx_data"] {
                let rel = format!("onnx/{stem}{suf}.{ext}");
                let path = snapshot_dir.join(rel);
                if !path.exists() {
                    continue;
                }
                println!(
                    "{} {}",
                    if dry_run { "Would delete" } else { "Deleting" },
                    path.display()
                );
                if !dry_run {
                    std::fs::remove_file(&path)
                        .with_context(|| format!("failed to delete {}", path.display()))?;
                }
                removed.push(path);
            }
        }
    }

    // If this snapshot is now missing variants that `cbx list` would otherwise report,
    // that's expected: users often want to keep only one variant around.
    Ok(removed)
}

fn print_repo_cache_listing(repo_id: &str, long: bool, show_sizes: bool) -> anyhow::Result<()> {
    let cache = Cache::from_env();
    let repo = HfRepo::new(repo_id.to_string(), RepoType::Model);

    let mut repo_dir = cache.path().clone();
    repo_dir.push(repo.folder_name());

    println!("HF cache root: {}", cache.path().display());
    println!("Repo cache dir: {}", repo_dir.display());

    if !repo_dir.exists() {
        println!("No cache found for this repo.");
        return Ok(());
    }

    // Print refs (revision -> commit hash) if present.
    let refs_dir = repo_dir.join("refs");
    if refs_dir.exists() {
        println!("Refs:");
        let mut refs: Vec<_> = std::fs::read_dir(&refs_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        refs.sort_by_key(|e| e.file_name());
        for e in refs {
            let name = e.file_name().to_string_lossy().to_string();
            let hash = std::fs::read_to_string(e.path()).unwrap_or_default();
            let hash = hash.trim();
            if !hash.is_empty() {
                println!("  {name} -> {hash}");
            }
        }
    }

    let snapshots_dir = repo_dir.join("snapshots");
    if !snapshots_dir.exists() {
        println!("No snapshots folder found (nothing downloaded yet).");
        return Ok(());
    }

    let mut snapshots: Vec<_> = std::fs::read_dir(&snapshots_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    snapshots.sort_by_key(|e| e.file_name());

    if snapshots.is_empty() {
        println!("No snapshots found (nothing downloaded yet).");
        return Ok(());
    }

    println!("Snapshots:");
    for snap in snapshots {
        let commit = snap.file_name().to_string_lossy().to_string();
        let snap_path = snap.path();

        let dtypes = list_dtypes_in_snapshot(&snap_path);
        if dtypes.is_empty() {
            println!("  {commit}: (no complete variants)");
            if long {
                println!("    path: {}", snap_path.display());
            }
            continue;
        }

        if show_sizes {
            let mut parts = Vec::new();
            for dt in &dtypes {
                let bytes = variant_bytes_in_snapshot(&snap_path, *dt).unwrap_or(0);
                parts.push(format!(
                    "{} ({})",
                    dt.label(),
                    hf_sizes::format_bytes(bytes)
                ));
            }
            println!("  {commit}: {}", parts.join(", "));
        } else {
            let labels: Vec<_> = dtypes.iter().map(|d| d.label()).collect();
            println!("  {commit}: {}", labels.join(", "));
        }

        if long {
            println!("    path: {}", snap_path.display());
            if show_sizes {
                let tok = snap_path.join("tokenizer.json");
                if tok.exists() {
                    let sz = std::fs::metadata(&tok).map(|m| m.len()).unwrap_or(0);
                    println!("    tokenizer.json: {}", hf_sizes::format_bytes(sz));
                }
            }
        }
    }

    Ok(())
}

fn list_dtypes_in_snapshot(snapshot_dir: &std::path::Path) -> Vec<DType> {
    let tokenizer_ok = snapshot_dir.join("tokenizer.json").exists();
    let onnx_dir = snapshot_dir.join("onnx");

    const VARIANTS: [(DType, &str); 7] = [
        (DType::Fp32, ""),
        (DType::Fp16, "_fp16"),
        (DType::Quantized, "_quantized"),
        (DType::Q4, "_q4"),
        (DType::Q4f16, "_q4f16"),
        (DType::Q8, "_q8"),
        (DType::Q8f16, "_q8f16"),
    ];

    fn has_pair(onnx_dir: &std::path::Path, stem: &str, suffix: &str) -> bool {
        let onnx = onnx_dir.join(format!("{stem}{suffix}.onnx"));
        let data = onnx_dir.join(format!("{stem}{suffix}.onnx_data"));
        onnx.exists() && data.exists()
    }

    let mut out = Vec::new();
    if !tokenizer_ok || !onnx_dir.exists() {
        return out;
    }

    for (dtype, suffix) in VARIANTS {
        let ok = has_pair(&onnx_dir, "conditional_decoder", suffix)
            && has_pair(&onnx_dir, "speech_encoder", suffix)
            && has_pair(&onnx_dir, "embed_tokens", suffix)
            && has_pair(&onnx_dir, "language_model", suffix);
        if ok {
            out.push(dtype);
        }
    }
    out
}

fn variant_bytes_in_snapshot(snapshot_dir: &std::path::Path, dtype: DType) -> Option<u64> {
    let suf = dtype.suffix();
    let stems = [
        "conditional_decoder",
        "speech_encoder",
        "embed_tokens",
        "language_model",
    ];
    let mut total = 0u64;
    for stem in stems {
        for ext in ["onnx", "onnx_data"] {
            let p = snapshot_dir.join(format!("onnx/{stem}{suf}.{ext}"));
            let sz = std::fs::metadata(&p).ok()?.len();
            total = total.saturating_add(sz);
        }
    }
    Some(total)
}

fn prune_unreferenced_hf_blobs(
    repo_dir: &std::path::Path,
    dry_run: bool,
    ignore_snapshot_paths: &[std::path::PathBuf],
    ignore_snapshot_dirs: &[std::path::PathBuf],
) -> anyhow::Result<()> {
    let blobs_dir = repo_dir.join("blobs");
    let snapshots_dir = repo_dir.join("snapshots");
    if !blobs_dir.exists() || !snapshots_dir.exists() {
        return Ok(());
    }

    // Only attempt blob pruning if snapshots actually symlink into blobs.
    // Some setups disable symlinks (or use hardlinks/copies), and in that case pruning blobs
    // can remove a cache optimization without necessarily reclaiming disk.
    let blobs_dir_canon = std::fs::canonicalize(&blobs_dir).unwrap_or(blobs_dir.clone());

    let mut referenced = std::collections::HashSet::<std::path::PathBuf>::new();
    let mut saw_blob_symlink = false;

    fn visit_dir(
        dir: &std::path::Path,
        blobs_dir_canon: &std::path::Path,
        referenced: &mut std::collections::HashSet<std::path::PathBuf>,
        saw_blob_symlink: &mut bool,
        ignore_paths: &std::collections::HashSet<std::path::PathBuf>,
        ignore_dirs: &std::collections::HashSet<std::path::PathBuf>,
    ) -> anyhow::Result<()> {
        if ignore_dirs.contains(dir) {
            return Ok(());
        }
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let p = e.path();
            if ignore_paths.contains(&p) {
                continue;
            }
            let md = std::fs::symlink_metadata(&p)?;
            if md.is_dir() {
                visit_dir(
                    &p,
                    blobs_dir_canon,
                    referenced,
                    saw_blob_symlink,
                    ignore_paths,
                    ignore_dirs,
                )?;
                continue;
            }
            if !md.file_type().is_symlink() {
                continue;
            }
            let target = std::fs::read_link(&p)?;
            let abs = p.parent().unwrap_or(dir).join(target);
            let abs = std::fs::canonicalize(&abs).unwrap_or(abs);
            if abs.starts_with(blobs_dir_canon) {
                *saw_blob_symlink = true;
                referenced.insert(abs);
            }
        }
        Ok(())
    }

    // Collect referenced blob paths.
    let ignore_paths: std::collections::HashSet<std::path::PathBuf> =
        ignore_snapshot_paths.iter().cloned().collect();
    let ignore_dirs: std::collections::HashSet<std::path::PathBuf> =
        ignore_snapshot_dirs.iter().cloned().collect();
    visit_dir(
        &snapshots_dir,
        &blobs_dir_canon,
        &mut referenced,
        &mut saw_blob_symlink,
        &ignore_paths,
        &ignore_dirs,
    )?;

    if !saw_blob_symlink {
        return Ok(());
    }

    // Delete any blob files not referenced by any remaining snapshots.
    for e in std::fs::read_dir(&blobs_dir)? {
        let e = e?;
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if name.ends_with(".lock") || name.ends_with(".incomplete") {
            continue;
        }
        let pc = std::fs::canonicalize(&p).unwrap_or(p.clone());
        if referenced.contains(&pc) {
            continue;
        }
        println!(
            "{} {}",
            if dry_run {
                "Would delete unreferenced blob"
            } else {
                "Deleting unreferenced blob"
            },
            p.display()
        );
        if !dry_run {
            let _ = std::fs::remove_file(&p);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn clean_deletes_selected_variant_and_prunes_blobs() -> anyhow::Result<()> {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir()?;
        let repo_dir = tmp.path().join("models--ORG--REPO");
        let blobs_dir = repo_dir.join("blobs");
        let snap1_dir = repo_dir.join("snapshots").join("snap1");
        let snap2_dir = repo_dir.join("snapshots").join("snap2");

        std::fs::create_dir_all(&blobs_dir)?;
        std::fs::create_dir_all(snap1_dir.join("onnx"))?;
        std::fs::create_dir_all(snap2_dir.join("onnx"))?;

        // Create one "kept" blob referenced by a different snapshot.
        let kept_blob = blobs_dir.join("kept_blob");
        std::fs::write(&kept_blob, b"kept")?;
        symlink(
            "../../../blobs/kept_blob",
            snap2_dir.join("onnx/language_model.onnx"),
        )?;

        // Create fp32 variant symlinks in snap1 pointing to unique blobs.
        let stems = [
            "conditional_decoder",
            "speech_encoder",
            "embed_tokens",
            "language_model",
        ];
        for stem in stems {
            for ext in ["onnx", "onnx_data"] {
                let blob_name = format!("blob_{stem}.{ext}");
                let blob_path = blobs_dir.join(&blob_name);
                std::fs::write(&blob_path, b"x")?;
                let link_path = snap1_dir.join(format!("onnx/{stem}.{ext}"));
                symlink(format!("../../../blobs/{blob_name}"), link_path)?;
            }
        }

        // Delete only fp32 from snap1 (should remove symlinks).
        let removed = delete_variant_files_in_snapshot(&snap1_dir, &[DType::Fp32], false)?;
        assert!(!removed.is_empty());

        // Prune unreferenced blobs (should delete the fp32 blobs, but keep kept_blob).
        prune_unreferenced_hf_blobs(&repo_dir, false, &[], &[])?;

        assert!(kept_blob.exists());
        for stem in stems {
            for ext in ["onnx", "onnx_data"] {
                let blob_name = format!("blob_{stem}.{ext}");
                let blob_path = blobs_dir.join(blob_name);
                assert!(
                    !blob_path.exists(),
                    "expected blob to be pruned: {}",
                    blob_path.display()
                );
            }
        }

        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn downloads_reports_variant_sizes_from_symlink_targets() -> anyhow::Result<()> {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir()?;
        let repo_dir = tmp.path().join("models--ORG--REPO");
        let blobs_dir = repo_dir.join("blobs");
        let snap_dir = repo_dir.join("snapshots").join("snap1");
        std::fs::create_dir_all(&blobs_dir)?;
        std::fs::create_dir_all(snap_dir.join("onnx"))?;
        std::fs::write(snap_dir.join("tokenizer.json"), b"tok")?;

        // Create fp16 variant symlinks pointing at blobs with known sizes.
        let stems = [
            "conditional_decoder",
            "speech_encoder",
            "embed_tokens",
            "language_model",
        ];
        let mut expected = 0u64;
        for (i, stem) in stems.iter().enumerate() {
            for ext in ["onnx", "onnx_data"] {
                let blob_name = format!("fp16_{stem}.{ext}");
                let blob_path = blobs_dir.join(&blob_name);
                let size = 10u64 + (i as u64);
                std::fs::write(&blob_path, vec![0u8; size as usize])?;
                expected += size;

                let link_path = snap_dir.join(format!("onnx/{stem}_fp16.{ext}"));
                symlink(format!("../../../blobs/{blob_name}"), link_path)?;
            }
        }

        let got = variant_bytes_in_snapshot(&snap_dir, DType::Fp16).unwrap_or(0);
        assert_eq!(got, expected);
        Ok(())
    }
}
