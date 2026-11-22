//! Handler for `vox serve` — HTTP API server.

use super::models::{self, BundleOptions};
use vox::server::{LoadOptions, ServerOptions};

/// Arguments of `vox serve`.
#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    /// Deprecated, ignored by the server.
    pub cache_models: bool,
    /// `-y` / `--download`: fetch missing default models before starting.
    pub download: bool,
    /// `--language` (falls back to `VOX_LANGUAGE`).
    pub language: Option<String>,
    /// `--stt-model` (falls back to `VOX_STT_MODEL`).
    pub stt_model: Option<String>,
    /// `--piper-voice` (falls back to `VOX_PIPER_VOICE`).
    pub piper_voice: Option<String>,
    /// `--ollama-host` (falls back to `VOX_OLLAMA_HOST`, then `localhost:11434`).
    pub ollama_host: Option<String>,
}

/// Merge `--ollama-host` with the `VOX_*` server options.
pub fn server_options(args: &ServeArgs, env: ServerOptions) -> ServerOptions {
    ServerOptions {
        ollama_host: args
            .ollama_host
            .as_deref()
            .map(str::trim)
            .filter(|h| !h.is_empty())
            .map(str::to_string)
            .or(env.ollama_host),
        ..env
    }
}

/// Merge command-line flags with the `VOX_*` environment fallbacks.
pub fn load_options(args: &ServeArgs, env: LoadOptions) -> LoadOptions {
    LoadOptions {
        language: args.language.clone(),
        stt_model: args.stt_model.clone(),
        piper_voice: args.piper_voice.clone(),
    }
    .or(env)
}

fn bundle_options(opts: &LoadOptions) -> BundleOptions {
    BundleOptions {
        language: opts.language.clone(),
        stt_model: opts.stt_model.clone(),
        piper_voice: opts.piper_voice.clone(),
    }
}

/// Start the server, downloading the default models first when asked to.
pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let opts = load_options(&args, LoadOptions::from_env());
    let server_opts = server_options(&args, ServerOptions::from_env());

    // Refuse an exposed bind address before downloading or loading anything.
    vox::server::check_bind_host(&args.host, server_opts.allow_remote)?;
    // Offline mode: refuse a remote Ollama before downloading or loading anything.
    vox::privacy::ensure_ollama_host_allowed(&server_opts.resolved_ollama_host())?;

    // Reject an unknown language up front instead of starting a server
    // whose STT silently failed to load.
    #[cfg(feature = "whisper")]
    vox::WhisperConfig::new("")
        .with_language(opts.language.clone())
        .validate()?;

    let bundle_opts = bundle_options(&opts);
    if args.download {
        let failed = models::ensure_default_bundle(&bundle_opts).await;
        if !failed.is_empty() {
            eprintln!(
                "warning: starting without {} — the matching endpoints will return 503. \
                 Run `vox doctor` for details.",
                failed.join(", ")
            );
        }
    } else {
        let profile = vox::system_profile::SystemProfile::detect();
        let dir = models::models_dir();
        let bundle = models::default_bundle(&profile, &bundle_opts);
        let missing = models::missing_bundle_models(&dir, &bundle);
        if !missing.is_empty() {
            eprintln!(
                "note: {} default model(s) not found in {} ({}).\n\
                 \x20     Run `vox serve --download` to fetch them, or `vox doctor` for details.",
                missing.len(),
                dir.display(),
                missing.join(", ")
            );
        }
    }

    vox::server::run_with_config(&args.host, args.port, args.cache_models, opts, server_opts).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(language: Option<&str>, piper_voice: Option<&str>) -> ServeArgs {
        ServeArgs {
            host: "127.0.0.1".into(),
            port: 3000,
            cache_models: false,
            download: false,
            language: language.map(str::to_string),
            stt_model: None,
            piper_voice: piper_voice.map(str::to_string),
            ollama_host: None,
        }
    }

    #[test]
    fn ollama_host_flag_overrides_environment() {
        let env = ServerOptions {
            ollama_host: Some("env-host:11434".into()),
            ..ServerOptions::default()
        };
        let mut a = args(None, None);
        assert_eq!(
            server_options(&a, env.clone()).resolved_ollama_host(),
            "env-host:11434"
        );
        a.ollama_host = Some("http://gpu-box:11434/".into());
        assert_eq!(
            server_options(&a, env).resolved_ollama_host(),
            "gpu-box:11434"
        );
        assert_eq!(
            server_options(&args(None, None), ServerOptions::default()).resolved_ollama_host(),
            "localhost:11434"
        );
    }

    #[test]
    fn flags_override_environment_fallbacks() {
        let env = LoadOptions {
            language: Some("en".into()),
            stt_model: Some("base".into()),
            piper_voice: Some("de".into()),
        };
        let opts = load_options(&args(Some("tr"), None), env);
        assert_eq!(opts.language.as_deref(), Some("tr"));
        assert_eq!(opts.stt_model.as_deref(), Some("base"));
        assert_eq!(opts.piper_voice.as_deref(), Some("de"));
    }

    #[test]
    fn language_flag_reaches_server_and_bundle_options() {
        let opts = load_options(&args(Some("tr"), Some("tr")), LoadOptions::default());
        assert_eq!(opts.language.as_deref(), Some("tr"));
        let bundle = bundle_options(&opts);
        assert_eq!(bundle.language.as_deref(), Some("tr"));
        assert_eq!(models::bundle_piper_voice(&bundle), "piper-tr");
    }
}
