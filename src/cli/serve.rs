//! Handler for `vox serve` — HTTP API server.

use super::models::{self, BundleOptions};
use crate::server::LoadOptions;

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

    crate::server::run_with_options(&args.host, args.port, args.cache_models, opts).await
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
        }
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
