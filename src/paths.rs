//! Filesystem locations shared by the CLI, the server and the Python bindings.
//!
//! This module is the single source of truth for *where vox keeps things*.
//! It is dependency-free and always compiled, so every front end resolves the
//! same directories regardless of which cargo features are enabled.
//!
//! # Models directory
//!
//! [`models_dir`] resolves in this order (first match wins):
//!
//! 1. The `VOX_MODELS_DIR` environment variable (explicit override).
//! 2. A source checkout: a `models/` directory that sits next to vox's own
//!    `Cargo.toml`, found by walking up from the running executable and from
//!    the current working directory. This is detected at **run time**; nothing
//!    about the build machine is baked into the binary.
//! 3. The per-user OS data directory: `<data_local_dir>/vox/models`
//!    (`~/Library/Application Support/vox/models` on macOS,
//!    `$XDG_DATA_HOME/vox/models` or `~/.local/share/vox/models` on Linux,
//!    `%LOCALAPPDATA%\vox\models` on Windows).
//!
//! The data directory root ([`data_dir`]) is the same one the speaker
//! database lives in, so an installed binary keeps all of its state under a
//! single `vox/` folder.
//!
//! Resolution never creates directories; callers that write (downloads)
//! create what they need.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Environment variable that overrides the models directory.
pub const MODELS_DIR_ENV: &str = "VOX_MODELS_DIR";

/// Package names that identify vox's own `Cargo.toml`.
const VOX_PACKAGE_NAMES: &[&str] = &["vox", "vox-voice"];

/// Why a particular models directory was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelsDirSource {
    /// `VOX_MODELS_DIR` was set.
    Env,
    /// `models/` next to vox's `Cargo.toml` (running from a source checkout).
    Repo,
    /// Per-user OS data directory.
    DataDir,
    /// No home/data directory could be determined; `./models` is used.
    Fallback,
}

impl ModelsDirSource {
    /// Human-readable explanation, used by `vox doctor`.
    pub fn describe(self) -> &'static str {
        match self {
            ModelsDirSource::Env => "VOX_MODELS_DIR environment variable",
            ModelsDirSource::Repo => "models/ next to vox's Cargo.toml (source checkout)",
            ModelsDirSource::DataDir => "per-user data directory",
            ModelsDirSource::Fallback => "no user data directory found, using ./models",
        }
    }
}

/// A resolved models directory together with the rule that selected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsDirResolution {
    pub path: PathBuf,
    pub source: ModelsDirSource,
}

/// Resolve the models directory. See the module docs for the order.
pub fn models_dir() -> PathBuf {
    resolve_models_dir().path
}

/// Resolve the models directory and report which rule selected it.
pub fn resolve_models_dir() -> ModelsDirResolution {
    let mut starts = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        starts.push(dir.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    resolve_models_dir_from(std::env::var_os(MODELS_DIR_ENV), &starts, data_local_dir())
}

/// Pure resolution logic behind [`resolve_models_dir`].
///
/// - `env_override`: value of `VOX_MODELS_DIR` (empty counts as unset).
/// - `repo_search_starts`: directories to walk up from when looking for a
///   source checkout.
/// - `data_local`: the OS per-user data directory (without the `vox` suffix).
pub fn resolve_models_dir_from(
    env_override: Option<OsString>,
    repo_search_starts: &[PathBuf],
    data_local: Option<PathBuf>,
) -> ModelsDirResolution {
    if let Some(explicit) = env_override.filter(|v| !v.is_empty()) {
        return ModelsDirResolution {
            path: PathBuf::from(explicit),
            source: ModelsDirSource::Env,
        };
    }

    for start in repo_search_starts {
        if let Some(root) = find_repo_root(start) {
            let candidate = root.join("models");
            if candidate.is_dir() {
                return ModelsDirResolution {
                    path: candidate,
                    source: ModelsDirSource::Repo,
                };
            }
        }
    }

    match data_local {
        Some(dir) => ModelsDirResolution {
            path: dir.join("vox").join("models"),
            source: ModelsDirSource::DataDir,
        },
        None => ModelsDirResolution {
            path: PathBuf::from("models"),
            source: ModelsDirSource::Fallback,
        },
    }
}

/// Root of vox's per-user data (`<data_local_dir>/vox`).
///
/// Holds `models/` (unless overridden) and the speaker database. Returns
/// `None` when no home directory can be determined.
pub fn data_dir() -> Option<PathBuf> {
    data_local_dir().map(|d| d.join("vox"))
}

/// The OS per-user *local* data directory.
///
/// Equivalent to `dirs::data_local_dir()`, implemented here so that path
/// resolution does not depend on an optional crate.
pub fn data_local_dir() -> Option<PathBuf> {
    fn env_path(key: &str) -> Option<PathBuf> {
        std::env::var_os(key)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    }

    if cfg!(target_os = "windows") {
        env_path("LOCALAPPDATA")
            .or_else(|| env_path("USERPROFILE").map(|h| h.join("AppData").join("Local")))
    } else if cfg!(target_os = "macos") {
        env_path("HOME").map(|h| h.join("Library").join("Application Support"))
    } else {
        env_path("XDG_DATA_HOME")
            .filter(|p| p.is_absolute())
            .or_else(|| env_path("HOME").map(|h| h.join(".local").join("share")))
    }
}

/// Walk up from `start` looking for the directory that holds vox's own
/// `Cargo.toml`. Manifests of other packages (including the `python/`
/// bindings crate) are skipped.
fn find_repo_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| is_vox_manifest(&dir.join("Cargo.toml")))
        .map(Path::to_path_buf)
}

/// Does `manifest` declare a `[package]` named like vox?
fn is_vox_manifest(manifest: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(manifest) else {
        return false;
    };
    let mut in_package = false;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some((key, value)) = line.split_once('=')
            && key.trim() == "name"
        {
            let name = value.trim().trim_matches('"');
            return VOX_PACKAGE_NAMES.contains(&name);
        }
    }
    false
}

/// Map a user-facing Whisper model name (`tiny.en`, `base`, `turbo`, ...) to
/// the GGML file name stored in the models directory.
///
/// Names without `.en` are multilingual models. A value that already ends in
/// `.bin` is treated as a file name and returned unchanged.
pub fn whisper_model_filename(model: &str) -> String {
    match model {
        // Quantized variants whose on-disk name does not follow `ggml-<name>.bin`.
        "small-q5" => "ggml-small-q5_1.bin".into(),
        "medium-q5" => "ggml-medium-q5_0.bin".into(),
        // Whisper large-v3-turbo aliases.
        "turbo" | "large-v3-turbo" => "ggml-large-v3-turbo.bin".into(),
        "turbo-q5" | "large-v3-turbo-q5" => "ggml-large-v3-turbo-q5_0.bin".into(),
        "turbo-q8" | "large-v3-turbo-q8" => "ggml-large-v3-turbo-q8_0.bin".into(),
        other if other.ends_with(".bin") => other.to_string(),
        other => format!("ggml-{other}.bin"),
    }
}

/// Is this Whisper model (name or file name) English-only?
pub fn whisper_model_is_english_only(model: &str) -> bool {
    let stem = model.strip_suffix(".bin").unwrap_or(model);
    stem.ends_with(".en") || stem.contains(".en-")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "vox-paths-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Build a fake checkout: `<root>/Cargo.toml` (+ optional `models/`) and
    /// return a nested directory that mimics `target/release`.
    fn fake_repo(root: &Path, package: &str, with_models: bool) -> PathBuf {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            format!(
                "[workspace]\nmembers = [\"python\"]\n\n[package]\nname = \"{package}\"\nversion = \"0.0.0\"\n\n[lib]\nname = \"vox\"\n"
            ),
        )
        .unwrap();
        if with_models {
            std::fs::create_dir_all(root.join("models")).unwrap();
        }
        let bin_dir = root.join("target").join("release");
        std::fs::create_dir_all(&bin_dir).unwrap();
        bin_dir
    }

    #[test]
    fn env_override_wins_over_repo_and_data_dir() {
        let tmp = Scratch::new("env");
        let bin_dir = fake_repo(&tmp.0.join("repo"), "vox", true);
        let got = resolve_models_dir_from(
            Some(OsString::from("/custom/models")),
            &[bin_dir],
            Some(tmp.0.join("data")),
        );
        assert_eq!(got.path, PathBuf::from("/custom/models"));
        assert_eq!(got.source, ModelsDirSource::Env);
    }

    #[test]
    fn empty_env_override_is_ignored() {
        let tmp = Scratch::new("emptyenv");
        let got = resolve_models_dir_from(Some(OsString::new()), &[], Some(tmp.0.join("data")));
        assert_eq!(got.source, ModelsDirSource::DataDir);
    }

    #[test]
    fn repo_models_wins_over_data_dir() {
        let tmp = Scratch::new("repo");
        let root = tmp.0.join("repo");
        let bin_dir = fake_repo(&root, "vox", true);
        let got = resolve_models_dir_from(None, &[bin_dir], Some(tmp.0.join("data")));
        assert_eq!(got.path, root.join("models"));
        assert_eq!(got.source, ModelsDirSource::Repo);
    }

    #[test]
    fn renamed_package_is_still_recognised_as_repo() {
        let tmp = Scratch::new("renamed");
        let root = tmp.0.join("repo");
        let bin_dir = fake_repo(&root, "vox-voice", true);
        let got = resolve_models_dir_from(None, &[bin_dir], None);
        assert_eq!(got.path, root.join("models"));
        assert_eq!(got.source, ModelsDirSource::Repo);
    }

    #[test]
    fn falls_back_to_data_dir_outside_a_checkout() {
        let tmp = Scratch::new("data");
        let installed = tmp.0.join("usr").join("local").join("bin");
        std::fs::create_dir_all(&installed).unwrap();
        let data = tmp.0.join("data");
        let got = resolve_models_dir_from(None, &[installed], Some(data.clone()));
        assert_eq!(got.path, data.join("vox").join("models"));
        assert_eq!(got.source, ModelsDirSource::DataDir);
    }

    #[test]
    fn checkout_without_models_dir_uses_data_dir() {
        let tmp = Scratch::new("nomodels");
        let bin_dir = fake_repo(&tmp.0.join("repo"), "vox", false);
        let data = tmp.0.join("data");
        let got = resolve_models_dir_from(None, &[bin_dir], Some(data.clone()));
        assert_eq!(got.path, data.join("vox").join("models"));
        assert_eq!(got.source, ModelsDirSource::DataDir);
    }

    #[test]
    fn foreign_cargo_project_is_not_mistaken_for_vox() {
        let tmp = Scratch::new("foreign");
        let bin_dir = fake_repo(&tmp.0.join("other"), "some-other-crate", true);
        let data = tmp.0.join("data");
        let got = resolve_models_dir_from(None, &[bin_dir], Some(data.clone()));
        assert_eq!(got.source, ModelsDirSource::DataDir);
        assert_eq!(got.path, data.join("vox").join("models"));
    }

    #[test]
    fn nested_bindings_crate_resolves_to_the_outer_checkout() {
        let tmp = Scratch::new("nested");
        let root = tmp.0.join("repo");
        fake_repo(&root, "vox", true);
        let python = root.join("python");
        std::fs::create_dir_all(&python).unwrap();
        std::fs::write(
            python.join("Cargo.toml"),
            "[package]\nname = \"vox-python\"\n",
        )
        .unwrap();
        let got = resolve_models_dir_from(None, &[python], None);
        assert_eq!(got.path, root.join("models"));
    }

    #[test]
    fn no_data_dir_falls_back_to_relative_models() {
        let got = resolve_models_dir_from(None, &[], None);
        assert_eq!(got.path, PathBuf::from("models"));
        assert_eq!(got.source, ModelsDirSource::Fallback);
    }

    #[test]
    fn data_dir_is_the_vox_folder_of_data_local_dir() {
        if let (Some(root), Some(local)) = (data_dir(), data_local_dir()) {
            assert_eq!(root, local.join("vox"));
        }
    }

    #[test]
    fn whisper_filenames_cover_english_multilingual_and_turbo() {
        assert_eq!(whisper_model_filename("tiny.en"), "ggml-tiny.en.bin");
        assert_eq!(whisper_model_filename("small.en"), "ggml-small.en.bin");
        assert_eq!(whisper_model_filename("base"), "ggml-base.bin");
        assert_eq!(whisper_model_filename("small"), "ggml-small.bin");
        assert_eq!(whisper_model_filename("small-q5"), "ggml-small-q5_1.bin");
        assert_eq!(whisper_model_filename("turbo"), "ggml-large-v3-turbo.bin");
        assert_eq!(
            whisper_model_filename("turbo-q5"),
            "ggml-large-v3-turbo-q5_0.bin"
        );
        assert_eq!(whisper_model_filename("ggml-custom.bin"), "ggml-custom.bin");
    }

    #[test]
    fn english_only_detection() {
        assert!(whisper_model_is_english_only("tiny.en"));
        assert!(whisper_model_is_english_only("ggml-base.en.bin"));
        assert!(whisper_model_is_english_only("ggml-tiny.en-int8.bin"));
        assert!(!whisper_model_is_english_only("base"));
        assert!(!whisper_model_is_english_only("ggml-large-v3-turbo.bin"));
    }
}
