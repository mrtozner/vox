//! Model directory resolution and STT/TTS backend loading for the server.

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[allow(unused_imports)]
use tracing::info;

/// Startup choices that influence which models the server loads.
///
/// Filled from `vox serve` flags, with environment variables as fallback
/// (see [`LoadOptions::from_env`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// Spoken language as a Whisper code (`en`, `tr`, ...) or `auto`.
    /// `None` keeps the historical behaviour: prefer English-only models,
    /// auto-detect when only a multilingual model is installed.
    pub language: Option<String>,
    /// Whisper model to load (`base`, `small.en`, `turbo`, or a `.bin` file
    /// name). `None` picks the best installed model for this hardware.
    pub stt_model: Option<String>,
    /// Piper voice to use for conversation TTS (`tr`, `en-us`, `piper-de`,
    /// or a voice file stem such as `tr_TR-dfki-medium`).
    pub piper_voice: Option<String>,
}

impl LoadOptions {
    /// Read `VOX_LANGUAGE`, `VOX_STT_MODEL` and `VOX_PIPER_VOICE`.
    pub fn from_env() -> Self {
        fn var(key: &str) -> Option<String> {
            std::env::var(key)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        }
        Self {
            language: var("VOX_LANGUAGE"),
            stt_model: var("VOX_STT_MODEL"),
            piper_voice: var("VOX_PIPER_VOICE"),
        }
    }

    /// Fill unset fields from `fallback` (explicit flags win over env).
    pub fn or(self, fallback: Self) -> Self {
        Self {
            language: self.language.or(fallback.language),
            stt_model: self.stt_model.or(fallback.stt_model),
            piper_voice: self.piper_voice.or(fallback.piper_voice),
        }
    }
}

/// Resolve the models directory for the server.
///
/// Delegates to [`vox::paths::models_dir`], the single source of truth
/// shared with the CLI: `VOX_MODELS_DIR` → `models/` next to vox's
/// `Cargo.toml` (source checkout) → per-user OS data directory.
pub(super) fn resolve_models_dir() -> PathBuf {
    vox::paths::models_dir()
}

/// `(backend, model name, model size in MB)` for the loaded STT model.
pub(super) type SttLoadResult = (
    Option<Arc<dyn vox::traits::SttBackend>>,
    Option<String>,
    Option<u64>,
);

/// Whisper GGML file names to try, best first.
///
/// An explicitly requested model comes first; the rest follows the
/// hardware- and language-aware order from
/// [`vox::system_profile::SystemProfile::recommended_whisper_candidates`],
/// which covers English-only, multilingual and turbo models.
pub(super) fn stt_candidates(
    profile: &vox::system_profile::SystemProfile,
    opts: &LoadOptions,
) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(model) = opts.stt_model.as_deref() {
        candidates.push(vox::paths::whisper_model_filename(model));
    }
    for name in profile.recommended_whisper_candidates(opts.language.as_deref()) {
        if !candidates.iter().any(|c| c == name) {
            candidates.push(name.to_string());
        }
    }
    candidates
}

/// Whisper configuration for a model file: language from the startup
/// options, thread count from the hardware profile.
#[cfg(feature = "whisper")]
pub(super) fn whisper_config(
    model_path: &Path,
    profile: &vox::system_profile::SystemProfile,
    opts: &LoadOptions,
) -> vox::WhisperConfig {
    vox::WhisperConfig::new(model_path)
        .with_language(opts.language.clone())
        .with_threads(profile.recommended_inference_threads())
}

/// Load the first available Whisper model from [`stt_candidates`].
#[cfg(feature = "whisper")]
pub(super) fn load_stt(
    models_dir: &Path,
    profile: &vox::system_profile::SystemProfile,
    opts: &LoadOptions,
) -> SttLoadResult {
    let candidates = stt_candidates(profile, opts);
    if let Some(requested) = opts.stt_model.as_deref()
        && !models_dir.join(&candidates[0]).exists()
    {
        tracing::warn!(
            model = requested,
            file = %candidates[0],
            "requested whisper model is not installed, falling back to the best available one"
        );
    }

    for name in &candidates {
        let path = models_dir.join(name);
        if !path.exists() {
            continue;
        }
        let config = whisper_config(&path, profile, opts);
        let threads = config.n_threads;
        match vox::WhisperBackend::with_config(config) {
            Ok(backend) => {
                info!(
                    model = %path.display(),
                    language = opts.language.as_deref().unwrap_or("auto"),
                    multilingual = backend.is_multilingual(),
                    threads,
                    "loaded whisper STT backend"
                );
                let variant = name
                    .strip_prefix("ggml-")
                    .unwrap_or(name)
                    .strip_suffix(".bin")
                    .unwrap_or(name)
                    .to_string();
                let size = std::fs::metadata(&path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                return (
                    Some(Arc::new(backend) as Arc<dyn vox::traits::SttBackend>),
                    Some(variant),
                    size,
                );
            }
            Err(e) => {
                tracing::warn!(model = %path.display(), error = %e, "failed to load whisper model");
            }
        }
    }

    match opts.language.as_deref() {
        Some(lang) if !lang.eq_ignore_ascii_case("en") => tracing::warn!(
            "no multilingual whisper model found in {} for language '{lang}', STT disabled \
             (run `vox serve --download --language {lang}` or `vox models download whisper-base`)",
            models_dir.display()
        ),
        _ => tracing::warn!(
            "no whisper model found in {}, STT disabled (run `vox serve --download` or `vox doctor`)",
            models_dir.display()
        ),
    }
    (None, None, None)
}

#[cfg(not(feature = "whisper"))]
pub(super) fn load_stt(
    _models_dir: &Path,
    _profile: &vox::system_profile::SystemProfile,
    _opts: &LoadOptions,
) -> SttLoadResult {
    tracing::warn!("STT disabled (compiled without 'whisper' feature)");
    (None, None, None)
}

#[cfg(feature = "qwen3")]
#[allow(unused_variables)]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
    opts: &LoadOptions,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let _ = models_dir;

    // Try Qwen3 first (if enabled)
    match vox::tts::Qwen3Backend::new().await {
        Ok(backend) => {
            info!("loaded qwen3 TTS backend");
            return (
                Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                Some("qwen3".to_string()),
                None, // Model size not applicable (auto-downloaded to HF cache)
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load qwen3 TTS backend, trying fallback");
        }
    }

    // Fallback: try Kokoro if Qwen3 is unavailable
    #[cfg(feature = "kokoro")]
    {
        let model_path = models_dir.join("kokoro-v1.0.onnx");
        let voices_path = models_dir.join("voices.bin");
        if model_path.exists() && voices_path.exists() {
            match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
                Ok(backend) => {
                    info!("loaded kokoro TTS backend");
                    let model_name = model_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.replace('-', " "))
                        .unwrap_or_else(|| "kokoro".to_string());
                    let model_size = std::fs::metadata(&model_path)
                        .ok()
                        .map(|m| m.len() / (1024 * 1024));
                    return (
                        Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                        Some(model_name),
                        model_size,
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load kokoro TTS backend");
                }
            }
        }
    }

    // Fallback: try Piper if Kokoro is unavailable
    #[cfg(feature = "piper")]
    {
        if let Some(result) = try_load_piper(models_dir, opts) {
            return result;
        }
    }

    (None, None, None)
}

#[cfg(all(feature = "kokoro", not(feature = "qwen3")))]
#[allow(unused_variables)]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
    opts: &LoadOptions,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    let model_path = models_dir.join("kokoro-v1.0.onnx");
    let voices_path = models_dir.join("voices.bin");
    if model_path.exists() && voices_path.exists() {
        match vox::tts::KokoroBackend::new(&model_path, &voices_path).await {
            Ok(backend) => {
                info!("loaded kokoro TTS backend");
                let model_name = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.replace('-', " "))
                    .unwrap_or_else(|| "kokoro".to_string());
                let model_size = std::fs::metadata(&model_path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                return (
                    Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                    Some(model_name),
                    model_size,
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load kokoro TTS backend");
            }
        }
    } else {
        tracing::warn!(
            "kokoro model files not found in {}, trying piper fallback",
            models_dir.display()
        );
    }

    // Fallback: try Piper if Kokoro is unavailable
    #[cfg(feature = "piper")]
    {
        if let Some(result) = try_load_piper(models_dir, opts) {
            return result;
        }
    }

    (None, None, None)
}

#[cfg(all(feature = "piper", not(any(feature = "kokoro", feature = "qwen3"))))]
#[allow(unused_variables)]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
    opts: &LoadOptions,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    if let Some(result) = try_load_piper(models_dir, opts) {
        return result;
    }
    tracing::warn!(
        "no piper model found in {}, TTS disabled",
        models_dir.display()
    );
    (None, None, None)
}

#[cfg(not(any(feature = "kokoro", feature = "piper", feature = "qwen3")))]
#[allow(unused_variables)]
pub(super) async fn load_tts(
    models_dir: &std::path::Path,
    opts: &LoadOptions,
) -> (
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
) {
    tracing::warn!(
        "TTS disabled (compiled without 'kokoro', 'piper', or 'qwen3' feature); models dir: {}",
        models_dir.display()
    );
    (None, None, None)
}

#[cfg(feature = "piper")]
pub(super) type PiperLoadResult = Option<(
    Option<Arc<dyn vox::traits::TtsBackend>>,
    Option<String>,
    Option<u64>,
)>;

/// Normalise a voice name for comparison: lower-case, `-` → `_`, without a
/// `piper-` registry prefix or `.onnx(.json)` suffix.
#[cfg(any(feature = "piper", test))]
fn normalize_voice(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    let lower = lower.strip_prefix("piper-").unwrap_or(&lower);
    let lower = lower.strip_suffix(".json").unwrap_or(lower);
    let lower = lower.strip_suffix(".onnx").unwrap_or(lower);
    lower.replace('-', "_")
}

/// Does the voice file stem (already normalised) match `wanted`
/// (already normalised)? `tr` matches `tr_tr_dfki_medium`, `en_us` matches
/// `en_us_lessac_medium`, but `e` does not match `en_us_...`.
#[cfg(any(feature = "piper", test))]
fn voice_matches(stem: &str, wanted: &str) -> bool {
    !wanted.is_empty()
        && stem
            .strip_prefix(wanted)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
}

/// Piper voice configs (`*.onnx.json`) in `piper_dir`, in the order they
/// should be tried. The order is deterministic:
///
/// 1. the voice requested with `--piper-voice` / `VOX_PIPER_VOICE`,
/// 2. voices for the configured language (`--language tr` → `tr_*`),
/// 3. `en_US`, then other English voices,
/// 4. everything else,
///
/// with ties broken by file name. (Directory iteration order is
/// OS-dependent, so it is never relied upon.)
#[cfg(any(feature = "piper", test))]
pub(super) fn piper_candidates(piper_dir: &Path, opts: &LoadOptions) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(piper_dir) else {
        return Vec::new();
    };
    let wanted_voice = opts.piper_voice.as_deref().map(normalize_voice);
    let wanted_lang = opts
        .language
        .as_deref()
        .map(|l| {
            normalize_voice(l)
                .chars()
                .take_while(|c| *c != '_')
                .collect::<String>()
        })
        // English needs no special casing: the default order is already
        // en_US first, then the other English voices.
        .filter(|l| !l.is_empty() && l != "auto" && l != "en");

    let mut ranked: Vec<(u8, String, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?.to_string();
            if !file_name.ends_with(".onnx.json") {
                return None;
            }
            let stem = normalize_voice(&file_name);
            let rank = if wanted_voice
                .as_deref()
                .is_some_and(|w| voice_matches(&stem, w))
            {
                0
            } else if wanted_lang
                .as_deref()
                .is_some_and(|l| voice_matches(&stem, l))
            {
                1
            } else if voice_matches(&stem, "en_us") {
                2
            } else if voice_matches(&stem, "en") {
                3
            } else {
                4
            };
            Some((rank, file_name, path))
        })
        .collect();
    ranked.sort();
    ranked.into_iter().map(|(_, _, path)| path).collect()
}

/// Load a Piper voice from `models_dir/piper/`, trying the configs in the
/// deterministic order given by [`piper_candidates`].
#[cfg(feature = "piper")]
pub(super) fn try_load_piper(models_dir: &Path, opts: &LoadOptions) -> PiperLoadResult {
    let piper_dir = models_dir.join("piper");
    if !piper_dir.exists() {
        return None;
    }

    let candidates = piper_candidates(&piper_dir, opts);
    if let (Some(voice), Some(first)) = (opts.piper_voice.as_deref(), candidates.first()) {
        let first_stem = first
            .file_name()
            .and_then(|n| n.to_str())
            .map(normalize_voice)
            .unwrap_or_default();
        if !voice_matches(&first_stem, &normalize_voice(voice)) {
            tracing::warn!(
                requested = voice,
                using = %first.display(),
                "requested piper voice is not installed, falling back \
                 (install it with `vox models download piper-<lang>` and its `-config`)"
            );
        }
    }

    for path in candidates {
        match vox::tts::PiperBackend::new(&path) {
            Ok(backend) => {
                info!(model = %path.display(), "loaded piper TTS backend");
                let model_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| {
                        // "en_US-lessac-medium.onnx" -> "piper en_US-lessac-medium"
                        format!("piper {}", s.strip_suffix(".onnx").unwrap_or(s))
                    })
                    .unwrap_or_else(|| "piper".to_string());
                // Size of the .onnx model file (not the json config)
                let onnx_path = path.with_extension(""); // strips .json -> .onnx
                let model_size = std::fs::metadata(&onnx_path)
                    .ok()
                    .map(|m| m.len() / (1024 * 1024));
                return Some((
                    Some(Arc::new(backend) as Arc<dyn vox::traits::TtsBackend>),
                    Some(model_name),
                    model_size,
                ));
            }
            Err(e) => {
                tracing::warn!(model = %path.display(), error = %e, "failed to load piper model");
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox::system_profile::{Arch, Device, HardwareClass, Os, SystemProfile};

    fn profile(class: HardwareClass, num_cpus: usize) -> SystemProfile {
        SystemProfile {
            os: Os::Linux,
            arch: Arch::X86_64,
            device: Device::LinuxDesktop,
            total_ram_mb: 16384,
            available_ram_mb: 8000,
            num_cpus,
            class,
        }
    }

    fn opts(language: Option<&str>, stt_model: Option<&str>, piper: Option<&str>) -> LoadOptions {
        LoadOptions {
            language: language.map(str::to_string),
            stt_model: stt_model.map(str::to_string),
            piper_voice: piper.map(str::to_string),
        }
    }

    /// Scratch directory holding empty voice config files.
    struct VoiceDir(PathBuf);

    impl VoiceDir {
        fn new(tag: &str, files: &[&str]) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "vox-loaders-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for f in files {
                std::fs::write(dir.join(f), b"{}").unwrap();
            }
            Self(dir)
        }

        fn order(&self, opts: &LoadOptions) -> Vec<String> {
            piper_candidates(&self.0, opts)
                .iter()
                .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
                .collect()
        }
    }

    impl Drop for VoiceDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const VOICES: &[&str] = &[
        "tr_TR-dfki-medium.onnx.json",
        "tr_TR-dfki-medium.onnx",
        "de_DE-thorsten-medium.onnx.json",
        "en_US-lessac-medium.onnx.json",
        "en_GB-alan-medium.onnx.json",
        "README.txt",
    ];

    #[test]
    fn stt_candidates_cover_multilingual_and_turbo_models() {
        let c = stt_candidates(&profile(HardwareClass::Medium, 8), &LoadOptions::default());
        assert_eq!(c[0], "ggml-base.en.bin");
        for expected in ["ggml-base.bin", "ggml-small.bin", "ggml-large-v3-turbo.bin"] {
            assert!(c.iter().any(|n| n == expected), "missing {expected}: {c:?}");
        }
    }

    #[test]
    fn stt_candidates_for_turkish_exclude_english_only_models() {
        let c = stt_candidates(
            &profile(HardwareClass::Large, 8),
            &opts(Some("tr"), None, None),
        );
        assert_eq!(c[0], "ggml-small.bin");
        assert!(c.iter().all(|n| !n.contains(".en")), "{c:?}");
    }

    #[test]
    fn explicit_stt_model_is_tried_first_without_duplicates() {
        let c = stt_candidates(
            &profile(HardwareClass::Large, 8),
            &opts(None, Some("turbo-q5"), None),
        );
        assert_eq!(c[0], "ggml-large-v3-turbo-q5_0.bin");
        assert_eq!(
            c.iter()
                .filter(|n| *n == "ggml-large-v3-turbo-q5_0.bin")
                .count(),
            1
        );
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn language_and_threads_reach_whisper_config() {
        let config = whisper_config(
            Path::new("ggml-base.bin"),
            &profile(HardwareClass::Large, 6),
            &opts(Some("tr"), None, None),
        );
        assert_eq!(config.language.as_deref(), Some("tr"));
        assert_eq!(config.n_threads, 5);

        let auto = whisper_config(
            Path::new("ggml-base.bin"),
            &profile(HardwareClass::Large, 32),
            &LoadOptions::default(),
        );
        assert_eq!(auto.language, None);
        assert_eq!(auto.n_threads, 8);
    }

    #[test]
    fn piper_default_order_is_deterministic_and_english_first() {
        let dir = VoiceDir::new("default", VOICES);
        assert_eq!(
            dir.order(&LoadOptions::default()),
            [
                "en_US-lessac-medium.onnx.json",
                "en_GB-alan-medium.onnx.json",
                "de_DE-thorsten-medium.onnx.json",
                "tr_TR-dfki-medium.onnx.json",
            ]
        );
    }

    #[test]
    fn piper_language_selects_matching_voice() {
        let dir = VoiceDir::new("lang", VOICES);
        let order = dir.order(&opts(Some("tr"), None, None));
        assert_eq!(order[0], "tr_TR-dfki-medium.onnx.json");
        assert_eq!(order[1], "en_US-lessac-medium.onnx.json");
    }

    #[test]
    fn piper_voice_flag_beats_language_and_accepts_aliases() {
        let dir = VoiceDir::new("voice", VOICES);
        for alias in ["de", "piper-de", "de_DE-thorsten-medium", "DE_de"] {
            let order = dir.order(&opts(Some("tr"), None, Some(alias)));
            assert_eq!(order[0], "de_DE-thorsten-medium.onnx.json", "alias {alias}");
            assert_eq!(order[1], "tr_TR-dfki-medium.onnx.json", "alias {alias}");
        }
        let order = dir.order(&opts(None, None, Some("en-gb")));
        assert_eq!(order[0], "en_GB-alan-medium.onnx.json");
    }

    #[test]
    fn piper_unknown_voice_falls_back_to_default_order() {
        let dir = VoiceDir::new("unknown", VOICES);
        assert_eq!(
            dir.order(&opts(None, None, Some("xx"))),
            dir.order(&LoadOptions::default())
        );
    }

    #[test]
    fn piper_missing_dir_yields_no_candidates() {
        let missing = std::env::temp_dir().join("vox-loaders-definitely-missing-dir");
        assert!(piper_candidates(&missing, &LoadOptions::default()).is_empty());
    }

    #[test]
    fn voice_matching_requires_a_whole_segment() {
        assert!(voice_matches("tr_tr_dfki_medium", "tr"));
        assert!(voice_matches("en_us_lessac_medium", "en_us"));
        assert!(!voice_matches("en_us_lessac_medium", "e"));
        assert!(!voice_matches("es_es_davefx_medium", "en"));
        assert!(!voice_matches("tr_tr_dfki_medium", ""));
    }

    #[test]
    fn explicit_options_win_over_fallback() {
        let merged = opts(Some("tr"), None, None).or(opts(Some("en"), Some("base"), Some("de")));
        assert_eq!(merged, opts(Some("tr"), Some("base"), Some("de")));
    }
}
