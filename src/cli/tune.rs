//! Handler for `vox tune` — measure this machine and pick models for it.
//!
//! `vox benchmark` reports numbers for one fixed setup. `vox tune` answers
//! the question behind it: which of the models installed here should be
//! used. It
//!
//! 1. builds a speech sample (a sentence spoken by the installed TTS, or the
//!    WAV file given with `--sample`), because Whisper returns early on
//!    silence and silence says nothing about accuracy,
//! 2. transcribes it with every installed Whisper model that suits the
//!    language, recording speed and how many words came out right,
//! 3. times every installed TTS backend on the same sentence,
//! 4. streams a short reply from the installed Ollama models, recording time
//!    to first token and tokens per second,
//! 5. picks one of each and writes the result to the `[tune]` table of
//!    `config.toml`, which `vox chat` reads for its defaults.
//!
//! `--dry-run` skips steps 1 to 4 and the write: it prints what the hardware
//! profile and the installed models suggest, without loading a model or
//! opening a connection.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::{self, TuneConfig};
use super::models::{self, models_dir};
use vox::system_profile::SystemProfile;
use vox::traits::{SttBackend, TtsBackend};
use vox::types::{AudioChunk, TtsRequest, Utterance};

/// STT slower than this share of real time makes a conversation feel laggy.
pub const STT_RTF_BUDGET: f64 = 0.30;
/// Accuracy differences below this are treated as a tie (one word in a
/// fourteen-word sentence is 0.07).
pub const STT_ACCURACY_TIE: f64 = 0.02;
/// TTS must be faster than real time to stream without gaps.
pub const TTS_RTF_BUDGET: f64 = 1.0;
/// Slower than this and the reply is spoken faster than it is generated.
pub const LLM_MIN_TOKENS_PER_SEC: f64 = 12.0;
/// Longest acceptable wait for the first token from a loaded model.
pub const LLM_MAX_FIRST_TOKEN_MS: u64 = 1500;
/// How many Whisper models are measured at most (in preference order).
pub const MAX_STT_CANDIDATES: usize = 4;
/// How many Ollama models are measured at most when `--llm` is not given.
pub const MAX_LLM_CANDIDATES: usize = 4;

const LLM_PROMPT: &str = "In two short sentences, explain why the sky is blue.";

/// Arguments of `vox tune`.
#[derive(Debug, Clone, Default)]
pub struct TuneArgs {
    /// Recommend from the hardware profile and installed models only.
    pub dry_run: bool,
    /// Download the recommended models that are missing.
    pub yes: bool,
    /// WAV file with speech to transcribe instead of a synthesized sentence.
    pub sample: Option<PathBuf>,
    /// What is said in `sample`, to score accuracy.
    pub reference: Option<String>,
    /// Spoken language (`en`, `tr`, ..., `auto`).
    pub language: Option<String>,
    /// Ollama models to measure; all installed ones (up to a cap) if empty.
    pub llm: Vec<String>,
    /// `--ollama-host`.
    pub ollama_host: Option<String>,
    /// Timed repetitions per model.
    pub runs: usize,
    /// Skip the LLM measurement.
    pub no_llm: bool,
    /// Measure but do not write `config.toml`.
    pub no_save: bool,
}

// ---------------------------------------------------------------------------
// Language and reference text
// ---------------------------------------------------------------------------

/// Primary language subtag in lower case; English for `None`, empty or `auto`.
pub fn sample_language(language: Option<&str>) -> String {
    let lang = language.unwrap_or("").trim().to_ascii_lowercase();
    let primary = lang.split(['-', '_']).next().unwrap_or("");
    match primary {
        "" | "auto" => "en".to_string(),
        other => other.to_string(),
    }
}

/// A sentence to synthesize and transcribe, for the languages vox ships a
/// Piper voice for.
pub fn reference_sentence(lang: &str) -> Option<&'static str> {
    Some(match lang {
        "en" => {
            "The weather is nice today, so we will walk to the old bridge and have lunch by the river."
        }
        "tr" => {
            "Bugün hava çok güzel, bu yüzden eski köprüye yürüyüp nehir kenarında yemek yiyeceğiz."
        }
        "de" => {
            "Das Wetter ist heute schön, deshalb gehen wir zur alten Brücke und essen am Fluss zu Mittag."
        }
        "es" => {
            "Hoy hace buen tiempo, así que caminaremos hasta el puente viejo y comeremos junto al río."
        }
        "fr" => {
            "Il fait beau aujourd'hui, alors nous marcherons jusqu'au vieux pont et déjeunerons au bord de la rivière."
        }
        _ => return None,
    })
}

/// User-facing Whisper model name for a GGML file name; the inverse of
/// `vox::paths::whisper_model_filename`.
pub fn whisper_name_from_file(file: &str) -> String {
    match file {
        "ggml-small-q5_1.bin" => "small-q5".into(),
        "ggml-medium-q5_0.bin" => "medium-q5".into(),
        "ggml-large-v3-turbo.bin" => "turbo".into(),
        "ggml-large-v3-turbo-q5_0.bin" => "turbo-q5".into(),
        "ggml-large-v3-turbo-q8_0.bin" => "turbo-q8".into(),
        other => other
            .strip_prefix("ggml-")
            .and_then(|s| s.strip_suffix(".bin"))
            .unwrap_or(other)
            .to_string(),
    }
}

// ---------------------------------------------------------------------------
// Word accuracy
// ---------------------------------------------------------------------------

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Share of the reference words the transcript got right: `1 - WER`, floored
/// at 0. Case and punctuation are ignored. An empty reference scores 0.
pub fn word_accuracy(reference: &str, transcript: &str) -> f64 {
    let r = words(reference);
    let h = words(transcript);
    if r.is_empty() {
        return 0.0;
    }
    // Word-level edit distance, one row at a time.
    let mut row: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let substitution = diagonal + usize::from(rw != hw);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(diagonal + 1);
        }
    }
    (1.0 - row[h.len()] as f64 / r.len() as f64).max(0.0)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Inventory and the static (dry-run) recommendation
// ---------------------------------------------------------------------------

/// What is installed and compiled in, as far as tuning is concerned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inventory {
    /// Whisper GGML files suited to the language, best first for this hardware.
    pub stt_candidates: Vec<String>,
    /// The subset of `stt_candidates` present in the models directory.
    pub stt_installed: Vec<String>,
    /// TTS backends compiled into this binary that can speak the language.
    pub tts_compiled: Vec<&'static str>,
    /// The subset of `tts_compiled` whose model files are present.
    pub tts_installed: Vec<&'static str>,
}

fn kokoro_speaks(lang: &str) -> bool {
    lang == "en"
}

/// Registry name of the Piper voice for `lang`, if vox ships one.
fn piper_voice_for(lang: &str) -> Option<String> {
    let voice = models::piper_voice_alias(lang);
    // The alias falls back to English for languages it does not know.
    (lang == "en" || voice != "piper-en-us").then_some(voice)
}

fn piper_installed(dir: &Path, lang: &str) -> bool {
    let Some(voice) = piper_voice_for(lang) else {
        return false;
    };
    let config = format!("{voice}-config");
    [voice.as_str(), config.as_str()]
        .iter()
        .all(|name| models::find_model(name).is_some_and(|m| dir.join(m.filename).exists()))
}

impl Inventory {
    /// Look at `dir` for the models that matter for `language`.
    pub fn scan(dir: &Path, profile: &SystemProfile, language: Option<&str>) -> Self {
        let lang = sample_language(language);
        let stt_candidates: Vec<String> = profile
            .recommended_whisper_candidates(language)
            .into_iter()
            .map(str::to_string)
            .collect();
        let stt_installed = stt_candidates
            .iter()
            .filter(|f| dir.join(f).exists())
            .cloned()
            .collect();

        let mut tts_compiled = Vec::new();
        let mut tts_installed = Vec::new();
        if cfg!(feature = "piper") && piper_voice_for(&lang).is_some() {
            tts_compiled.push("piper");
            if piper_installed(dir, &lang) {
                tts_installed.push("piper");
            }
        }
        if cfg!(feature = "kokoro") && kokoro_speaks(&lang) {
            tts_compiled.push("kokoro");
            if dir.join("kokoro-v1.0.onnx").exists() && dir.join("voices.bin").exists() {
                tts_installed.push("kokoro");
            }
        }
        Self {
            stt_candidates,
            stt_installed,
            tts_compiled,
            tts_installed,
        }
    }
}

/// One choice and why it was made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pick {
    pub choice: Option<String>,
    pub reason: String,
}

impl Pick {
    fn new(choice: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            choice: Some(choice.into()),
            reason: reason.into(),
        }
    }
    fn none(reason: impl Into<String>) -> Self {
        Self {
            choice: None,
            reason: reason.into(),
        }
    }
}

/// The models to use: a Whisper model name, a TTS backend, an Ollama tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recommendation {
    pub stt: Pick,
    pub tts: Pick,
    pub llm: Pick,
}

/// Recommend without measuring: the hardware class decides the order, the
/// models directory decides what is available.
pub fn recommend_static(profile: &SystemProfile, inv: &Inventory) -> Recommendation {
    let class = profile.class;
    let stt = match (inv.stt_installed.first(), inv.stt_candidates.first()) {
        (Some(file), _) => Pick::new(
            whisper_name_from_file(file),
            format!("best installed Whisper model for a {class} machine"),
        ),
        (None, Some(file)) => {
            let name = whisper_name_from_file(file);
            let download = models::whisper_download_name(&name);
            Pick::new(
                name,
                format!("not installed yet; get it with: vox models download {download}"),
            )
        }
        (None, None) => Pick::none("no Whisper model suits this language"),
    };

    let tts = match (inv.tts_installed.first(), inv.tts_compiled.first()) {
        (Some(backend), _) => Pick::new(*backend, "installed and compiled into this build"),
        (None, Some(backend)) => Pick::new(
            *backend,
            "compiled into this build, voice not installed yet; `vox serve --download` fetches it",
        ),
        (None, None) => Pick::none("this build has no TTS backend for the language"),
    };

    let llm = match profile.recommended_ollama_model() {
        Some(tag) => Pick::new(
            tag,
            format!(
                "fits a {class} machine ({} MB RAM); get it with: ollama pull {tag}",
                profile.total_ram_mb
            ),
        ),
        None => Pick::none(
            "too little RAM for a local LLM; point --ollama-host at another machine \
             (not possible in offline mode)",
        ),
    };

    Recommendation { stt, tts, llm }
}

// ---------------------------------------------------------------------------
// Measurements and the measured recommendation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SttMeasurement {
    /// User-facing model name (`base.en`).
    pub model: String,
    pub load_ms: u64,
    /// Median transcription time for the sample.
    pub transcribe_ms: f64,
    /// `transcribe_ms` divided by the sample's duration.
    pub rtf: f64,
    /// `None` when there is no reference text to compare against.
    pub word_accuracy: Option<f64>,
    pub transcript: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsMeasurement {
    pub backend: String,
    pub load_ms: u64,
    /// Median synthesis time for the reference sentence.
    pub synth_ms: f64,
    pub audio_ms: u64,
    pub rtf: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmMeasurement {
    pub model: String,
    pub size_mb: Option<u64>,
    /// Time for the first reply, which includes loading the model.
    pub cold_first_token_ms: u64,
    /// Time to first token once the model is loaded.
    pub first_token_ms: u64,
    pub tokens_per_sec: f64,
}

/// Index of the Whisper model to use. `measurements` is in preference order.
///
/// Among the models within [`STT_RTF_BUDGET`], the most accurate wins; models
/// within [`STT_ACCURACY_TIE`] of it count as equally accurate and the
/// fastest of those is taken. Without accuracy figures the first model within
/// budget wins. If none is within budget, the fastest one is taken.
pub fn pick_stt(measurements: &[SttMeasurement]) -> Option<usize> {
    let fastest = |indices: &mut dyn Iterator<Item = usize>| {
        indices.min_by(|a, b| measurements[*a].rtf.total_cmp(&measurements[*b].rtf))
    };
    let in_budget: Vec<usize> = (0..measurements.len())
        .filter(|i| measurements[*i].rtf <= STT_RTF_BUDGET)
        .collect();
    if in_budget.is_empty() {
        return fastest(&mut (0..measurements.len()));
    }
    let best = in_budget
        .iter()
        .filter_map(|i| measurements[*i].word_accuracy)
        .max_by(f64::total_cmp);
    match best {
        None => in_budget.first().copied(),
        Some(best) => fastest(&mut in_budget.iter().copied().filter(|i| {
            measurements[*i]
                .word_accuracy
                .is_some_and(|a| a >= best - STT_ACCURACY_TIE)
        })),
    }
}

/// Index of the fastest TTS backend.
pub fn pick_tts(measurements: &[TtsMeasurement]) -> Option<usize> {
    (0..measurements.len()).min_by(|a, b| measurements[*a].rtf.total_cmp(&measurements[*b].rtf))
}

/// Whether an LLM is fast enough to hold a spoken conversation.
pub fn llm_is_conversational(m: &LlmMeasurement) -> bool {
    m.tokens_per_sec >= LLM_MIN_TOKENS_PER_SEC && m.first_token_ms <= LLM_MAX_FIRST_TOKEN_MS
}

/// Index of the Ollama model to use: the largest one that is fast enough
/// (size stands in for quality), or the fastest if none is.
pub fn pick_llm(measurements: &[LlmMeasurement]) -> Option<usize> {
    let by_speed = |a: &usize, b: &usize| {
        measurements[*a]
            .tokens_per_sec
            .total_cmp(&measurements[*b].tokens_per_sec)
    };
    let usable: Vec<usize> = (0..measurements.len())
        .filter(|i| llm_is_conversational(&measurements[*i]))
        .collect();
    if usable.is_empty() {
        return (0..measurements.len()).max_by(by_speed);
    }
    usable.into_iter().max_by(|a, b| {
        let size = |i: &usize| measurements[*i].size_mb.unwrap_or(0);
        size(a).cmp(&size(b)).then_with(|| by_speed(a, b))
    })
}

/// Combine measurements into a recommendation. A category that was not
/// measured keeps the static pick.
pub fn recommend_measured(
    fallback: &Recommendation,
    stt: &[SttMeasurement],
    tts: &[TtsMeasurement],
    llm: &[LlmMeasurement],
) -> Recommendation {
    let not_measured = |pick: &Pick| Pick {
        choice: pick.choice.clone(),
        reason: format!("not measured; {}", pick.reason),
    };

    let stt_pick = match pick_stt(stt) {
        Some(i) => {
            let m = &stt[i];
            let accuracy = m
                .word_accuracy
                .map(|a| format!(", {:.0}% of words correct", a * 100.0))
                .unwrap_or_default();
            let caveat = if m.rtf > STT_RTF_BUDGET {
                format!("; slower than the {STT_RTF_BUDGET:.2} target, but the fastest installed")
            } else {
                String::new()
            };
            Pick::new(
                m.model.clone(),
                format!("RTF {:.3}{accuracy}{caveat}", m.rtf),
            )
        }
        None => not_measured(&fallback.stt),
    };

    let tts_pick = match pick_tts(tts) {
        Some(i) => {
            let m = &tts[i];
            let caveat = if m.rtf > TTS_RTF_BUDGET {
                "; slower than real time, speech will have gaps"
            } else {
                ""
            };
            Pick::new(m.backend.clone(), format!("RTF {:.3}{caveat}", m.rtf))
        }
        None => not_measured(&fallback.tts),
    };

    let llm_pick = match pick_llm(llm) {
        Some(i) => {
            let m = &llm[i];
            let caveat = if llm_is_conversational(m) {
                ""
            } else {
                "; too slow for a fluent conversation, but the fastest installed"
            };
            Pick::new(
                m.model.clone(),
                format!(
                    "{:.1} tokens/s, first token after {} ms{caveat}",
                    m.tokens_per_sec, m.first_token_ms
                ),
            )
        }
        None => not_measured(&fallback.llm),
    };

    Recommendation {
        stt: stt_pick,
        tts: tts_pick,
        llm: llm_pick,
    }
}

/// The `[tune]` table for a measured recommendation.
pub fn tune_config(
    profile: &SystemProfile,
    language: Option<&str>,
    rec: &Recommendation,
    stt: &[SttMeasurement],
    tts: &[TtsMeasurement],
    llm: &[LlmMeasurement],
) -> TuneConfig {
    let stt_m = pick_stt(stt).map(|i| &stt[i]);
    let tts_m = pick_tts(tts).map(|i| &tts[i]);
    let llm_m = pick_llm(llm).map(|i| &llm[i]);
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    TuneConfig {
        measured_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        machine: profile.summary(),
        language: language.map(str::to_string),
        // Only measured picks are written: a guess must not look like a result.
        stt_model: stt_m.and(rec.stt.choice.clone()),
        stt_rtf: stt_m.map(|m| round(m.rtf)),
        stt_word_accuracy: stt_m.and_then(|m| m.word_accuracy).map(round),
        tts_backend: tts_m.and(rec.tts.choice.clone()),
        tts_rtf: tts_m.map(|m| round(m.rtf)),
        llm_model: llm_m.and(rec.llm.choice.clone()),
        llm_tokens_per_sec: llm_m.map(|m| round(m.tokens_per_sec)),
        llm_first_token_ms: llm_m.map(|m| m.first_token_ms),
    }
}

// ---------------------------------------------------------------------------
// Speech sample
// ---------------------------------------------------------------------------

/// Speech to transcribe: 16 kHz mono.
#[derive(Debug, Clone)]
pub struct SpeechSample {
    pub samples: Vec<f32>,
    /// What is said, when known.
    pub reference: Option<String>,
    /// Where the audio came from, for the report.
    pub source: String,
}

impl SpeechSample {
    pub fn duration_ms(&self) -> u64 {
        self.samples.len() as u64 * 1000 / 16_000
    }

    fn utterance(&self) -> Utterance {
        Utterance {
            audio: AudioChunk {
                samples: self.samples.clone(),
                sample_rate: 16_000,
                channels: 1,
            },
            duration_ms: self.duration_ms(),
            #[cfg(feature = "diarization")]
            speaker_id: None,
        }
    }
}

/// Read a WAV file (any sample rate, channel count, integer or float) as a
/// 16 kHz mono sample.
pub fn load_wav_sample(path: &Path, reference: Option<&str>) -> anyhow::Result<SpeechSample> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow::anyhow!("cannot read WAV file {}: {e}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()
        }
    }
    .map_err(|e| anyhow::anyhow!("cannot decode WAV file {}: {e}", path.display()))?;
    let samples = vox::audio::to_mono_16k(&samples, spec.sample_rate, spec.channels)?;
    if samples.len() < 8_000 {
        anyhow::bail!(
            "{} holds less than half a second of audio; record a full sentence",
            path.display()
        );
    }
    Ok(SpeechSample {
        samples,
        reference: reference
            .map(str::trim)
            .filter(|r| !r.is_empty())
            .map(str::to_string),
        source: path.display().to_string(),
    })
}

/// A TTS backend ready to be measured.
pub struct LoadedTts {
    pub backend: &'static str,
    pub voice: Option<String>,
    pub tts: Arc<dyn TtsBackend>,
    pub load_ms: u64,
}

/// Find a model file: downloaded if missing when `yes`, otherwise only if
/// it is already installed.
#[cfg(any(feature = "kokoro", feature = "piper"))]
async fn model_file(name: &str, filename: &str, yes: bool) -> Option<PathBuf> {
    if yes {
        match models::ensure_model(name, filename, true).await {
            Ok(path) => Some(path),
            Err(e) => {
                eprintln!("  warning: {name} is unavailable: {e}");
                None
            }
        }
    } else {
        models::resolve_model(filename)
    }
}

/// Load every TTS backend that is compiled in, installed (or downloadable
/// when `yes`) and able to speak `lang`.
#[allow(unused_variables, unused_mut)]
pub async fn load_tts_backends(lang: &str, yes: bool) -> Vec<LoadedTts> {
    let mut out: Vec<LoadedTts> = Vec::new();

    #[cfg(feature = "piper")]
    if let Some(voice) = piper_voice_for(lang) {
        let config_name = format!("{voice}-config");
        let files = match (models::find_model(&voice), models::find_model(&config_name)) {
            (Some(model), Some(cfg)) => {
                let onnx = model_file(&voice, model.filename, yes).await;
                let json = model_file(&config_name, cfg.filename, yes).await;
                onnx.and(json)
            }
            _ => None,
        };
        if let Some(config_path) = files {
            let started = Instant::now();
            match vox::PiperBackend::new(&config_path) {
                Ok(tts) => out.push(LoadedTts {
                    backend: "piper",
                    voice: None,
                    tts: Arc::new(tts),
                    load_ms: started.elapsed().as_millis() as u64,
                }),
                Err(e) => eprintln!("  warning: Piper failed to load: {e}"),
            }
        }
    }

    #[cfg(feature = "kokoro")]
    if kokoro_speaks(lang) {
        let model = model_file("kokoro", "kokoro-v1.0.onnx", yes).await;
        let voices = model_file("kokoro-voices", "voices.bin", yes).await;
        if let (Some(model), Some(voices)) = (model, voices) {
            let started = Instant::now();
            match vox::KokoroBackend::new(&model, &voices).await {
                Ok(tts) => out.push(LoadedTts {
                    backend: "kokoro",
                    voice: Some("af_heart".to_string()),
                    tts: Arc::new(tts),
                    load_ms: started.elapsed().as_millis() as u64,
                }),
                Err(e) => eprintln!("  warning: Kokoro failed to load: {e}"),
            }
        }
    }

    out
}

/// Speak `text` with `tts` and return it as a 16 kHz mono sample.
pub async fn synthesize_sample(tts: &LoadedTts, text: &str) -> anyhow::Result<SpeechSample> {
    let output = tts
        .tts
        .synthesize(&TtsRequest {
            text: text.to_string(),
            voice: tts.voice.clone(),
            seed: None,
        })
        .await?;
    let samples = vox::audio::to_mono_16k(
        &output.audio.samples,
        output.audio.sample_rate,
        output.audio.channels,
    )?;
    if samples.is_empty() {
        anyhow::bail!("{} produced no audio", tts.backend);
    }
    Ok(SpeechSample {
        samples,
        reference: Some(text.to_string()),
        source: format!("synthesized with {}", tts.backend),
    })
}

/// An English speech sample for `vox benchmark`, or `None` when no TTS
/// backend is available to produce one.
pub async fn english_sample(yes: bool) -> Option<SpeechSample> {
    let text = reference_sentence("en")?;
    for tts in load_tts_backends("en", yes).await {
        match synthesize_sample(&tts, text).await {
            Ok(sample) => return Some(sample),
            Err(e) => eprintln!("  warning: could not synthesize a speech sample: {e}"),
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------

async fn measure_stt(
    path: &Path,
    file: &str,
    language: Option<&str>,
    sample: &SpeechSample,
    runs: usize,
) -> anyhow::Result<SttMeasurement> {
    let started = Instant::now();
    let stt = vox::WhisperBackend::with_config(
        vox::WhisperConfig::new(path).with_language(language.map(str::to_string)),
    )?;
    let load_ms = started.elapsed().as_millis() as u64;
    time_stt(&stt, &whisper_name_from_file(file), load_ms, sample, runs).await
}

/// Transcribe `sample` once to warm up and `runs` more times for the clock.
async fn time_stt(
    stt: &dyn SttBackend,
    model: &str,
    load_ms: u64,
    sample: &SpeechSample,
    runs: usize,
) -> anyhow::Result<SttMeasurement> {
    let utterance = sample.utterance();
    // The first run pays one-time costs (graph setup, caches).
    let mut transcript = stt.transcribe(&utterance).await?.text;
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        transcript = stt.transcribe(&utterance).await?.text;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let transcribe_ms = median(&mut times);
    let transcript = transcript.trim().to_string();
    Ok(SttMeasurement {
        model: model.to_string(),
        load_ms,
        transcribe_ms,
        rtf: transcribe_ms / sample.duration_ms().max(1) as f64,
        word_accuracy: sample
            .reference
            .as_deref()
            .map(|r| word_accuracy(r, &transcript)),
        transcript,
    })
}

async fn measure_tts(tts: &LoadedTts, text: &str, runs: usize) -> anyhow::Result<TtsMeasurement> {
    let request = TtsRequest {
        text: text.to_string(),
        voice: tts.voice.clone(),
        seed: None,
    };
    tts.tts.synthesize(&request).await?; // warm-up
    let mut times = Vec::with_capacity(runs);
    let mut audio_ms = 0;
    for _ in 0..runs {
        let started = Instant::now();
        let output = tts.tts.synthesize(&request).await?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        audio_ms = output.audio.duration_ms();
    }
    let synth_ms = median(&mut times);
    Ok(TtsMeasurement {
        backend: tts.backend.to_string(),
        load_ms: tts.load_ms,
        synth_ms,
        audio_ms,
        rtf: synth_ms / audio_ms.max(1) as f64,
    })
}

/// One streamed reply: (time to first token, tokens per second).
async fn time_reply(
    client: &reqwest::Client,
    host: &str,
    model: &str,
) -> anyhow::Result<(Duration, f64)> {
    use vox::conversation::ChatMessage;

    let messages = [
        ChatMessage::system("You are a concise voice assistant."),
        ChatMessage::user(LLM_PROMPT),
    ];
    let started = Instant::now();
    let mut first: Option<Duration> = None;
    let mut last = Duration::ZERO;
    let mut tokens = 0u32;
    // The same streaming path Live Talk uses, so the figures are the ones a
    // conversation would see (and the request shows up in the privacy ledger).
    vox::streaming_chat::stream_chat(
        client,
        host,
        model,
        &messages,
        &vox::streaming_chat::StreamOptions::default(),
        tokio_util::sync::CancellationToken::new(),
        |_token| {
            let now = started.elapsed();
            first.get_or_insert(now);
            last = now;
            tokens += 1;
        },
        |_sentence| async { Ok(()) },
    )
    .await?;

    let first = first.ok_or_else(|| anyhow::anyhow!("'{model}' produced no output"))?;
    let generating = last.saturating_sub(first).as_secs_f64();
    let tokens_per_sec = if tokens > 1 && generating > 0.0 {
        f64::from(tokens - 1) / generating
    } else {
        0.0
    };
    Ok((first, tokens_per_sec))
}

async fn measure_llm(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    size_mb: Option<u64>,
) -> anyhow::Result<LlmMeasurement> {
    let (cold, _) = time_reply(client, host, model).await?;
    let (first, tokens_per_sec) = time_reply(client, host, model).await?;
    Ok(LlmMeasurement {
        model: model.to_string(),
        size_mb,
        cold_first_token_ms: cold.as_millis() as u64,
        first_token_ms: first.as_millis() as u64,
        tokens_per_sec,
    })
}

/// Which Ollama models to measure: the requested ones, or the installed
/// ones that fit in RAM, smallest first, up to [`MAX_LLM_CANDIDATES`].
pub fn llm_candidates(
    requested: &[String],
    installed: &[vox::OllamaModelSummary],
    total_ram_mb: u64,
) -> Vec<(String, Option<u64>)> {
    if !requested.is_empty() {
        return requested
            .iter()
            .map(|name| {
                let size = installed
                    .iter()
                    .find(|m| m.name == *name || m.name == format!("{name}:latest"))
                    .and_then(|m| m.size_mb);
                (name.clone(), size)
            })
            .collect();
    }
    // A model needs its file size in RAM plus context; leave room for the
    // OS, STT and TTS.
    let budget_mb = total_ram_mb * 6 / 10;
    let mut fitting: Vec<&vox::OllamaModelSummary> = installed
        .iter()
        .filter(|m| m.size_mb.is_none_or(|s| s <= budget_mb))
        .collect();
    fitting.sort_by_key(|m| m.size_mb.unwrap_or(0));
    fitting
        .into_iter()
        .take(MAX_LLM_CANDIDATES)
        .map(|m| (m.name.clone(), m.size_mb))
        .collect()
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Text of the recommendation block.
pub fn render_recommendation(rec: &Recommendation) -> String {
    let line = |label: &str, pick: &Pick| {
        format!(
            "  {label:<4} {:<22} {}\n",
            pick.choice.as_deref().unwrap_or("(none)"),
            pick.reason
        )
    };
    let mut out = String::from("Recommendation\n");
    out.push_str(&line("STT", &rec.stt));
    out.push_str(&line("TTS", &rec.tts));
    out.push_str(&line("LLM", &rec.llm));
    out
}

/// Text of the `--dry-run` report.
pub fn render_dry_run(
    profile: &SystemProfile,
    dir: &Path,
    inv: &Inventory,
    ollama_host: &str,
    offline: bool,
) -> String {
    let list = |items: Vec<String>| {
        if items.is_empty() {
            "none".to_string()
        } else {
            items.join(", ")
        }
    };
    let mut out = String::new();
    out.push_str(&format!("Machine: {}\n", profile.summary()));
    out.push_str(&format!("Models directory: {}\n", dir.display()));
    out.push_str(&format!(
        "Installed Whisper models: {}\n",
        list(
            inv.stt_installed
                .iter()
                .map(|f| whisper_name_from_file(f))
                .collect()
        )
    ));
    out.push_str(&format!(
        "Installed TTS backends: {}\n",
        list(inv.tts_installed.iter().map(|b| b.to_string()).collect())
    ));
    out.push_str(&format!(
        "Ollama: {ollama_host}{}\n\n",
        if offline && !vox::privacy::is_loopback_host(ollama_host) {
            " (refused in offline mode: not a loopback address)"
        } else if offline {
            " (offline mode)"
        } else {
            ""
        }
    ));
    out.push_str(&render_recommendation(&recommend_static(profile, inv)));
    out.push_str(
        "\nDry run: nothing was loaded, measured, downloaded or written.\n\
         Run `vox tune` to measure this machine and save the result.\n",
    );
    out
}

fn print_network_summary() {
    let snapshot = vox::privacy::ledger().snapshot();
    if snapshot.destinations.is_empty() {
        println!("Network: no outbound connections were made.");
        return;
    }
    println!("Network: connections made by this run");
    for row in &snapshot.destinations {
        println!(
            "  {:<28} {:<30} {} connection(s), {} B sent, {} B received{}{}",
            row.destination,
            row.purpose,
            row.connections,
            row.bytes_sent,
            row.bytes_received,
            if row.loopback { " (this machine)" } else { "" },
            if row.blocked > 0 {
                format!(", {} refused by offline mode", row.blocked)
            } else {
                String::new()
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// Run `vox tune`.
pub async fn run(args: TuneArgs) -> anyhow::Result<()> {
    let profile = SystemProfile::detect();
    let dir = models_dir();
    let language = args
        .language
        .as_deref()
        .map(str::trim)
        .filter(|l| !l.is_empty());
    let lang = sample_language(language);
    let config = config::load();
    let host = vox::privacy::resolve_ollama_host(
        args.ollama_host.as_deref(),
        config.ollama_host.as_deref(),
    );
    let offline = vox::privacy::offline();
    let runs = args.runs.clamp(1, 20);

    println!("=== vox tune ===\n");

    let inv = Inventory::scan(&dir, &profile, language);
    if args.dry_run {
        print!(
            "{}",
            render_dry_run(&profile, &dir, &inv, &host.host, offline)
        );
        return Ok(());
    }

    println!("Machine: {}", profile.summary());
    println!("Models directory: {}\n", dir.display());
    let fallback = recommend_static(&profile, &inv);

    // --- TTS backends and the speech sample -------------------------------
    println!("Loading TTS backends...");
    let tts_backends = load_tts_backends(&lang, args.yes).await;
    let sentence = reference_sentence(&lang);

    let sample = match &args.sample {
        Some(path) => Some(load_wav_sample(path, args.reference.as_deref())?),
        None => {
            let mut sample = None;
            if let Some(text) = sentence {
                for tts in &tts_backends {
                    match synthesize_sample(tts, text).await {
                        Ok(s) => {
                            sample = Some(s);
                            break;
                        }
                        Err(e) => eprintln!("  warning: {} could not speak: {e}", tts.backend),
                    }
                }
            }
            sample
        }
    };

    // --- STT ---------------------------------------------------------------
    let mut stt_results = Vec::new();
    match &sample {
        None => println!(
            "STT: skipped. There is no TTS backend to synthesize a speech sample for '{lang}'.\n\
             \x20    Record a sentence and pass it with: vox tune --sample speech.wav --reference \"what you said\""
        ),
        Some(sample) => {
            let mut files = inv.stt_installed.clone();
            if files.is_empty()
                && args.yes
                && let Some(file) = inv.stt_candidates.first()
            {
                let name = models::whisper_download_name(&whisper_name_from_file(file));
                models::ensure_model(&name, file, true).await?;
                files.push(file.clone());
            }
            if files.is_empty() {
                println!(
                    "STT: skipped. No Whisper model for '{lang}' is installed. \
                     Run `vox tune -y` to download the recommended one."
                );
            } else {
                println!(
                    "STT: transcribing {:.1} s of speech ({}), {runs} run(s) per model",
                    sample.duration_ms() as f64 / 1000.0,
                    sample.source
                );
                if files.len() > MAX_STT_CANDIDATES {
                    println!(
                        "  measuring the first {MAX_STT_CANDIDATES} of {} installed models",
                        files.len()
                    );
                }
            }
            for file in files.iter().take(MAX_STT_CANDIDATES) {
                match measure_stt(&dir.join(file), file, language, sample, runs).await {
                    Ok(m) => {
                        println!(
                            "  {:<12} load {:>5} ms  transcribe {:>6.0} ms  RTF {:.3}{}",
                            m.model,
                            m.load_ms,
                            m.transcribe_ms,
                            m.rtf,
                            m.word_accuracy
                                .map(|a| format!("  words correct {:.0}%", a * 100.0))
                                .unwrap_or_default()
                        );
                        stt_results.push(m);
                    }
                    Err(e) => eprintln!("  {file}: failed: {e}"),
                }
            }
        }
    }
    println!();

    // --- TTS ---------------------------------------------------------------
    let mut tts_results = Vec::new();
    match sentence {
        Some(text) if !tts_backends.is_empty() => {
            println!("TTS: synthesizing one sentence, {runs} run(s) per backend");
            for tts in &tts_backends {
                match measure_tts(tts, text, runs).await {
                    Ok(m) => {
                        println!(
                            "  {:<12} load {:>5} ms  synthesize {:>6.0} ms for {:.1} s of audio  RTF {:.3}",
                            m.backend,
                            m.load_ms,
                            m.synth_ms,
                            m.audio_ms as f64 / 1000.0,
                            m.rtf
                        );
                        tts_results.push(m);
                    }
                    Err(e) => eprintln!("  {}: failed: {e}", tts.backend),
                }
            }
        }
        Some(_) => {
            println!("TTS: skipped. No TTS model is installed. Run `vox tune -y` to download one.")
        }
        None => println!("TTS: skipped. There is no test sentence for language '{lang}'."),
    }
    drop(tts_backends);
    println!();

    // --- LLM ---------------------------------------------------------------
    let mut llm_results = Vec::new();
    if args.no_llm {
        println!("LLM: skipped (--no-llm).");
    } else if let Err(denied) = vox::privacy::ensure_ollama_host_allowed(&host.host) {
        println!("LLM: skipped. {denied}");
    } else {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let installed = vox::capabilities::ollama::probe(&client, &host.host)
            .await
            .unwrap_or_default();
        let candidates = llm_candidates(&args.llm, &installed, profile.total_ram_mb);
        if candidates.is_empty() {
            println!(
                "LLM: skipped. No Ollama models found at {} (from {}). Is Ollama running?",
                host.host, host.source
            );
        } else {
            println!(
                "LLM: streaming a short reply from {} model(s) at {}",
                candidates.len(),
                host.host
            );
            for (model, size_mb) in candidates {
                match measure_llm(&client, &host.host, &model, size_mb).await {
                    Ok(m) => {
                        println!(
                            "  {:<22} first token {:>5} ms (cold {:>6} ms)  {:>6.1} tokens/s",
                            m.model, m.first_token_ms, m.cold_first_token_ms, m.tokens_per_sec
                        );
                        llm_results.push(m);
                    }
                    Err(e) => eprintln!("  {model}: failed: {e}"),
                }
            }
        }
    }
    println!();

    // --- Result ------------------------------------------------------------
    let rec = recommend_measured(&fallback, &stt_results, &tts_results, &llm_results);
    print!("{}", render_recommendation(&rec));
    println!();

    let measured_anything =
        !(stt_results.is_empty() && tts_results.is_empty() && llm_results.is_empty());
    if args.no_save {
        println!("Not saved (--no-save).");
    } else if !measured_anything {
        println!("Nothing could be measured, so the configuration was left unchanged.");
    } else {
        let path = config::config_path();
        let mut config = config;
        config.tune = Some(tune_config(
            &profile,
            language,
            &rec,
            &stt_results,
            &tts_results,
            &llm_results,
        ));
        config::save_to(&config, &path)?;
        println!("Saved to the [tune] table of {}", path.display());
        println!("`vox chat` uses the tuned Whisper and Ollama models unless told otherwise.");
    }
    println!();
    print_network_summary();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox::system_profile::HardwareClass;

    fn profile(class: HardwareClass, ram_mb: u64) -> SystemProfile {
        let mut p = SystemProfile::detect();
        p.class = class;
        p.total_ram_mb = ram_mb;
        p
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vox-tune-test-{}-{name}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn stt(model: &str, rtf: f64, accuracy: Option<f64>) -> SttMeasurement {
        SttMeasurement {
            model: model.into(),
            load_ms: 100,
            transcribe_ms: rtf * 5000.0,
            rtf,
            word_accuracy: accuracy,
            transcript: String::new(),
        }
    }

    fn llm(model: &str, size_mb: u64, first_token_ms: u64, tps: f64) -> LlmMeasurement {
        LlmMeasurement {
            model: model.into(),
            size_mb: Some(size_mb),
            cold_first_token_ms: first_token_ms * 10,
            first_token_ms,
            tokens_per_sec: tps,
        }
    }

    #[test]
    fn word_accuracy_ignores_case_and_punctuation() {
        assert_eq!(word_accuracy("Hello, world!", "hello world"), 1.0);
        assert_eq!(word_accuracy("we'll go", "We'll go."), 1.0);
        assert_eq!(word_accuracy("one two three four", "one two three"), 0.75);
        assert_eq!(
            word_accuracy("one two three four", "one too three four"),
            0.75
        );
        assert_eq!(word_accuracy("one two", "zero one two"), 0.5);
        assert_eq!(word_accuracy("one two", ""), 0.0);
        assert_eq!(word_accuracy("a", "b c d e"), 0.0, "never negative");
        assert_eq!(word_accuracy("", "anything"), 0.0);
        assert_eq!(word_accuracy("Bugün hava güzel", "bugün hava güzel"), 1.0);
    }

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn whisper_names_round_trip_through_file_names() {
        for class in [HardwareClass::Tiny, HardwareClass::Large] {
            let p = profile(class, 8000);
            for language in [None, Some("tr"), Some("auto")] {
                for file in p.recommended_whisper_candidates(language) {
                    let name = whisper_name_from_file(file);
                    assert!(!name.ends_with(".bin"), "{file} -> {name}");
                    assert_eq!(vox::paths::whisper_model_filename(&name), file);
                }
            }
        }
    }

    #[test]
    fn sample_language_defaults_to_english() {
        assert_eq!(sample_language(None), "en");
        assert_eq!(sample_language(Some("auto")), "en");
        assert_eq!(sample_language(Some(" ")), "en");
        assert_eq!(sample_language(Some("TR")), "tr");
        assert_eq!(sample_language(Some("de-DE")), "de");
        assert_eq!(sample_language(Some("en_US")), "en");
        for lang in ["en", "tr", "de", "es", "fr"] {
            assert!(reference_sentence(lang).is_some(), "{lang}");
        }
        assert!(reference_sentence("ja").is_none());
    }

    #[test]
    fn dry_run_recommends_without_any_model_installed() {
        let dir = scratch("empty");
        let p = profile(HardwareClass::Large, 32_000);
        let inv = Inventory::scan(&dir, &p, None);
        assert!(inv.stt_installed.is_empty());
        assert!(inv.tts_installed.is_empty());

        let rec = recommend_static(&p, &inv);
        assert_eq!(rec.stt.choice.as_deref(), Some("small.en"));
        assert!(
            rec.stt
                .reason
                .contains("vox models download whisper-small.en"),
            "{}",
            rec.stt.reason
        );
        assert_eq!(rec.llm.choice.as_deref(), Some("llama3.2:3b"));
        assert!(rec.llm.reason.contains("ollama pull llama3.2:3b"));

        let report = render_dry_run(&p, &dir, &inv, "localhost:11434", false);
        assert!(report.contains("Recommendation"), "{report}");
        assert!(report.contains("small.en"), "{report}");
        assert!(report.contains("llama3.2:3b"), "{report}");
        assert!(
            report.contains("Installed Whisper models: none"),
            "{report}"
        );
        assert!(report.contains("Dry run: nothing was loaded"), "{report}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dry_run_prefers_what_is_installed() {
        let dir = scratch("installed");
        std::fs::write(dir.join("ggml-tiny.en.bin"), b"x").unwrap();
        std::fs::write(dir.join("ggml-base.en.bin"), b"x").unwrap();
        let p = profile(HardwareClass::Large, 32_000);
        let inv = Inventory::scan(&dir, &p, Some("en"));
        // Large hardware prefers small, then base, then tiny.
        assert_eq!(inv.stt_installed, ["ggml-base.en.bin", "ggml-tiny.en.bin"]);
        let rec = recommend_static(&p, &inv);
        assert_eq!(rec.stt.choice.as_deref(), Some("base.en"));
        assert!(
            rec.stt.reason.contains("best installed"),
            "{}",
            rec.stt.reason
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dry_run_never_offers_an_english_only_model_for_another_language() {
        let dir = scratch("turkish");
        std::fs::write(dir.join("ggml-base.en.bin"), b"x").unwrap();
        let p = profile(HardwareClass::Medium, 8_000);
        let inv = Inventory::scan(&dir, &p, Some("tr"));
        assert!(inv.stt_installed.is_empty());
        let rec = recommend_static(&p, &inv);
        let choice = rec.stt.choice.unwrap();
        assert!(
            !vox::paths::whisper_model_is_english_only(&choice),
            "{choice}"
        );
        assert!(!inv.tts_compiled.contains(&"kokoro"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dry_run_on_a_tiny_machine_has_no_local_llm() {
        let p = profile(HardwareClass::Tiny, 512);
        let rec = recommend_static(&p, &Inventory::default());
        assert_eq!(rec.llm.choice, None);
        assert!(
            rec.llm.reason.contains("--ollama-host"),
            "{}",
            rec.llm.reason
        );
        assert_eq!(rec.stt.choice, None);
        assert_eq!(rec.tts.choice, None);
    }

    #[test]
    fn dry_run_reports_a_remote_host_refused_by_offline_mode() {
        let p = profile(HardwareClass::Large, 32_000);
        let dir = PathBuf::from("/nonexistent");
        let inv = Inventory::default();
        let report = render_dry_run(&p, &dir, &inv, "gpu-box:11434", true);
        assert!(report.contains("refused in offline mode"), "{report}");
        let report = render_dry_run(&p, &dir, &inv, "localhost:11434", true);
        assert!(!report.contains("refused"), "{report}");
        assert!(report.contains("(offline mode)"), "{report}");
    }

    #[test]
    fn stt_pick_prefers_accuracy_within_the_speed_budget() {
        let ms = [
            stt("small.en", 0.25, Some(1.0)),
            stt("base.en", 0.10, Some(0.90)),
            stt("tiny.en", 0.05, Some(0.80)),
        ];
        assert_eq!(pick_stt(&ms), Some(0));
    }

    #[test]
    fn stt_pick_takes_the_faster_of_equally_accurate_models() {
        let ms = [
            stt("small.en", 0.25, Some(1.0)),
            stt("base.en", 0.10, Some(0.99)),
            stt("tiny.en", 0.05, Some(0.80)),
        ];
        assert_eq!(pick_stt(&ms), Some(1));
    }

    #[test]
    fn stt_pick_rejects_models_over_budget_even_if_more_accurate() {
        let ms = [
            stt("medium.en", 0.90, Some(1.0)),
            stt("base.en", 0.10, Some(0.85)),
        ];
        assert_eq!(pick_stt(&ms), Some(1));
    }

    #[test]
    fn stt_pick_falls_back_to_the_fastest_when_nothing_is_in_budget() {
        let ms = [
            stt("small", 1.4, Some(1.0)),
            stt("base", 0.6, Some(0.9)),
            stt("tiny", 0.8, Some(0.7)),
        ];
        assert_eq!(pick_stt(&ms), Some(1));
        assert_eq!(pick_stt(&[]), None);
    }

    #[test]
    fn stt_pick_without_accuracy_keeps_the_preference_order() {
        let ms = [
            stt("medium", 0.50, None),
            stt("small", 0.20, None),
            stt("base", 0.05, None),
        ];
        assert_eq!(pick_stt(&ms), Some(1));
    }

    #[test]
    fn llm_pick_takes_the_largest_model_that_is_fast_enough() {
        let ms = [
            llm("qwen2.5:0.5b", 400, 150, 90.0),
            llm("llama3.2:3b", 2000, 300, 35.0),
            llm("llama3.1:70b", 40_000, 4000, 2.0),
        ];
        assert_eq!(pick_llm(&ms), Some(1));
    }

    #[test]
    fn llm_pick_falls_back_to_the_fastest_when_all_are_slow() {
        let ms = [llm("big", 9000, 3000, 3.0), llm("small", 1000, 2500, 8.0)];
        assert_eq!(pick_llm(&ms), Some(1));
        assert_eq!(pick_llm(&[]), None);
        assert!(!llm_is_conversational(&ms[1]));
    }

    #[test]
    fn llm_candidates_are_the_requested_ones_or_the_small_installed_ones() {
        let installed: Vec<vox::OllamaModelSummary> = [
            ("huge:70b", 40_000),
            ("b:3b", 2000),
            ("a:1b", 800),
            ("c:7b", 4500),
            ("d:8b", 4700),
            ("e:9b", 4800),
        ]
        .into_iter()
        .map(|(name, size)| vox::OllamaModelSummary {
            name: name.into(),
            size_mb: Some(size),
            family: None,
        })
        .collect();

        let auto = llm_candidates(&[], &installed, 16_000);
        let names: Vec<&str> = auto.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["a:1b", "b:3b", "c:7b", "d:8b"],
            "smallest first, capped, fits RAM"
        );

        let asked = llm_candidates(&["huge:70b".into(), "missing".into()], &installed, 16_000);
        assert_eq!(
            asked,
            [
                ("huge:70b".to_string(), Some(40_000)),
                ("missing".to_string(), None)
            ]
        );
    }

    #[test]
    fn measured_recommendation_replaces_the_static_one_per_category() {
        let p = profile(HardwareClass::Large, 32_000);
        let fallback = recommend_static(&p, &Inventory::default());
        let stt_ms = [stt("base.en", 0.08, Some(0.95))];
        let llm_ms = [llm("llama3.2:3b", 2000, 250, 40.0)];

        let rec = recommend_measured(&fallback, &stt_ms, &[], &llm_ms);
        assert_eq!(rec.stt.choice.as_deref(), Some("base.en"));
        assert!(rec.stt.reason.contains("RTF 0.080"), "{}", rec.stt.reason);
        assert!(rec.stt.reason.contains("95%"), "{}", rec.stt.reason);
        assert_eq!(rec.tts.choice, fallback.tts.choice);
        assert!(
            rec.tts.reason.starts_with("not measured"),
            "{}",
            rec.tts.reason
        );
        assert_eq!(rec.llm.choice.as_deref(), Some("llama3.2:3b"));
        assert!(
            rec.llm.reason.contains("40.0 tokens/s"),
            "{}",
            rec.llm.reason
        );

        // Only what was measured reaches the config file.
        let cfg = tune_config(&p, Some("en"), &rec, &stt_ms, &[], &llm_ms);
        assert_eq!(cfg.stt_model.as_deref(), Some("base.en"));
        assert_eq!(cfg.stt_rtf, Some(0.08));
        assert_eq!(cfg.stt_word_accuracy, Some(0.95));
        assert_eq!(cfg.tts_backend, None);
        assert_eq!(cfg.tts_rtf, None);
        assert_eq!(cfg.llm_model.as_deref(), Some("llama3.2:3b"));
        assert_eq!(cfg.llm_first_token_ms, Some(250));
        assert_eq!(cfg.language.as_deref(), Some("en"));
        assert!(cfg.measured_unix > 0);
        assert!(!cfg.machine.is_empty());
    }

    #[test]
    fn measured_recommendation_flags_picks_that_miss_the_targets() {
        let p = profile(HardwareClass::Small, 4_000);
        let fallback = recommend_static(&p, &Inventory::default());
        let rec = recommend_measured(
            &fallback,
            &[stt("base", 0.7, None)],
            &[TtsMeasurement {
                backend: "kokoro".into(),
                load_ms: 1,
                synth_ms: 6000.0,
                audio_ms: 4000,
                rtf: 1.5,
            }],
            &[llm("m", 900, 2000, 5.0)],
        );
        assert!(rec.stt.reason.contains("slower than"), "{}", rec.stt.reason);
        assert!(
            rec.tts.reason.contains("slower than real time"),
            "{}",
            rec.tts.reason
        );
        assert!(rec.llm.reason.contains("too slow"), "{}", rec.llm.reason);
    }

    #[test]
    fn wav_sample_is_downmixed_and_resampled_to_16k_mono() {
        let dir = scratch("wav");
        let path = dir.join("speech.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48_000 {
            let v = ((i as f32 * 0.05).sin() * 12_000.0) as i16;
            writer.write_sample(v).unwrap();
            writer.write_sample(v).unwrap();
        }
        writer.finalize().unwrap();

        let sample = load_wav_sample(&path, Some("  hello there ")).unwrap();
        assert!(
            (sample.samples.len() as i64 - 16_000).abs() <= 16,
            "{}",
            sample.samples.len()
        );
        assert!((990..=1010).contains(&sample.duration_ms()));
        assert_eq!(sample.reference.as_deref(), Some("hello there"));
        assert!(sample.samples.iter().all(|s| s.abs() <= 1.0));
        assert!(
            sample.samples.iter().any(|s| s.abs() > 0.1),
            "audio must not be silent"
        );

        assert!(load_wav_sample(&dir.join("missing.wav"), None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn too_short_wav_sample_is_rejected() {
        let dir = scratch("short");
        let path = dir.join("short.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..1600 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        let err = load_wav_sample(&path, None).unwrap_err().to_string();
        assert!(err.contains("less than half a second"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- The measuring code, against stand-in backends ----------------------

    struct FakeTts {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl TtsBackend for FakeTts {
        async fn synthesize(
            &self,
            request: &TtsRequest,
        ) -> Result<vox::types::TtsOutput, vox::VoxError> {
            tokio::time::sleep(self.delay).await;
            assert!(!request.text.is_empty());
            // Two seconds of a 220 Hz tone at Piper's sample rate.
            let samples = (0..44_100)
                .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / 22_050.0).sin() * 0.3)
                .collect();
            Ok(vox::types::TtsOutput::from_mono(samples, 22_050))
        }
    }

    fn fake_tts(delay_ms: u64) -> LoadedTts {
        LoadedTts {
            backend: "fake",
            voice: Some("v".into()),
            tts: Arc::new(FakeTts {
                delay: Duration::from_millis(delay_ms),
            }),
            load_ms: 7,
        }
    }

    struct FakeStt {
        text: &'static str,
        delay: Duration,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl SttBackend for FakeStt {
        async fn transcribe(&self, audio: &Utterance) -> Result<vox::SttResult, vox::VoxError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(audio.audio.sample_rate, 16_000);
            assert_eq!(audio.audio.channels, 1);
            assert!(
                audio.audio.samples.iter().any(|s| s.abs() > 0.05),
                "not silence"
            );
            tokio::time::sleep(self.delay).await;
            Ok(vox::SttResult {
                text: format!("  {} ", self.text),
                language: None,
                duration_ms: audio.duration_ms,
                processing_time_ms: 0,
                #[cfg(feature = "diarization")]
                speaker_id: None,
            })
        }
    }

    #[tokio::test]
    async fn synthesized_sample_is_16k_speech_with_its_reference() {
        let sample = synthesize_sample(&fake_tts(0), "hello there")
            .await
            .unwrap();
        assert!(
            (1990..=2010).contains(&sample.duration_ms()),
            "{}",
            sample.duration_ms()
        );
        assert_eq!(sample.reference.as_deref(), Some("hello there"));
        assert_eq!(sample.source, "synthesized with fake");
        assert!(sample.samples.iter().any(|s| s.abs() > 0.05));
    }

    #[tokio::test]
    async fn tts_timing_reports_rtf_against_the_audio_produced() {
        let m = measure_tts(&fake_tts(40), "hello there", 3).await.unwrap();
        assert_eq!(m.backend, "fake");
        assert_eq!(m.load_ms, 7);
        assert_eq!(m.audio_ms, 2000);
        assert!(m.synth_ms >= 40.0, "{}", m.synth_ms);
        assert!(m.synth_ms < 1000.0, "{}", m.synth_ms);
        assert!((m.rtf - m.synth_ms / 2000.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn stt_timing_scores_the_transcript_against_the_reference() {
        let sample = synthesize_sample(&fake_tts(0), "one two three four")
            .await
            .unwrap();
        let stt = FakeStt {
            text: "One, two, three for.",
            delay: Duration::from_millis(20),
            calls: Default::default(),
        };
        let m = time_stt(&stt, "base.en", 11, &sample, 3).await.unwrap();
        assert_eq!(
            stt.calls.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "warm-up + 3 runs"
        );
        assert_eq!(m.model, "base.en");
        assert_eq!(m.load_ms, 11);
        assert_eq!(m.transcript, "One, two, three for.");
        assert_eq!(m.word_accuracy, Some(0.75));
        assert!(m.transcribe_ms >= 20.0, "{}", m.transcribe_ms);
        assert!((m.rtf - m.transcribe_ms / sample.duration_ms() as f64).abs() < 1e-9);

        // Without a reference there is no accuracy figure.
        let unknown = SpeechSample {
            reference: None,
            ..sample
        };
        let m = time_stt(&stt, "base.en", 0, &unknown, 1).await.unwrap();
        assert_eq!(m.word_accuracy, None);
    }

    /// A stand-in Ollama `/api/chat`: `gap_ms` before the first token and
    /// between tokens. Serves any number of requests.
    async fn fake_ollama_chat(tokens: usize, gap_ms: u64) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let host = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    // Read the request head and its JSON body.
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                            let len: usize = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse().ok())
                                .unwrap_or(0);
                            if buf.len() >= end + 4 + len {
                                break;
                            }
                        }
                        match sock.read(&mut tmp).await {
                            Ok(n) if n > 0 => buf.extend_from_slice(&tmp[..n]),
                            _ => return,
                        }
                    }
                    let head = "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\n\
                                connection: close\r\n\r\n";
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.flush().await;
                    for i in 0..tokens {
                        tokio::time::sleep(Duration::from_millis(gap_ms)).await;
                        let line = serde_json::json!({
                            "message": { "role": "assistant", "content": format!("word{i} ") },
                            "done": false,
                        });
                        let _ = sock.write_all(format!("{line}\n").as_bytes()).await;
                        let _ = sock.flush().await;
                    }
                    let _ = sock.write_all(b"{\"done\":true}\n").await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        host
    }

    #[tokio::test]
    async fn llm_timing_measures_first_token_and_rate_from_the_stream() {
        let host = fake_ollama_chat(11, 20).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let m = measure_llm(&client, &host, "stand-in:1b", Some(800))
            .await
            .unwrap();

        assert_eq!(m.model, "stand-in:1b");
        assert_eq!(m.size_mb, Some(800));
        // The first token arrives one gap after the request.
        assert!(
            (15..1000).contains(&m.first_token_ms),
            "{}",
            m.first_token_ms
        );
        assert!(m.cold_first_token_ms >= 15, "{}", m.cold_first_token_ms);
        // 10 intervals of at least 20 ms each: at most 50 tokens per second.
        assert!(
            m.tokens_per_sec > 5.0 && m.tokens_per_sec <= 50.5,
            "{}",
            m.tokens_per_sec
        );

        // Both replies went through the egress choke point.
        let snapshot = vox::privacy::ledger().snapshot();
        let row = snapshot
            .destinations
            .iter()
            .find(|row| row.destination == host && row.purpose == "llm chat")
            .expect("requests are in the privacy ledger");
        assert_eq!(row.connections, 2, "warm-up + measured reply");
        assert!(row.bytes_sent > 0 && row.bytes_received > 0);
        assert!(row.loopback);
    }

    #[tokio::test]
    async fn llm_timing_fails_for_a_model_that_says_nothing() {
        let host = fake_ollama_chat(0, 0).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let err = measure_llm(&client, &host, "mute", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("produced no output"), "{err}");
    }
}
