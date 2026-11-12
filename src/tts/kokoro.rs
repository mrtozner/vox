//! Kokoro TTS backend using the kokoro-tts crate.
//!
//! The crate is used as published, so this backend works around four of its
//! limits rather than fixing them at the source:
//!
//! - **Numbers.** Its G2P rewrites the first number in the text as Chinese
//!   numerals and drops the rest. Text is run through
//!   [`normalize`](super::normalize) first, so no digits reach it.
//! - **Length.** A v1.0 voice pack holds one style vector per token count.
//!   Text that phonemizes to more tokens than the pack has entries indexes
//!   out of bounds and panics. Text is split into chunks that fit, with the
//!   pack size read from the voices file.
//! - **Concurrency.** The English G2P is C code with global state. Every
//!   phonemization in the process goes through one mutex.
//! - **Scripts.** Characters outside Latin-1 and CJK are silently dropped.
//!   Common Latin letters are folded to ASCII; anything else is reported.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use kokoro_tts::{KokoroTts, Voice as KokoroVoice};

use super::normalize::{self, NormLang, NormalizeMode};
use crate::error::VoxError;
use crate::traits::TtsBackend;
use crate::types::{TtsOutput, TtsRequest, VoiceInfo};

/// Kokoro output sample rate.
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Tokens per inference assumed when the voices file cannot be inspected.
/// Every published v1.0 voice pack has this many entries.
const FALLBACK_PACK_LEN: usize = 510;

/// Voices that only work with the v1.1 model.
const V11_ONLY_VOICES: [&str; 3] = ["af_maple", "af_sol", "bf_vale"];

/// Languages the bundled G2P can phonemize.
const G2P_LANGUAGES: [&str; 2] = ["en", "zh"];

/// Serializes every G2P call and inference in the process.
///
/// The English G2P keeps its state in C globals, so two phonemizations must
/// never overlap, including ones from separate `KokoroBackend` instances.
/// `KokoroTts::synth` runs G2P and inference in one call, so the lock covers
/// both. It is taken per chunk, which lets a short request run between the
/// chunks of a long one.
///
/// It is an owned lock so the guard can travel into `spawn_blocking`: a
/// caller that gives up mid-request must not release the gate while its G2P
/// is still running on the blocking pool.
static SYNTH_GATE: LazyLock<Arc<tokio::sync::Mutex<()>>> =
    LazyLock::new(|| Arc::new(tokio::sync::Mutex::new(())));

/// Configuration for the Kokoro TTS backend.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    /// Path to the Kokoro ONNX model file.
    pub model_path: PathBuf,
    /// Path to the voices binary file.
    pub voices_path: PathBuf,
    /// Default voice to use when none is specified.
    pub default_voice: String,
    /// Speed multiplier (1.0 = normal).
    pub speed: f32,
}

/// The part of Kokoro that needs model files. Tests substitute a fake.
#[async_trait]
trait SynthEngine: Send + Sync {
    /// Number of model tokens `text` phonemizes to, padding included.
    ///
    /// Runs G2P, so it blocks and must be called with [`SYNTH_GATE`] held.
    fn token_count(&self, text: &str) -> Result<usize, VoxError>;

    /// Synthesize one chunk. Must be called with [`SYNTH_GATE`] held.
    async fn synth(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, VoxError>;
}

struct OnnxEngine {
    tts: KokoroTts,
}

#[async_trait]
impl SynthEngine for OnnxEngine {
    fn token_count(&self, text: &str) -> Result<usize, VoxError> {
        let phonemes = kokoro_tts::g2p(text, false)
            .map_err(|e| VoxError::Tts(format!("kokoro phonemization failed: {e}")))?;
        Ok(kokoro_tts::get_token_ids(&phonemes, false).len())
    }

    async fn synth(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, VoxError> {
        let voice = KokoroBackend::parse_voice(voice, speed)?;
        // The second value is the time inference took, not the audio length.
        let (samples, _inference_time) = self
            .tts
            .synth(text, voice)
            .await
            .map_err(|e| VoxError::Tts(format!("kokoro synthesis failed: {e}")))?;
        Ok(samples)
    }
}

/// Read the number of style vectors each voice pack holds.
///
/// The voices file is a bincode (standard config) `HashMap<String,
/// Vec<Vec<Vec<f32>>>>`. Only the lengths matter here, so the floats are
/// skipped rather than decoded.
fn parse_pack_lens(bytes: &[u8]) -> Result<HashMap<String, usize>, String> {
    struct Reader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            let end = self
                .pos
                .checked_add(n)
                .filter(|&end| end <= self.bytes.len())
                .ok_or_else(|| format!("truncated at byte {}", self.pos))?;
            let slice = &self.bytes[self.pos..end];
            self.pos = end;
            Ok(slice)
        }

        /// bincode's variable-length unsigned integer.
        fn varint(&mut self) -> Result<usize, String> {
            let width = match self.take(1)?[0] {
                small @ 0..=250 => return Ok(usize::from(small)),
                251 => 2,
                252 => 4,
                253 => 8,
                other => return Err(format!("unsupported length marker {other}")),
            };
            let mut buf = [0u8; 8];
            buf[..width].copy_from_slice(self.take(width)?);
            usize::try_from(u64::from_le_bytes(buf)).map_err(|e| e.to_string())
        }
    }

    let mut reader = Reader { bytes, pos: 0 };
    let voices = reader.varint()?;
    let mut packs = HashMap::new();
    for _ in 0..voices {
        let name_len = reader.varint()?;
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|e| format!("voice name is not UTF-8: {e}"))?
            .to_string();
        let pack_len = reader.varint()?;
        for _ in 0..pack_len {
            let styles = reader.varint()?;
            for _ in 0..styles {
                let floats = reader.varint()?;
                let byte_len = floats
                    .checked_mul(4)
                    .ok_or_else(|| "style vector length overflows".to_string())?;
                reader.take(byte_len)?;
            }
        }
        packs.insert(name, pack_len);
    }
    if packs.is_empty() {
        return Err("no voices in file".into());
    }
    if reader.pos != bytes.len() {
        return Err(format!(
            "{} trailing bytes after the last voice",
            bytes.len() - reader.pos
        ));
    }
    Ok(packs)
}

/// Language of a Kokoro voice, from the first letter of its id.
fn voice_language(voice: &str) -> &'static str {
    match voice.as_bytes().first().map(u8::to_ascii_lowercase) {
        Some(b'z') => "zh",
        Some(b'j') => "ja",
        Some(b'e') => "es",
        Some(b'f') => "fr",
        Some(b'h') => "hi",
        Some(b'i') => "it",
        Some(b'p') => "pt",
        // 'a' and 'b' are American and British English.
        _ => "en",
    }
}

/// True for characters the crate's G2P keeps: Latin-1, CJK ideographs and
/// CJK punctuation. It drops everything else without saying so.
fn g2p_accepts(c: char) -> bool {
    c as u32 <= 0xFF
        || ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || "，。：·？、！《》（）【】〖〗〔〕“”‘’〈〉…—　".contains(c)
}

/// ASCII stand-in for a Latin letter outside Latin-1.
///
/// The result is a rough approximation: it keeps the word intact so the
/// English G2P reads something, where dropping the letter would cut it.
fn fold_latin(c: char) -> Option<&'static str> {
    Some(match c {
        'ā' | 'ă' | 'ą' => "a",
        'Ā' | 'Ă' | 'Ą' => "A",
        'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'Ć' | 'Ĉ' | 'Ċ' | 'Č' => "C",
        'ď' | 'đ' => "d",
        'Ď' | 'Đ' => "D",
        'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => "e",
        'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => "E",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'Ĝ' | 'Ğ' | 'Ġ' | 'Ģ' => "G",
        'ĥ' | 'ħ' => "h",
        'Ĥ' | 'Ħ' => "H",
        'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => "i",
        'Ĩ' | 'Ī' | 'Ĭ' | 'Į' | 'İ' => "I",
        'ĵ' => "j",
        'Ĵ' => "J",
        'ķ' => "k",
        'Ķ' => "K",
        'ĺ' | 'ļ' | 'ľ' | 'ł' => "l",
        'Ĺ' | 'Ļ' | 'Ľ' | 'Ł' => "L",
        'ń' | 'ņ' | 'ň' => "n",
        'Ń' | 'Ņ' | 'Ň' => "N",
        'ō' | 'ŏ' | 'ő' => "o",
        'Ō' | 'Ŏ' | 'Ő' => "O",
        'œ' => "oe",
        'Œ' => "OE",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'Ŕ' | 'Ŗ' | 'Ř' => "R",
        'ś' | 'ŝ' | 'ş' | 'š' | 'ș' => "s",
        'Ś' | 'Ŝ' | 'Ş' | 'Š' | 'Ș' => "S",
        'ţ' | 'ť' | 'ŧ' | 'ț' => "t",
        'Ţ' | 'Ť' | 'Ŧ' | 'Ț' => "T",
        'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => "U",
        'ŵ' => "w",
        'Ŵ' => "W",
        'ŷ' => "y",
        'Ŷ' | 'Ÿ' => "Y",
        'ź' | 'ż' | 'ž' => "z",
        'Ź' | 'Ż' | 'Ž' => "Z",
        _ => return None,
    })
}

/// Text ready for the crate's G2P, plus what had to be left out.
#[derive(Debug, PartialEq)]
struct Prepared {
    text: String,
    /// Characters with no G2P support, in order of first appearance.
    unsupported: Vec<char>,
    /// How many characters were left out in total.
    unsupported_count: usize,
}

/// Normalize `text` for `voice` and remove what the G2P cannot read.
///
/// Fails when the input had content but none of it is in a supported script;
/// synthesizing silence for it would hide the problem.
fn prepare_text(text: &str, voice: &str) -> Result<Prepared, VoxError> {
    // The crate converts digits itself for Chinese. Every other voice goes
    // through the English G2P, so numbers are spelled out in English.
    let normalized = if voice_language(voice) == "zh" {
        normalize::clean(text)
    } else {
        normalize::normalize(text, NormLang::English, NormalizeMode::Full)
    };

    let mut out = String::with_capacity(normalized.len());
    let mut unsupported = Vec::new();
    let mut unsupported_count = 0;
    for c in normalized.chars() {
        if g2p_accepts(c) {
            out.push(c);
        } else if let Some(folded) = fold_latin(c) {
            out.push_str(folded);
        } else {
            unsupported_count += 1;
            if !unsupported.contains(&c) && unsupported.len() < 8 {
                unsupported.push(c);
            }
            // Keep the neighbours apart, as the dropped character did.
            out.push(' ');
        }
    }
    let text = out.split_whitespace().collect::<Vec<_>>().join(" ");

    if unsupported_count > 0 && !text.chars().any(char::is_alphanumeric) {
        let sample: String = unsupported.iter().collect();
        return Err(VoxError::Tts(format!(
            "kokoro cannot speak this text: its G2P covers Latin-1 and Chinese characters only, \
             and none of the input is in either (found {sample:?})"
        )));
    }
    Ok(Prepared {
        text,
        unsupported,
        unsupported_count,
    })
}

/// Kokoro TTS backend powered by the kokoro-tts crate.
///
/// Uses the Kokoro-82M model (82 million parameters) for high-quality
/// text-to-speech synthesis. Runs entirely locally via ONNX Runtime.
///
/// Requests are synthesized one chunk at a time, process-wide; see the
/// module documentation.
pub struct KokoroBackend {
    engine: Arc<dyn SynthEngine>,
    config: KokoroConfig,
    /// Style vectors per voice, which is the most tokens one inference may
    /// carry. Empty when the voices file could not be inspected.
    packs: HashMap<String, usize>,
    /// Voice languages already warned about for lacking G2P support.
    warned_languages: std::sync::Mutex<HashSet<&'static str>>,
}

impl KokoroBackend {
    /// Create a new Kokoro TTS backend.
    ///
    /// `model_path` should point to `kokoro-v1.0.onnx` (or int8 variant).
    /// `voices_path` should point to `voices-v1.0.bin`.
    pub async fn new(
        model_path: impl AsRef<Path>,
        voices_path: impl AsRef<Path>,
    ) -> Result<Self, VoxError> {
        Self::with_config(KokoroConfig {
            model_path: model_path.as_ref().to_path_buf(),
            voices_path: voices_path.as_ref().to_path_buf(),
            default_voice: "af_heart".into(),
            speed: 1.0,
        })
        .await
    }

    /// Create with custom configuration.
    pub async fn with_config(config: KokoroConfig) -> Result<Self, VoxError> {
        if !config.model_path.exists() {
            return Err(VoxError::ModelNotFound(config.model_path.clone()));
        }
        if !config.voices_path.exists() {
            return Err(VoxError::ModelNotFound(config.voices_path.clone()));
        }

        let tts = KokoroTts::new(&config.model_path, &config.voices_path)
            .await
            .map_err(|e| VoxError::Tts(format!("failed to load kokoro model: {e}")))?;

        let packs = match tokio::fs::read(&config.voices_path).await {
            Ok(bytes) => parse_pack_lens(&bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    fallback = FALLBACK_PACK_LEN,
                    "could not read voice pack sizes; assuming the fallback"
                );
                HashMap::new()
            }),
            Err(e) => {
                tracing::warn!(error = %e, "could not re-read kokoro voices file");
                HashMap::new()
            }
        };

        Ok(Self::from_parts(
            Arc::new(OnnxEngine { tts }),
            config,
            packs,
        ))
    }

    fn from_parts(
        engine: Arc<dyn SynthEngine>,
        config: KokoroConfig,
        packs: HashMap<String, usize>,
    ) -> Self {
        Self {
            engine,
            config,
            packs,
            warned_languages: std::sync::Mutex::new(HashSet::new()),
        }
    }

    /// Languages the bundled G2P can phonemize (`en`, `zh`).
    ///
    /// Voices in other languages are still listed and usable, but their text
    /// is read with English pronunciation rules.
    pub fn g2p_languages() -> &'static [&'static str] {
        &G2P_LANGUAGES
    }

    /// Most tokens one inference may carry for `voice`.
    fn max_tokens(&self, voice: &str) -> usize {
        self.packs
            .get(voice)
            .copied()
            // A voice missing from the file fails in the engine with a clear
            // error; until then, plan for the smallest pack.
            .or_else(|| self.packs.values().copied().min())
            .unwrap_or(FALLBACK_PACK_LEN)
    }

    fn warn_if_no_g2p(&self, voice: &str) {
        let language = voice_language(voice);
        if G2P_LANGUAGES.contains(&language) {
            return;
        }
        let first_time = self
            .warned_languages
            .lock()
            .map(|mut seen| seen.insert(language))
            .unwrap_or(false);
        if first_time {
            tracing::warn!(
                voice,
                language,
                "kokoro has no G2P for this language; text is read with English pronunciation"
            );
        }
    }

    /// Synthesize `text`, already prepared, as a sequence of chunks that each
    /// fit the voice pack.
    async fn synth_chunked(&self, text: &str, voice: &str) -> Result<Vec<f32>, VoxError> {
        let max_tokens = self.max_tokens(voice);
        // English phonemizes to a little over one token per character, so
        // half the token budget in characters rarely needs a second split.
        let mut queue: VecDeque<String> =
            normalize::split_text(text, (max_tokens / 2).max(1)).into();
        let mut samples = Vec::new();

        while let Some(chunk) = queue.pop_front() {
            let gate = Arc::clone(&SYNTH_GATE).lock_owned().await;

            // The only way to learn the token count is to phonemize, which
            // the engine then repeats. That costs one extra G2P pass per
            // chunk; an oversized chunk would panic inside the crate.
            //
            // The guard goes into the blocking task and comes back with the
            // result. If this future is dropped while the task runs, the
            // gate stays held until the G2P has finished.
            let engine = Arc::clone(&self.engine);
            let counted = chunk.clone();
            let (_gate, tokens) =
                tokio::task::spawn_blocking(move || (gate, engine.token_count(&counted)))
                    .await
                    .map_err(|e| VoxError::Tts(format!("kokoro phonemization task failed: {e}")))?;
            let tokens = tokens?;

            if tokens > max_tokens {
                let chars = chunk.chars().count();
                if chars < 2 {
                    return Err(VoxError::Tts(format!(
                        "kokoro cannot fit {chunk:?} ({tokens} tokens) in a voice pack of {max_tokens}"
                    )));
                }
                for part in normalize::split_text(&chunk, chars / 2).into_iter().rev() {
                    queue.push_front(part);
                }
                continue;
            }
            // Two tokens are the padding alone: nothing to say.
            if tokens <= 2 {
                continue;
            }

            samples.extend(self.engine.synth(&chunk, voice, self.config.speed).await?);
        }
        Ok(samples)
    }

    /// Parse a voice name string into a Kokoro Voice enum variant.
    fn parse_voice(name: &str, speed: f32) -> Result<KokoroVoice, VoxError> {
        match name.to_lowercase().as_str() {
            // American female (v1.0)
            "af_heart" => Ok(KokoroVoice::AfHeart(speed)),
            "af_sky" => Ok(KokoroVoice::AfSky(speed)),
            "af_bella" => Ok(KokoroVoice::AfBella(speed)),
            "af_sarah" => Ok(KokoroVoice::AfSarah(speed)),
            "af_nicole" => Ok(KokoroVoice::AfNicole(speed)),
            "af_nova" => Ok(KokoroVoice::AfNova(speed)),
            "af_river" => Ok(KokoroVoice::AfRiver(speed)),
            "af_alloy" => Ok(KokoroVoice::AfAlloy(speed)),
            "af_aoede" => Ok(KokoroVoice::AfAoede(speed)),
            "af_jessica" => Ok(KokoroVoice::AfJessica(speed)),
            "af_kore" => Ok(KokoroVoice::AfKore(speed)),
            // American male (v1.0)
            "am_adam" => Ok(KokoroVoice::AmAdam(speed)),
            "am_echo" => Ok(KokoroVoice::AmEcho(speed)),
            "am_eric" => Ok(KokoroVoice::AmEric(speed)),
            "am_liam" => Ok(KokoroVoice::AmLiam(speed)),
            "am_michael" => Ok(KokoroVoice::AmMichael(speed)),
            "am_onyx" => Ok(KokoroVoice::AmOnyx(speed)),
            "am_puck" => Ok(KokoroVoice::AmPuck(speed)),
            "am_fenrir" => Ok(KokoroVoice::AmFenrir(speed)),
            "am_santa" => Ok(KokoroVoice::AmSanta(speed)),
            // British female (v1.0)
            "bf_alice" => Ok(KokoroVoice::BfAlice(speed)),
            "bf_emma" => Ok(KokoroVoice::BfEmma(speed)),
            "bf_isabella" => Ok(KokoroVoice::BfIsabella(speed)),
            "bf_lily" => Ok(KokoroVoice::BfLily(speed)),
            // British male (v1.0)
            "bm_daniel" => Ok(KokoroVoice::BmDaniel(speed)),
            "bm_fable" => Ok(KokoroVoice::BmFable(speed)),
            "bm_george" => Ok(KokoroVoice::BmGeorge(speed)),
            "bm_lewis" => Ok(KokoroVoice::BmLewis(speed)),
            // Japanese (v1.0)
            "jf_alpha" => Ok(KokoroVoice::JfAlpha(speed)),
            "jf_gongitsune" => Ok(KokoroVoice::JfGongitsune(speed)),
            "jf_tebukuro" => Ok(KokoroVoice::JfTebukuro(speed)),
            "jf_nezumi" => Ok(KokoroVoice::JfNezumi(speed)),
            "jm_kumo" => Ok(KokoroVoice::JmKumo(speed)),
            // Chinese (v1.0)
            "zf_xiaobei" => Ok(KokoroVoice::ZfXiaobei(speed)),
            "zf_xiaoni" => Ok(KokoroVoice::ZfXiaoni(speed)),
            "zf_xiaoxiao" => Ok(KokoroVoice::ZfXiaoxiao(speed)),
            "zf_xiaoyi" => Ok(KokoroVoice::ZfXiaoyi(speed)),
            "zm_yunjian" => Ok(KokoroVoice::ZmYunjian(speed)),
            "zm_yunxi" => Ok(KokoroVoice::ZmYunxi(speed)),
            "zm_yunxia" => Ok(KokoroVoice::ZmYunxia(speed)),
            "zm_yunyang" => Ok(KokoroVoice::ZmYunyang(speed)),
            // Spanish (v1.0)
            "ef_dora" => Ok(KokoroVoice::EfDora(speed)),
            "em_alex" => Ok(KokoroVoice::EmAlex(speed)),
            "em_santa" => Ok(KokoroVoice::EmSanta(speed)),
            // French (v1.0)
            "ff_siwis" => Ok(KokoroVoice::FfSiwis(speed)),
            // Hindi (v1.0)
            "hf_alpha" => Ok(KokoroVoice::HfAlpha(speed)),
            "hf_beta" => Ok(KokoroVoice::HfBeta(speed)),
            "hm_omega" => Ok(KokoroVoice::HmOmega(speed)),
            "hm_psi" => Ok(KokoroVoice::HmPsi(speed)),
            // Italian (v1.0)
            "if_sara" => Ok(KokoroVoice::IfSara(speed)),
            "im_nicola" => Ok(KokoroVoice::ImNicola(speed)),
            // Portuguese (v1.0)
            "pf_dora" => Ok(KokoroVoice::PfDora(speed)),
            "pm_alex" => Ok(KokoroVoice::PmAlex(speed)),
            "pm_santa" => Ok(KokoroVoice::PmSanta(speed)),
            // These ship in the v1.1 voice pack and need the v1.1 model, which
            // has different input names. Vox loads the v1.0 model.
            v if V11_ONLY_VOICES.contains(&v) => Err(VoxError::Tts(format!(
                "voice {name} needs the Kokoro v1.1 model; vox ships v1.0"
            ))),
            _ => Err(VoxError::Tts(format!("unknown voice: {name}"))),
        }
    }

    /// Build the full list of supported named voices with metadata.
    ///
    /// Voices outside English and Chinese are labelled "(English phonemes)":
    /// the bundled G2P has no rules for those languages, so their text is
    /// read with English pronunciation in that voice's timbre.
    fn voice_list() -> Vec<VoiceInfo> {
        vec![
            // American English female
            VoiceInfo {
                id: "af_heart".into(),
                name: "Heart".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_alloy".into(),
                name: "Alloy".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_aoede".into(),
                name: "Aoede".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_bella".into(),
                name: "Bella".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_jessica".into(),
                name: "Jessica".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_kore".into(),
                name: "Kore".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_nicole".into(),
                name: "Nicole".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_nova".into(),
                name: "Nova".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_river".into(),
                name: "River".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_sarah".into(),
                name: "Sarah".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "af_sky".into(),
                name: "Sky".into(),
                gender: "female".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // American English male
            VoiceInfo {
                id: "am_adam".into(),
                name: "Adam".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_echo".into(),
                name: "Echo".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_eric".into(),
                name: "Eric".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_fenrir".into(),
                name: "Fenrir".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_liam".into(),
                name: "Liam".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_michael".into(),
                name: "Michael".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_onyx".into(),
                name: "Onyx".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_puck".into(),
                name: "Puck".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            VoiceInfo {
                id: "am_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "en-US".into(),
                accent: "American".into(),
            },
            // British English female
            VoiceInfo {
                id: "bf_alice".into(),
                name: "Alice".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_emma".into(),
                name: "Emma".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_isabella".into(),
                name: "Isabella".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bf_lily".into(),
                name: "Lily".into(),
                gender: "female".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // British English male
            VoiceInfo {
                id: "bm_daniel".into(),
                name: "Daniel".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_fable".into(),
                name: "Fable".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_george".into(),
                name: "George".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            VoiceInfo {
                id: "bm_lewis".into(),
                name: "Lewis".into(),
                gender: "male".into(),
                language: "en-GB".into(),
                accent: "British".into(),
            },
            // Japanese
            VoiceInfo {
                id: "jf_alpha".into(),
                name: "Alpha".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese (English phonemes)".into(),
            },
            VoiceInfo {
                id: "jf_gongitsune".into(),
                name: "Gongitsune".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese (English phonemes)".into(),
            },
            VoiceInfo {
                id: "jf_tebukuro".into(),
                name: "Tebukuro".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese (English phonemes)".into(),
            },
            VoiceInfo {
                id: "jf_nezumi".into(),
                name: "Nezumi".into(),
                gender: "female".into(),
                language: "ja".into(),
                accent: "Japanese (English phonemes)".into(),
            },
            VoiceInfo {
                id: "jm_kumo".into(),
                name: "Kumo".into(),
                gender: "male".into(),
                language: "ja".into(),
                accent: "Japanese (English phonemes)".into(),
            },
            // Chinese
            VoiceInfo {
                id: "zf_xiaobei".into(),
                name: "Xiaobei".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoni".into(),
                name: "Xiaoni".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoxiao".into(),
                name: "Xiaoxiao".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zf_xiaoyi".into(),
                name: "Xiaoyi".into(),
                gender: "female".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunjian".into(),
                name: "Yunjian".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunxi".into(),
                name: "Yunxi".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunxia".into(),
                name: "Yunxia".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            VoiceInfo {
                id: "zm_yunyang".into(),
                name: "Yunyang".into(),
                gender: "male".into(),
                language: "zh".into(),
                accent: "Chinese".into(),
            },
            // Spanish
            VoiceInfo {
                id: "ef_dora".into(),
                name: "Dora".into(),
                gender: "female".into(),
                language: "es".into(),
                accent: "Spanish (English phonemes)".into(),
            },
            VoiceInfo {
                id: "em_alex".into(),
                name: "Alex".into(),
                gender: "male".into(),
                language: "es".into(),
                accent: "Spanish (English phonemes)".into(),
            },
            VoiceInfo {
                id: "em_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "es".into(),
                accent: "Spanish (English phonemes)".into(),
            },
            // French
            VoiceInfo {
                id: "ff_siwis".into(),
                name: "Siwis".into(),
                gender: "female".into(),
                language: "fr".into(),
                accent: "French (English phonemes)".into(),
            },
            // Hindi
            VoiceInfo {
                id: "hf_alpha".into(),
                name: "Alpha".into(),
                gender: "female".into(),
                language: "hi".into(),
                accent: "Hindi (English phonemes)".into(),
            },
            VoiceInfo {
                id: "hf_beta".into(),
                name: "Beta".into(),
                gender: "female".into(),
                language: "hi".into(),
                accent: "Hindi (English phonemes)".into(),
            },
            VoiceInfo {
                id: "hm_omega".into(),
                name: "Omega".into(),
                gender: "male".into(),
                language: "hi".into(),
                accent: "Hindi (English phonemes)".into(),
            },
            VoiceInfo {
                id: "hm_psi".into(),
                name: "Psi".into(),
                gender: "male".into(),
                language: "hi".into(),
                accent: "Hindi (English phonemes)".into(),
            },
            // Italian
            VoiceInfo {
                id: "if_sara".into(),
                name: "Sara".into(),
                gender: "female".into(),
                language: "it".into(),
                accent: "Italian (English phonemes)".into(),
            },
            VoiceInfo {
                id: "im_nicola".into(),
                name: "Nicola".into(),
                gender: "male".into(),
                language: "it".into(),
                accent: "Italian (English phonemes)".into(),
            },
            // Portuguese (Brazilian)
            VoiceInfo {
                id: "pf_dora".into(),
                name: "Dora".into(),
                gender: "female".into(),
                language: "pt-BR".into(),
                accent: "Brazilian (English phonemes)".into(),
            },
            VoiceInfo {
                id: "pm_alex".into(),
                name: "Alex".into(),
                gender: "male".into(),
                language: "pt-BR".into(),
                accent: "Brazilian (English phonemes)".into(),
            },
            VoiceInfo {
                id: "pm_santa".into(),
                name: "Santa".into(),
                gender: "male".into(),
                language: "pt-BR".into(),
                accent: "Brazilian (English phonemes)".into(),
            },
        ]
    }
}

#[async_trait]
impl TtsBackend for KokoroBackend {
    async fn synthesize(&self, request: &TtsRequest) -> Result<TtsOutput, VoxError> {
        if request.seed.is_some() {
            tracing::debug!("kokoro backend does not support seed; ignoring");
        }

        let voice = request
            .voice
            .as_deref()
            .unwrap_or(&self.config.default_voice)
            .to_lowercase();
        // Reject an unknown voice before doing any work.
        Self::parse_voice(&voice, self.config.speed)?;
        self.warn_if_no_g2p(&voice);

        let prepared = prepare_text(&request.text, &voice)?;
        if prepared.unsupported_count > 0 {
            tracing::warn!(
                voice = %voice,
                skipped = prepared.unsupported_count,
                characters = %prepared.unsupported.iter().collect::<String>(),
                "kokoro G2P cannot read some characters; they were left out"
            );
        }

        let samples = self.synth_chunked(&prepared.text, &voice).await?;
        Ok(TtsOutput::from_mono(samples, KOKORO_SAMPLE_RATE))
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        let all = Self::voice_list();
        if self.packs.is_empty() {
            return all;
        }
        // Only offer voices the loaded file can actually serve.
        let present: Vec<VoiceInfo> = all
            .iter()
            .filter(|v| self.packs.contains_key(&v.id))
            .cloned()
            .collect();
        if present.is_empty() { all } else { present }
    }

    fn backend_name(&self) -> &str {
        "kokoro"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Samples the fake engine emits per input character.
    const SAMPLES_PER_CHAR: usize = 100;

    /// Stand-in for the ONNX engine.
    ///
    /// Like the real one, it cannot synthesize more tokens than the pack
    /// holds; where the real one panics, this returns an error the tests
    /// treat as a failure. It also tracks how many calls overlap.
    struct FakeEngine {
        pack_len: usize,
        tokens_per_char: usize,
        delay: Duration,
        active: AtomicUsize,
        max_active: AtomicUsize,
        chunks: StdMutex<Vec<String>>,
    }

    impl FakeEngine {
        fn new(pack_len: usize, tokens_per_char: usize) -> Self {
            Self {
                pack_len,
                tokens_per_char,
                delay: Duration::ZERO,
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
                chunks: StdMutex::new(Vec::new()),
            }
        }

        fn enter(&self) {
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now, Ordering::SeqCst);
        }

        fn leave(&self) {
            self.active.fetch_sub(1, Ordering::SeqCst);
        }

        fn tokens(&self, text: &str) -> usize {
            text.chars().count() * self.tokens_per_char + 2
        }
    }

    #[async_trait]
    impl SynthEngine for FakeEngine {
        fn token_count(&self, text: &str) -> Result<usize, VoxError> {
            self.enter();
            std::thread::sleep(self.delay);
            self.leave();
            Ok(self.tokens(text))
        }

        async fn synth(&self, text: &str, _voice: &str, _speed: f32) -> Result<Vec<f32>, VoxError> {
            self.enter();
            tokio::time::sleep(self.delay).await;
            self.leave();
            if self.tokens(text) > self.pack_len {
                return Err(VoxError::Tts(format!(
                    "chunk of {} tokens exceeds pack of {}: the real engine panics here",
                    self.tokens(text),
                    self.pack_len
                )));
            }
            self.chunks.lock().unwrap().push(text.to_string());
            Ok(vec![0.1; text.chars().count() * SAMPLES_PER_CHAR])
        }
    }

    fn backend(engine: Arc<FakeEngine>, packs: &[(&str, usize)]) -> KokoroBackend {
        KokoroBackend::from_parts(
            engine,
            KokoroConfig {
                model_path: PathBuf::new(),
                voices_path: PathBuf::new(),
                default_voice: "af_heart".into(),
                speed: 1.0,
            },
            packs.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        )
    }

    fn request(text: &str, voice: Option<&str>) -> TtsRequest {
        TtsRequest {
            text: text.to_string(),
            voice: voice.map(str::to_string),
            seed: None,
        }
    }

    fn encode_varint(out: &mut Vec<u8>, n: usize) {
        match n {
            0..=250 => out.push(n as u8),
            251..=65_535 => {
                out.push(251);
                out.extend((n as u16).to_le_bytes());
            }
            _ => {
                out.push(252);
                out.extend((n as u32).to_le_bytes());
            }
        }
    }

    /// Encode a voices file the way bincode's standard config does:
    /// `(name, pack length, style vector length)` per voice.
    fn encode_voices(voices: &[(&str, usize, usize)]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_varint(&mut out, voices.len());
        for &(name, pack_len, style_dim) in voices {
            encode_varint(&mut out, name.len());
            out.extend(name.as_bytes());
            encode_varint(&mut out, pack_len);
            for i in 0..pack_len {
                encode_varint(&mut out, 1);
                encode_varint(&mut out, style_dim);
                for _ in 0..style_dim {
                    out.extend((i as f32).to_le_bytes());
                }
            }
        }
        out
    }

    #[test]
    fn pack_lens_are_read_from_the_voices_file() {
        // 510 and 256 need the multi-byte length form, as in the real file.
        let bytes = encode_voices(&[("af_heart", 510, 256), ("bm_george", 3, 4)]);
        let packs = parse_pack_lens(&bytes).unwrap();
        assert_eq!(packs.len(), 2);
        assert_eq!(packs["af_heart"], 510);
        assert_eq!(packs["bm_george"], 3);
    }

    #[test]
    fn pack_len_parser_rejects_garbage() {
        let bytes = encode_voices(&[("af_heart", 510, 256)]);
        assert!(parse_pack_lens(&bytes[..bytes.len() - 1]).is_err());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(parse_pack_lens(&padded).is_err());
        assert!(parse_pack_lens(&[]).is_err());
        assert!(parse_pack_lens(&[0]).is_err());
        // A zip header: the NumPy archive some Kokoro guides distribute.
        assert!(parse_pack_lens(b"PK\x03\x04\x14\x00\x00\x00").is_err());
    }

    #[test]
    fn max_tokens_comes_from_the_pack() {
        let engine = Arc::new(FakeEngine::new(0, 1));
        let b = backend(engine.clone(), &[("af_heart", 510), ("bm_george", 120)]);
        assert_eq!(b.max_tokens("af_heart"), 510);
        assert_eq!(b.max_tokens("bm_george"), 120);
        // Not in the file: plan for the smallest pack.
        assert_eq!(b.max_tokens("am_adam"), 120);
        // File unreadable: documented fallback.
        assert_eq!(
            backend(engine, &[]).max_tokens("af_heart"),
            FALLBACK_PACK_LEN
        );
    }

    #[tokio::test]
    async fn long_text_is_split_into_chunks_that_fit_the_pack() {
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank. ";
        let text = sentence.repeat(80);
        assert!(text.len() >= 5000);

        // Three tokens per character: the first split, sized in characters, is
        // always too optimistic, so the token check has to split again.
        let engine = Arc::new(FakeEngine::new(510, 3));
        let b = backend(engine.clone(), &[("af_heart", 510)]);
        let out = b.synthesize(&request(&text, None)).await.unwrap();

        let chunks = engine.chunks.lock().unwrap().clone();
        assert!(
            chunks.len() > 1,
            "expected several chunks, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            assert!(engine.tokens(chunk) <= 510, "oversized chunk: {chunk:?}");
        }
        // Every word is spoken exactly once, in order.
        assert_eq!(
            chunks.join(" ").split_whitespace().collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
        let spoken: usize = chunks.iter().map(|c| c.chars().count()).sum();
        assert_eq!(out.audio.samples.len(), spoken * SAMPLES_PER_CHAR);
    }

    #[tokio::test]
    async fn unbroken_text_is_still_split() {
        let engine = Arc::new(FakeEngine::new(64, 3));
        let b = backend(engine.clone(), &[("af_heart", 64)]);
        let text = "a".repeat(5000);
        b.synthesize(&request(&text, None)).await.unwrap();
        let chunks = engine.chunks.lock().unwrap().clone();
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
    }

    #[tokio::test]
    async fn a_chunk_that_can_never_fit_is_an_error_not_a_panic() {
        // Pack too small for even one character.
        let engine = Arc::new(FakeEngine::new(4, 5));
        let b = backend(engine, &[("af_heart", 4)]);
        let err = b.synthesize(&request("hello", None)).await.unwrap_err();
        assert!(err.to_string().contains("cannot fit"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_run_one_at_a_time() {
        let mut engine = FakeEngine::new(510, 1);
        engine.delay = Duration::from_millis(15);
        let engine = Arc::new(engine);
        // Two backends, as the server builds when Kokoro serves both the
        // Speak tab and conversation: the gate is process-wide.
        let first = Arc::new(backend(engine.clone(), &[("af_heart", 510)]));
        let second = Arc::new(backend(engine.clone(), &[("af_heart", 510)]));

        let text = "One sentence here. Another sentence there. And a third one.";
        let tasks: Vec<_> = [first.clone(), second, first]
            .into_iter()
            .map(|b| {
                let req = request(text, None);
                tokio::spawn(async move { b.synthesize(&req).await })
            })
            .collect();
        for task in tasks {
            task.await.unwrap().unwrap();
        }

        assert_eq!(engine.chunks.lock().unwrap().len(), 3);
        assert_eq!(
            engine.max_active.load(Ordering::SeqCst),
            1,
            "G2P or inference overlapped"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_abandoned_request_keeps_the_gate_until_its_g2p_finishes() {
        let mut engine = FakeEngine::new(510, 1);
        engine.delay = Duration::from_millis(60);
        let engine = Arc::new(engine);
        let b = Arc::new(backend(engine.clone(), &[("af_heart", 510)]));

        // Start a request and drop it while its G2P is on the blocking pool.
        let abandoned = {
            let b = b.clone();
            tokio::spawn(async move { b.synthesize(&request("First request.", None)).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        abandoned.abort();
        let _ = abandoned.await;

        // The next request must wait for that G2P instead of overlapping it.
        b.synthesize(&request("Second request.", None))
            .await
            .unwrap();
        assert_eq!(
            engine.max_active.load(Ordering::SeqCst),
            1,
            "G2P of an abandoned request overlapped the next one"
        );
        assert_eq!(*engine.chunks.lock().unwrap(), ["Second request."]);
    }

    #[tokio::test]
    async fn duration_matches_sample_count() {
        let engine = Arc::new(FakeEngine::new(510, 1));
        let b = backend(engine, &[("af_heart", 510)]);
        let out = b.synthesize(&request("Hello there.", None)).await.unwrap();
        let samples = out.audio.samples.len() as u64;
        assert_eq!(samples, 12 * SAMPLES_PER_CHAR as u64);
        assert_eq!(
            out.duration_ms,
            samples * 1000 / u64::from(KOKORO_SAMPLE_RATE)
        );
        assert_eq!(out.audio.sample_rate, KOKORO_SAMPLE_RATE);
        assert_eq!(out.audio.channels, 1);
    }

    #[tokio::test]
    async fn numbers_are_spelled_out_before_synthesis() {
        let engine = Arc::new(FakeEngine::new(510, 1));
        let b = backend(engine.clone(), &[]);
        b.synthesize(&request("It is 3° and 25% humid, Dr. Lee.", None))
            .await
            .unwrap();
        assert_eq!(
            engine.chunks.lock().unwrap().concat(),
            "It is three degrees and twenty-five percent humid, Doctor Lee."
        );
    }

    #[tokio::test]
    async fn empty_text_yields_empty_audio() {
        let engine = Arc::new(FakeEngine::new(510, 1));
        let b = backend(engine.clone(), &[]);
        let out = b.synthesize(&request("  \n ", None)).await.unwrap();
        assert!(out.audio.samples.is_empty());
        assert_eq!(out.duration_ms, 0);
        assert!(engine.chunks.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unknown_and_v11_voices_are_rejected() {
        let engine = Arc::new(FakeEngine::new(510, 1));
        let b = backend(engine.clone(), &[]);
        let err = b
            .synthesize(&request("hi", Some("nobody")))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown voice"), "{err}");
        for voice in V11_ONLY_VOICES {
            let err = b.synthesize(&request("hi", Some(voice))).await.unwrap_err();
            assert!(err.to_string().contains("v1.1"), "{err}");
        }
        assert!(engine.chunks.lock().unwrap().is_empty());
    }

    #[test]
    fn voice_list_matches_what_can_be_synthesized() {
        let voices = KokoroBackend::voice_list();
        for v in &voices {
            assert!(
                KokoroBackend::parse_voice(&v.id, 1.0).is_ok(),
                "listed voice {} cannot be parsed",
                v.id
            );
            assert!(!V11_ONLY_VOICES.contains(&v.id.as_str()));
            // Language metadata agrees with the id prefix.
            assert!(
                v.language.to_lowercase().starts_with(voice_language(&v.id)),
                "{}: {} vs {}",
                v.id,
                v.language,
                voice_language(&v.id)
            );
            // Voices without G2P say so.
            let has_g2p = G2P_LANGUAGES.contains(&voice_language(&v.id));
            assert_eq!(v.accent.contains("English phonemes"), !has_g2p, "{}", v.id);
        }
        let mut ids: Vec<_> = voices.iter().map(|v| v.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), voices.len(), "duplicate voice ids");
    }

    #[test]
    fn list_voices_is_limited_to_the_loaded_file() {
        let engine = Arc::new(FakeEngine::new(510, 1));
        let b = backend(engine.clone(), &[("af_heart", 510), ("bm_george", 510)]);
        let ids: Vec<_> = b.list_voices().into_iter().map(|v| v.id).collect();
        assert_eq!(ids, ["af_heart", "bm_george"]);
        // Unknown file contents: fall back to the static list.
        assert_eq!(
            backend(engine, &[]).list_voices().len(),
            KokoroBackend::voice_list().len()
        );
    }

    #[test]
    fn prepare_folds_latin_and_reports_the_rest() {
        // Turkish letters outside Latin-1 are folded, not dropped.
        let p = prepare_text("Işık ağır şey", "af_heart").unwrap();
        assert_eq!(p.text, "Isik agir sey");
        assert_eq!(p.unsupported_count, 0);

        // Mixed scripts: the readable part survives, the rest is reported.
        let p = prepare_text("Hello мир", "af_heart").unwrap();
        assert_eq!(p.text, "Hello");
        assert_eq!(p.unsupported_count, 3);
        assert_eq!(p.unsupported, ['м', 'и', 'р']);

        // Nothing readable at all is an error.
        let err = prepare_text("こんにちは", "jf_alpha").unwrap_err();
        assert!(err.to_string().contains("cannot speak"), "{err}");

        // Chinese is supported and keeps its digits for the crate's own rules.
        let p = prepare_text("你好 3", "zf_xiaoxiao").unwrap();
        assert_eq!(p.text, "你好 3");
        assert_eq!(p.unsupported_count, 0);

        // Other voices never pass digits through.
        let p = prepare_text("Tengo 3 gatos", "ef_dora").unwrap();
        assert_eq!(p.text, "Tengo three gatos");
    }
}
