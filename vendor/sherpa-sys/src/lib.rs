//! Raw FFI bindings to the sherpa-onnx C API (offline recognition).
//!
//! These bindings target the sherpa-onnx offline recognizer for
//! non-streaming speech-to-text (STT) using models like SenseVoice,
//! Zipformer transducer, Paraformer, or Whisper.
//!
//! Struct layouts match sherpa-onnx v1.12.24 C API (verified via bindgen).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_float, c_int};

// ---------------------------------------------------------------------------
// Config structs (all fields zero-initializable)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineTransducerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub joiner: *const c_char,
}

impl Default for SherpaOnnxOfflineTransducerModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
            joiner: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineParaformerModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOfflineParaformerModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineNemoEncDecCtcModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOfflineNemoEncDecCtcModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineWhisperModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub language: *const c_char,
    pub task: *const c_char,
    pub tail_paddings: c_int,
}

impl Default for SherpaOnnxOfflineWhisperModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
            language: std::ptr::null(),
            task: std::ptr::null(),
            tail_paddings: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineTdnnModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOfflineTdnnModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineSenseVoiceModelConfig {
    pub model: *const c_char,
    pub language: *const c_char,
    pub use_itn: c_int,
}

impl Default for SherpaOnnxOfflineSenseVoiceModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
            language: std::ptr::null(),
            use_itn: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineMoonshineModelConfig {
    pub preprocessor: *const c_char,
    pub encoder: *const c_char,
    pub uncached_decoder: *const c_char,
    pub cached_decoder: *const c_char,
}

impl Default for SherpaOnnxOfflineMoonshineModelConfig {
    fn default() -> Self {
        Self {
            preprocessor: std::ptr::null(),
            encoder: std::ptr::null(),
            uncached_decoder: std::ptr::null(),
            cached_decoder: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineFireRedAsrModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
}

impl Default for SherpaOnnxOfflineFireRedAsrModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineDolphinModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOfflineDolphinModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineZipformerCtcModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOfflineZipformerCtcModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineCanaryModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub src_lang: *const c_char,
    pub tgt_lang: *const c_char,
    pub use_pnc: c_int,
}

impl Default for SherpaOnnxOfflineCanaryModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
            src_lang: std::ptr::null(),
            tgt_lang: std::ptr::null(),
            use_pnc: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SherpaOnnxOfflineModelConfig - 272 bytes (v1.12.24)
//
// Field order MUST match the C header exactly. The previous bindings were
// missing nemo_ctc, fire_red_asr, dolphin, zipformer_ctc, and canary,
// and had whisper/tdnn before nemo_ctc instead of after.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineModelConfig {
    pub transducer: SherpaOnnxOfflineTransducerModelConfig,
    pub paraformer: SherpaOnnxOfflineParaformerModelConfig,
    pub nemo_ctc: SherpaOnnxOfflineNemoEncDecCtcModelConfig,
    pub whisper: SherpaOnnxOfflineWhisperModelConfig,
    pub tdnn: SherpaOnnxOfflineTdnnModelConfig,
    pub tokens: *const c_char,
    pub num_threads: c_int,
    pub debug: c_int,
    pub provider: *const c_char,
    pub model_type: *const c_char,
    pub modeling_unit: *const c_char,
    pub bpe_vocab: *const c_char,
    pub telespeech_ctc: *const c_char,
    pub sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig,
    pub moonshine: SherpaOnnxOfflineMoonshineModelConfig,
    pub fire_red_asr: SherpaOnnxOfflineFireRedAsrModelConfig,
    pub dolphin: SherpaOnnxOfflineDolphinModelConfig,
    pub zipformer_ctc: SherpaOnnxOfflineZipformerCtcModelConfig,
    pub canary: SherpaOnnxOfflineCanaryModelConfig,
}

impl Default for SherpaOnnxOfflineModelConfig {
    fn default() -> Self {
        Self {
            transducer: SherpaOnnxOfflineTransducerModelConfig::default(),
            paraformer: SherpaOnnxOfflineParaformerModelConfig::default(),
            nemo_ctc: SherpaOnnxOfflineNemoEncDecCtcModelConfig::default(),
            whisper: SherpaOnnxOfflineWhisperModelConfig::default(),
            tdnn: SherpaOnnxOfflineTdnnModelConfig::default(),
            tokens: std::ptr::null(),
            num_threads: 0,
            debug: 0,
            provider: std::ptr::null(),
            model_type: std::ptr::null(),
            modeling_unit: std::ptr::null(),
            bpe_vocab: std::ptr::null(),
            telespeech_ctc: std::ptr::null(),
            sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig::default(),
            moonshine: SherpaOnnxOfflineMoonshineModelConfig::default(),
            fire_red_asr: SherpaOnnxOfflineFireRedAsrModelConfig::default(),
            dolphin: SherpaOnnxOfflineDolphinModelConfig::default(),
            zipformer_ctc: SherpaOnnxOfflineZipformerCtcModelConfig::default(),
            canary: SherpaOnnxOfflineCanaryModelConfig::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineLMConfig {
    pub model: *const c_char,
    pub scale: c_float,
}

impl Default for SherpaOnnxOfflineLMConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
            scale: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct SherpaOnnxFeatureConfig {
    pub sample_rate: c_int,
    pub feature_dim: c_int,
}

// ---------------------------------------------------------------------------
// SherpaOnnxHomophoneReplacerConfig - new in v1.12.24
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxHomophoneReplacerConfig {
    pub dict_dir: *const c_char,
    pub lexicon: *const c_char,
    pub rule_fsts: *const c_char,
}

impl Default for SherpaOnnxHomophoneReplacerConfig {
    fn default() -> Self {
        Self {
            dict_dir: std::ptr::null(),
            lexicon: std::ptr::null(),
            rule_fsts: std::ptr::null(),
        }
    }
}

// ---------------------------------------------------------------------------
// SherpaOnnxOfflineRecognizerConfig - 376 bytes (v1.12.24)
//
// Adds the `hr` (homophone replacer) field at the end.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOfflineRecognizerConfig {
    pub feat_config: SherpaOnnxFeatureConfig,
    pub model_config: SherpaOnnxOfflineModelConfig,
    pub lm_config: SherpaOnnxOfflineLMConfig,
    pub decoding_method: *const c_char,
    pub max_active_paths: c_int,
    pub hotwords_file: *const c_char,
    pub hotwords_score: c_float,
    pub rule_fsts: *const c_char,
    pub rule_fars: *const c_char,
    pub blank_penalty: c_float,
    pub hr: SherpaOnnxHomophoneReplacerConfig,
}

impl Default for SherpaOnnxOfflineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: SherpaOnnxFeatureConfig::default(),
            model_config: SherpaOnnxOfflineModelConfig::default(),
            lm_config: SherpaOnnxOfflineLMConfig::default(),
            decoding_method: std::ptr::null(),
            max_active_paths: 0,
            hotwords_file: std::ptr::null(),
            hotwords_score: 0.0,
            rule_fsts: std::ptr::null(),
            rule_fars: std::ptr::null(),
            blank_penalty: 0.0,
            hr: SherpaOnnxHomophoneReplacerConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

pub enum SherpaOnnxOfflineRecognizer {}
pub enum SherpaOnnxOfflineStream {}

// ---------------------------------------------------------------------------
// SherpaOnnxOfflineRecognizerResult - 72 bytes (v1.12.24)
//
// Critical fixes:
//   - `tokens` is *const c_char (single string), NOT *const *const c_char
//   - `json` field added between `tokens_arr` and `lang`
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct SherpaOnnxOfflineRecognizerResult {
    pub text: *const c_char,
    pub timestamps: *const c_float,
    pub count: c_int,
    pub tokens: *const c_char,
    pub tokens_arr: *const *const c_char,
    pub json: *const c_char,
    pub lang: *const c_char,
    pub emotion: *const c_char,
    pub event: *const c_char,
}

// ---------------------------------------------------------------------------
// FFI function declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn SherpaOnnxCreateOfflineRecognizer(
        config: *const SherpaOnnxOfflineRecognizerConfig,
    ) -> *mut SherpaOnnxOfflineRecognizer;

    pub fn SherpaOnnxDestroyOfflineRecognizer(recognizer: *mut SherpaOnnxOfflineRecognizer);

    pub fn SherpaOnnxCreateOfflineStream(
        recognizer: *const SherpaOnnxOfflineRecognizer,
    ) -> *mut SherpaOnnxOfflineStream;

    pub fn SherpaOnnxDestroyOfflineStream(stream: *mut SherpaOnnxOfflineStream);

    pub fn SherpaOnnxAcceptWaveformOffline(
        stream: *mut SherpaOnnxOfflineStream,
        sample_rate: c_int,
        samples: *const c_float,
        n: c_int,
    );

    pub fn SherpaOnnxDecodeOfflineStream(
        recognizer: *const SherpaOnnxOfflineRecognizer,
        stream: *mut SherpaOnnxOfflineStream,
    );

    pub fn SherpaOnnxGetOfflineStreamResult(
        stream: *const SherpaOnnxOfflineStream,
    ) -> *const SherpaOnnxOfflineRecognizerResult;

    pub fn SherpaOnnxDestroyOfflineRecognizerResult(
        result: *const SherpaOnnxOfflineRecognizerResult,
    );
}
