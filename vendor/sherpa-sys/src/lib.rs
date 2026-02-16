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

// ---------------------------------------------------------------------------
// Online/Streaming Recognition
// ---------------------------------------------------------------------------

/// SherpaOnnxOnlineTransducerModelConfig — 24 bytes
#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineTransducerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
    pub joiner: *const c_char,
}

impl Default for SherpaOnnxOnlineTransducerModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
            joiner: std::ptr::null(),
        }
    }
}

/// SherpaOnnxOnlineParaformerModelConfig — 16 bytes
#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineParaformerModelConfig {
    pub encoder: *const c_char,
    pub decoder: *const c_char,
}

impl Default for SherpaOnnxOnlineParaformerModelConfig {
    fn default() -> Self {
        Self {
            encoder: std::ptr::null(),
            decoder: std::ptr::null(),
        }
    }
}

/// SherpaOnnxOnlineZipformer2CtcModelConfig — 8 bytes
#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineZipformer2CtcModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOnlineZipformer2CtcModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

/// SherpaOnnxOnlineNemoCtcModelConfig — 8 bytes
#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineNemoCtcModelConfig {
    pub model: *const c_char,
}

impl Default for SherpaOnnxOnlineNemoCtcModelConfig {
    fn default() -> Self {
        Self {
            model: std::ptr::null(),
        }
    }
}

// ---------------------------------------------------------------------------
// SherpaOnnxOnlineModelConfig — 128 bytes (v1.12.24)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineModelConfig {
    pub transducer: SherpaOnnxOnlineTransducerModelConfig,
    pub paraformer: SherpaOnnxOnlineParaformerModelConfig,
    pub zipformer2_ctc: SherpaOnnxOnlineZipformer2CtcModelConfig,
    pub tokens: *const c_char,
    pub num_threads: c_int,
    pub provider: *const c_char,
    pub debug: c_int,
    pub model_type: *const c_char,
    pub modeling_unit: *const c_char,
    pub bpe_vocab: *const c_char,
    pub tokens_buf: *const c_char,
    pub tokens_buf_size: c_int,
    pub nemo_ctc: SherpaOnnxOnlineNemoCtcModelConfig,
}

impl Default for SherpaOnnxOnlineModelConfig {
    fn default() -> Self {
        Self {
            transducer: SherpaOnnxOnlineTransducerModelConfig::default(),
            paraformer: SherpaOnnxOnlineParaformerModelConfig::default(),
            zipformer2_ctc: SherpaOnnxOnlineZipformer2CtcModelConfig::default(),
            tokens: std::ptr::null(),
            num_threads: 0,
            provider: std::ptr::null(),
            debug: 0,
            model_type: std::ptr::null(),
            modeling_unit: std::ptr::null(),
            bpe_vocab: std::ptr::null(),
            tokens_buf: std::ptr::null(),
            tokens_buf_size: 0,
            nemo_ctc: SherpaOnnxOnlineNemoCtcModelConfig::default(),
        }
    }
}

/// SherpaOnnxOnlineCtcFstDecoderConfig — 16 bytes
#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineCtcFstDecoderConfig {
    pub graph: *const c_char,
    pub max_active: c_int,
}

impl Default for SherpaOnnxOnlineCtcFstDecoderConfig {
    fn default() -> Self {
        Self {
            graph: std::ptr::null(),
            max_active: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SherpaOnnxOnlineRecognizerConfig — 264 bytes (v1.12.24)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineRecognizerConfig {
    pub feat_config: SherpaOnnxFeatureConfig,
    pub model_config: SherpaOnnxOnlineModelConfig,
    pub decoding_method: *const c_char,
    pub max_active_paths: c_int,
    pub enable_endpoint: c_int,
    pub rule1_min_trailing_silence: c_float,
    pub rule2_min_trailing_silence: c_float,
    pub rule3_min_utterance_length: c_float,
    pub hotwords_file: *const c_char,
    pub hotwords_score: c_float,
    pub ctc_fst_decoder_config: SherpaOnnxOnlineCtcFstDecoderConfig,
    pub rule_fsts: *const c_char,
    pub rule_fars: *const c_char,
    pub blank_penalty: c_float,
    pub hotwords_buf: *const c_char,
    pub hotwords_buf_size: c_int,
    pub hr: SherpaOnnxHomophoneReplacerConfig,
}

impl Default for SherpaOnnxOnlineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: SherpaOnnxFeatureConfig::default(),
            model_config: SherpaOnnxOnlineModelConfig::default(),
            decoding_method: std::ptr::null(),
            max_active_paths: 0,
            enable_endpoint: 0,
            rule1_min_trailing_silence: 0.0,
            rule2_min_trailing_silence: 0.0,
            rule3_min_utterance_length: 0.0,
            hotwords_file: std::ptr::null(),
            hotwords_score: 0.0,
            ctc_fst_decoder_config: SherpaOnnxOnlineCtcFstDecoderConfig::default(),
            rule_fsts: std::ptr::null(),
            rule_fars: std::ptr::null(),
            blank_penalty: 0.0,
            hotwords_buf: std::ptr::null(),
            hotwords_buf_size: 0,
            hr: SherpaOnnxHomophoneReplacerConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// SherpaOnnxOnlineRecognizerResult — 48 bytes
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct SherpaOnnxOnlineRecognizerResult {
    pub text: *const c_char,
    pub tokens: *const c_char,
    pub tokens_arr: *const *const c_char,
    pub timestamps: *mut c_float,
    pub count: c_int,
    pub json: *const c_char,
}

// ---------------------------------------------------------------------------
// Opaque handles for online recognition
// ---------------------------------------------------------------------------

pub enum SherpaOnnxOnlineRecognizer {}
pub enum SherpaOnnxOnlineStream {}

// ---------------------------------------------------------------------------
// Compile-time struct size assertions (catches layout drift)
//
// Note: size_of catches missing/extra fields but NOT reordered fields at the
// same total size. A field swap would silently corrupt data. Full protection
// requires CI bindgen diffing, but this catches the most common mistake
// (adding a field upstream that we don't have).
// ---------------------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<SherpaOnnxOnlineModelConfig>() == 128);
const _: () = assert!(std::mem::size_of::<SherpaOnnxOnlineRecognizerConfig>() == 264);
const _: () = assert!(std::mem::size_of::<SherpaOnnxOnlineRecognizerResult>() == 48);

// ---------------------------------------------------------------------------
// Online FFI function declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn SherpaOnnxCreateOnlineRecognizer(
        config: *const SherpaOnnxOnlineRecognizerConfig,
    ) -> *const SherpaOnnxOnlineRecognizer;

    pub fn SherpaOnnxDestroyOnlineRecognizer(recognizer: *const SherpaOnnxOnlineRecognizer);

    pub fn SherpaOnnxCreateOnlineStream(
        recognizer: *const SherpaOnnxOnlineRecognizer,
    ) -> *const SherpaOnnxOnlineStream;

    pub fn SherpaOnnxCreateOnlineStreamWithHotwords(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        hotwords: *const c_char,
    ) -> *const SherpaOnnxOnlineStream;

    pub fn SherpaOnnxDestroyOnlineStream(stream: *const SherpaOnnxOnlineStream);

    pub fn SherpaOnnxOnlineStreamAcceptWaveform(
        stream: *const SherpaOnnxOnlineStream,
        sample_rate: c_int,
        samples: *const c_float,
        n: c_int,
    );

    pub fn SherpaOnnxOnlineStreamInputFinished(stream: *const SherpaOnnxOnlineStream);

    pub fn SherpaOnnxIsOnlineStreamReady(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    ) -> c_int;

    pub fn SherpaOnnxDecodeOnlineStream(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    );

    pub fn SherpaOnnxDecodeMultipleOnlineStreams(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        streams: *mut *const SherpaOnnxOnlineStream,
        n: c_int,
    );

    pub fn SherpaOnnxGetOnlineStreamResult(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    ) -> *const SherpaOnnxOnlineRecognizerResult;

    pub fn SherpaOnnxDestroyOnlineRecognizerResult(r: *const SherpaOnnxOnlineRecognizerResult);

    pub fn SherpaOnnxGetOnlineStreamResultAsJson(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    ) -> *const c_char;

    pub fn SherpaOnnxDestroyOnlineStreamResultJson(s: *const c_char);

    pub fn SherpaOnnxOnlineStreamReset(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    );

    pub fn SherpaOnnxOnlineStreamIsEndpoint(
        recognizer: *const SherpaOnnxOnlineRecognizer,
        stream: *const SherpaOnnxOnlineStream,
    ) -> c_int;
}
