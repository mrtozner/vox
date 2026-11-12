//! Handler for `vox models` and `vox doctor` — model registry, download,
//! integrity verification and first-run diagnostics.

use std::fmt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub use vox::paths::models_dir;

/// Metadata for a downloadable model.
pub struct ModelInfo {
    /// Human-readable name (used in `vox models download <name>`).
    pub name: &'static str,
    /// Filename on disk inside the models directory.
    pub filename: &'static str,
    /// Download URL, pinned to an immutable revision (commit hash or
    /// release tag) so the bytes behind it cannot change.
    pub url: &'static str,
    /// Exact file size in bytes.
    pub size_bytes: u64,
    /// Lower-case hex SHA-256 of the file.
    pub sha256: &'static str,
    /// Model kind label (VAD, STT, TTS, Diarization).
    pub kind: &'static str,
}

/// All known models that can be managed through the CLI.
///
/// Every entry is pinned: Hugging Face URLs use `resolve/<commit>/`, GitHub
/// URLs use a commit hash or a release tag, and each file carries its exact
/// size and SHA-256. Downloads are verified against both before they are
/// moved into place. To bump a model, update URL, size and hash together
/// (`cargo test --features cli --bin vox -- --ignored registry_urls`
/// checks every URL against the network).
///
/// Names: `whisper-<size>.en` are English-only, `whisper-<size>` without
/// `.en` are multilingual (Turkish, German, ...). Silero VAD is pinned to
/// the v5.1.2 release of snakers4/silero-vad.
pub const MODELS: &[ModelInfo] = &[
    // Voice activity detection — Silero VAD v5.1.2
    ModelInfo {
        name: "silero-vad",
        filename: "silero_vad.onnx",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/6478567951ae5c9979ad7b234185b5515f4be7a1/src/silero_vad/data/silero_vad.onnx",
        size_bytes: 2_327_524,
        sha256: "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
        kind: "VAD",
    },
    // Whisper (whisper.cpp GGML) — English-only
    ModelInfo {
        name: "whisper-tiny.en",
        filename: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en.bin",
        size_bytes: 77_704_715,
        sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-tiny.en-int8",
        filename: "ggml-tiny.en-int8.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q8_0.bin",
        size_bytes: 43_550_795,
        sha256: "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-base.en",
        filename: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin",
        size_bytes: 147_964_211,
        sha256: "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-base.en-int8",
        filename: "ggml-base.en-int8.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en-q8_0.bin",
        size_bytes: 81_781_811,
        sha256: "a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-small.en",
        filename: "ggml-small.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.en.bin",
        size_bytes: 487_614_201,
        sha256: "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-medium.en",
        filename: "ggml-medium.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium.en.bin",
        size_bytes: 1_533_774_781,
        sha256: "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
        kind: "STT",
    },
    // Whisper (whisper.cpp GGML) — multilingual
    ModelInfo {
        name: "whisper-tiny",
        filename: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin",
        size_bytes: 77_691_713,
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-base",
        filename: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin",
        size_bytes: 147_951_465,
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-small",
        filename: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin",
        size_bytes: 487_601_967,
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-small-q5",
        filename: "ggml-small-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin",
        size_bytes: 190_085_487,
        sha256: "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-medium",
        filename: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium.bin",
        size_bytes: 1_533_763_059,
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-medium-q5",
        filename: "ggml-medium-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium-q5_0.bin",
        size_bytes: 539_212_467,
        sha256: "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f",
        kind: "STT",
    },
    // Whisper large-v3-turbo — multilingual, most accurate, slow on CPU
    ModelInfo {
        name: "whisper-large-v3-turbo",
        filename: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin",
        size_bytes: 1_624_555_275,
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo-q5",
        filename: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin",
        size_bytes: 574_041_195,
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        kind: "STT",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo-q8",
        filename: "ggml-large-v3-turbo-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q8_0.bin",
        size_bytes: 874_188_075,
        sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        kind: "STT",
    },
    // Distil-Whisper (English-only GGML conversions published upstream)
    ModelInfo {
        name: "distil-whisper-small.en",
        filename: "ggml-distil-small.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-small.en/resolve/9e4a67ca4569c30be43a3fe7fba1621e504f0093/ggml-distil-small.en.bin",
        size_bytes: 336_191_657,
        sha256: "7691eb11167ab7aaf6b3e05d8266f2fd9ad89c550e433f86ac266ebdee6c970a",
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-medium.en",
        filename: "ggml-distil-medium.en.bin",
        url: "https://huggingface.co/distil-whisper/distil-medium.en/resolve/6e61418885eaf4d5cc9f64e508e80ac5b4c052b7/ggml-medium-32-2.en.bin",
        size_bytes: 794_018_180,
        sha256: "ad53ccb618188b210550e98cc32bf5a13188d86635e395bb11115ed275d6e7aa",
        kind: "STT",
    },
    ModelInfo {
        name: "distil-whisper-large-v3",
        filename: "ggml-distil-large-v3.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/0d78dd96ed9fc152325f63b53788fec3b43de031/ggml-distil-large-v3.bin",
        size_bytes: 1_519_521_155,
        sha256: "2883a11b90fb10ed592d826edeaee7d2929bf1ab985109fe9e1e7b4d2b69a298",
        kind: "STT",
    },
    // Kokoro TTS
    ModelInfo {
        name: "kokoro",
        filename: "kokoro-v1.0.onnx",
        url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        size_bytes: 325_532_387,
        sha256: "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
        kind: "TTS",
    },
    ModelInfo {
        name: "kokoro-int8",
        filename: "kokoro-v1.0.int8.onnx",
        url: "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/f46687f7e41512228ae953af24a11b2640ea0f22/onnx/model_quantized.onnx",
        size_bytes: 92_360_543,
        sha256: "0d55b15d4b735d61a21b0105136bc81b8768c4db94753193c19354fa863cd556",
        kind: "TTS",
    },
    ModelInfo {
        name: "kokoro-voices",
        filename: "voices.bin",
        // The bincode voices file kokoro-tts reads. The same-named file in
        // thewh1teagle/kokoro-onnx is a NumPy archive and does not load.
        url: "https://github.com/mzdk100/kokoro/releases/download/V1.0/voices.bin",
        size_bytes: 28_311_781,
        sha256: "988549ccf1d68a80a31de1b9d00ac59d5dceb69743cbc607d1de17c7c4cdccf4",
        kind: "TTS",
    },
    // Sherpa-ONNX SenseVoice (batch STT)
    ModelInfo {
        name: "sherpa-sensevoice",
        filename: "sensevoice/model.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/2365baeacb507f821a0c8120fcee3d484dba7a07/model.int8.onnx",
        size_bytes: 239_233_841,
        sha256: "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-sensevoice-tokens",
        filename: "sensevoice/tokens.txt",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/2365baeacb507f821a0c8120fcee3d484dba7a07/tokens.txt",
        size_bytes: 315_894,
        sha256: "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
        kind: "STT",
    },
    // Sherpa-ONNX streaming zipformer transducer
    ModelInfo {
        name: "sherpa-streaming-encoder",
        filename: "sherpa-streaming/encoder.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/672fbf1b30579d6585301139bb363f42a0ad4a24/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 71_083_163,
        sha256: "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-decoder",
        filename: "sherpa-streaming/decoder.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/672fbf1b30579d6585301139bb363f42a0ad4a24/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 1_307_236,
        sha256: "98da299f471e38bb4e1a8df579b8cc9122d6039576a77e357b3c60f17dd83b02",
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-joiner",
        filename: "sherpa-streaming/joiner.int8.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/672fbf1b30579d6585301139bb363f42a0ad4a24/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        size_bytes: 259_335,
        sha256: "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297",
        kind: "STT",
    },
    ModelInfo {
        name: "sherpa-streaming-tokens",
        filename: "sherpa-streaming/tokens.txt",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/672fbf1b30579d6585301139bb363f42a0ad4a24/tokens.txt",
        size_bytes: 5_048,
        sha256: "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
        kind: "STT",
    },
    // Piper TTS voices (model + config pairs)
    ModelInfo {
        name: "piper-en-us",
        filename: "piper/en_US-lessac-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        size_bytes: 63_201_294,
        sha256: "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-en-us-config",
        filename: "piper/en_US-lessac-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        size_bytes: 4_885,
        sha256: "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-de",
        filename: "piper/de_DE-thorsten-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        size_bytes: 63_221_984,
        sha256: "fd270eb3d1ebf91d763f8a0f949b1b1bebdfefaea18c3aaf8577dd34536e064f",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-de-config",
        filename: "piper/de_DE-thorsten-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
        size_bytes: 5_460,
        sha256: "cf245c91668e74cb6716f38647ae3d6d9bac42965872596f65f9ab2b7d7221ed",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-es",
        filename: "piper/es_ES-davefx-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        size_bytes: 63_221_984,
        sha256: "d2c9224e122321aac3c33343957fff4d0f465226cb74c66b0f520ca3f8c0dd07",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-es-config",
        filename: "piper/es_ES-davefx-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
        size_bytes: 5_458,
        sha256: "531c4051502e1acbc10b42437f11679bfce9e3919e1796e51fabd73e01e7fd2e",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-fr",
        filename: "piper/fr_FR-siwis-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
        size_bytes: 63_221_984,
        sha256: "b1d946fd68d0347155fa59bce0a018924abbef3c1561181db45ca463d83963ef",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-fr-config",
        filename: "piper/fr_FR-siwis-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
        size_bytes: 5_458,
        sha256: "df814aa3dfedd00510a07144442cc0a220ff21e43d03c3b078444fc0be68fc37",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-tr",
        filename: "piper/tr_TR-dfki-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx",
        size_bytes: 63_201_294,
        sha256: "2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-tr-config",
        filename: "piper/tr_TR-dfki-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json",
        size_bytes: 4_960,
        sha256: "13ebd7810f1b61b5027583cf3131a0a233b6ea81c38f2200ebc4ff41c3cca039",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-zh",
        filename: "piper/zh_CN-huayan-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        size_bytes: 63_201_294,
        sha256: "9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d",
        kind: "TTS",
    },
    ModelInfo {
        name: "piper-zh-config",
        filename: "piper/zh_CN-huayan-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
        size_bytes: 4_822,
        sha256: "d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b",
        kind: "TTS",
    },
    // Speaker embedding model (WeSpeaker ResNet34, VoxCeleb)
    ModelInfo {
        name: "speaker-encoder",
        filename: "speaker_encoder.onnx",
        url: "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx",
        size_bytes: 26_535_549,
        sha256: "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
        kind: "Diarization",
    },
];

/// Look up a registry entry by name.
pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

/// Look up a registry entry by its on-disk file name.
pub fn find_model_by_filename(filename: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.filename == filename)
}

// ============================================================================
// Integrity verification
// ============================================================================

/// Why a model file failed verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The file does not exist.
    Missing,
    /// The file size differs from the registry.
    Size { expected: u64, actual: u64 },
    /// The SHA-256 differs from the registry.
    Checksum { expected: String, actual: String },
    /// The file could not be read.
    Io(String),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::Missing => write!(f, "file not found"),
            VerifyError::Size { expected, actual } => write!(
                f,
                "size mismatch: file is {} ({actual} bytes), expected {} ({expected} bytes)",
                format_bytes(*actual),
                format_bytes(*expected),
            ),
            VerifyError::Checksum { expected, actual } => write!(
                f,
                "checksum mismatch: sha256 is {actual}, expected {expected}"
            ),
            VerifyError::Io(e) => write!(f, "cannot read file: {e}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Compute the lower-case hex SHA-256 of a file (streaming, constant memory).
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Cheap check: does the file exist with exactly the expected size?
pub fn check_size(path: &Path, expected_size: u64) -> Result<(), VerifyError> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(VerifyError::Missing),
        Err(e) => return Err(VerifyError::Io(e.to_string())),
    };
    if meta.len() != expected_size {
        return Err(VerifyError::Size {
            expected: expected_size,
            actual: meta.len(),
        });
    }
    Ok(())
}

/// Full check: exact size, then SHA-256.
pub fn verify_file(
    path: &Path,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), VerifyError> {
    check_size(path, expected_size)?;
    let actual = sha256_file(path).map_err(|e| VerifyError::Io(e.to_string()))?;
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        return Err(VerifyError::Checksum {
            expected: expected_sha256.to_ascii_lowercase(),
            actual,
        });
    }
    Ok(())
}

/// Path of the temporary file a download is streamed into
/// (`<dest>.part`, kept across runs so interrupted downloads can resume).
fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ============================================================================
// `vox models list`
// ============================================================================

/// Status column for one registry entry.
fn model_status(dir: &Path, model: &ModelInfo) -> String {
    let file_path = dir.join(model.filename);
    match check_size(&file_path, model.size_bytes) {
        Ok(()) => "downloaded".to_string(),
        Err(VerifyError::Size { actual, .. }) => format!(
            "size mismatch ({}; run: vox models download {} --force)",
            format_bytes(actual),
            model.name
        ),
        Err(VerifyError::Missing) => match std::fs::metadata(part_path(&file_path)) {
            Ok(part) => format!(
                "partial ({} of {}; re-run download to resume)",
                format_bytes(part.len()),
                format_bytes(model.size_bytes)
            ),
            Err(_) => "not downloaded".to_string(),
        },
        Err(other) => other.to_string(),
    }
}

/// Render the `vox models list` table for a models directory.
pub fn render_list(dir: &Path) -> String {
    use fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "Models directory: {}\n", dir.display());
    let _ = writeln!(out, "  {:<28} {:<12} {:<10} STATUS", "NAME", "KIND", "SIZE");
    let _ = writeln!(out, "  {}", "-".repeat(66));
    for model in MODELS {
        let _ = writeln!(
            out,
            "  {:<28} {:<12} {:<10} {}",
            model.name,
            model.kind,
            format_bytes(model.size_bytes),
            model_status(dir, model),
        );
    }
    let _ = writeln!(
        out,
        "\nWhisper models ending in .en are English-only; the others are multilingual\n\
         (use them with --language, e.g. `vox listen --model base --language tr`)."
    );
    out
}

/// List all available models, showing which are downloaded.
pub fn list() -> anyhow::Result<()> {
    print!("{}", render_list(&models_dir()));
    Ok(())
}

// ============================================================================
// `vox models download`
// ============================================================================

/// Download a model by name with a progress bar.
///
/// The file is streamed to `<dest>.part`; an interrupted download resumes
/// from where it stopped (HTTP range request). The finished file must match
/// the registry's exact size and SHA-256 before it is moved into place.
pub async fn download(name: &str, force: bool) -> anyhow::Result<()> {
    let model = find_model(name).ok_or_else(|| {
        let available: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        anyhow::anyhow!(
            "Unknown model: '{name}'\n\nAvailable models:\n  {}",
            available.join("\n  ")
        )
    })?;
    download_to(model, &models_dir(), force).await
}

async fn download_to(model: &ModelInfo, dir: &Path, force: bool) -> anyhow::Result<()> {
    let dest = dir.join(model.filename);
    let part = part_path(&dest);

    if dest.exists() && !force {
        match check_size(&dest, model.size_bytes) {
            Ok(()) => {
                println!("Model '{}' is already downloaded at:", model.name);
                println!("  {}", dest.display());
                println!(
                    "\nTo re-download, use: vox models download {} --force",
                    model.name
                );
                return Ok(());
            }
            Err(e) => anyhow::bail!(
                "Model '{}' exists at {} but looks wrong ({e}).\n\n\
                 It may be a truncated download or a different revision.\n\
                 To replace it, run: vox models download {} --force",
                model.name,
                dest.display(),
                model.name
            ),
        }
    }

    if force {
        if dest.exists() {
            println!("Removing existing file for forced re-download...");
            tokio::fs::remove_file(&dest).await?;
        }
        let _ = tokio::fs::remove_file(&part).await;
    }

    // Ensure parent directories exist (models dir itself, and subdirectories
    // for filenames like sensevoice/model.int8.onnx).
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to create models directory {}: {e}\n\n\
                 Set VOX_MODELS_DIR to a writable location to store models elsewhere.",
                parent.display()
            )
        })?;
    }

    println!(
        "Downloading {} ({})...",
        model.name,
        format_bytes(model.size_bytes)
    );
    println!("  URL: {}", model.url);
    println!("  Destination: {}", dest.display());
    println!();

    fetch_to_part(model, &part).await?;
    finalize_download(model, &part, &dest).await?;

    println!(
        "Successfully downloaded {} to {}",
        model.name,
        dest.display()
    );
    println!(
        "Verified: {} ({} bytes), sha256 {}",
        format_bytes(model.size_bytes),
        model.size_bytes,
        model.sha256
    );
    Ok(())
}

/// Stream `model.url` into `part`, resuming if `part` already holds a prefix.
async fn fetch_to_part(model: &ModelInfo, part: &Path) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    let mut offset = tokio::fs::metadata(part)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if offset > model.size_bytes {
        // Leftover from a different revision; start over.
        let _ = tokio::fs::remove_file(part).await;
        offset = 0;
    }
    if offset == model.size_bytes {
        return Ok(()); // Fully downloaded earlier; only verification is left.
    }
    // The only place a model is fetched: refused in offline mode, recorded otherwise.
    let egress = vox::privacy::download_request(model.url, model.name)?;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("vox/", env!("CARGO_PKG_VERSION")))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut request = client.get(model.url);
    if offset > 0 {
        println!("  Resuming partial download at {}", format_bytes(offset));
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let response = request
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Download failed: {e}"))?;

    egress.served_by(response.url().as_str());
    let status = response.status();
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = tokio::fs::remove_file(part).await;
        anyhow::bail!(
            "The server rejected the resume request; the partial file was removed.\n\
             To retry, run: vox models download {}",
            model.name
        );
    }
    if !status.is_success() {
        anyhow::bail!("Download failed with HTTP {}: {}", status, model.url);
    }
    if offset > 0 && status != reqwest::StatusCode::PARTIAL_CONTENT {
        // Server ignored the Range header and is sending the whole file.
        offset = 0;
    }

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(offset > 0)
        .truncate(offset == 0)
        .open(part)
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to create temporary download file {}: {e}\n\n\
                 This may indicate insufficient disk space or permission issues.",
                part.display()
            )
        })?;

    let pb = indicatif::ProgressBar::new(model.size_bytes);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_position(offset);

    let mut written = offset;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            anyhow::anyhow!(
                "Download stream error: {e}\n\n\
                 The download was interrupted. The partial file is kept at\n  {}\n\
                 To resume, run: vox models download {}",
                part.display(),
                model.name
            )
        })?;

        egress.add_received(chunk.len() as u64);
        written += chunk.len() as u64;
        if written > model.size_bytes {
            drop(file);
            let _ = tokio::fs::remove_file(part).await;
            anyhow::bail!(
                "Download of '{}' is larger than the expected {} bytes.\n\
                 The upstream file does not match the pinned registry entry; \
                 the partial file was removed.",
                model.name,
                model.size_bytes
            );
        }

        file.write_all(&chunk).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to write downloaded data: {e}\n\n\
                 This typically indicates insufficient disk space.\n\
                 Partial download location: {}\n\
                 Required space: {}\n\n\
                 To resume after freeing space, run: vox models download {}",
                part.display(),
                format_bytes(model.size_bytes),
                model.name
            )
        })?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);
    pb.finish_and_clear();
    Ok(())
}

/// Verify a completed `.part` file against the registry and move it into
/// place. A truncated file is kept for resuming; a file that fails the
/// checksum is deleted.
async fn finalize_download(model: &ModelInfo, part: &Path, dest: &Path) -> anyhow::Result<()> {
    let (size, sha256) = (model.size_bytes, model.sha256);
    let part_owned = part.to_path_buf();
    let verdict = tokio::task::spawn_blocking(move || verify_file(&part_owned, size, sha256))
        .await
        .map_err(|e| anyhow::anyhow!("verification task failed: {e}"))?;

    match verdict {
        Ok(()) => {}
        Err(VerifyError::Size { expected, actual }) if actual < expected => anyhow::bail!(
            "Download of '{}' is incomplete ({actual} of {expected} bytes).\n\
             The partial file is kept at\n  {}\n\
             To resume, run: vox models download {}",
            model.name,
            part.display(),
            model.name
        ),
        Err(e) => {
            let _ = tokio::fs::remove_file(part).await;
            anyhow::bail!(
                "Downloaded file for '{}' failed verification: {e}\n\n\
                 The download was corrupted or the upstream file changed.\n\
                 The bad file has been removed; nothing was installed.\n\
                 To retry, run: vox models download {}",
                model.name,
                model.name
            );
        }
    }

    tokio::fs::rename(part, dest).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to finalize download: {e}\n\n\
             The file was downloaded and verified but could not be moved into place.\n\
             Temporary file: {}\n\
             Final location: {}",
            part.display(),
            dest.display()
        )
    })
}

/// Print the models directory path.
pub fn path() -> anyhow::Result<()> {
    println!("{}", models_dir().display());
    Ok(())
}

// ============================================================================
// Resolution helpers used by the other commands
// ============================================================================

/// Resolve a model file: check the resolved `models_dir()` first, then fall
/// back to the current working directory.
pub fn resolve_model(filename: &str) -> Option<PathBuf> {
    let models = models_dir();
    let candidate = models.join(filename);
    if candidate.exists() {
        return Some(candidate);
    }
    let local = PathBuf::from(filename);
    if local.exists() {
        return Some(local);
    }
    None
}

/// Map user-facing Whisper model name to GGML filename.
///
/// Names without `.en` (`tiny`, `base`, `small`, `medium`, `turbo`) are
/// multilingual.
///
/// WARNING: Whisper Turbo models (large-v3-turbo*) are far slower on CPU
/// than tiny/base. Prefer the quantized `turbo-q5`.
pub fn model_filename(model: &str) -> String {
    vox::paths::whisper_model_filename(model)
}

/// Map user model name to the download registry name.
pub fn whisper_download_name(model: &str) -> String {
    match model {
        // Whisper Turbo aliases
        "turbo" => "whisper-large-v3-turbo".into(),
        "turbo-q5" => "whisper-large-v3-turbo-q5".into(),
        "turbo-q8" => "whisper-large-v3-turbo-q8".into(),
        other => format!("whisper-{other}"),
    }
}

/// Map user-facing Distil-Whisper model name to GGML filename.
#[cfg(feature = "distil-whisper")]
pub fn distil_whisper_model_filename(model: &str) -> String {
    match model {
        "tiny" => "ggml-distil-tiny.bin".into(),
        "tiny.en" => "ggml-distil-tiny.en.bin".into(),
        "base" => "ggml-distil-base.bin".into(),
        "base.en" => "ggml-distil-base.en.bin".into(),
        "small" => "ggml-distil-small.bin".into(),
        "small.en" => "ggml-distil-small.en.bin".into(),
        "medium" => "ggml-distil-medium.bin".into(),
        "medium.en" => "ggml-distil-medium.en.bin".into(),
        other => {
            if other.ends_with(".bin") {
                other.to_string()
            } else {
                format!("ggml-distil-{other}.bin")
            }
        }
    }
}

/// Map user model name to the distil-whisper download registry name.
///
/// Upstream publishes GGML conversions only for `small.en`, `medium.en`
/// and `large-v3`; other sizes are not in the registry.
#[cfg(feature = "distil-whisper")]
pub fn distil_whisper_download_name(model: &str) -> String {
    format!("distil-whisper-{model}")
}

/// Prompt user to download a missing model. Returns true if yes.
pub fn prompt_download(model_name: &str, filename: &str, auto_yes: bool) -> anyhow::Result<bool> {
    if auto_yes {
        println!("Model '{model_name}' not found. Downloading automatically...");
        return Ok(true);
    }

    use std::io::Write;
    print!("Model '{model_name}' ({filename}) not found. Download now? [Y/n] ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    Ok(input.is_empty() || input == "y" || input == "yes")
}

/// Resolve a model, offering to download if missing.
pub async fn ensure_model(
    model_name: &str,
    filename: &str,
    auto_yes: bool,
) -> anyhow::Result<PathBuf> {
    if let Some(p) = resolve_model(filename) {
        // A wrong size usually means a truncated download from an older
        // version or a different model revision. Loading may still work, so
        // warn instead of failing; `vox doctor` reports it as a problem.
        if let Some(model) = find_model(model_name) {
            if let Err(e @ VerifyError::Size { .. }) = check_size(&p, model.size_bytes) {
                eprintln!(
                    "warning: {} does not match the '{model_name}' registry entry ({e}).\n\
                     \x20        If loading fails, run: vox models download {model_name} --force",
                    p.display()
                );
            }
        }
        return Ok(p);
    }
    if find_model(model_name).is_none() {
        anyhow::bail!(
            "Model file '{filename}' was not found in {} and '{model_name}' is not a \
             downloadable model.\n\nRun `vox models list` to see what can be downloaded.",
            models_dir().display()
        );
    }
    if prompt_download(model_name, filename, auto_yes)? {
        download(model_name, false).await?;
        resolve_model(filename).ok_or_else(|| {
            anyhow::anyhow!("Download of '{model_name}' succeeded but file not found")
        })
    } else {
        anyhow::bail!(
            "Required model '{model_name}' not found.\n\n\
             Download with:\n\
             \n  vox models download {model_name}\n"
        );
    }
}

/// Download both SenseVoice model files and return the model directory path.
#[cfg(feature = "sherpa")]
pub async fn ensure_sherpa_sensevoice(yes: bool) -> anyhow::Result<PathBuf> {
    let model_path = ensure_model("sherpa-sensevoice", "sensevoice/model.int8.onnx", yes).await?;
    ensure_model("sherpa-sensevoice-tokens", "sensevoice/tokens.txt", yes).await?;
    // Return the parent directory (sensevoice/)
    Ok(model_path.parent().unwrap().to_path_buf())
}

/// Download streaming zipformer model files and return the model directory path.
#[cfg(feature = "sherpa")]
pub async fn ensure_sherpa_streaming(yes: bool) -> anyhow::Result<PathBuf> {
    let encoder_path = ensure_model(
        "sherpa-streaming-encoder",
        "sherpa-streaming/encoder.int8.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-decoder",
        "sherpa-streaming/decoder.int8.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-joiner",
        "sherpa-streaming/joiner.int8.onnx",
        yes,
    )
    .await?;
    ensure_model(
        "sherpa-streaming-tokens",
        "sherpa-streaming/tokens.txt",
        yes,
    )
    .await?;
    // Return the parent directory (sherpa-streaming/)
    Ok(encoder_path.parent().unwrap().to_path_buf())
}

/// Download both Piper voice model files (.onnx + .onnx.json) and return the config path.
///
/// `voice_name` is the registry prefix (e.g. "piper-en-us", "piper-de").
/// Returns the path to the `.onnx.json` config file.
#[cfg(feature = "piper")]
pub async fn ensure_piper_voice(voice_name: &str, yes: bool) -> anyhow::Result<PathBuf> {
    let model_registry_name = voice_name;
    let config_registry_name = format!("{voice_name}-config");

    // Look up the model entry to get the filename
    let model_info = find_model(model_registry_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown piper voice: '{voice_name}'"))?;
    let config_info = find_model(&config_registry_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown piper voice config: '{config_registry_name}'"))?;

    // Download both files
    ensure_model(model_registry_name, model_info.filename, yes).await?;
    let config_path = ensure_model(&config_registry_name, config_info.filename, yes).await?;

    Ok(config_path)
}

/// Map a short voice alias to a piper model registry name.
///
/// Supports shorthand like "en", "de", "fr", "tr", locale forms like
/// "tr_TR" / "tr-TR", as well as the full registry names like
/// "piper-en-us". Unknown aliases fall back to English.
pub fn piper_voice_alias(voice: &str) -> String {
    let lower = voice.trim().to_lowercase().replace('_', "-");
    match lower.as_str() {
        "en" | "en-us" | "english" => "piper-en-us".to_string(),
        "de" | "de-de" | "german" | "deutsch" => "piper-de".to_string(),
        "es" | "es-es" | "spanish" | "espanol" => "piper-es".to_string(),
        "fr" | "fr-fr" | "french" | "francais" => "piper-fr".to_string(),
        "tr" | "tr-tr" | "turkish" | "turkce" | "türkçe" => "piper-tr".to_string(),
        "zh" | "zh-cn" | "chinese" | "mandarin" => "piper-zh".to_string(),
        _ if lower.starts_with("piper-") => lower,
        _ => "piper-en-us".to_string(), // default to English
    }
}

// ============================================================================
// Default bundle (`vox serve --download`, `vox doctor`)
// ============================================================================

/// What the user asked for at startup; drives which models make up the
/// default bundle.
#[derive(Debug, Clone, Default)]
pub struct BundleOptions {
    /// Spoken language (`en`, `tr`, `auto`, ...). `None` = English defaults.
    pub language: Option<String>,
    /// Explicit Whisper model (`base`, `small.en`, `turbo`, ...).
    pub stt_model: Option<String>,
    /// Explicit Piper voice (`tr`, `en-us`, `piper-de`, ...).
    pub piper_voice: Option<String>,
}

/// What a bundle entry is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Vad,
    Stt,
    Tts,
    ConversationTts,
    Diarization,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::Vad => "VAD",
            Role::Stt => "STT",
            Role::Tts => "TTS",
            Role::ConversationTts => "TTS (Live Talk)",
            Role::Diarization => "Speaker ID",
        }
    }

    /// What stops working when this model is missing.
    fn impact(self) -> &'static str {
        match self {
            Role::Vad => {
                "microphone endpoints (/v1/listen, Live Talk, `vox listen`) are unavailable"
            }
            Role::Stt => "speech-to-text is disabled (transcribe endpoints return 503)",
            Role::Tts => "Kokoro text-to-speech is unavailable (Speak tab, /v1/synthesize)",
            Role::ConversationTts => "Piper voice for Live Talk/Converse is unavailable",
            Role::Diarization => "speaker identification is disabled",
        }
    }
}

/// One model of the default bundle.
pub struct BundleItem {
    pub role: Role,
    pub model: &'static ModelInfo,
    /// Other file names that satisfy this slot (e.g. any suitable Whisper
    /// model the server would load instead of the recommended one).
    pub alternatives: Vec<&'static str>,
    /// Missing required items make `vox doctor` fail.
    pub required: bool,
}

impl BundleItem {
    fn single(role: Role, name: &str, required: bool) -> Option<Self> {
        Some(Self {
            role,
            model: find_model(name)?,
            alternatives: Vec::new(),
            required,
        })
    }

    /// The installed file that satisfies this slot, if any: the bundle's
    /// own model first, then the alternatives in order.
    pub fn installed(&self, dir: &Path) -> Option<&'static str> {
        std::iter::once(self.model.filename)
            .chain(self.alternatives.iter().copied())
            .find(|f| dir.join(f).exists())
    }
}

/// Registry name of the Piper voice for the given options.
pub fn bundle_piper_voice(opts: &BundleOptions) -> String {
    if let Some(voice) = opts.piper_voice.as_deref() {
        return piper_voice_alias(voice);
    }
    match opts.language.as_deref().map(str::trim) {
        Some(lang) if !lang.is_empty() && !lang.eq_ignore_ascii_case("auto") => {
            // Unknown languages fall back to the English voice.
            let short: String = lang.chars().take_while(|c| c.is_alphabetic()).collect();
            piper_voice_alias(&short)
        }
        _ => "piper-en-us".to_string(),
    }
}

/// The models a fresh install needs for `vox serve`, given what this binary
/// was compiled with, the hardware it runs on and the requested language.
pub fn default_bundle(
    profile: &vox::system_profile::SystemProfile,
    opts: &BundleOptions,
) -> Vec<BundleItem> {
    let mut items = Vec::new();
    items.extend(BundleItem::single(Role::Vad, "silero-vad", true));

    // STT: the explicitly requested model, else the best one for this
    // hardware and language. Any other suitable installed model counts.
    let candidates = profile.recommended_whisper_candidates(opts.language.as_deref());
    let wanted = opts
        .stt_model
        .as_deref()
        .and_then(|m| find_model(&whisper_download_name(m)))
        .or_else(|| candidates.iter().find_map(|f| find_model_by_filename(f)));
    if let Some(model) = wanted {
        items.push(BundleItem {
            role: Role::Stt,
            model,
            alternatives: candidates
                .into_iter()
                .filter(|f| *f != model.filename)
                .collect(),
            required: true,
        });
    }

    if cfg!(feature = "kokoro") {
        items.extend(BundleItem::single(Role::Tts, "kokoro", true));
        items.extend(BundleItem::single(Role::Tts, "kokoro-voices", true));
    }
    if cfg!(feature = "piper") {
        let voice = bundle_piper_voice(opts);
        // Piper is the only TTS when Kokoro is not compiled in.
        let required = !cfg!(feature = "kokoro");
        items.extend(BundleItem::single(Role::ConversationTts, &voice, required));
        items.extend(BundleItem::single(
            Role::ConversationTts,
            &format!("{voice}-config"),
            required,
        ));
    }
    if cfg!(feature = "server") {
        items.extend(BundleItem::single(
            Role::Diarization,
            "speaker-encoder",
            false,
        ));
    }
    items
}

/// Names of bundle models that are not installed in `dir`.
#[cfg(any(feature = "server", test))]
pub fn missing_bundle_models(dir: &Path, bundle: &[BundleItem]) -> Vec<&'static str> {
    bundle
        .iter()
        .filter(|item| item.installed(dir).is_none())
        .map(|item| item.model.name)
        .collect()
}

/// Download every missing model of the default bundle.
///
/// Failures are reported but do not abort: the server starts with whatever
/// is available and the affected endpoints return 503. Returns the names of
/// the models that could not be installed.
#[cfg(feature = "server")]
pub async fn ensure_default_bundle(opts: &BundleOptions) -> Vec<&'static str> {
    let profile = vox::system_profile::SystemProfile::detect();
    let dir = models_dir();
    let bundle = default_bundle(&profile, opts);
    let missing = missing_bundle_models(&dir, &bundle);
    if missing.is_empty() {
        return Vec::new();
    }

    let total: u64 = missing
        .iter()
        .filter_map(|n| find_model(n))
        .map(|m| m.size_bytes)
        .sum();
    println!(
        "Downloading {} missing model(s) ({}) to {}\n",
        missing.len(),
        format_bytes(total),
        dir.display()
    );

    let mut failed = Vec::new();
    for name in missing {
        if let Err(e) = download(name, false).await {
            eprintln!("warning: could not download '{name}': {e}\n");
            failed.push(name);
        }
    }
    failed
}

// ============================================================================
// `vox doctor`
// ============================================================================

/// Cargo features this binary was compiled with (the ones that matter for
/// which models are needed).
fn compiled_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    for (enabled, name) in [
        (cfg!(feature = "whisper"), "whisper"),
        (cfg!(feature = "distil-whisper"), "distil-whisper"),
        (cfg!(feature = "sherpa"), "sherpa"),
        (cfg!(feature = "silero"), "silero"),
        (cfg!(feature = "kokoro"), "kokoro"),
        (cfg!(feature = "piper"), "piper"),
        (cfg!(feature = "qwen3"), "qwen3"),
        (cfg!(feature = "server"), "server"),
    ] {
        if enabled {
            features.push(name);
        }
    }
    features
}

/// Result of a `vox doctor` run.
pub struct DoctorReport {
    /// Full text to print.
    pub text: String,
    /// Number of required models that are missing or corrupt.
    pub required_problems: usize,
}

/// Inspect `dir` against the default bundle and explain every problem.
///
/// With `verify`, every installed registry model is also hashed and compared
/// with its pinned SHA-256 (slow for large models).
pub fn diagnose(
    resolution: &vox::paths::ModelsDirResolution,
    profile: &vox::system_profile::SystemProfile,
    opts: &BundleOptions,
    verify: bool,
) -> DoctorReport {
    use fmt::Write;
    let dir = &resolution.path;
    let mut out = String::new();
    let mut required_problems = 0;
    let mut optional_problems = 0;

    let _ = writeln!(out, "vox {} doctor\n", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(out, "Models directory: {}", dir.display());
    let _ = writeln!(out, "  chosen by:      {}", resolution.source.describe());
    if !dir.is_dir() {
        let _ = writeln!(
            out,
            "  status:         does not exist yet (created on first download)"
        );
    }
    let _ = writeln!(out, "Compiled features: {}", compiled_features().join(", "));
    let _ = writeln!(
        out,
        "Hardware: {} (whisper threads: {})",
        profile.summary(),
        profile.recommended_inference_threads()
    );
    let _ = writeln!(
        out,
        "Language: {}",
        opts.language
            .as_deref()
            .unwrap_or("not set (English defaults)")
    );
    if !cfg!(feature = "kokoro") && !cfg!(feature = "piper") {
        let _ = writeln!(
            out,
            "\nNote: this binary was built without a TTS backend (kokoro/piper); \
             speech synthesis is unavailable regardless of installed models."
        );
    }

    let _ = writeln!(out, "\nDefault models for `vox serve`:");
    for item in default_bundle(profile, opts) {
        let role = item.role.label();
        let mut problem: Option<String> = None;
        match item.installed(dir) {
            None => {
                problem = Some(format!(
                    "{} is not downloaded; {}.\n             fix: vox models download {}",
                    item.model.name,
                    item.role.impact(),
                    item.model.name
                ));
            }
            Some(file) => {
                let path = dir.join(file);
                let known = find_model_by_filename(file);
                let check = match known {
                    Some(m) if verify => verify_file(&path, m.size_bytes, m.sha256),
                    Some(m) => check_size(&path, m.size_bytes),
                    None => Ok(()),
                };
                match check {
                    Ok(()) => {
                        let note = if file == item.model.filename {
                            String::new()
                        } else {
                            format!(" (recommended for this setup: {})", item.model.name)
                        };
                        let verified = if verify { ", sha256 ok" } else { "" };
                        let _ = writeln!(out, "  [ok]      {role:<16} {file}{verified}{note}");
                    }
                    Err(e) => {
                        let name = known.map(|m| m.name).unwrap_or(item.model.name);
                        problem = Some(format!(
                            "{file} looks wrong ({e}); truncated download or a different revision.\n             \
                             fix: vox models download {name} --force"
                        ));
                    }
                }
            }
        }
        if let Some(problem) = problem {
            let tag = if item.required {
                "[MISSING]"
            } else {
                "[warn]   "
            };
            let _ = writeln!(out, "  {tag} {role:<16} {problem}");
            if item.required {
                required_problems += 1;
            } else {
                optional_problems += 1;
            }
        }
    }

    // Anything half-downloaded?
    let partials: Vec<String> = MODELS
        .iter()
        .filter_map(|m| {
            let part = part_path(&dir.join(m.filename));
            let len = std::fs::metadata(&part).ok()?.len();
            Some(format!(
                "  {} — {} of {} (resume: vox models download {})",
                m.name,
                format_bytes(len),
                format_bytes(m.size_bytes),
                m.name
            ))
        })
        .collect();
    if !partials.is_empty() {
        let _ = writeln!(out, "\nInterrupted downloads:\n{}", partials.join("\n"));
    }

    let _ = writeln!(out);
    if required_problems == 0 && optional_problems == 0 {
        let _ = writeln!(out, "All default models are installed.");
    } else {
        let _ = writeln!(
            out,
            "{required_problems} required and {optional_problems} optional model(s) need attention."
        );
        let mut fix = String::from("vox serve --download");
        if let Some(lang) = opts.language.as_deref() {
            let _ = write!(fix, " --language {lang}");
        }
        let _ = writeln!(out, "Download everything that is missing with: {fix}");
    }

    DoctorReport {
        text: out,
        required_problems,
    }
}

/// `vox doctor`: print what is installed, what is missing and why.
/// Fails (non-zero exit) when a required model is missing or corrupt.
pub fn doctor(opts: &BundleOptions, verify: bool) -> anyhow::Result<()> {
    let resolution = vox::paths::resolve_models_dir();
    let profile = vox::system_profile::SystemProfile::detect();
    if verify {
        println!("Verifying checksums (this can take a while for large models)...\n");
    }
    let report = diagnose(&resolution, &profile, opts, verify);
    print!("{}", report.text);
    if report.required_problems > 0 {
        anyhow::bail!(
            "{} required model(s) missing or corrupt",
            report.required_problems
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use vox::system_profile::{Arch, Device, HardwareClass, Os, SystemProfile};

    /// Unique scratch directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "vox-models-{tag}-{}-{:?}",
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

    fn profile(class: HardwareClass) -> SystemProfile {
        SystemProfile {
            os: Os::Linux,
            arch: Arch::X86_64,
            device: Device::LinuxDesktop,
            total_ram_mb: 16384,
            available_ram_mb: 8000,
            num_cpus: 8,
            class,
        }
    }

    fn lang(language: &str) -> BundleOptions {
        BundleOptions {
            language: Some(language.to_string()),
            ..Default::default()
        }
    }

    /// sha256("hello world")
    const HELLO_SHA: &str = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    fn fake_model(sha256: &'static str, size_bytes: u64) -> ModelInfo {
        ModelInfo {
            name: "fake",
            filename: "fake.bin",
            url: "http://127.0.0.1:9/unused",
            size_bytes,
            sha256,
            kind: "STT",
        }
    }

    // ---- registry invariants ------------------------------------------------

    #[test]
    fn registry_names_and_filenames_are_unique() {
        let mut names = HashSet::new();
        let mut files = HashSet::new();
        for m in MODELS {
            assert!(names.insert(m.name), "duplicate name {}", m.name);
            assert!(
                files.insert(m.filename),
                "duplicate filename {}",
                m.filename
            );
        }
    }

    #[test]
    fn every_registry_entry_has_checksum_and_exact_size() {
        for m in MODELS {
            assert_eq!(
                m.sha256.len(),
                64,
                "{}: sha256 must be 64 hex chars",
                m.name
            );
            assert!(
                m.sha256
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{}: sha256 must be lower-case hex",
                m.name
            );
            assert!(m.size_bytes > 0, "{}: size missing", m.name);
        }
    }

    #[test]
    fn every_registry_url_is_pinned_to_a_revision() {
        fn is_commit(s: &str) -> bool {
            s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
        }
        for m in MODELS {
            let url = m.url;
            assert!(url.starts_with("https://"), "{}: {url}", m.name);
            if let Some(rest) = url.strip_prefix("https://huggingface.co/") {
                let rev = rest
                    .split("/resolve/")
                    .nth(1)
                    .and_then(|r| r.split('/').next())
                    .unwrap_or("");
                assert!(
                    is_commit(rev),
                    "{}: HF url not pinned to a commit: {url}",
                    m.name
                );
            } else if let Some(rest) = url.strip_prefix("https://raw.githubusercontent.com/") {
                let rev = rest.split('/').nth(2).unwrap_or("");
                assert!(
                    is_commit(rev),
                    "{}: raw GitHub url not pinned: {url}",
                    m.name
                );
            } else {
                assert!(
                    url.contains("/releases/download/"),
                    "{}: unexpected host or unpinned url: {url}",
                    m.name
                );
            }
        }
    }

    #[test]
    fn registry_has_small_en_and_multilingual_whisper_models() {
        for name in [
            "whisper-tiny.en",
            "whisper-base.en",
            "whisper-small.en",
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-large-v3-turbo",
            "whisper-large-v3-turbo-q5",
        ] {
            assert!(find_model(name).is_some(), "registry is missing {name}");
        }
    }

    #[test]
    fn models_list_shows_small_en_and_multilingual_models() {
        let tmp = Scratch::new("list");
        let listing = render_list(&tmp.0);
        for name in [
            "whisper-small.en",
            "whisper-base ",
            "whisper-small ",
            "whisper-large-v3-turbo ",
            "piper-tr ",
        ] {
            assert!(
                listing.contains(name),
                "`vox models list` lacks {name:?}:\n{listing}"
            );
        }
        assert!(listing.contains("not downloaded"));
    }

    #[test]
    fn list_reports_partial_and_mismatched_files() {
        let tmp = Scratch::new("liststatus");
        std::fs::write(tmp.0.join("silero_vad.onnx"), b"truncated").unwrap();
        std::fs::write(tmp.0.join("ggml-base.bin.part"), b"half").unwrap();
        let listing = render_list(&tmp.0);
        assert!(listing.contains("size mismatch"), "{listing}");
        assert!(listing.contains("partial"), "{listing}");
    }

    #[test]
    fn whisper_names_map_to_registry_entries() {
        for model in [
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "tiny",
            "base",
            "small",
            "medium",
            "small-q5",
            "medium-q5",
            "turbo",
            "turbo-q5",
            "turbo-q8",
        ] {
            let entry = find_model(&whisper_download_name(model))
                .unwrap_or_else(|| panic!("no registry entry for whisper model '{model}'"));
            assert_eq!(
                entry.filename,
                model_filename(model),
                "filename mismatch for '{model}'"
            );
        }
    }

    #[test]
    fn every_whisper_candidate_the_server_tries_is_downloadable() {
        for class in [
            HardwareClass::Tiny,
            HardwareClass::Small,
            HardwareClass::Large,
        ] {
            for language in [None, Some("auto"), Some("tr")] {
                for file in profile(class).recommended_whisper_candidates(language) {
                    assert!(
                        find_model_by_filename(file).is_some(),
                        "{file} is a server candidate but not in the registry"
                    );
                }
            }
        }
    }

    #[test]
    fn piper_voices_come_in_model_and_config_pairs() {
        for m in MODELS.iter().filter(|m| m.name.starts_with("piper-")) {
            if m.name.ends_with("-config") {
                assert!(m.filename.ends_with(".onnx.json"), "{}", m.name);
                let voice = m.name.trim_end_matches("-config");
                let model = find_model(voice).expect("config without voice");
                assert_eq!(format!("{}.json", model.filename), m.filename);
            } else {
                assert!(
                    find_model(&format!("{}-config", m.name)).is_some(),
                    "{}",
                    m.name
                );
            }
        }
        assert_eq!(
            find_model("piper-tr").unwrap().filename,
            "piper/tr_TR-dfki-medium.onnx"
        );
    }

    #[test]
    fn piper_aliases_include_turkish() {
        assert_eq!(piper_voice_alias("tr"), "piper-tr");
        assert_eq!(piper_voice_alias("tr_TR"), "piper-tr");
        assert_eq!(piper_voice_alias("Turkish"), "piper-tr");
        assert_eq!(piper_voice_alias("en"), "piper-en-us");
        assert_eq!(piper_voice_alias("piper-de"), "piper-de");
        assert_eq!(piper_voice_alias("klingon"), "piper-en-us");
    }

    // ---- verification ---------------------------------------------------------

    #[test]
    fn verify_accepts_matching_file() {
        let tmp = Scratch::new("verify-ok");
        let path = tmp.0.join("model.bin");
        std::fs::write(&path, b"hello world").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), HELLO_SHA);
        assert_eq!(verify_file(&path, 11, HELLO_SHA), Ok(()));
        assert_eq!(verify_file(&path, 11, &HELLO_SHA.to_uppercase()), Ok(()));
    }

    #[test]
    fn verify_rejects_corrupt_file_with_checksum_error() {
        let tmp = Scratch::new("verify-corrupt");
        let path = tmp.0.join("model.bin");
        // Same length as "hello world", one byte flipped.
        std::fs::write(&path, b"hello_world").unwrap();
        match verify_file(&path, 11, HELLO_SHA) {
            Err(VerifyError::Checksum { expected, actual }) => {
                assert_eq!(expected, HELLO_SHA);
                assert_ne!(actual, HELLO_SHA);
            }
            other => panic!("expected checksum error, got {other:?}"),
        }
    }

    #[test]
    fn verify_rejects_truncated_file_with_size_error() {
        let tmp = Scratch::new("verify-size");
        let path = tmp.0.join("model.bin");
        std::fs::write(&path, b"hello").unwrap();
        assert_eq!(
            verify_file(&path, 11, HELLO_SHA),
            Err(VerifyError::Size {
                expected: 11,
                actual: 5
            })
        );
        assert_eq!(
            verify_file(&tmp.0.join("absent.bin"), 11, HELLO_SHA),
            Err(VerifyError::Missing)
        );
    }

    #[tokio::test]
    async fn corrupt_download_is_rejected_and_never_installed() {
        let tmp = Scratch::new("finalize-corrupt");
        let model = fake_model(HELLO_SHA, 11);
        let dest = tmp.0.join(model.filename);
        let part = part_path(&dest);
        std::fs::write(&part, b"hello_world").unwrap();

        let err = finalize_download(&model, &part, &dest)
            .await
            .expect_err("corrupt file must not be installed");
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        assert!(!dest.exists(), "corrupt file was moved into place");
        assert!(!part.exists(), "corrupt partial file was left behind");
    }

    #[tokio::test]
    async fn truncated_download_is_kept_for_resume() {
        let tmp = Scratch::new("finalize-short");
        let model = fake_model(HELLO_SHA, 11);
        let dest = tmp.0.join(model.filename);
        let part = part_path(&dest);
        std::fs::write(&part, b"hello").unwrap();

        let err = finalize_download(&model, &part, &dest)
            .await
            .expect_err("incomplete file must not be installed");
        assert!(err.to_string().contains("incomplete"), "{err}");
        assert!(!dest.exists());
        assert!(
            part.exists(),
            "partial file should be kept so it can resume"
        );
    }

    #[tokio::test]
    async fn verified_download_is_moved_into_place() {
        let tmp = Scratch::new("finalize-ok");
        let model = fake_model(HELLO_SHA, 11);
        let dest = tmp.0.join(model.filename);
        let part = part_path(&dest);
        std::fs::write(&part, b"hello world").unwrap();

        finalize_download(&model, &part, &dest).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello world");
        assert!(!part.exists());
    }

    #[tokio::test]
    async fn complete_part_file_is_not_fetched_again() {
        // The URL is unreachable: success proves no request was made.
        let tmp = Scratch::new("fetch-skip");
        let model = fake_model(HELLO_SHA, 11);
        let part = part_path(&tmp.0.join(model.filename));
        std::fs::write(&part, b"hello world").unwrap();
        fetch_to_part(&model, &part).await.unwrap();
    }

    #[tokio::test]
    async fn existing_file_with_wrong_size_is_reported_not_silently_accepted() {
        let tmp = Scratch::new("existing-bad");
        let model = fake_model(HELLO_SHA, 11);
        std::fs::write(tmp.0.join(model.filename), b"short").unwrap();
        let err = download_to(&model, &tmp.0, false).await.unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
    }

    #[test]
    fn part_path_appends_suffix_without_clobbering_extensions() {
        assert_eq!(
            part_path(Path::new("piper/en_US-lessac-medium.onnx.json")),
            PathBuf::from("piper/en_US-lessac-medium.onnx.json.part")
        );
        assert_ne!(
            part_path(Path::new("piper/en_US-lessac-medium.onnx")),
            part_path(Path::new("piper/en_US-lessac-medium.onnx.json"))
        );
    }

    // ---- default bundle / doctor ---------------------------------------------

    #[test]
    fn default_bundle_has_vad_and_recommended_english_stt() {
        let bundle = default_bundle(&profile(HardwareClass::Medium), &BundleOptions::default());
        let names: Vec<&str> = bundle.iter().map(|i| i.model.name).collect();
        assert!(names.contains(&"silero-vad"));
        assert!(names.contains(&"whisper-base.en"), "{names:?}");
        assert_eq!(names.contains(&"kokoro"), cfg!(feature = "kokoro"));
        assert_eq!(names.contains(&"piper-en-us"), cfg!(feature = "piper"));
        assert_eq!(names.contains(&"speaker-encoder"), cfg!(feature = "server"));
    }

    #[test]
    fn turkish_bundle_uses_multilingual_stt_and_turkish_voice() {
        let bundle = default_bundle(&profile(HardwareClass::Large), &lang("tr"));
        let stt = bundle.iter().find(|i| i.role == Role::Stt).unwrap();
        assert_eq!(stt.model.name, "whisper-small");
        assert!(stt.alternatives.iter().all(|f| !f.contains(".en")));
        assert_eq!(bundle_piper_voice(&lang("tr")), "piper-tr");
        assert_eq!(bundle_piper_voice(&lang("tr-TR")), "piper-tr");
        assert_eq!(bundle_piper_voice(&lang("auto")), "piper-en-us");
        assert_eq!(bundle_piper_voice(&lang("xx")), "piper-en-us");
        if cfg!(feature = "piper") {
            assert!(bundle.iter().any(|i| i.model.name == "piper-tr"));
            assert!(bundle.iter().any(|i| i.model.name == "piper-tr-config"));
        }
    }

    #[test]
    fn explicit_stt_model_and_voice_override_the_bundle() {
        let opts = BundleOptions {
            language: Some("tr".into()),
            stt_model: Some("turbo-q5".into()),
            piper_voice: Some("de".into()),
        };
        let bundle = default_bundle(&profile(HardwareClass::Small), &opts);
        let stt = bundle.iter().find(|i| i.role == Role::Stt).unwrap();
        assert_eq!(stt.model.name, "whisper-large-v3-turbo-q5");
        assert_eq!(bundle_piper_voice(&opts), "piper-de");
    }

    #[test]
    fn any_suitable_installed_whisper_model_satisfies_the_stt_slot() {
        let tmp = Scratch::new("bundle-alt");
        let bundle = default_bundle(&profile(HardwareClass::Large), &BundleOptions::default());
        assert!(missing_bundle_models(&tmp.0, &bundle).contains(&"whisper-small.en"));

        std::fs::write(tmp.0.join("ggml-tiny.en.bin"), b"x").unwrap();
        let missing = missing_bundle_models(&tmp.0, &bundle);
        assert!(
            !missing.iter().any(|n| n.starts_with("whisper-")),
            "{missing:?}"
        );

        // ...but an English-only model does not satisfy a Turkish setup.
        let tr = default_bundle(&profile(HardwareClass::Large), &lang("tr"));
        assert!(missing_bundle_models(&tmp.0, &tr).contains(&"whisper-small"));
    }

    fn resolution(dir: &Path) -> vox::paths::ModelsDirResolution {
        vox::paths::ModelsDirResolution {
            path: dir.to_path_buf(),
            source: vox::paths::ModelsDirSource::Env,
        }
    }

    #[test]
    fn doctor_lists_missing_models_with_reason_and_fix() {
        let tmp = Scratch::new("doctor-empty");
        let report = diagnose(
            &resolution(&tmp.0),
            &profile(HardwareClass::Medium),
            &BundleOptions::default(),
            false,
        );
        assert!(report.required_problems >= 2, "{}", report.text);
        assert!(report.text.contains("VOX_MODELS_DIR"), "{}", report.text);
        assert!(report.text.contains("silero-vad"), "{}", report.text);
        assert!(
            report.text.contains("vox models download whisper-base.en"),
            "{}",
            report.text
        );
        assert!(
            report.text.contains("speech-to-text is disabled"),
            "{}",
            report.text
        );
        assert!(
            report.text.contains("vox serve --download"),
            "{}",
            report.text
        );
    }

    #[test]
    fn doctor_flags_truncated_files_and_explains_why() {
        let tmp = Scratch::new("doctor-truncated");
        std::fs::write(tmp.0.join("silero_vad.onnx"), b"truncated").unwrap();
        let report = diagnose(
            &resolution(&tmp.0),
            &profile(HardwareClass::Medium),
            &lang("tr"),
            false,
        );
        assert!(report.text.contains("size mismatch"), "{}", report.text);
        assert!(
            report
                .text
                .contains("vox models download silero-vad --force"),
            "{}",
            report.text
        );
        assert!(
            report.text.contains("vox serve --download --language tr"),
            "{}",
            report.text
        );
    }

    #[test]
    fn doctor_verify_detects_same_size_corruption() {
        let tmp = Scratch::new("doctor-verify");
        let vad = find_model("silero-vad").unwrap();
        std::fs::write(tmp.0.join(vad.filename), vec![0u8; vad.size_bytes as usize]).unwrap();
        let p = profile(HardwareClass::Medium);
        let quick = diagnose(&resolution(&tmp.0), &p, &BundleOptions::default(), false);
        assert!(quick.text.contains("[ok]      VAD"), "{}", quick.text);
        let full = diagnose(&resolution(&tmp.0), &p, &BundleOptions::default(), true);
        assert!(full.text.contains("checksum mismatch"), "{}", full.text);
        assert_eq!(full.required_problems, quick.required_problems + 1);
    }

    // ---- network (run explicitly; needs internet) ----------------------------

    /// Checks every registry URL with a 1-byte range request and compares the
    /// advertised total size with the registry. Run in a dedicated CI job:
    /// `cargo test --features cli --bin vox -- --ignored registry_urls`
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn registry_urls_are_reachable_and_sizes_match() {
        let client = reqwest::Client::builder()
            .user_agent(concat!(
                "vox/",
                env!("CARGO_PKG_VERSION"),
                " registry-check"
            ))
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let mut failures = Vec::new();
        for m in MODELS {
            let result = client
                .get(m.url)
                .header(reqwest::header::RANGE, "bytes=0-0")
                .send()
                .await;
            match result {
                Ok(resp) if resp.status() == reqwest::StatusCode::PARTIAL_CONTENT => {
                    let total = resp
                        .headers()
                        .get(reqwest::header::CONTENT_RANGE)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.rsplit('/').next())
                        .and_then(|v| v.parse::<u64>().ok());
                    if total != Some(m.size_bytes) {
                        failures.push(format!(
                            "{}: size {total:?} != registry {}",
                            m.name, m.size_bytes
                        ));
                    }
                }
                Ok(resp) => {
                    failures.push(format!("{}: HTTP {} for {}", m.name, resp.status(), m.url))
                }
                Err(e) => failures.push(format!("{}: {e}", m.name)),
            }
        }
        assert!(
            failures.is_empty(),
            "registry problems:\n{}",
            failures.join("\n")
        );
    }
}
