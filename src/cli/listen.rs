//! Handler for `vox listen` — real-time microphone transcription.

#[cfg(feature = "distil-whisper")]
use super::models::{distil_whisper_download_name, distil_whisper_model_filename};
use super::models::{ensure_model, model_filename, whisper_download_name};
use vox::{SileroVad, Vox};

/// Whisper model used when `--model` is not given: `tiny.en` for English,
/// the multilingual `base` for any other language (including `auto`).
pub fn default_whisper_model(language: Option<&str>) -> &'static str {
    match language.map(|l| l.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("en") => "tiny.en",
        Some(_) => "base",
    }
}

/// Refuse combinations that cannot work: a non-English language with an
/// English-only model would silently transcribe gibberish.
pub fn check_model_language(model: &str, language: Option<&str>) -> anyhow::Result<()> {
    let Some(lang) = language.map(|l| l.trim().to_ascii_lowercase()) else {
        return Ok(());
    };
    if lang.is_empty() || lang == "en" || !vox::paths::whisper_model_is_english_only(model) {
        return Ok(());
    }
    let multilingual = model.replace(".en", "");
    anyhow::bail!(
        "Model '{model}' is English-only and cannot transcribe language '{lang}'.\n\n\
         Use a multilingual model instead, for example:\n\
         \n  vox listen --model {multilingual} --language {lang}\n"
    )
}

/// Whisper configuration for `vox listen`: the requested language (or
/// auto-detect) with the machine's default thread count.
pub fn whisper_config(model_path: &std::path::Path, language: Option<&str>) -> vox::WhisperConfig {
    vox::WhisperConfig::new(model_path).with_language(language)
}

/// Run the listen command.
pub async fn run(
    model: Option<&str>,
    stt_backend: &str,
    language: Option<&str>,
    yes: bool,
) -> anyhow::Result<()> {
    let model = model.unwrap_or_else(|| default_whisper_model(language));
    if stt_backend == "whisper" {
        // Fail on impossible model/language combinations and typos before
        // downloading or loading anything.
        check_model_language(model, language)?;
        whisper_config(std::path::Path::new(""), language).validate()?;
    } else if language.is_some() {
        eprintln!("warning: --language is only used by the whisper backend; ignoring it");
    }

    let vad_path = ensure_model("silero-vad", "silero_vad.onnx", yes).await?;

    println!("Loading VAD model...");
    let vad = SileroVad::new(&vad_path)?;

    match stt_backend {
        "whisper" => run_whisper(model, language, vad, yes).await,
        #[cfg(feature = "distil-whisper")]
        "distil-whisper" => run_distil_whisper(model, vad, yes).await,
        #[cfg(not(feature = "distil-whisper"))]
        "distil-whisper" => anyhow::bail!(
            "Distil-Whisper STT requires the 'distil-whisper' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,distil-whisper --release\n"
        ),
        #[cfg(feature = "sherpa")]
        "sherpa" => run_sherpa(vad, yes).await,
        #[cfg(not(feature = "sherpa"))]
        "sherpa" => anyhow::bail!(
            "Sherpa STT requires the 'sherpa' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,sherpa --release\n"
        ),
        #[cfg(feature = "sherpa")]
        "sherpa-streaming" => run_sherpa_streaming(vad, yes).await,
        #[cfg(not(feature = "sherpa"))]
        "sherpa-streaming" => anyhow::bail!(
            "Sherpa streaming STT requires the 'sherpa' feature.\n\n\
             Rebuild with:\n\
             \n  cargo build --features cli,sherpa --release\n"
        ),
        other => anyhow::bail!(
            "Unknown STT backend: '{other}'. Use 'whisper', 'distil-whisper', 'sherpa', or 'sherpa-streaming'."
        ),
    }
}

async fn run_whisper(
    model: &str,
    language: Option<&str>,
    vad: SileroVad,
    yes: bool,
) -> anyhow::Result<()> {
    let whisper_file = model_filename(model);
    let whisper_name = whisper_download_name(model);
    let whisper_path = ensure_model(&whisper_name, &whisper_file, yes).await?;

    let config = whisper_config(&whisper_path, language);
    println!(
        "Loading Whisper model ({model}, language: {}, {} threads)...",
        language.unwrap_or("auto"),
        config.n_threads
    );
    let stt = vox::WhisperBackend::with_config(config)?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;
    Ok(())
}

#[cfg(feature = "distil-whisper")]
async fn run_distil_whisper(model: &str, vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    let distil_whisper_file = distil_whisper_model_filename(model);
    let distil_whisper_name = distil_whisper_download_name(model);
    let distil_whisper_path = ensure_model(&distil_whisper_name, &distil_whisper_file, yes).await?;

    println!("Loading Distil-Whisper model ({model})...");
    let stt = vox::DistilWhisperBackend::from_model(&distil_whisper_path)?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;
    Ok(())
}

#[cfg(feature = "sherpa")]
async fn run_sherpa(vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    let model_dir = super::models::ensure_sherpa_sensevoice(yes).await?;

    println!("Loading Sherpa SenseVoice model...");
    let stt = vox::SherpaBackend::from_sensevoice(&model_dir)?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(|result, _ctx| {
            println!("[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;
    Ok(())
}

#[cfg(feature = "sherpa")]
async fn run_sherpa_streaming(vad: SileroVad, yes: bool) -> anyhow::Result<()> {
    use std::io::Write;

    // Download streaming model.
    let model_dir = super::models::ensure_sherpa_streaming(yes).await?;

    println!("Loading Sherpa streaming zipformer model...");
    let streaming_backend = vox::SherpaStreamingBackend::from_transducer(&model_dir)?;

    // Also load a batch STT backend as fallback.
    let sensevoice_dir = super::models::ensure_sherpa_sensevoice(yes).await?;
    println!("Loading Sherpa SenseVoice model (batch fallback)...");
    let batch_stt = vox::SherpaBackend::from_sensevoice(&sensevoice_dir)?;

    let vox = Vox::builder()
        .vad(vad)
        .stt(batch_stt)
        .streaming_stt(streaming_backend)
        .on_partial(|text| {
            print!("\r\x1b[2K[partial] {text}");
            let _ = std::io::stdout().flush();
        })
        .on_utterance(|result, _ctx| {
            println!("\r\x1b[2K[transcript] {}", result.text);
        })
        .build()?;

    println!("Listening on default microphone... (Ctrl+C to stop)\n");
    vox.listen().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_flag_reaches_whisper_config() {
        let config = whisper_config(std::path::Path::new("ggml-base.bin"), Some("tr"));
        assert_eq!(config.language.as_deref(), Some("tr"));
        assert!(config.n_threads >= 1);
        assert_eq!(
            whisper_config(std::path::Path::new("ggml-base.bin"), None).language,
            None
        );
    }

    #[test]
    fn default_model_follows_language() {
        assert_eq!(default_whisper_model(None), "tiny.en");
        assert_eq!(default_whisper_model(Some("en")), "tiny.en");
        assert_eq!(default_whisper_model(Some("tr")), "base");
        assert_eq!(default_whisper_model(Some("auto")), "base");
    }

    #[test]
    fn english_only_model_is_rejected_for_other_languages() {
        assert!(check_model_language("tiny.en", None).is_ok());
        assert!(check_model_language("tiny.en", Some("en")).is_ok());
        assert!(check_model_language("base", Some("tr")).is_ok());
        assert!(check_model_language("turbo", Some("auto")).is_ok());
        let err = check_model_language("small.en", Some("tr")).unwrap_err();
        assert!(
            err.to_string().contains("--model small --language tr"),
            "{err}"
        );
        assert!(check_model_language("tiny.en", Some("auto")).is_err());
    }

    #[test]
    fn unknown_language_fails_validation() {
        assert!(
            whisper_config(std::path::Path::new(""), Some("tr"))
                .validate()
                .is_ok()
        );
        assert!(
            whisper_config(std::path::Path::new(""), Some("klingon"))
                .validate()
                .is_err()
        );
    }
}
