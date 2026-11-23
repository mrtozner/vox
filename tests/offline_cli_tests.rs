//! The `vox` binary in offline mode, and `vox tune --dry-run`.
//!
//! These run the real executable. Every invocation gets its own empty
//! models and config directories, and none of them may touch the network:
//! the commands under test must fail (or finish) before any connection.
//!
//! Run with: `cargo test --test offline_cli_tests --features cli`

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("vox-cli-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::create_dir_all(dir.join("config")).unwrap();
        Self(dir)
    }
    fn models(&self) -> PathBuf {
        self.0.join("models")
    }
    fn config(&self) -> PathBuf {
        self.0.join("config")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn vox(scratch: &Scratch) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vox"));
    cmd.env("VOX_MODELS_DIR", scratch.models())
        .env("VOX_CONFIG_DIR", scratch.config())
        .env_remove("VOX_OFFLINE")
        .env_remove("VOX_OLLAMA_HOST")
        .env_remove("VOX_LLM_BACKEND")
        .env_remove("VOX_LLM_HOST")
        .env_remove("VOX_LLM_MODEL")
        .env_remove("RUST_LOG")
        .stdin(std::process::Stdio::null());
    cmd
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn is_empty_dir(dir: &Path) -> bool {
    std::fs::read_dir(dir).unwrap().next().is_none()
}

#[test]
fn offline_flag_refuses_models_download() {
    let scratch = Scratch::new("download-flag");
    let output = vox(&scratch)
        .args(["--offline", "models", "download", "silero-vad"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("offline mode"), "{text}");
    assert!(text.contains("silero-vad"), "{text}");
    assert!(is_empty_dir(&scratch.models()), "nothing may be written");
}

#[test]
fn offline_env_refuses_models_download() {
    let scratch = Scratch::new("download-env");
    let output = vox(&scratch)
        .env("VOX_OFFLINE", "1")
        .args(["models", "download", "silero-vad"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("offline mode"), "{text}");
    assert!(is_empty_dir(&scratch.models()));
}

#[test]
fn offline_env_set_to_zero_is_not_offline() {
    // `models list` needs no network; it only has to keep working, and the
    // tune report must not claim offline mode.
    let scratch = Scratch::new("env-zero");
    let output = vox(&scratch)
        .env("VOX_OFFLINE", "0")
        .args(["tune", "--dry-run"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(output.status.success(), "{text}");
    assert!(!text.contains("(offline mode)"), "{text}");
}

#[test]
fn offline_refuses_a_remote_ollama_for_chat() {
    let scratch = Scratch::new("chat-flag");
    let output = vox(&scratch)
        .args(["--offline", "chat", "--ollama-host", "10.255.255.1:11434"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("offline mode"), "{text}");
    assert!(text.contains("10.255.255.1:11434"), "{text}");
    assert!(
        is_empty_dir(&scratch.models()),
        "must fail before any model is fetched"
    );
}

#[test]
fn offline_checks_the_active_llm_endpoint_before_loading_speech_models() {
    for subcommand in [
        "chat",
        #[cfg(feature = "server")]
        "serve",
    ] {
        let scratch = Scratch::new(&format!("active-llm-{subcommand}"));
        let output = vox(&scratch)
            .env("VOX_LLM_BACKEND", "openai-compatible")
            .env("VOX_LLM_HOST", "llm-box.invalid:8080")
            .args(["--offline", subcommand, "--ollama-host", "localhost:11434"])
            .output()
            .unwrap();
        let text = text(&output);
        assert!(!output.status.success(), "{text}");
        assert!(text.contains("offline mode"), "{text}");
        assert!(text.contains("llm-box.invalid:8080"), "{text}");
        assert!(is_empty_dir(&scratch.models()), "{text}");
    }
}

#[test]
fn offline_refuses_a_remote_ollama_from_the_environment_and_the_config_file() {
    let scratch = Scratch::new("chat-env");
    let output = vox(&scratch)
        .env("VOX_OFFLINE", "true")
        .env("VOX_OLLAMA_HOST", "http://gpu-box.invalid:11434/")
        .args(["chat"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("offline mode"), "{text}");
    assert!(text.contains("gpu-box.invalid:11434"), "{text}");

    std::fs::write(
        scratch.config().join("config.toml"),
        "ollama_host = \"cfg-box.invalid:11434\"\n",
    )
    .unwrap();
    let output = vox(&scratch).args(["--offline", "chat"]).output().unwrap();
    let text = self::text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("cfg-box.invalid:11434"), "{text}");
}

#[cfg(feature = "server")]
#[test]
fn offline_refuses_a_remote_ollama_for_serve() {
    let scratch = Scratch::new("serve");
    let output = vox(&scratch)
        .args([
            "--offline",
            "serve",
            "--port",
            "0",
            "--ollama-host",
            "gpu-box.invalid:11434",
        ])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("offline mode"), "{text}");
    assert!(text.contains("gpu-box.invalid:11434"), "{text}");
}

#[test]
fn tune_dry_run_prints_a_recommendation_and_changes_nothing() {
    let scratch = Scratch::new("tune");
    let output = vox(&scratch).args(["tune", "--dry-run"]).output().unwrap();
    let text = text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("Machine:"), "{text}");
    assert!(text.contains("Recommendation"), "{text}");
    for label in ["STT", "TTS", "LLM"] {
        assert!(
            text.lines().any(|l| l.trim_start().starts_with(label)),
            "no {label} line in:\n{text}"
        );
    }
    assert!(text.contains("Installed Whisper models: none"), "{text}");
    assert!(text.contains("Ollama: localhost:11434"), "{text}");
    assert!(text.contains("Dry run: nothing was loaded"), "{text}");
    assert!(is_empty_dir(&scratch.models()));
    assert!(
        is_empty_dir(&scratch.config()),
        "no config.toml on a dry run"
    );
}

#[test]
fn tune_dry_run_reports_the_configured_llm_without_discovery() {
    let scratch = Scratch::new("tune-compatible");
    let output = vox(&scratch)
        .env("VOX_LLM_BACKEND", "openai-compatible")
        .env("VOX_LLM_HOST", "llm-box.invalid:8080")
        .env("VOX_LLM_MODEL", "custom-model")
        .args(["--offline", "tune", "--dry-run"])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("llm-box.invalid:8080"), "{text}");
    assert!(text.contains("refused in offline mode"), "{text}");
    assert!(text.contains("custom-model"), "{text}");
    assert!(is_empty_dir(&scratch.models()), "{text}");
    assert!(is_empty_dir(&scratch.config()), "{text}");
}

#[test]
fn tune_dry_run_sees_installed_models_and_the_offline_host_check() {
    let scratch = Scratch::new("tune-installed");
    std::fs::write(scratch.models().join("ggml-base.bin"), b"placeholder").unwrap();
    let output = vox(&scratch)
        .args([
            "--offline",
            "tune",
            "--dry-run",
            "--language",
            "tr",
            "--ollama-host",
            "gpu-box.invalid:11434",
        ])
        .output()
        .unwrap();
    let text = text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("Installed Whisper models: base"), "{text}");
    assert!(text.contains("refused in offline mode"), "{text}");
    let stt = text
        .lines()
        .find(|l| l.trim_start().starts_with("STT"))
        .unwrap();
    assert!(stt.contains("base"), "{stt}");
    assert!(
        !stt.contains(".en"),
        "English-only model offered for Turkish: {stt}"
    );
}
