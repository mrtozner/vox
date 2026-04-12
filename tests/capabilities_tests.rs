//! Unit tests for the capability registry.
//!
//! These tests don't require a running Ollama instance — they build a
//! registry with an unreachable host and assert that probes degrade
//! gracefully to empty lists.

use std::time::Duration;
use vox::capabilities::{CapabilityRegistry, FeatureFlags, HardwareFacts, ModelInventory};
use vox::system_profile::SystemProfile;

#[tokio::test]
async fn hardware_facts_match_system_profile() {
    let profile = SystemProfile::detect();
    let hw = HardwareFacts::detect(&profile).await;
    assert_eq!(hw.cpu_cores, profile.num_cpus);
    assert_eq!(hw.total_ram_mb, profile.total_ram_mb);
    assert!(!hw.os.is_empty());
    assert!(!hw.arch.is_empty());
    assert!(!hw.hardware_class.is_empty());
}

#[test]
fn feature_flags_reflect_compile_features() {
    let flags = FeatureFlags::detect();
    // On a server build these two must be on
    assert!(
        flags.whisper,
        "whisper feature should be on in server build"
    );
    assert!(flags.silero, "silero feature should be on in server build");
}

#[test]
fn model_inventory_from_parts_handles_missing() {
    let inv = ModelInventory::from_parts(None, None, None, None);
    assert!(inv.stt.is_none());
    assert!(inv.streaming_stt.is_none());
    assert!(inv.tts.is_none());
    assert!(inv.vad.is_none());
}

#[test]
fn model_inventory_from_parts_populates_fields() {
    let inv = ModelInventory::from_parts(
        Some(("whisper", "base.en", Some(142))),
        None,
        Some(("piper", "en_US-lessac-medium", Some(62))),
        None,
    );
    let stt = inv.stt.expect("stt should be set");
    assert_eq!(stt.backend, "whisper");
    assert_eq!(stt.model_name, "base.en");
    assert_eq!(stt.size_mb, Some(142));

    let tts = inv.tts.expect("tts should be set");
    assert_eq!(tts.backend, "piper");
    assert_eq!(tts.size_mb, Some(62));
}

#[tokio::test]
async fn registry_build_tolerates_unreachable_ollama() {
    let profile = SystemProfile::detect();
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(250))
        .build()
        .unwrap();
    // 127.0.0.1:1 is almost certainly closed.
    let registry = CapabilityRegistry::build(
        &profile,
        Some(("whisper", "base.en", Some(142))),
        None,
        Some(("piper", "test", Some(62))),
        None,
        "127.0.0.1:1",
        &http,
    )
    .await;

    assert!(registry.ollama_models.is_empty());
    assert!(registry.hardware.cpu_cores > 0);
    assert!(registry.features.whisper);
    assert!(registry.generated_at_unix > 0);
}

#[tokio::test]
async fn inject_into_prompt_appends_fact_block() {
    let profile = SystemProfile::detect();
    let http = reqwest::Client::new();
    let registry = CapabilityRegistry::build(
        &profile,
        Some(("whisper", "base.en", Some(142))),
        None,
        Some(("piper", "en_US-lessac-medium", Some(62))),
        None,
        "127.0.0.1:1",
        &http,
    )
    .await;

    let base = "You are a test assistant.";
    let full = registry.inject_into_prompt(base);
    assert!(full.starts_with(base));
    assert!(full.contains("Host environment"));
    assert!(full.contains("whisper"));
    assert!(full.contains("piper"));
    assert!(full.contains("base.en"));
    assert!(full.contains("Speech to text"));
    assert!(full.contains("Text to speech"));
}

#[tokio::test]
async fn flat_list_includes_backends() {
    let profile = SystemProfile::detect();
    let http = reqwest::Client::new();
    let registry = CapabilityRegistry::build(
        &profile,
        Some(("whisper", "base.en", Some(142))),
        None,
        Some(("piper", "en_US-lessac-medium", Some(62))),
        None,
        "127.0.0.1:1",
        &http,
    )
    .await;

    let flat = registry.to_flat_list();
    assert!(flat.iter().any(|c| c.id == "hw.host"));
    assert!(flat.iter().any(|c| c.id == "stt"));
    assert!(flat.iter().any(|c| c.id == "tts"));
}

#[tokio::test]
async fn registry_is_serializable() {
    let profile = SystemProfile::detect();
    let http = reqwest::Client::new();
    let registry =
        CapabilityRegistry::build(&profile, None, None, None, None, "127.0.0.1:1", &http).await;

    let json = serde_json::to_string(&registry).unwrap();
    assert!(json.contains("hardware"));
    assert!(json.contains("features"));
    assert!(json.contains("ollama_models"));
    assert!(json.contains("generated_at_unix"));
}
