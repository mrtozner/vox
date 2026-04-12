//! Tests for the intelligence layer.

#![cfg(feature = "intelligence")]

use std::time::Duration;
use vox::intelligence::{
    PreferenceType, SemanticCache, SemanticCacheConfig, UserModel, VoiceMemory,
};

#[test]
fn test_semantic_cache_basic() {
    let cache = SemanticCache::new();

    cache.put("What is the weather today?", "It's sunny.");

    let result = cache.get("What is the weather today?");
    assert_eq!(result, Some("It's sunny.".to_string()));

    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 1);
    assert_eq!(metrics.misses, 0);
}

#[test]
fn test_semantic_cache_normalization() {
    let cache = SemanticCache::new();

    cache.put("What is the weather today?", "It's sunny.");

    // Should hit cache due to normalization
    let result = cache.get("WHAT IS THE WEATHER TODAY?");
    assert_eq!(result, Some("It's sunny.".to_string()));

    let result = cache.get("what is the weather today");
    assert_eq!(result, Some("It's sunny.".to_string()));
}

#[test]
fn test_cache_ttl_expiration() {
    let config = SemanticCacheConfig {
        max_entries: 100,
        ttl: Duration::from_millis(100),
        enable_metrics: true,
    };
    let cache = SemanticCache::with_config(config);

    cache.put("test", "response");

    // Should hit immediately
    assert!(cache.get("test").is_some());

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(150));

    // Should miss after TTL
    assert!(cache.get("test").is_none());
}

#[test]
fn test_cache_lru_eviction() {
    let config = SemanticCacheConfig {
        max_entries: 2,
        ttl: Duration::from_secs(3600),
        enable_metrics: true,
    };
    let cache = SemanticCache::with_config(config);

    cache.put("q1", "a1");
    cache.put("q2", "a2");

    // Access q1 to make it more recently used
    let _ = cache.get("q1");

    // Add q3, should evict q2 (least recently used)
    cache.put("q3", "a3");

    assert!(cache.get("q1").is_some());
    assert!(cache.get("q2").is_none());
    assert!(cache.get("q3").is_some());
}

#[test]
fn test_user_model_basic() {
    let model = UserModel::new();

    model.set_preference(
        "user123",
        PreferenceType::ResponseStyle("casual".to_string()),
    );
    model.set_preference("user123", PreferenceType::Verbosity(5));

    let profile = model.get_profile("user123").unwrap();
    assert_eq!(profile.preferences.len(), 2);
    assert_eq!(profile.user_id, "user123");
}

#[test]
fn test_user_model_interactions() {
    let model = UserModel::new();

    model.record_interaction(
        "user123",
        "What is the weather like today?",
        "It's sunny and warm.",
        true,
    );

    let profile = model.get_profile("user123").unwrap();
    assert_eq!(profile.conversation_metadata.total_conversations, 1);
    assert!(profile.conversation_metadata.avg_preferred_response_length > 0.0);
}

#[test]
fn test_user_model_system_prompt() {
    let model = UserModel::new();

    model.set_preference(
        "user123",
        PreferenceType::ResponseStyle("casual".to_string()),
    );
    model.set_preference("user123", PreferenceType::Verbosity(8));

    let modifier = model.build_system_prompt_modifier("user123");
    assert!(modifier.contains("casual"));
    assert!(modifier.contains("detailed"));
}

#[test]
fn test_user_model_delete() {
    let model = UserModel::new();

    model.set_preference("user123", PreferenceType::Verbosity(5));
    assert!(model.get_profile("user123").is_some());

    let deleted = model.delete_profile("user123");
    assert!(deleted);
    assert!(model.get_profile("user123").is_none());
}

#[tokio::test]
async fn test_voice_memory_new_user() {
    let memory = VoiceMemory::new();

    let generate_response =
        |_modifier: String| async { Ok::<String, vox::VoxError>("Hello!".to_string()) };

    let (response, from_cache, welcome_msg) = memory
        .process_interaction(Some("user123"), "Hi", generate_response)
        .await
        .unwrap();

    assert_eq!(response, "Hello!");
    assert!(!from_cache);
    assert!(welcome_msg.is_none()); // No welcome for new users
}

#[tokio::test]
async fn test_voice_memory_cache() {
    let memory = VoiceMemory::new();

    // First interaction - cache miss
    let generate_response = |_modifier: String| async {
        Ok::<String, vox::VoxError>("The weather is sunny.".to_string())
    };

    let (response1, from_cache1, _) = memory
        .process_interaction(Some("user123"), "What's the weather?", generate_response)
        .await
        .unwrap();

    assert_eq!(response1, "The weather is sunny.");
    assert!(!from_cache1);

    // Second interaction - cache hit
    let generate_response = |_modifier: String| async {
        Ok::<String, vox::VoxError>("Should not be called".to_string())
    };

    let (response2, from_cache2, _) = memory
        .process_interaction(Some("user123"), "What's the weather?", generate_response)
        .await
        .unwrap();

    assert_eq!(response2, "The weather is sunny.");
    assert!(from_cache2);
}

#[test]
fn test_voice_memory_speaker_management() {
    let memory = VoiceMemory::new();

    memory.enroll_speaker("user123", Some("Alice"));

    let info = memory.get_speaker_info("user123").unwrap();
    assert_eq!(info.speaker_id, "user123");
    assert_eq!(info.display_name, Some("Alice".to_string()));

    let speakers = memory.list_speakers();
    assert_eq!(speakers.len(), 1);

    let deleted = memory.delete_speaker_data("user123");
    assert!(deleted);

    assert!(memory.get_speaker_info("user123").is_none());
}

#[test]
fn test_voice_memory_privacy_dashboard() {
    let memory = VoiceMemory::new();

    memory.enroll_speaker("user123", Some("Alice"));
    memory.set_preference("user123", PreferenceType::Verbosity(5));

    let dashboard = memory.get_privacy_dashboard("user123").unwrap();
    assert_eq!(dashboard.speaker_id, "user123");
    assert_eq!(dashboard.display_name, Some("Alice".to_string()));
    assert_eq!(dashboard.preferences.len(), 1);

    // Test JSON export
    let json = dashboard.to_json();
    assert!(json.is_ok());
}
