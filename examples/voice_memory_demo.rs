//! Example demonstrating the voice memory system.
//!
//! Shows how to use semantic caching, user preferences, and personalized interactions.

use std::time::Duration;
use vox::VoxError;
use vox::intelligence::{PreferenceType, SemanticCacheConfig, VoiceMemory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the voice memory system with custom configuration
    let cache_config = SemanticCacheConfig {
        max_entries: 500,
        ttl: Duration::from_secs(1800), // 30 minutes
        enable_metrics: true,
    };

    let memory = VoiceMemory::builder()
        .cache_config(cache_config)
        .auto_enrollment(true)
        .welcome_messages(true)
        .build();

    println!("=== Voice Memory Demo ===\n");

    // Simulate a conversation with a new user
    println!("--- First Interaction (New User) ---");
    let (response, from_cache, welcome) = memory
        .process_interaction(
            Some("speaker_001"),
            "What's the weather like today?",
            simulate_llm_response,
        )
        .await?;

    println!("Question: What's the weather like today?");
    println!("Response: {}", response);
    println!("From cache: {}", from_cache);
    println!("Welcome message: {:?}", welcome);
    println!();

    // Set user preferences
    println!("--- Setting User Preferences ---");
    memory.set_preference(
        "speaker_001",
        PreferenceType::ResponseStyle("casual".to_string()),
    );
    memory.set_preference("speaker_001", PreferenceType::Verbosity(3)); // Concise
    memory.enroll_speaker("speaker_001", Some("Alice"));
    println!("Set preferences: casual style, concise verbosity");
    println!("Enrolled as: Alice");
    println!();

    // Second interaction - should get welcome message
    println!("--- Second Interaction (Returning User) ---");
    let (response, from_cache, welcome) = memory
        .process_interaction(
            Some("speaker_001"),
            "Tell me about the weather tomorrow",
            simulate_llm_response,
        )
        .await?;

    println!("Question: Tell me about the weather tomorrow");
    println!("Response: {}", response);
    println!("From cache: {}", from_cache);
    println!("Welcome message: {:?}", welcome);
    println!();

    // Third interaction - cache hit (semantically similar question)
    println!("--- Third Interaction (Cache Hit) ---");
    let (response, from_cache, welcome) = memory
        .process_interaction(
            Some("speaker_001"),
            "WHAT'S THE WEATHER LIKE TODAY?", // Same question, different case
            simulate_llm_response,
        )
        .await?;

    println!("Question: WHAT'S THE WEATHER LIKE TODAY?");
    println!("Response: {}", response);
    println!("From cache: {}", from_cache);
    println!("Welcome message: {:?}", welcome);
    println!();

    // Show cache metrics
    println!("--- Cache Metrics ---");
    let metrics = memory.cache_metrics();
    println!("Cache hits: {}", metrics.hits);
    println!("Cache misses: {}", metrics.misses);
    println!("Hit rate: {:.2}%", metrics.hit_rate());
    println!("Average hit latency: {:.2}µs", metrics.avg_hit_latency_us);
    println!("Average miss latency: {:.2}µs", metrics.avg_miss_latency_us);
    println!("Latency reduction: {:.2}µs", metrics.latency_reduction_us());
    println!();

    // Show privacy dashboard
    println!("--- Privacy Dashboard ---");
    if let Some(dashboard) = memory.get_privacy_dashboard("speaker_001") {
        println!("Speaker ID: {}", dashboard.speaker_id);
        println!("Display Name: {:?}", dashboard.display_name);
        println!("Total Conversations: {}", dashboard.total_conversations);
        println!("Preferences: {}", dashboard.preferences.len());

        if !dashboard.frequent_topics.is_empty() {
            println!("Frequent Topics:");
            for (topic, count) in dashboard.frequent_topics.iter().take(5) {
                println!("  - {}: {} times", topic, count);
            }
        }

        // Export as JSON
        if let Ok(json) = dashboard.to_json() {
            println!("\nJSON Export (first 200 chars):");
            println!("{}", &json[..json.len().min(200)]);
        }
    }
    println!();

    // List all speakers
    println!("--- All Speakers ---");
    for speaker_id in memory.list_speakers() {
        if let Some(info) = memory.get_speaker_info(&speaker_id) {
            println!(
                "Speaker: {} ({})",
                info.speaker_id,
                info.display_name.unwrap_or_default()
            );
            println!("  Conversations: {}", info.total_conversations);
            println!("  Preferences: {}", info.preferences_count);
        }
    }
    println!();

    // Demonstrate privacy: delete user data
    println!("--- Privacy Feature: Delete User Data ---");
    println!("Deleting data for speaker_001...");
    let deleted = memory.delete_speaker_data("speaker_001");
    println!("Deleted: {}", deleted);

    let remaining = memory.list_speakers();
    println!("Remaining speakers: {}", remaining.len());

    Ok(())
}

/// Simulate LLM response generation.
///
/// In a real application, this would call an actual LLM API with the
/// system prompt modifier to personalize the response.
async fn simulate_llm_response(system_prompt_modifier: String) -> Result<String, VoxError> {
    // Simulate some processing time
    tokio::time::sleep(Duration::from_millis(100)).await;

    // In a real application, you would:
    // 1. Build the full system prompt with the modifier
    // 2. Call the LLM API (e.g., Ollama, OpenAI, Claude)
    // 3. Return the response

    if !system_prompt_modifier.is_empty() {
        println!(
            "  [LLM] Using personalization: {}",
            system_prompt_modifier.lines().next().unwrap_or("")
        );
    }

    Ok(
        "It's sunny and warm with a high of 75°F. Perfect weather for outdoor activities!"
            .to_string(),
    )
}
