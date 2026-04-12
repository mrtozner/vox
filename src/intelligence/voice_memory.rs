//! Voice memory integration layer.
//!
//! Unified interface that combines speaker recognition, conversation memory,
//! and user preferences to provide a personalized voice interaction experience.
//!
//! # Features
//!
//! - Automatic speaker enrollment on first use
//! - "Welcome back" messages for returning users
//! - Context restoration across sessions
//! - Privacy dashboard (view/delete data)
//! - Integration with Vox pipeline
//!
//! # Privacy
//!
//! All data is stored locally. No external services are used.

use super::cache::{SemanticCache, SemanticCacheConfig};
use super::user_model::{PreferenceType, UserModel};
use crate::error::VoxError;
use std::sync::Arc;

/// Voice memory system that integrates speaker recognition, caching, and preferences.
pub struct VoiceMemory {
    /// Semantic cache for LLM responses.
    cache: Arc<SemanticCache>,
    /// User modeling system.
    user_model: Arc<UserModel>,
    /// Enable automatic enrollment.
    auto_enrollment: bool,
    /// Enable welcome back messages.
    welcome_messages: bool,
}

impl VoiceMemory {
    /// Create a new voice memory system with default configuration.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(SemanticCache::new()),
            user_model: Arc::new(UserModel::new()),
            auto_enrollment: true,
            welcome_messages: true,
        }
    }

    /// Create a voice memory system with custom cache configuration.
    pub fn with_cache_config(cache_config: SemanticCacheConfig) -> Self {
        Self {
            cache: Arc::new(SemanticCache::with_config(cache_config)),
            user_model: Arc::new(UserModel::new()),
            auto_enrollment: true,
            welcome_messages: true,
        }
    }

    /// Create a builder for configuring the voice memory system.
    pub fn builder() -> VoiceMemoryBuilder {
        VoiceMemoryBuilder::default()
    }

    /// Process a voice interaction with memory and personalization.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - Unique identifier for the speaker (from speaker recognition)
    /// * `question` - The user's question
    /// * `generate_response` - Async function to generate a response if not cached
    ///
    /// # Returns
    ///
    /// Returns a tuple of (response, from_cache, welcome_message)
    pub async fn process_interaction<F, Fut>(
        &self,
        speaker_id: Option<&str>,
        question: &str,
        generate_response: F,
    ) -> Result<(String, bool, Option<String>), VoxError>
    where
        F: FnOnce(String) -> Fut,
        Fut: std::future::Future<Output = Result<String, VoxError>>,
    {
        // Use "anonymous" if no speaker ID
        let speaker_id = speaker_id.unwrap_or("anonymous");

        // Check if this is a new user
        let is_new_user = self.user_model.get_profile(speaker_id).is_none();

        // Auto-enroll new users
        if is_new_user && self.auto_enrollment {
            self.enroll_speaker(speaker_id, None);
        }

        // Generate welcome message for returning users
        let welcome_message = if !is_new_user && self.welcome_messages {
            Some(self.generate_welcome_message(speaker_id))
        } else {
            None
        };

        // Check cache first
        if let Some(cached_response) = self.cache.get(question) {
            // Record cache hit as interaction (positive feedback)
            self.user_model
                .record_interaction(speaker_id, question, &cached_response, true);

            return Ok((cached_response, true, welcome_message));
        }

        // Build personalized system prompt
        let system_prompt_modifier = self.user_model.build_system_prompt_modifier(speaker_id);

        // Generate response with personalization
        let response = generate_response(system_prompt_modifier).await?;

        // Cache the response
        self.cache.put(question, &response);

        // Record interaction (assume positive feedback by default)
        self.user_model
            .record_interaction(speaker_id, question, &response, true);

        Ok((response, false, welcome_message))
    }

    /// Enroll a new speaker.
    pub fn enroll_speaker(&self, speaker_id: &str, display_name: Option<&str>) -> String {
        let mut profile = self.user_model.get_or_create_profile(speaker_id);

        if let Some(name) = display_name {
            profile.display_name = Some(name.to_string());
        }

        self.user_model.update_profile(profile);

        speaker_id.to_string()
    }

    /// Set a user preference.
    pub fn set_preference(&self, speaker_id: &str, preference: PreferenceType) {
        self.user_model.set_preference(speaker_id, preference);
    }

    /// Get user profile information.
    pub fn get_speaker_info(&self, speaker_id: &str) -> Option<SpeakerInfo> {
        self.user_model
            .get_profile(speaker_id)
            .map(|profile| SpeakerInfo {
                speaker_id: profile.user_id,
                display_name: profile.display_name,
                total_conversations: profile.conversation_metadata.total_conversations,
                last_interaction: profile.last_interaction,
                preferences_count: profile.preferences.len(),
            })
    }

    /// Delete all data for a speaker (privacy feature).
    pub fn delete_speaker_data(&self, speaker_id: &str) -> bool {
        self.user_model.delete_profile(speaker_id)
    }

    /// Get cache metrics.
    pub fn cache_metrics(&self) -> super::cache::CacheMetrics {
        self.cache.metrics()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// List all known speakers.
    pub fn list_speakers(&self) -> Vec<String> {
        self.user_model.list_users()
    }

    /// Generate a welcome message for a returning user.
    fn generate_welcome_message(&self, speaker_id: &str) -> String {
        if let Some(profile) = self.user_model.get_profile(speaker_id) {
            let name = profile.display_name.as_deref().unwrap_or(speaker_id);
            let conversation_count = profile.conversation_metadata.total_conversations;

            if conversation_count == 0 {
                format!("Welcome, {}! It's great to meet you.", name)
            } else if conversation_count < 5 {
                format!("Welcome back, {}!", name)
            } else {
                format!("Hello again, {}! Nice to see you.", name)
            }
        } else {
            "Hello!".to_string()
        }
    }

    /// Get privacy dashboard data for a speaker.
    pub fn get_privacy_dashboard(&self, speaker_id: &str) -> Option<PrivacyDashboard> {
        self.user_model
            .get_profile(speaker_id)
            .map(|profile| PrivacyDashboard {
                speaker_id: profile.user_id.clone(),
                display_name: profile.display_name.clone(),
                total_conversations: profile.conversation_metadata.total_conversations,
                preferences: profile.preferences.clone(),
                frequent_topics: profile.conversation_metadata.frequent_topics.clone(),
                last_interaction: profile.last_interaction,
            })
    }
}

impl Default for VoiceMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a VoiceMemory system.
pub struct VoiceMemoryBuilder {
    cache_config: SemanticCacheConfig,
    auto_enrollment: bool,
    welcome_messages: bool,
}

impl Default for VoiceMemoryBuilder {
    fn default() -> Self {
        Self {
            cache_config: SemanticCacheConfig::default(),
            auto_enrollment: true,
            welcome_messages: true,
        }
    }
}

impl VoiceMemoryBuilder {
    /// Set the cache configuration.
    pub fn cache_config(mut self, config: SemanticCacheConfig) -> Self {
        self.cache_config = config;
        self
    }

    /// Enable or disable automatic enrollment.
    pub fn auto_enrollment(mut self, enabled: bool) -> Self {
        self.auto_enrollment = enabled;
        self
    }

    /// Enable or disable welcome messages.
    pub fn welcome_messages(mut self, enabled: bool) -> Self {
        self.welcome_messages = enabled;
        self
    }

    /// Build the VoiceMemory system.
    pub fn build(self) -> VoiceMemory {
        VoiceMemory {
            cache: Arc::new(SemanticCache::with_config(self.cache_config)),
            user_model: Arc::new(UserModel::new()),
            auto_enrollment: self.auto_enrollment,
            welcome_messages: self.welcome_messages,
        }
    }
}

/// Information about a speaker.
#[derive(Debug, Clone)]
pub struct SpeakerInfo {
    /// Unique speaker identifier.
    pub speaker_id: String,
    /// Display name (if set).
    pub display_name: Option<String>,
    /// Total number of conversations.
    pub total_conversations: u64,
    /// Last interaction timestamp.
    pub last_interaction: Option<std::time::SystemTime>,
    /// Number of preferences set.
    pub preferences_count: usize,
}

/// Privacy dashboard data for a speaker.
#[derive(Debug, Clone)]
pub struct PrivacyDashboard {
    /// Speaker identifier.
    pub speaker_id: String,
    /// Display name.
    pub display_name: Option<String>,
    /// Total conversations.
    pub total_conversations: u64,
    /// User preferences.
    pub preferences: Vec<PreferenceType>,
    /// Frequently discussed topics.
    pub frequent_topics: std::collections::HashMap<String, u32>,
    /// Last interaction timestamp.
    pub last_interaction: Option<std::time::SystemTime>,
}

impl PrivacyDashboard {
    /// Export as JSON string for user review.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl serde::Serialize for PrivacyDashboard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("PrivacyDashboard", 6)?;
        state.serialize_field("speaker_id", &self.speaker_id)?;
        state.serialize_field("display_name", &self.display_name)?;
        state.serialize_field("total_conversations", &self.total_conversations)?;
        state.serialize_field("preferences", &self.preferences)?;
        state.serialize_field("frequent_topics", &self.frequent_topics)?;

        // Convert SystemTime to readable string
        let last_interaction_str = self.last_interaction.map(|t| format!("{:?}", t));
        state.serialize_field("last_interaction", &last_interaction_str)?;

        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_new_user_enrollment() {
        let memory = VoiceMemory::new();

        let generate_response =
            |_modifier: String| async { Ok::<String, VoxError>("Hello!".to_string()) };

        let (response, from_cache, welcome_msg) = memory
            .process_interaction(Some("user123"), "Hi", generate_response)
            .await
            .unwrap();

        assert_eq!(response, "Hello!");
        assert!(!from_cache);
        assert!(welcome_msg.is_none()); // No welcome for new users
    }

    #[tokio::test]
    async fn test_returning_user_welcome() {
        let memory = VoiceMemory::new();

        // First interaction
        let generate_response =
            |_modifier: String| async { Ok::<String, VoxError>("Hello!".to_string()) };

        memory
            .process_interaction(Some("user123"), "Hi", generate_response)
            .await
            .unwrap();

        // Second interaction
        let generate_response =
            |_modifier: String| async { Ok::<String, VoxError>("How are you?".to_string()) };

        let (_, _, welcome_msg) = memory
            .process_interaction(Some("user123"), "How are you?", generate_response)
            .await
            .unwrap();

        assert!(welcome_msg.is_some());
        assert!(welcome_msg.unwrap().contains("Welcome back"));
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let memory = VoiceMemory::new();

        // First interaction - cache miss
        let generate_response = |_modifier: String| async {
            Ok::<String, VoxError>("The weather is sunny.".to_string())
        };

        let (response1, from_cache1, _) = memory
            .process_interaction(Some("user123"), "What's the weather?", generate_response)
            .await
            .unwrap();

        assert_eq!(response1, "The weather is sunny.");
        assert!(!from_cache1);

        // Second interaction - cache hit
        let generate_response = |_modifier: String| async {
            Ok::<String, VoxError>("Should not be called".to_string())
        };

        let (response2, from_cache2, _) = memory
            .process_interaction(Some("user123"), "What's the weather?", generate_response)
            .await
            .unwrap();

        assert_eq!(response2, "The weather is sunny.");
        assert!(from_cache2);
    }

    #[test]
    fn test_set_preference() {
        let memory = VoiceMemory::new();

        memory.set_preference(
            "user123",
            PreferenceType::ResponseStyle("casual".to_string()),
        );

        let profile = memory.user_model.get_profile("user123").unwrap();
        assert_eq!(profile.preferences.len(), 1);
    }

    #[test]
    fn test_speaker_info() {
        let memory = VoiceMemory::new();

        memory.enroll_speaker("user123", Some("Alice"));

        let info = memory.get_speaker_info("user123").unwrap();
        assert_eq!(info.speaker_id, "user123");
        assert_eq!(info.display_name, Some("Alice".to_string()));
    }

    #[test]
    fn test_delete_speaker_data() {
        let memory = VoiceMemory::new();

        memory.enroll_speaker("user123", Some("Alice"));
        assert!(memory.get_speaker_info("user123").is_some());

        let deleted = memory.delete_speaker_data("user123");
        assert!(deleted);
        assert!(memory.get_speaker_info("user123").is_none());
    }

    #[test]
    fn test_list_speakers() {
        let memory = VoiceMemory::new();

        memory.enroll_speaker("user1", None);
        memory.enroll_speaker("user2", None);

        let speakers = memory.list_speakers();
        assert_eq!(speakers.len(), 2);
        assert!(speakers.contains(&"user1".to_string()));
        assert!(speakers.contains(&"user2".to_string()));
    }

    #[test]
    fn test_privacy_dashboard() {
        let memory = VoiceMemory::new();

        memory.enroll_speaker("user123", Some("Alice"));
        memory.set_preference("user123", PreferenceType::Verbosity(5));

        let dashboard = memory.get_privacy_dashboard("user123").unwrap();
        assert_eq!(dashboard.speaker_id, "user123");
        assert_eq!(dashboard.display_name, Some("Alice".to_string()));
        assert_eq!(dashboard.preferences.len(), 1);
    }

    #[test]
    fn test_builder() {
        let cache_config = SemanticCacheConfig {
            max_entries: 500,
            ttl: Duration::from_secs(1800),
            enable_metrics: false,
        };

        let memory = VoiceMemory::builder()
            .cache_config(cache_config)
            .auto_enrollment(false)
            .welcome_messages(false)
            .build();

        assert!(!memory.auto_enrollment);
        assert!(!memory.welcome_messages);
    }

    #[test]
    fn test_cache_metrics() {
        let memory = VoiceMemory::new();

        memory.cache.put("q1", "a1");
        let _ = memory.cache.get("q1");

        let metrics = memory.cache_metrics();
        assert_eq!(metrics.hits, 1);
    }
}
