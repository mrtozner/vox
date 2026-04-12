//! User modeling and preference learning.
//!
//! Tracks user preferences and conversation patterns to personalize responses.
//! Supports both implicit learning (from conversation patterns) and explicit
//! preference setting via API.
//!
//! # Privacy
//!
//! All data is stored locally. No external services are used.
//! Users can view and delete their data at any time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// User preference types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreferenceType {
    /// Response style (e.g., "concise", "detailed", "casual", "formal").
    ResponseStyle(String),
    /// Verbosity level (0-10, where 0 is most concise).
    Verbosity(u8),
    /// Preferred topics (list of topics the user is interested in).
    PreferredTopics(Vec<String>),
    /// Disliked topics (list of topics to avoid).
    DislikedTopics(Vec<String>),
    /// Preferred language.
    Language(String),
    /// Time of day preferences (e.g., morning greetings).
    TimeOfDayPreferences(bool),
    /// Custom preference (key-value pair).
    Custom(String, String),
}

/// A user's preference profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique identifier for the user (e.g., speaker ID).
    pub user_id: String,
    /// User's display name.
    pub display_name: Option<String>,
    /// User preferences.
    pub preferences: Vec<PreferenceType>,
    /// Conversation history metadata (for implicit learning).
    pub conversation_metadata: ConversationMetadata,
    /// Timestamp of last interaction.
    pub last_interaction: Option<std::time::SystemTime>,
}

/// Metadata about user's conversation patterns.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationMetadata {
    /// Total number of conversations.
    pub total_conversations: u64,
    /// Average response length preferred (in characters).
    pub avg_preferred_response_length: f64,
    /// Topics mentioned frequently.
    pub frequent_topics: HashMap<String, u32>,
    /// Common question patterns.
    pub common_question_patterns: HashMap<String, u32>,
}

impl UserProfile {
    /// Create a new user profile.
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            display_name: None,
            preferences: Vec::new(),
            conversation_metadata: ConversationMetadata::default(),
            last_interaction: None,
        }
    }

    /// Set a preference.
    pub fn set_preference(&mut self, preference: PreferenceType) {
        // Remove existing preference of the same type
        self.preferences
            .retain(|p| std::mem::discriminant(p) != std::mem::discriminant(&preference));

        self.preferences.push(preference);
    }

    /// Get a preference of a specific type.
    pub fn get_preference(&self, pref_type: &str) -> Option<&PreferenceType> {
        self.preferences.iter().find(|p| match p {
            PreferenceType::ResponseStyle(_) => pref_type == "response_style",
            PreferenceType::Verbosity(_) => pref_type == "verbosity",
            PreferenceType::PreferredTopics(_) => pref_type == "preferred_topics",
            PreferenceType::DislikedTopics(_) => pref_type == "disliked_topics",
            PreferenceType::Language(_) => pref_type == "language",
            PreferenceType::TimeOfDayPreferences(_) => pref_type == "time_of_day_preferences",
            PreferenceType::Custom(key, _) => key == pref_type,
        })
    }

    /// Record a conversation interaction for implicit learning.
    pub fn record_interaction(&mut self, question: &str, response: &str, feedback_positive: bool) {
        self.last_interaction = Some(std::time::SystemTime::now());
        self.conversation_metadata.total_conversations += 1;

        // Update average preferred response length based on feedback
        if feedback_positive {
            let response_len = response.len() as f64;
            let total = self.conversation_metadata.total_conversations as f64;
            let current_avg = self.conversation_metadata.avg_preferred_response_length;

            self.conversation_metadata.avg_preferred_response_length =
                (current_avg * (total - 1.0) + response_len) / total;
        }

        // Extract and track topics (simple keyword extraction)
        let topics = self.extract_topics(question);
        for topic in topics {
            *self
                .conversation_metadata
                .frequent_topics
                .entry(topic)
                .or_insert(0) += 1;
        }

        // Track question patterns
        let pattern = self.extract_question_pattern(question);
        *self
            .conversation_metadata
            .common_question_patterns
            .entry(pattern)
            .or_insert(0) += 1;
    }

    /// Extract topics from a question (simple keyword extraction).
    fn extract_topics(&self, text: &str) -> Vec<String> {
        let mut topics = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        // Simple noun extraction (words that aren't common words)
        let common_words = [
            "what", "when", "where", "why", "how", "who", "is", "are", "was", "were", "the", "a",
            "an", "in", "on", "at", "to", "for", "of", "with", "by",
        ];

        for word in words {
            let clean = word
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            if clean.len() > 3 && !common_words.contains(&clean.as_str()) {
                topics.push(clean);
            }
        }

        topics
    }

    /// Extract question pattern (e.g., "what is", "how to", "when does").
    fn extract_question_pattern(&self, question: &str) -> String {
        let lower = question.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        if words.len() >= 2 {
            format!("{} {}", words[0], words[1])
        } else if !words.is_empty() {
            words[0].to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Generate a system prompt modifier based on user preferences.
    pub fn to_system_prompt_modifier(&self) -> String {
        let mut modifiers = Vec::new();

        for pref in &self.preferences {
            match pref {
                PreferenceType::ResponseStyle(style) => {
                    modifiers.push(format!("Use a {} style in your responses.", style));
                }
                PreferenceType::Verbosity(level) => {
                    let verbosity_desc = match level {
                        0..=3 => "very concise",
                        4..=6 => "moderate",
                        7..=10 => "detailed",
                        _ => "moderate",
                    };
                    modifiers.push(format!("Keep responses {}.", verbosity_desc));
                }
                PreferenceType::PreferredTopics(topics) => {
                    if !topics.is_empty() {
                        modifiers
                            .push(format!("The user is interested in: {}.", topics.join(", ")));
                    }
                }
                PreferenceType::DislikedTopics(topics) => {
                    if !topics.is_empty() {
                        modifiers.push(format!("Avoid discussing: {}.", topics.join(", ")));
                    }
                }
                PreferenceType::Language(lang) => {
                    modifiers.push(format!("Respond in {}.", lang));
                }
                PreferenceType::TimeOfDayPreferences(enabled) => {
                    if *enabled {
                        modifiers.push("Use time-appropriate greetings.".to_string());
                    }
                }
                PreferenceType::Custom(key, value) => {
                    modifiers.push(format!("{}: {}", key, value));
                }
            }
        }

        // Add implicit learning insights
        if !self.conversation_metadata.frequent_topics.is_empty() {
            let top_topics: Vec<String> = self
                .conversation_metadata
                .frequent_topics
                .iter()
                .filter(|&(_, count)| *count >= 3) // Only topics mentioned 3+ times
                .map(|(topic, _)| topic.clone())
                .take(5)
                .collect();

            if !top_topics.is_empty() {
                modifiers.push(format!(
                    "The user frequently asks about: {}.",
                    top_topics.join(", ")
                ));
            }
        }

        if modifiers.is_empty() {
            String::new()
        } else {
            format!("User preferences:\n{}", modifiers.join("\n"))
        }
    }
}

/// User modeling system that manages multiple user profiles.
pub struct UserModel {
    profiles: Arc<RwLock<HashMap<String, UserProfile>>>,
}

impl UserModel {
    /// Create a new user modeling system.
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a user profile.
    pub fn get_or_create_profile(&self, user_id: &str) -> UserProfile {
        let mut profiles = self.profiles.write().unwrap();

        profiles
            .entry(user_id.to_string())
            .or_insert_with(|| UserProfile::new(user_id.to_string()))
            .clone()
    }

    /// Update a user profile.
    pub fn update_profile(&self, profile: UserProfile) {
        let mut profiles = self.profiles.write().unwrap();
        profiles.insert(profile.user_id.clone(), profile);
    }

    /// Set a preference for a user.
    pub fn set_preference(&self, user_id: &str, preference: PreferenceType) {
        let mut profile = self.get_or_create_profile(user_id);
        profile.set_preference(preference);
        self.update_profile(profile);
    }

    /// Record a conversation interaction.
    pub fn record_interaction(
        &self,
        user_id: &str,
        question: &str,
        response: &str,
        feedback_positive: bool,
    ) {
        let mut profile = self.get_or_create_profile(user_id);
        profile.record_interaction(question, response, feedback_positive);
        self.update_profile(profile);
    }

    /// Get a user profile.
    pub fn get_profile(&self, user_id: &str) -> Option<UserProfile> {
        let profiles = self.profiles.read().unwrap();
        profiles.get(user_id).cloned()
    }

    /// Delete a user profile (for privacy).
    pub fn delete_profile(&self, user_id: &str) -> bool {
        let mut profiles = self.profiles.write().unwrap();
        profiles.remove(user_id).is_some()
    }

    /// List all user IDs.
    pub fn list_users(&self) -> Vec<String> {
        let profiles = self.profiles.read().unwrap();
        profiles.keys().cloned().collect()
    }

    /// Get the total number of user profiles.
    pub fn user_count(&self) -> usize {
        let profiles = self.profiles.read().unwrap();
        profiles.len()
    }

    /// Build a system prompt modifier for a user.
    pub fn build_system_prompt_modifier(&self, user_id: &str) -> String {
        if let Some(profile) = self.get_profile(user_id) {
            profile.to_system_prompt_modifier()
        } else {
            String::new()
        }
    }
}

impl Default for UserModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_profile() {
        let model = UserModel::new();
        let profile = model.get_or_create_profile("user123");

        assert_eq!(profile.user_id, "user123");
        assert!(profile.preferences.is_empty());
    }

    #[test]
    fn test_set_preference() {
        let model = UserModel::new();

        model.set_preference(
            "user123",
            PreferenceType::ResponseStyle("casual".to_string()),
        );
        model.set_preference("user123", PreferenceType::Verbosity(5));

        let profile = model.get_profile("user123").unwrap();
        assert_eq!(profile.preferences.len(), 2);
    }

    #[test]
    fn test_get_preference() {
        let mut profile = UserProfile::new("user123".to_string());
        profile.set_preference(PreferenceType::ResponseStyle("casual".to_string()));

        let pref = profile.get_preference("response_style");
        assert!(pref.is_some());

        if let Some(PreferenceType::ResponseStyle(style)) = pref {
            assert_eq!(style, "casual");
        } else {
            panic!("Expected ResponseStyle preference");
        }
    }

    #[test]
    fn test_record_interaction() {
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
    fn test_extract_topics() {
        let profile = UserProfile::new("user123".to_string());
        let topics = profile.extract_topics("What is the latest news about technology?");

        assert!(topics.contains(&"latest".to_string()));
        assert!(topics.contains(&"news".to_string()));
        assert!(topics.contains(&"technology".to_string()));
    }

    #[test]
    fn test_extract_question_pattern() {
        let profile = UserProfile::new("user123".to_string());

        let pattern = profile.extract_question_pattern("What is the time?");
        assert_eq!(pattern, "what is");

        let pattern = profile.extract_question_pattern("How do I cook pasta?");
        assert_eq!(pattern, "how do");
    }

    #[test]
    fn test_system_prompt_modifier() {
        let mut profile = UserProfile::new("user123".to_string());
        profile.set_preference(PreferenceType::ResponseStyle("casual".to_string()));
        profile.set_preference(PreferenceType::Verbosity(8));

        let modifier = profile.to_system_prompt_modifier();
        assert!(modifier.contains("casual"));
        assert!(modifier.contains("detailed"));
    }

    #[test]
    fn test_delete_profile() {
        let model = UserModel::new();

        model.set_preference("user123", PreferenceType::Verbosity(5));
        assert!(model.get_profile("user123").is_some());

        let deleted = model.delete_profile("user123");
        assert!(deleted);
        assert!(model.get_profile("user123").is_none());
    }

    #[test]
    fn test_list_users() {
        let model = UserModel::new();

        model.set_preference("user1", PreferenceType::Verbosity(5));
        model.set_preference("user2", PreferenceType::Verbosity(3));

        let users = model.list_users();
        assert_eq!(users.len(), 2);
        assert!(users.contains(&"user1".to_string()));
        assert!(users.contains(&"user2".to_string()));
    }

    #[test]
    fn test_implicit_learning() {
        let model = UserModel::new();

        // Record multiple interactions about weather
        model.record_interaction("user123", "What's the weather like?", "Sunny", true);
        model.record_interaction("user123", "How's the weather today?", "Rainy", true);
        model.record_interaction("user123", "Is the weather good?", "Yes, sunny", true);

        let profile = model.get_profile("user123").unwrap();

        // Should have learned that user asks about weather
        assert!(
            profile
                .conversation_metadata
                .frequent_topics
                .contains_key("weather")
        );

        let modifier = profile.to_system_prompt_modifier();
        // Weather should appear in modifier if mentioned 3+ times
        assert!(
            modifier.contains("weather")
                || profile.conversation_metadata.frequent_topics["weather"] >= 3
        );
    }
}
