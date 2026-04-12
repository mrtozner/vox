//! Intelligence layer for Vox voice assistant.
//!
//! Provides semantic caching, user modeling, and voice memory integration
//! to make conversations more intelligent and personalized.
//!
//! # Features
//!
//! - **Semantic Caching**: Cache LLM responses based on semantic similarity
//! - **User Modeling**: Learn user preferences and conversation patterns
//! - **Voice Memory**: Unified interface for personalized voice interactions
//!
//! # Privacy
//!
//! All data is stored locally. No external services are used.
//! Users have full control over their data via the privacy dashboard.
//!
//! # Example
//!
//! ```no_run
//! use vox::intelligence::{VoiceMemory, PreferenceType};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let memory = VoiceMemory::new();
//!
//! // Enroll a new speaker
//! memory.enroll_speaker("speaker_001", Some("Alice"));
//!
//! // Set preferences
//! memory.set_preference("speaker_001", PreferenceType::ResponseStyle("casual".to_string()));
//!
//! // Process an interaction with memory
//! let (response, from_cache, welcome) = memory.process_interaction(
//!     Some("speaker_001"),
//!     "What's the weather like?",
//!     |system_prompt_modifier| async move {
//!         // Generate LLM response with personalization
//!         Ok("It's sunny and warm today!".to_string())
//!     }
//! ).await?;
//!
//! println!("Response: {}", response);
//! println!("From cache: {}", from_cache);
//! if let Some(msg) = welcome {
//!     println!("Welcome message: {}", msg);
//! }
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod user_model;
pub mod voice_memory;

pub use cache::{CacheMetrics, SemanticCache, SemanticCacheConfig};
pub use user_model::{ConversationMetadata, PreferenceType, UserModel, UserProfile};
pub use voice_memory::{PrivacyDashboard, SpeakerInfo, VoiceMemory, VoiceMemoryBuilder};
