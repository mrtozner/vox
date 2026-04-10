//! Voice-optimized system prompts for LLM chat.
//!
//! Provides prompts optimized for text-to-speech (TTS) output:
//! - Short sentences for better pacing
//! - Conversational tone without markdown
//! - Natural speech patterns

/// Prompt optimization mode for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePromptMode {
    /// Standard mode: technical responses with markdown formatting.
    Standard,
    /// Voice mode: conversational responses optimized for TTS.
    Voice,
}

/// Build a system prompt based on the selected mode.
///
/// # Arguments
/// * `mode` - The prompt optimization mode
///
/// # Returns
/// A system prompt string suitable for LLM chat
pub fn build_system_prompt(mode: VoicePromptMode) -> String {
    match mode {
        VoicePromptMode::Standard => STANDARD_PROMPT.to_string(),
        VoicePromptMode::Voice => VOICE_PROMPT.to_string(),
    }
}

/// Standard system prompt for text-based chat.
///
/// Allows markdown, code blocks, and technical formatting.
const STANDARD_PROMPT: &str = "\
You are a helpful AI assistant. Provide clear and accurate responses. \
You may use markdown formatting, code blocks, and technical language \
as appropriate for the user's questions.";

/// Voice-optimized system prompt for TTS output.
///
/// Based on 2026 TTS best practices:
/// - Short sentences (max 15-20 words)
/// - Conversational tone with contractions
/// - No markdown formatting
/// - Natural speech patterns
/// - Spell out acronyms
const VOICE_PROMPT: &str = "\
You are a voice assistant. Keep responses conversational.

Rules for natural speech:
- Use short, clear sentences. Max twenty words per sentence.
- Speak naturally. Use contractions like you're, it's, don't.
- Avoid markdown. No asterisks, hashes, or code blocks.
- List items as first, second, third. Not bullet points.
- Spell out acronyms. Say A-P-I, not API.
- Keep technical explanations simple.
- Sound like a helpful friend talking.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_prompt() {
        let prompt = build_system_prompt(VoicePromptMode::Standard);
        assert!(prompt.contains("markdown"));
        assert!(prompt.contains("helpful AI assistant"));
    }

    #[test]
    fn test_voice_prompt() {
        let prompt = build_system_prompt(VoicePromptMode::Voice);
        assert!(prompt.contains("voice assistant"));
        assert!(prompt.contains("short"));
        assert!(prompt.contains("conversational"));
        assert!(!prompt.contains("markdown formatting is allowed"));
    }

    #[test]
    fn test_different_modes_produce_different_prompts() {
        let standard = build_system_prompt(VoicePromptMode::Standard);
        let voice = build_system_prompt(VoicePromptMode::Voice);
        assert_ne!(standard, voice);
    }
}
