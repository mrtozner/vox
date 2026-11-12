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

/// Build a system prompt that also fixes the reply language.
///
/// `language` is a language code or name as given to `--language` (for
/// example `"tr"` or `"german"`). `None`, an empty string, `"auto"` and
/// English leave the base prompt unchanged, so the default prompt stays
/// byte-identical and its cached prefix stays valid.
pub fn build_system_prompt_for_language(mode: VoicePromptMode, language: Option<&str>) -> String {
    let base = build_system_prompt(mode);
    let Some(lang) = language.map(str::trim).filter(|l| !l.is_empty()) else {
        return base;
    };
    let lower = lang.to_lowercase();
    if lower == "auto" || lower == "english" || lower == "en" || lower.starts_with("en-") {
        return base;
    }
    format!(
        "{base}\n\nThe user speaks the language with code or name \"{lang}\". \
         Always reply in that language."
    )
}

/// Build a system prompt with environment facts injected.
///
/// Used when the server/CLI has an environment-aware `CapabilityRegistry`
/// available. Falls back to identical output as `build_system_prompt` if
/// the registry is None-ish.
#[cfg(any(feature = "cli", feature = "server"))]
pub fn build_system_prompt_with_registry(
    mode: VoicePromptMode,
    registry: &crate::capabilities::CapabilityRegistry,
) -> String {
    let base = build_system_prompt(mode);
    registry.inject_into_prompt(&base)
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
/// - Short sentences (max 15-20 words)
/// - Conversational tone with contractions
/// - No markdown, emoji, links or code: the reply is read aloud
/// - Start with a short opening clause so speech can begin early
/// - Spell out acronyms
const VOICE_PROMPT: &str = "\
You are a voice assistant. Your reply is read aloud. Keep responses conversational.

Rules for natural speech:
- Use short, clear sentences. Max twenty words per sentence.
- Start with a short first sentence.
- Speak naturally. Use contractions like you're, it's, don't.
- Avoid markdown. No asterisks, hashes, or code blocks.
- No emoji, no links, no tables.
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
    fn language_prompt_is_unchanged_for_default_and_english() {
        let base = build_system_prompt(VoicePromptMode::Voice);
        for lang in [
            None,
            Some(""),
            Some("auto"),
            Some("en"),
            Some("en-US"),
            Some("English"),
        ] {
            assert_eq!(
                build_system_prompt_for_language(VoicePromptMode::Voice, lang),
                base
            );
        }
    }

    #[test]
    fn language_prompt_names_the_language() {
        let base = build_system_prompt(VoicePromptMode::Voice);
        let prompt = build_system_prompt_for_language(VoicePromptMode::Voice, Some("tr"));
        assert!(prompt.starts_with(&base));
        assert!(prompt.contains("\"tr\""));
        assert!(prompt.contains("reply in that language"));
    }

    #[test]
    fn test_different_modes_produce_different_prompts() {
        let standard = build_system_prompt(VoicePromptMode::Standard);
        let voice = build_system_prompt(VoicePromptMode::Voice);
        assert_ne!(standard, voice);
    }
}
