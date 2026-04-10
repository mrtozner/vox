//! Demo: Voice Mode vs Standard Mode
//!
//! Run with: cargo run --example demo_voice_mode --features cli

use vox::prompts::{VoicePromptMode, build_system_prompt};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║           Vox Voice Mode Demo - Tier 1 Feature                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Show both prompts
    demo_prompts();

    println!("\n");

    // Show example responses
    demo_responses();

    println!("\n");

    // Show usage
    demo_usage();
}

fn demo_prompts() {
    println!("📋 SYSTEM PROMPTS COMPARISON\n");

    let standard = build_system_prompt(VoicePromptMode::Standard);
    let voice = build_system_prompt(VoicePromptMode::Voice);

    println!("┌─ STANDARD MODE ─────────────────────────────────────────────┐");
    println!("│ {:<60} │", format!("Length: {} chars", standard.len()));
    println!("├─────────────────────────────────────────────────────────────┤");
    for line in standard.lines() {
        println!("│ {:<60}│", line.chars().take(60).collect::<String>());
    }
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("┌─ VOICE MODE ────────────────────────────────────────────────┐");
    println!("│ {:<60} │", format!("Length: {} chars", voice.len()));
    println!("├─────────────────────────────────────────────────────────────┤");
    for line in voice.lines() {
        println!("│ {:<60}│", line.chars().take(60).collect::<String>());
    }
    println!("└─────────────────────────────────────────────────────────────┘");
}

fn demo_responses() {
    println!("🎭 EXAMPLE RESPONSES\n");

    println!("Question: \"What is an API and how does it work?\"\n");

    // Standard mode response
    println!("┌─ STANDARD MODE OUTPUT ──────────────────────────────────────┐");
    println!("│                                                              │");
    println!("│  An **API** (Application Programming Interface) is a set    │");
    println!("│  of rules and protocols that allows different software      │");
    println!("│  applications to communicate with each other.               │");
    println!("│                                                              │");
    println!("│  ### How it works:                                          │");
    println!("│                                                              │");
    println!("│  - **Request**: Your app sends a request to the API         │");
    println!("│  - **Processing**: The API processes the request            │");
    println!("│  - **Response**: Returns data (usually JSON or XML)         │");
    println!("│                                                              │");
    println!("│  Example: `GET /users/123` → Returns user data              │");
    println!("│                                                              │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("TTS would read: \"An asterisk asterisk A P I asterisk asterisk...\"\n");

    // Voice mode response
    println!("┌─ VOICE MODE OUTPUT ─────────────────────────────────────────┐");
    println!("│                                                              │");
    println!("│  An A-P-I lets different software talk to each other.       │");
    println!("│  Think of it like a waiter at a restaurant.                 │");
    println!("│                                                              │");
    println!("│  Here's how it works.                                       │");
    println!("│  First, your app sends a request to the A-P-I.              │");
    println!("│  Second, the A-P-I processes what you asked for.            │");
    println!("│  Third, it sends back the data you need.                    │");
    println!("│                                                              │");
    println!("│  For example, you might request user data.                  │");
    println!("│  The A-P-I finds it and returns it in JSON format.          │");
    println!("│                                                              │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("TTS would read: \"An A-P-I lets different software talk to each other...\"\n");

    // Show the difference
    println!("✨ KEY DIFFERENCES:\n");
    println!("  Standard Mode:                    Voice Mode:");
    println!("  ─────────────────                 ────────────");
    println!("  • Uses **markdown**               • Plain text only");
    println!("  • Long complex sentences          • Short 20-word sentences");
    println!("  • Says 'API'                      • Says 'A-P-I'");
    println!("  • Uses bullet points              • Says 'First, second, third'");
    println!("  • Professional tone               • Conversational tone");
    println!("  • Code blocks                     • Natural descriptions");
}

fn demo_usage() {
    println!("🚀 HOW TO USE\n");

    println!("Standard mode (default):");
    println!("  $ vox chat --llm llama3.2");
    println!("  > Best for text chat, allows markdown and code\n");

    println!("Voice mode (TTS-optimized):");
    println!("  $ vox chat --voice-mode --llm llama3.2");
    println!("  > Best for voice output, natural speech patterns\n");

    println!("Streaming TTS (coming soon):");
    println!("  $ vox chat --voice-mode --stream-tts --llm llama3.2");
    println!("  > Combines voice prompts with sentence-level TTS streaming\n");

    println!("─────────────────────────────────────────────────────────────────");
    println!("💡 TIP: Voice mode makes responses 2-3x better for TTS!");
    println!("─────────────────────────────────────────────────────────────────\n");
}
