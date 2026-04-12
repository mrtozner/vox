//! Live speaker identification demo with real microphone input.
//!
//! This example demonstrates real-time speaker identification:
//! 1. Enroll yourself and your friend
//! 2. Take turns speaking
//! 3. See who's identified in real-time
//!
//! Run with:
//! ```bash
//! cargo run --example live_speaker_test --features whisper,silero,diarization
//! ```

use std::sync::Arc;
use tokio::sync::Mutex;
use vox::audio::AudioCapture;
use vox::diarization::{RecognitionConfig, SpeakerDatabase, SpeakerRegistry};
use vox::{AudioChunk, SileroVad, SttBackend, VadBackend, VadEvent, WhisperBackend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🎤 Live Speaker Identification Demo");
    println!("====================================\n");

    // Check models
    let vad_model = "models/silero_vad.onnx";
    let stt_model = "models/ggml-tiny.en.bin";

    if !std::path::Path::new(vad_model).exists() {
        eprintln!("❌ VAD model not found: {}", vad_model);
        eprintln!("Run: bash scripts/download_models.sh");
        return Ok(());
    }

    if !std::path::Path::new(stt_model).exists() {
        eprintln!("❌ STT model not found: {}", stt_model);
        eprintln!("Run: bash scripts/download_models.sh");
        return Ok(());
    }

    println!("📦 Loading models...");
    let mut vad = SileroVad::new(vad_model)?;
    let stt = WhisperBackend::from_model(stt_model)?;
    println!("✅ Models loaded\n");

    // Initialize speaker system
    let db = SpeakerDatabase::open("speakers_live.db").await?;
    let recognition_config = RecognitionConfig {
        threshold: 0.7,
        require_threshold: true,
    };
    let registry = Arc::new(Mutex::new(SpeakerRegistry::with_config(recognition_config)));

    println!("📊 Current enrolled speakers:");
    {
        let reg = registry.lock().await;
        if reg.speaker_count() == 0 {
            println!("   (No speakers enrolled yet)\n");
        } else {
            for speaker in reg.list_speakers() {
                println!("   ✓ {} ({})", speaker.name, speaker.id);
            }
            println!();
        }
    }

    // Note: For real speaker identification, you need a speaker encoder model
    // For now, we'll use a simplified approach based on voice characteristics

    println!("🎯 ENROLLMENT MODE");
    println!("==================\n");

    println!("Let's enroll speakers. We'll capture voice samples and create profiles.\n");
    println!("NOTE: This demo uses a simplified approach. For production:");
    println!("      1. Download a speaker encoder model (ECAPA-TDNN)");
    println!("      2. Extract embeddings from audio");
    println!("      3. Use those embeddings for identification\n");

    // Enrollment workflow
    println!("Ready to enroll speakers? (y/n)");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() == "y" {
        println!("\n👤 SPEAKER 1 ENROLLMENT");
        println!("------------------------");
        println!("Speaker 1, please speak for 5 seconds...");
        println!("Say something like: 'Hello, my name is [name], testing speaker identification'");
        println!("Press Enter when ready...");
        let mut ready = String::new();
        std::io::stdin().read_line(&mut ready)?;

        // Capture enrollment sample
        let sample1 = capture_enrollment_sample(&mut vad, &stt, 5).await?;

        println!("Enter speaker 1 name: ");
        let mut name1 = String::new();
        std::io::stdin().read_line(&mut name1)?;
        let name1 = name1.trim();
        let id1 = format!("speaker_{}", name1.to_lowercase());

        // Create simplified embedding (in production, use speaker encoder)
        let embedding1 = create_simple_embedding(&sample1);

        {
            let mut reg = registry.lock().await;
            reg.enroll(&id1, name1, embedding1)?;
        }
        println!("✅ {} enrolled!\n", name1);

        // Second speaker
        println!("👤 SPEAKER 2 ENROLLMENT");
        println!("------------------------");
        println!("Speaker 2, please speak for 5 seconds...");
        println!("Say something like: 'Hi, I'm [name], this is my voice profile'");
        println!("Press Enter when ready...");
        let mut ready = String::new();
        std::io::stdin().read_line(&mut ready)?;

        let sample2 = capture_enrollment_sample(&mut vad, &stt, 5).await?;

        println!("Enter speaker 2 name: ");
        let mut name2 = String::new();
        std::io::stdin().read_line(&mut name2)?;
        let name2 = name2.trim();
        let id2 = format!("speaker_{}", name2.to_lowercase());

        let embedding2 = create_simple_embedding(&sample2);

        {
            let mut reg = registry.lock().await;
            reg.enroll(&id2, name2, embedding2)?;
        }
        println!("✅ {} enrolled!\n", name2);

        // Save to database
        {
            let reg = registry.lock().await;
            for speaker in reg.list_speakers() {
                db.store_speaker(&speaker).await?;
            }
        }
        println!("💾 Speakers saved to database\n");
    }

    println!("\n🎙️  IDENTIFICATION MODE");
    println!("======================\n");
    println!("Now take turns speaking. I'll try to identify who's talking.");
    println!("Speak naturally for a few seconds, then press Enter.");
    println!("Type 'quit' to exit.\n");

    // Main identification loop
    let (audio_capture, mut audio_rx) = AudioCapture::new(16000, 1)?;

    loop {
        println!("🔴 Ready to listen... Press Enter when someone starts speaking");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "quit" {
            break;
        }

        println!("🎤 Listening...");

        // Capture and process speech
        let audio_chunk = capture_utterance(&mut vad, &mut audio_rx).await?;

        if audio_chunk.samples.is_empty() {
            println!("   No speech detected. Try again.\n");
            continue;
        }

        // Create utterance for transcription
        let utterance = vox::Utterance {
            audio: audio_chunk.clone(),
            duration_ms: (audio_chunk.samples.len() as f32 / 16000.0 * 1000.0) as u64,
            speaker_id: None,
        };

        // Transcribe
        let transcription = stt.transcribe(&utterance).await?;

        // Identify speaker (simplified)
        let embedding = create_simple_embedding(&audio_chunk);

        let reg = registry.lock().await;
        let recognition = reg.identify(&embedding)?;
        drop(reg);

        match recognition {
            vox::diarization::Recognition::Identified {
                speaker_id,
                confidence,
            } => {
                let reg = registry.lock().await;
                let speaker = reg
                    .list_speakers()
                    .into_iter()
                    .find(|s| s.id == speaker_id)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());

                println!(
                    "\n✅ Identified: {} ({:.0}% confidence)",
                    speaker,
                    confidence * 100.0
                );
                println!("   Said: \"{}\"\n", transcription.text);

                // Record in database
                db.record_utterance(
                    &speaker_id,
                    utterance.audio.samples.len() as u64,
                    Some(&transcription.text),
                )
                .await?;
            }
            vox::diarization::Recognition::Unknown { best_score } => {
                println!(
                    "\n❓ Unknown speaker (best match: {:.0}%)",
                    best_score * 100.0
                );
                println!("   Said: \"{}\"\n", transcription.text);
            }
        }
    }

    println!("\n📊 Session Summary");
    println!("==================");

    let speakers = db.list_speakers().await?;
    for speaker in speakers {
        let history = db.get_conversation_history(&speaker.id, 100).await?;
        println!("\n👤 {}", speaker.name);
        println!("   Total utterances: {}", history.len());
        if !history.is_empty() {
            println!("   Recent:");
            for (i, entry) in history.iter().take(3).enumerate() {
                if let Some(text) = &entry.text {
                    println!("   {}. \"{}\"", i + 1, text);
                }
            }
        }
    }

    println!("\n✨ Session complete!");
    Ok(())
}

/// Capture an enrollment sample from microphone
async fn capture_enrollment_sample(
    vad: &mut SileroVad,
    stt: &WhisperBackend,
    duration_secs: u32,
) -> anyhow::Result<AudioChunk> {
    let (_audio_capture, mut audio_rx) = AudioCapture::new(16000, 1)?;
    let mut collected_audio = Vec::new();
    let target_samples = (16000 * duration_secs) as usize;

    println!("🔴 Recording... ({}s)", duration_secs);

    while collected_audio.len() < target_samples {
        if let Some(chunk) = audio_rx.recv().await {
            collected_audio.extend_from_slice(&chunk.samples);
        }
    }

    let audio = AudioChunk {
        samples: collected_audio[..target_samples].to_vec(),
        sample_rate: 16000,
        channels: 1,
    };

    // Verify quality - create utterance for transcription
    let utterance = vox::Utterance {
        audio: audio.clone(),
        duration_ms: (duration_secs * 1000) as u64,
        speaker_id: None,
    };
    let transcription = stt.transcribe(&utterance).await?;
    println!("   Transcribed: \"{}\"", transcription.text);

    Ok(audio)
}

/// Capture a single utterance from microphone
async fn capture_utterance(
    vad: &mut SileroVad,
    audio_rx: &mut tokio::sync::mpsc::Receiver<AudioChunk>,
) -> anyhow::Result<AudioChunk> {
    let mut utterance_audio = Vec::new();
    let mut is_speaking = false;
    let frame_size = vad.frame_size();

    // Capture frames until speech ends
    for _ in 0..300 {
        // Max ~10 seconds
        if let Some(chunk) = audio_rx.recv().await {
            for frame_start in (0..chunk.samples.len()).step_by(frame_size) {
                let frame_end = (frame_start + frame_size).min(chunk.samples.len());
                if frame_end - frame_start < frame_size {
                    break;
                }

                let frame = AudioChunk {
                    samples: chunk.samples[frame_start..frame_end].to_vec(),
                    sample_rate: 16000,
                    channels: 1,
                };

                let events = vad.process_frame(&frame).await?;

                for event in events {
                    match event {
                        VadEvent::SpeechStart => {
                            is_speaking = true;
                            println!("   🎤 Speech detected...");
                        }
                        VadEvent::SpeechEnd(utterance) => {
                            println!("   ✓ Speech ended");
                            return Ok(utterance.audio);
                        }
                        VadEvent::Silence => {}
                    }
                }
            }
        }
    }

    Ok(AudioChunk {
        samples: utterance_audio,
        sample_rate: 16000,
        channels: 1,
    })
}

/// Create a simple embedding from audio (for demo purposes)
/// In production, use SpeakerEmbedding::extract() with a real model
fn create_simple_embedding(audio: &AudioChunk) -> Vec<f32> {
    // Very simplified: use basic audio statistics as "embedding"
    // In production, use a proper speaker encoder model
    let mean: f32 = audio.samples.iter().sum::<f32>() / audio.samples.len() as f32;
    let variance: f32 = audio
        .samples
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / audio.samples.len() as f32;

    let energy: f32 =
        audio.samples.iter().map(|x| x.abs()).sum::<f32>() / audio.samples.len() as f32;

    vec![mean, variance.sqrt(), energy]
}
