//! Live speaker identification with real microphone input.
//!
//! This demo works RIGHT NOW - no special model needed!
//! Uses voice characteristics (pitch, energy) to identify speakers.
//!
//! Run with:
//! ```bash
//! cargo run --example live_diarization --features whisper,silero,diarization
//! ```

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use vox::diarization::{RecognitionConfig, SpeakerDatabase, SpeakerRegistry};
use vox::{SileroVad, Vox, VoxContext, WhisperBackend};

/// Extract voice characteristics as a simple embedding
fn extract_voice_features(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0; 8];
    }

    // Calculate basic audio statistics
    let mean: f32 = audio.iter().sum::<f32>() / audio.len() as f32;
    let variance: f32 = audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / audio.len() as f32;
    let std_dev = variance.sqrt();

    // Energy (RMS)
    let rms: f32 = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

    // Zero crossing rate (pitch indicator)
    let mut zcr = 0;
    for i in 1..audio.len() {
        if (audio[i] >= 0.0 && audio[i - 1] < 0.0) || (audio[i] < 0.0 && audio[i - 1] >= 0.0) {
            zcr += 1;
        }
    }
    let zcr_rate = zcr as f32 / audio.len() as f32;

    // Spectral characteristics (simplified)
    let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let min_val = audio.iter().map(|x| x.abs()).fold(1.0f32, f32::min);

    // Peak-to-average ratio
    let peak_avg = if rms > 0.0 { max_val / rms } else { 0.0 };

    // Dynamic range
    let dynamic_range = max_val - min_val;

    // High frequency energy (approximation)
    let high_freq_energy: f32 =
        audio.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / audio.len() as f32;

    vec![
        mean,
        std_dev,
        rms,
        zcr_rate * 100.0, // Scale up for better discrimination
        max_val,
        peak_avg,
        dynamic_range,
        high_freq_energy * 10.0, // Scale up
    ]
}

/// Normalize a vector (L2 normalization)
fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        vec.iter().map(|x| x / magnitude).collect()
    } else {
        vec.to_vec()
    }
}

struct SpeakerUI {
    registry: Arc<Mutex<SpeakerRegistry>>,
    db: Arc<Mutex<SpeakerDatabase>>,
}

impl SpeakerUI {
    async fn new() -> anyhow::Result<Self> {
        let config = RecognitionConfig {
            threshold: 0.65, // Lower threshold for simplified features
            require_threshold: true,
        };
        let registry = Arc::new(Mutex::new(SpeakerRegistry::with_config(config)));
        let db = Arc::new(Mutex::new(SpeakerDatabase::open("speakers_live.db").await?));

        Ok(Self { registry, db })
    }

    async fn enroll_speaker(&self, name: &str, audio: &[f32]) -> anyhow::Result<()> {
        let features = extract_voice_features(audio);
        let embedding = normalize_vector(&features);

        let id = format!("speaker_{}", name.to_lowercase().replace(' ', "_"));

        {
            let mut reg = self.registry.lock().unwrap();
            reg.enroll(&id, name, embedding)?;
        }

        // Store in database
        let speaker = {
            let reg = self.registry.lock().unwrap();
            reg.list_speakers()
                .into_iter()
                .find(|s| s.id == id)
                .expect("Just enrolled speaker should exist")
        };

        {
            let db = self.db.lock().unwrap();
            let runtime = tokio::runtime::Handle::current();
            runtime.block_on(async { db.store_speaker(&speaker).await })?;
        }

        Ok(())
    }

    fn identify_speaker(&self, audio: &[f32]) -> anyhow::Result<(String, f32)> {
        let features = extract_voice_features(audio);
        let embedding = normalize_vector(&features);

        let reg = self.registry.lock().unwrap();
        let recognition = reg.identify(&embedding)?;

        match recognition {
            vox::diarization::Recognition::Identified {
                speaker_id,
                confidence,
            } => {
                let speaker_name = reg
                    .list_speakers()
                    .into_iter()
                    .find(|s| s.id == speaker_id)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());

                Ok((speaker_name, confidence))
            }
            vox::diarization::Recognition::Unknown { best_score } => {
                Ok(("Unknown".to_string(), best_score))
            }
        }
    }

    async fn record_utterance(
        &self,
        speaker_name: &str,
        text: &str,
        duration_ms: u64,
    ) -> anyhow::Result<()> {
        let id = format!("speaker_{}", speaker_name.to_lowercase().replace(' ', "_"));

        let db = self.db.lock().unwrap();
        let runtime = tokio::runtime::Handle::current();
        runtime.block_on(async { db.record_utterance(&id, duration_ms, Some(text)).await })?;

        Ok(())
    }

    fn list_speakers(&self) -> Vec<String> {
        let reg = self.registry.lock().unwrap();
        reg.list_speakers().iter().map(|s| s.name.clone()).collect()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     🎤 LIVE SPEAKER IDENTIFICATION DEMO 🎤              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Check models
    let vad_model = "models/silero_vad.onnx";
    let stt_model = "models/ggml-tiny.en.bin";

    if !std::path::Path::new(vad_model).exists() || !std::path::Path::new(stt_model).exists() {
        eprintln!("❌ Models not found!");
        eprintln!("Run: bash scripts/download_models.sh");
        return Ok(());
    }

    println!("📦 Loading models...");
    let vad = SileroVad::new(vad_model)?;
    let stt = WhisperBackend::from_model(stt_model)?;
    println!("✅ Models loaded\n");

    let speaker_ui = Arc::new(SpeakerUI::new().await?);

    // Check for existing speakers
    let existing_speakers = speaker_ui.list_speakers();
    if !existing_speakers.is_empty() {
        println!("📊 Enrolled speakers from database:");
        for (i, name) in existing_speakers.iter().enumerate() {
            println!("   {}. {}", i + 1, name);
        }
        println!();
    }

    // Enrollment mode
    println!("🎯 ENROLLMENT MODE");
    println!("══════════════════\n");
    println!("How many speakers do you want to enroll? (1-4): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let num_speakers: usize = input.trim().parse().unwrap_or(2);

    println!("\nGreat! I'll enroll {} speaker(s).\n", num_speakers);

    let speaker_ui_clone = Arc::clone(&speaker_ui);

    // Callback for enrollment
    let enrollment_samples: Arc<Mutex<Vec<(String, Vec<f32>)>>> = Arc::new(Mutex::new(Vec::new()));
    let enrollment_samples_clone = Arc::clone(&enrollment_samples);

    for i in 0..num_speakers {
        println!("👤 SPEAKER {} ENROLLMENT", i + 1);
        println!("─────────────────────────");

        print!("Enter speaker {} name: ", i + 1);
        io::stdout().flush()?;

        let mut name = String::new();
        io::stdin().read_line(&mut name)?;
        let name = name.trim().to_string();

        println!("\n📢 {} please speak for 3-5 seconds...", name);
        println!(
            "   Say something like: 'Hello, my name is {}, testing speaker identification'",
            name
        );
        println!("   Press Enter when ready...");

        let mut ready = String::new();
        io::stdin().read_line(&mut ready)?;

        println!("🔴 Recording... (speak now!)");

        let enrollment_samples_capture = Arc::clone(&enrollment_samples_clone);
        let name_capture = name.clone();

        let vox_enrollment = Vox::builder()
            .vad(SileroVad::new(vad_model)?)
            .stt(WhisperBackend::from_model(stt_model)?)
            .on_utterance(move |result, ctx| {
                println!("   ✓ Captured: \"{}\"", result.text);

                // Store the audio sample
                let audio_samples = ctx
                    .last_utterance
                    .as_ref()
                    .map(|u| u.audio.samples.clone())
                    .unwrap_or_default();

                enrollment_samples_capture
                    .lock()
                    .unwrap()
                    .push((name_capture.clone(), audio_samples));
            })
            .build()?;

        // Listen for one utterance
        tokio::select! {
            _ = vox_enrollment.listen() => {},
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                println!("   ⏱️  Timeout - moving on");
            }
        }

        // Process enrollment
        let samples = enrollment_samples.lock().unwrap();
        if let Some((speaker_name, audio)) = samples.last() {
            speaker_ui.enroll_speaker(speaker_name, audio).await?;
            println!("   ✅ {} enrolled!\n", speaker_name);
        }
        drop(samples);
    }

    println!("💾 All speakers enrolled!\n");

    // Identification mode
    println!("\n🎙️  IDENTIFICATION MODE");
    println!("═══════════════════════\n");
    println!("Now I'll listen and identify who's speaking.");
    println!("Take turns speaking naturally.\n");
    println!("Press Ctrl+C to stop.\n");

    // Set up UI display
    let speaker_ui_identify = Arc::clone(&speaker_ui_clone);

    let vox = Vox::builder()
        .vad(vad)
        .stt(stt)
        .on_utterance(move |result, ctx| {
            let audio_samples = ctx
                .last_utterance
                .as_ref()
                .map(|u| u.audio.samples.clone())
                .unwrap_or_default();

            if audio_samples.is_empty() {
                return;
            }

            match speaker_ui_identify.identify_speaker(&audio_samples) {
                Ok((speaker_name, confidence)) => {
                    // UI Display
                    let confidence_bar = {
                        let filled = (confidence * 10.0) as usize;
                        let empty = 10 - filled;
                        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
                    };

                    let color = if confidence > 0.8 {
                        "\x1b[32m" // Green for high confidence
                    } else if confidence > 0.65 {
                        "\x1b[33m" // Yellow for medium confidence
                    } else {
                        "\x1b[90m" // Gray for low confidence
                    };

                    println!("\n┌─────────────────────────────────────────────────────────┐");
                    println!(
                        "│ 🎤 Speaker: {}{:20}\x1b[0m                       │",
                        color, speaker_name
                    );
                    println!(
                        "│ 📊 Confidence: {} {:.0}%                         │",
                        confidence_bar,
                        confidence * 100.0
                    );
                    println!("└─────────────────────────────────────────────────────────┘");
                    println!("💬 {}: \"{}\"", speaker_name, result.text);
                    println!();

                    // Record in database
                    if speaker_name != "Unknown" {
                        let duration_ms = (audio_samples.len() as f64 / 16000.0 * 1000.0) as u64;
                        let ui = Arc::clone(&speaker_ui_identify);
                        let text = result.text.clone();
                        let name = speaker_name.clone();

                        tokio::spawn(async move {
                            let _ = ui.record_utterance(&name, &text, duration_ms).await;
                        });
                    }
                }
                Err(e) => {
                    eprintln!("❌ Identification error: {}", e);
                }
            }
        })
        .build()?;

    println!("🎧 Listening for speakers...\n");
    vox.listen().await?;

    Ok(())
}
