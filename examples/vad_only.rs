//! VAD-only example -- detect speech segments without transcription.
//!
//! Listens to the default microphone and prints when speech starts
//! and stops, along with the duration of each utterance.
//!
//! Requirements:
//!   - Download Silero VAD: `wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
//!
//! Run:
//!   cargo run --example vad_only

use vox::SileroVad;
use vox::audio::{AudioCapture, AudioResampler};
use vox::traits::{VadBackend, VadEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut vad = SileroVad::new("silero_vad.onnx")?;
    let target_rate = vad.sample_rate();
    let frame_size = vad.frame_size();

    let (capture, mut audio_rx) = AudioCapture::new(16000, 1)?;
    let mut resampler = AudioResampler::new(16000, target_rate)?;

    capture.start()?;
    println!("Listening for speech... (Ctrl+C to stop)");

    loop {
        tokio::select! {
            chunk = audio_rx.recv() => {
                let chunk = match chunk {
                    Some(c) => c,
                    None => break,
                };

                let resampled = resampler.process(&chunk)?;

                for frame_samples in resampled.samples.chunks(frame_size) {
                    if frame_samples.len() < frame_size {
                        continue;
                    }

                    let frame = vox::AudioChunk {
                        samples: frame_samples.to_vec(),
                        sample_rate: target_rate,
                        channels: 1,
                    };

                    let events = vad.process_frame(&frame).await?;
                    for event in events {
                        match event {
                            VadEvent::SpeechStart => {
                                println!("[SPEECH START]");
                            }
                            VadEvent::SpeechEnd(utterance) => {
                                println!(
                                    "[SPEECH END] duration: {}ms ({} samples)",
                                    utterance.duration_ms,
                                    utterance.audio.samples.len()
                                );
                            }
                            VadEvent::Silence => {}
                        }
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nStopping...");
                break;
            }
        }
    }

    capture.stop()?;
    Ok(())
}
