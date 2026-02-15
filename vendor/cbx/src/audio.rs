use std::path::Path;

use anyhow::{Context, Result, bail};

pub const TARGET_SAMPLE_RATE: u32 = 24_000;

pub fn read_wav_as_f32_mono_24k(path: &Path) -> Result<Vec<f32>> {
    let (mut samples, sr) = read_wav_as_f32_mono(path)?;
    if sr != TARGET_SAMPLE_RATE {
        samples = resample_linear_mono(&samples, sr, TARGET_SAMPLE_RATE);
    }
    Ok(samples)
}

pub fn write_wav_f32_mono_24k(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).with_context(|| "failed to create wav writer")?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

fn read_wav_as_f32_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav {}", path.display()))?;
    let spec = reader.spec();

    if spec.sample_format == hound::SampleFormat::Int {
        if spec.bits_per_sample == 16 {
            let mut out = Vec::new();
            let mut frame = Vec::with_capacity(spec.channels as usize);
            for s in reader.samples::<i16>() {
                frame.push(s? as f32 / i16::MAX as f32);
                if frame.len() == spec.channels as usize {
                    out.push(frame.iter().sum::<f32>() / spec.channels as f32);
                    frame.clear();
                }
            }
            Ok((out, spec.sample_rate))
        } else if spec.bits_per_sample == 32 {
            let mut out = Vec::new();
            let mut frame = Vec::with_capacity(spec.channels as usize);
            for s in reader.samples::<i32>() {
                frame.push(s? as f32 / i32::MAX as f32);
                if frame.len() == spec.channels as usize {
                    out.push(frame.iter().sum::<f32>() / spec.channels as f32);
                    frame.clear();
                }
            }
            Ok((out, spec.sample_rate))
        } else {
            bail!(
                "unsupported int wav bits_per_sample={}",
                spec.bits_per_sample
            );
        }
    } else if spec.sample_format == hound::SampleFormat::Float {
        if spec.bits_per_sample != 32 {
            bail!(
                "unsupported float wav bits_per_sample={}",
                spec.bits_per_sample
            );
        }
        let mut out = Vec::new();
        let mut frame = Vec::with_capacity(spec.channels as usize);
        for s in reader.samples::<f32>() {
            frame.push(s?);
            if frame.len() == spec.channels as usize {
                out.push(frame.iter().sum::<f32>() / spec.channels as f32);
                frame.clear();
            }
        }
        Ok((out, spec.sample_rate))
    } else {
        bail!("unsupported wav sample_format");
    }
}

fn resample_linear_mono(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if input.is_empty() || from_sr == to_sr {
        return input.to_vec();
    }

    let ratio = to_sr as f64 / from_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let src_i0 = src_pos.floor() as isize;
        let src_i1 = src_i0 + 1;
        let t = (src_pos - src_i0 as f64) as f32;

        let s0 = sample_clamped(input, src_i0);
        let s1 = sample_clamped(input, src_i1);
        out.push(s0 * (1.0 - t) + s1 * t);
    }

    out
}

fn sample_clamped(input: &[f32], idx: isize) -> f32 {
    if idx <= 0 {
        return input[0];
    }
    let idx = idx as usize;
    if idx >= input.len() {
        return input[input.len() - 1];
    }
    input[idx]
}
