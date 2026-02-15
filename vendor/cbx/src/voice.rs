use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::Cache;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorI64 {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub format_version: u32,
    pub repo_id: String,
    pub revision: String,
    pub dtype: String,
    pub sample_rate_hz: u32,

    pub audio_features: TensorF32,
    pub audio_tokens: TensorI64,
    pub speaker_embeddings: TensorF32,
    pub speaker_features: TensorF32,
}

pub fn voice_cache_dir() -> Result<PathBuf> {
    let hub = Cache::from_env().path().clone();
    let hf_home = hub
        .parent()
        .context("failed to resolve HF_HOME from hub cache path")?;
    Ok(hf_home.join("cbx").join("voices"))
}

pub fn list_voice_profiles(dir: &Path) -> Result<Vec<String>> {
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cbxvoice") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            names.push(stem.to_string());
        }
    }
    names.sort();
    Ok(names)
}

pub fn load_voice_profile(dir: &Path, name: &str) -> Result<VoiceProfile> {
    let path = profile_path(dir, name);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let profile: VoiceProfile =
        bincode::deserialize(&bytes).context("failed to decode voice profile")?;
    Ok(profile)
}

pub fn save_voice_profile(dir: &Path, name: &str, profile: &VoiceProfile) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = profile_path(dir, name);
    let tmp = path.with_extension("cbxvoice.tmp");
    let bytes = bincode::serialize(profile).context("failed to encode voice profile")?;
    fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| format!("failed to move into {}", path.display()))?;
    Ok(())
}

pub fn remove_voice_profile(dir: &Path, name: &str) -> Result<()> {
    let path = profile_path(dir, name);
    if path.exists() {
        fs::remove_file(&path).with_context(|| format!("failed to delete {}", path.display()))?;
    }
    Ok(())
}

pub fn remove_all_voices(dir: &Path) -> Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir).with_context(|| format!("failed to delete {}", dir.display()))?;
    }
    Ok(())
}

pub fn pick_voice_for_model(
    dir: &Path,
    repo_id: &str,
    revision: &str,
    dtype: &str,
) -> Result<Option<String>> {
    let names = list_voice_profiles(dir)?;
    if names.is_empty() {
        return Ok(None);
    }

    // Prefer a profile named "default" if it matches the requested model tuple.
    if names.iter().any(|n| n == "default") {
        if let Ok(p) = load_voice_profile(dir, "default") {
            if p.repo_id == repo_id && p.revision == revision && p.dtype == dtype {
                return Ok(Some("default".to_string()));
            }
        }
    }

    // Otherwise pick the first profile that matches the requested model tuple.
    for name in &names {
        let Ok(p) = load_voice_profile(dir, name) else {
            continue;
        };
        if p.repo_id == repo_id && p.revision == revision && p.dtype == dtype {
            return Ok(Some(name.clone()));
        }
    }

    Ok(None)
}

fn profile_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.cbxvoice"))
}
