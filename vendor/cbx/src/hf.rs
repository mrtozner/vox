use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType, api::sync::Api};

#[derive(Debug, Clone, Copy)]
pub enum ModelVariant {
    Fp32,
    Fp16,
    Quantized,
    Q4,
    Q4f16,
    Q8,
    Q8f16,
}

impl ModelVariant {
    fn suffix(self) -> &'static str {
        match self {
            ModelVariant::Fp32 => "",
            ModelVariant::Fp16 => "_fp16",
            ModelVariant::Quantized => "_quantized",
            ModelVariant::Q4 => "_q4",
            ModelVariant::Q4f16 => "_q4f16",
            ModelVariant::Q8 => "_q8",
            ModelVariant::Q8f16 => "_q8f16",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatterboxPaths {
    pub tokenizer_json: PathBuf,
    pub conditional_decoder: PathBuf,
    pub speech_encoder: PathBuf,
    pub embed_tokens: PathBuf,
    pub language_model: PathBuf,
}

/// Downloads the required files into the Hugging Face cache (via `hf-hub`) and returns absolute paths.
///
/// Notes:
/// - The `.onnx` graphs in this repo reference external tensor data by relative filename, so we also download the
///   matching `.onnx_data` file for each model.
pub fn download_chatterbox_assets(
    repo_id: &str,
    revision: &str,
    variant: ModelVariant,
) -> Result<ChatterboxPaths> {
    let api = Api::new().context("failed to create Hugging Face Hub client")?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_json = repo
        .get("tokenizer.json")
        .with_context(|| format!("failed to download tokenizer.json from {repo_id}@{revision}"))?;

    let conditional_decoder =
        get_onnx_pair(&repo, "conditional_decoder", variant).context("conditional_decoder")?;
    let speech_encoder =
        get_onnx_pair(&repo, "speech_encoder", variant).context("speech_encoder")?;
    let embed_tokens = get_onnx_pair(&repo, "embed_tokens", variant).context("embed_tokens")?;
    let language_model =
        get_onnx_pair(&repo, "language_model", variant).context("language_model")?;

    Ok(ChatterboxPaths {
        tokenizer_json,
        conditional_decoder,
        speech_encoder,
        embed_tokens,
        language_model,
    })
}

fn get_onnx_pair(
    repo: &hf_hub::api::sync::ApiRepo,
    stem: &str,
    variant: ModelVariant,
) -> Result<PathBuf> {
    let suffix = variant.suffix();
    let onnx = format!("onnx/{stem}{suffix}.onnx");
    let onnx_data = format!("onnx/{stem}{suffix}.onnx_data");

    let onnx_path = repo
        .get(&onnx)
        .with_context(|| format!("failed to download {onnx}"))?;

    // This matters: the .onnx references the external tensor data file by relative path.
    //
    // Avoid an extra hub call if the snapshot already contains the matching .onnx_data.
    let onnx_data_local = onnx_path
        .parent()
        .map(|p| p.join(format!("{stem}{suffix}.onnx_data")));
    let needs_data = onnx_data_local.as_ref().is_none_or(|p| !p.exists());
    if needs_data {
        let _ = repo
            .get(&onnx_data)
            .with_context(|| format!("failed to download {onnx_data}"))?;
    }

    Ok(onnx_path)
}

/// Best-effort extraction of the Hugging Face snapshot directory from an absolute cache path.
///
/// Example input:
/// `.../hub/models--ORG--REPO/snapshots/<hash>/tokenizer.json`
/// Returns:
/// `.../hub/models--ORG--REPO/snapshots/<hash>`
pub fn snapshot_root_from_cache_path(path: &Path) -> Option<PathBuf> {
    let components: Vec<_> = path.components().collect();
    for i in 0..components.len().saturating_sub(1) {
        if components[i].as_os_str() == "snapshots" {
            let mut out = PathBuf::new();
            for c in &components[..=i + 1] {
                out.push(c.as_os_str());
            }
            return Some(out);
        }
    }
    None
}
