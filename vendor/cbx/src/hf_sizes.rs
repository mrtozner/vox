use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::hf::ModelVariant;

#[derive(Debug, Clone)]
pub struct VariantSizes {
    pub variant: ModelVariant,
    pub file_count: usize,
    pub total_bytes: u64,
    pub missing: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SizeTable {
    pub tokenizer_bytes: Option<u64>,
    pub variants: Vec<VariantSizes>,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    path: String,
    size: u64,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Deserialize)]
struct LfsInfo {
    size: u64,
}

pub fn fetch_size_table(repo_id: &str, revision: &str) -> Result<SizeTable> {
    // Prefer the "tree" endpoint because it includes file sizes (and LFS sizes).
    // Example:
    //   https://huggingface.co/api/models/<org>/<name>/tree/<revision>?recursive=1
    //
    // Important: repo_id includes a slash (ORG/NAME) and must NOT be percent-encoded as a single path segment.
    // Revision *is* a single path segment and may contain slashes in branch names, so encode it.
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/{}?recursive=1",
        repo_id,
        urlencoding::encode(revision)
    );

    let resp = ureq::get(&url)
        .set("Accept", "application/json")
        .call()
        .context("failed to call Hugging Face model tree API")?;

    if resp.status() != 200 {
        bail!("Hugging Face API returned HTTP {}", resp.status());
    }

    let entries: Vec<TreeEntry> = resp
        .into_json()
        .context("failed to parse Hugging Face tree API JSON response")?;

    let mut sizes_by_file: HashMap<String, u64> = HashMap::new();
    for e in entries {
        // If a file is stored via LFS, `size` is still present, but `lfs.size` is the real payload size.
        let effective_size = e.lfs.as_ref().map(|l| l.size).unwrap_or(e.size);
        sizes_by_file.insert(e.path, effective_size);
    }

    let tokenizer_bytes = sizes_by_file.get("tokenizer.json").copied();

    let variants = [
        ModelVariant::Fp32,
        ModelVariant::Fp16,
        ModelVariant::Quantized,
        ModelVariant::Q4,
        ModelVariant::Q4f16,
        ModelVariant::Q8,
        ModelVariant::Q8f16,
    ];

    let mut out_variants = Vec::new();
    for v in variants {
        let suffix = match v {
            ModelVariant::Fp32 => "",
            ModelVariant::Fp16 => "_fp16",
            ModelVariant::Quantized => "_quantized",
            ModelVariant::Q4 => "_q4",
            ModelVariant::Q4f16 => "_q4f16",
            ModelVariant::Q8 => "_q8",
            ModelVariant::Q8f16 => "_q8f16",
        };

        let mut file_count = 0usize;
        let mut total = 0u64;
        let mut missing = Vec::new();

        // tokenizer.json is common to all variants (and required to run).
        file_count += 1;
        if let Some(sz) = tokenizer_bytes {
            total += sz;
        } else {
            missing.push("tokenizer.json".to_string());
        }

        for stem in [
            "conditional_decoder",
            "speech_encoder",
            "embed_tokens",
            "language_model",
        ] {
            let onnx = format!("onnx/{}{}.onnx", stem, suffix);
            let onnx_data = format!("onnx/{}{}.onnx_data", stem, suffix);

            file_count += 2;
            match sizes_by_file.get(&onnx) {
                Some(sz) => total += *sz,
                None => missing.push(onnx),
            }
            match sizes_by_file.get(&onnx_data) {
                Some(sz) => total += *sz,
                None => missing.push(onnx_data),
            }
        }

        out_variants.push(VariantSizes {
            variant: v,
            file_count,
            total_bytes: total,
            missing,
        });
    }

    Ok(SizeTable {
        tokenizer_bytes,
        variants: out_variants,
    })
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{bytes} B")
    }
}
