//! Environment-aware capability registry.
//!
//! Phase 1 (read-only): collects facts about the host OS/arch/RAM/GPU, the
//! STT/TTS/VAD backends currently loaded, any locally-installed Ollama
//! models, and the compile-time feature flags. The snapshot is intended to
//! be injected into the LLM system prompt so the assistant can answer
//! "what can you do on this machine?" truthfully.
//!
//! No tool dispatch, no side effects — Phase 2 concern.

use serde::Serialize;

pub mod hardware;
pub mod models;
pub mod ollama;

pub use hardware::{GpuFacts, HardwareFacts};
pub use models::{LoadedModel, ModelInventory};
pub use ollama::OllamaModelSummary;

/// One-line summary of a single capability the assistant can claim.
#[derive(Debug, Clone, Serialize)]
pub struct Capability {
    pub id: String,
    pub label: String,
    pub summary: String,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Compile-time feature flags.
#[derive(Debug, Clone, Serialize, Default)]
pub struct FeatureFlags {
    pub whisper: bool,
    pub distil_whisper: bool,
    pub sherpa: bool,
    pub silero: bool,
    pub kokoro: bool,
    pub piper: bool,
    pub pocket: bool,
    pub chatterbox: bool,
    pub qwen3: bool,
    pub diarization: bool,
    pub intelligence: bool,
    pub pocket_metal: bool,
    pub chatterbox_coreml: bool,
    pub qwen3_cuda: bool,
    pub qwen3_metal: bool,
}

impl FeatureFlags {
    pub fn detect() -> Self {
        Self {
            whisper: cfg!(feature = "whisper"),
            distil_whisper: cfg!(feature = "distil-whisper"),
            sherpa: cfg!(feature = "sherpa"),
            silero: cfg!(feature = "silero"),
            kokoro: cfg!(feature = "kokoro"),
            piper: cfg!(feature = "piper"),
            pocket: cfg!(feature = "pocket"),
            chatterbox: cfg!(feature = "chatterbox"),
            qwen3: cfg!(feature = "qwen3"),
            diarization: cfg!(feature = "diarization"),
            intelligence: cfg!(feature = "intelligence"),
            pocket_metal: cfg!(feature = "pocket-metal"),
            chatterbox_coreml: cfg!(feature = "chatterbox-coreml"),
            qwen3_cuda: cfg!(feature = "qwen3-cuda"),
            qwen3_metal: cfg!(feature = "qwen3-metal"),
        }
    }
}

/// Snapshot of everything the assistant knows about its own environment.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CapabilityRegistry {
    pub hardware: HardwareFacts,
    pub models: ModelInventory,
    pub ollama_models: Vec<OllamaModelSummary>,
    pub features: FeatureFlags,
    pub generated_at_unix: u64,
}

impl CapabilityRegistry {
    /// Build a live snapshot. Has a 2 second overall budget — any probe
    /// that hangs is silently replaced with a default.
    ///
    /// Arguments correspond to what the server already tracks:
    /// - stt / streaming_stt / tts: (backend_name, model_name, size_mb)
    /// - vad: path to silero_vad.onnx (existence indicates loaded)
    pub async fn build(
        profile: &crate::system_profile::SystemProfile,
        stt: Option<(&str, &str, Option<u64>)>,
        streaming_stt: Option<(&str, &str, Option<u64>)>,
        tts: Option<(&str, &str, Option<u64>)>,
        vad: Option<&std::path::Path>,
        ollama_host: &str,
        http: &reqwest::Client,
    ) -> Self {
        let hardware = hardware::HardwareFacts::detect(profile).await;
        let models = models::ModelInventory::from_parts(stt, streaming_stt, tts, vad);
        // Ollama probe is the only network-crossing op. 1.5s cap.
        let ollama_models = tokio::time::timeout(
            std::time::Duration::from_millis(1500),
            ollama::probe(http, ollama_host),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or_default();

        Self {
            hardware,
            models,
            ollama_models,
            features: FeatureFlags::detect(),
            generated_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Append a compact, LLM-friendly facts block onto a base system prompt.
    /// Ollama model list is capped at 12 entries to protect the context budget.
    pub fn inject_into_prompt(&self, base: &str) -> String {
        // Build a deterministic, short facts block. Order matters for
        // prompt-cache stability: host, backends, ollama, limits.
        let mut out = String::with_capacity(base.len() + 1024);
        out.push_str(base);
        out.push_str("\n\n===\nHost environment (do not mention unless asked):\n");

        // Hardware line
        out.push_str(&format!(
            "- Platform: {} on {}, {} cores, {} MB RAM, class {}\n",
            self.hardware.os,
            self.hardware.arch,
            self.hardware.cpu_cores,
            self.hardware.total_ram_mb,
            self.hardware.hardware_class,
        ));

        // GPU line
        if self.hardware.gpu.present {
            let gpu_name = self.hardware.gpu.name.as_deref().unwrap_or("GPU");
            let kind = self.hardware.gpu.kind.as_deref().unwrap_or("unknown");
            out.push_str(&format!(
                "- GPU: {gpu_name} ({kind}). Compiled backends: {}.\n",
                if self.hardware.gpu.compiled_backends.is_empty() {
                    "none".to_string()
                } else {
                    self.hardware.gpu.compiled_backends.join(", ")
                },
            ));
        } else {
            out.push_str("- GPU: none detected\n");
        }

        // Model lines
        if let Some(stt) = &self.models.stt {
            out.push_str(&format!(
                "- Speech to text: {} backend, {}{}\n",
                stt.backend,
                stt.model_name,
                stt.size_mb.map(|s| format!(", {s} MB")).unwrap_or_default(),
            ));
        } else {
            out.push_str("- Speech to text: not loaded\n");
        }

        if let Some(tts) = &self.models.tts {
            out.push_str(&format!(
                "- Text to speech: {} backend, {}{}\n",
                tts.backend,
                tts.model_name,
                tts.size_mb.map(|s| format!(", {s} MB")).unwrap_or_default(),
            ));
        } else {
            out.push_str("- Text to speech: not loaded\n");
        }

        if let Some(vad) = &self.models.vad {
            out.push_str(&format!(
                "- Voice activity detection: {}\n",
                vad.backend,
            ));
        } else {
            out.push_str("- Voice activity detection: not loaded\n");
        }

        if let Some(streaming) = &self.models.streaming_stt {
            out.push_str(&format!(
                "- Streaming STT: {} backend, {}\n",
                streaming.backend, streaming.model_name,
            ));
        }

        // Ollama models — cap at 12 to stay within the context budget.
        if self.ollama_models.is_empty() {
            out.push_str("- Local LLM runtime: none reachable\n");
        } else {
            let names: Vec<String> = self
                .ollama_models
                .iter()
                .take(12)
                .map(|m| m.name.clone())
                .collect();
            let suffix = if self.ollama_models.len() > 12 {
                format!(" (+{} more)", self.ollama_models.len() - 12)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "- Local LLM runtime: Ollama\n- Ollama models installed: {}{}\n",
                names.join(", "),
                suffix,
            ));
        }

        // Limitations (hard-coded for Phase 1 — Phase 2 will derive these)
        out.push_str("\nLimitations on this device (Phase 1 — read-only):\n");
        out.push_str("- File system access: not available yet.\n");
        out.push_str("- Shell execution: not available.\n");
        out.push_str("- Image generation: not installed.\n");

        out
    }

    /// Enumerate every capability as a flat list.
    pub fn to_flat_list(&self) -> Vec<Capability> {
        let mut out = Vec::new();

        // Hardware
        out.push(Capability {
            id: "hw.host".into(),
            label: "Host".into(),
            summary: format!(
                "{} on {}, {} cores, {} MB RAM",
                self.hardware.os,
                self.hardware.arch,
                self.hardware.cpu_cores,
                self.hardware.total_ram_mb,
            ),
            available: true,
            details: serde_json::to_value(&self.hardware).ok(),
        });

        // Backends
        if let Some(m) = &self.models.stt {
            out.push(Capability {
                id: "stt".into(),
                label: format!("STT ({})", m.backend),
                summary: format!("Speech-to-text via {} ({})", m.backend, m.model_name),
                available: true,
                details: serde_json::to_value(m).ok(),
            });
        }
        if let Some(m) = &self.models.tts {
            out.push(Capability {
                id: "tts".into(),
                label: format!("TTS ({})", m.backend),
                summary: format!("Text-to-speech via {} ({})", m.backend, m.model_name),
                available: true,
                details: serde_json::to_value(m).ok(),
            });
        }
        if let Some(m) = &self.models.vad {
            out.push(Capability {
                id: "vad".into(),
                label: format!("VAD ({})", m.backend),
                summary: format!("Voice activity detection via {}", m.backend),
                available: true,
                details: serde_json::to_value(m).ok(),
            });
        }
        if let Some(m) = &self.models.streaming_stt {
            out.push(Capability {
                id: "stt.streaming".into(),
                label: format!("Streaming STT ({})", m.backend),
                summary: format!("Real-time STT via {}", m.backend),
                available: true,
                details: serde_json::to_value(m).ok(),
            });
        }

        // Ollama
        for m in &self.ollama_models {
            out.push(Capability {
                id: format!("ollama.{}", m.name),
                label: m.name.clone(),
                summary: format!(
                    "Local Ollama model ({} MB)",
                    m.size_mb.unwrap_or(0),
                ),
                available: true,
                details: serde_json::to_value(m).ok(),
            });
        }

        out
    }

    /// Returns `true` if the snapshot is older than `max_age`.
    pub fn is_stale(&self, max_age: std::time::Duration) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.generated_at_unix) > max_age.as_secs()
    }
}
