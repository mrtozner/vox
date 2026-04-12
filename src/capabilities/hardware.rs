//! Hardware fact collection for the capability registry.
//!
//! Wraps [`crate::system_profile::SystemProfile`] with additional GPU
//! probes. All probes have hard timeouts and never panic — missing or
//! stalling probes fall back to [`GpuFacts::default`].

use serde::Serialize;
use std::time::Duration;
use tokio::process::Command;

use crate::system_profile::{Arch, Device, HardwareClass, Os, SystemProfile};

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuFacts {
    pub present: bool,
    pub kind: Option<String>, // "Metal" | "CUDA" | "ROCm"
    pub name: Option<String>,
    pub compiled_backends: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct HardwareFacts {
    pub os: String,
    pub arch: String,
    pub device_label: String,
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub hardware_class: String,
    pub gpu: GpuFacts,
}

impl HardwareFacts {
    /// Build from an existing SystemProfile and perform GPU probes.
    /// Total budget for GPU probes is ~500ms per probe.
    pub async fn detect(profile: &SystemProfile) -> Self {
        let os_label = match profile.os {
            Os::MacOs => "macOS",
            Os::Linux => "Linux",
            Os::Other => "other",
        }
        .to_string();

        let arch_label = match profile.arch {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
            Arch::Armv7 => "armv7",
            Arch::Other => "unknown",
        }
        .to_string();

        let device_label = format!("{}", profile.device);
        let hw_class = match profile.class {
            HardwareClass::Tiny => "tiny",
            HardwareClass::Constrained => "constrained",
            HardwareClass::Small => "small",
            HardwareClass::Medium => "medium",
            HardwareClass::Large => "large",
        }
        .to_string();

        let gpu = probe_gpu(profile).await;

        Self {
            os: os_label,
            arch: arch_label,
            device_label,
            cpu_cores: profile.num_cpus,
            total_ram_mb: profile.total_ram_mb,
            hardware_class: hw_class,
            gpu,
        }
    }
}

/// Derive the compiled GPU backends from Cargo features.
fn compiled_backends() -> Vec<String> {
    let mut out = Vec::new();
    if cfg!(feature = "pocket-metal") {
        out.push("metal".into());
    }
    if cfg!(feature = "chatterbox-coreml") {
        out.push("coreml".into());
    }
    if cfg!(feature = "qwen3-cuda") {
        out.push("cuda".into());
    }
    if cfg!(feature = "qwen3-metal") && !out.iter().any(|s| s == "metal") {
        out.push("metal".into());
    }
    out
}

async fn probe_gpu(profile: &SystemProfile) -> GpuFacts {
    let compiled = compiled_backends();

    match profile.os {
        Os::MacOs => probe_gpu_macos(compiled).await,
        Os::Linux => probe_gpu_linux(profile, compiled).await,
        Os::Other => GpuFacts {
            present: false,
            kind: None,
            name: None,
            compiled_backends: compiled,
        },
    }
}

async fn probe_gpu_macos(compiled_backends: Vec<String>) -> GpuFacts {
    // system_profiler SPDisplaysDataType -json (500ms cap)
    let probe = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output();

    let name = match tokio::time::timeout(Duration::from_millis(500), probe).await {
        Ok(Ok(output)) if output.status.success() => {
            serde_json::from_slice::<serde_json::Value>(&output.stdout)
                .ok()
                .and_then(|v| {
                    v.get("SPDisplaysDataType")?
                        .as_array()?
                        .first()?
                        .get("sppci_model")?
                        .as_str()
                        .map(String::from)
                })
        }
        _ => None,
    };

    GpuFacts {
        present: true,
        kind: Some("Metal".into()),
        name: Some(name.unwrap_or_else(|| "Apple GPU".into())),
        compiled_backends,
    }
}

async fn probe_gpu_linux(profile: &SystemProfile, compiled_backends: Vec<String>) -> GpuFacts {
    // 1. Try nvidia-smi (500ms cap)
    let nvidia_probe = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();

    if let Ok(Ok(output)) = tokio::time::timeout(Duration::from_millis(500), nvidia_probe).await
        && output.status.success()
    {
        if let Ok(s) = std::str::from_utf8(&output.stdout) {
            let name = s.trim().split('\n').next().unwrap_or("").trim().to_string();
            if !name.is_empty() {
                return GpuFacts {
                    present: true,
                    kind: Some("CUDA".into()),
                    name: Some(name),
                    compiled_backends,
                };
            }
        }
    }

    // 2. Try AMD via /sys/class/drm/card*/device/uevent
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let uevent = entry.path().join("device").join("uevent");
            if let Ok(contents) = std::fs::read_to_string(&uevent) {
                if contents.contains("DRIVER=amdgpu") {
                    let kind = if std::path::Path::new("/dev/kfd").exists() {
                        "ROCm"
                    } else {
                        "amdgpu"
                    };
                    return GpuFacts {
                        present: true,
                        kind: Some(kind.into()),
                        name: Some("AMD GPU".into()),
                        compiled_backends,
                    };
                }
            }
        }
    }

    // 3. Raspberry Pi: VideoCore is CPU-class for LLM purposes
    if matches!(
        profile.device,
        Device::RaspberryPi5
            | Device::RaspberryPi4
            | Device::RaspberryPi3
            | Device::RaspberryPiZero2W
            | Device::RaspberryPiOther(_)
    ) {
        return GpuFacts {
            present: false,
            kind: None,
            name: None,
            compiled_backends,
        };
    }

    GpuFacts {
        present: false,
        kind: None,
        name: None,
        compiled_backends,
    }
}
