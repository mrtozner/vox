//! Runtime hardware detection for automatic backend selection.
//!
//! At server startup we want to pick sensible defaults for STT, TTS, and
//! LLM backends based on the machine we're actually running on. A user with
//! a Raspberry Pi Zero 2 W should not get the same defaults as a user on an
//! M1 MacBook Pro — both because what *works* differs (Zero 2 can't run
//! local Ollama) and because what's *fast* differs.
//!
//! This module performs a one-shot probe at startup and returns a
//! [`SystemProfile`] that callers can consult for recommendations. It is
//! dependency-free: Linux info comes from `/proc/*` and
//! `/proc/device-tree/*`; macOS info comes from `sysctl`.

use std::fmt;

/// Coarse OS classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    MacOs,
    Linux,
    Other,
}

/// Coarse CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    Armv7,
    Other,
}

/// Identified hardware model when we can recognize it from device tree,
/// fallback to a generic class otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    RaspberryPi5,
    RaspberryPi4,
    RaspberryPi3,
    RaspberryPiZero2W,
    RaspberryPiOther(String),
    MacOs,
    LinuxDesktop,
    Unknown,
}

/// Hardware capability class used to pick sensible model defaults.
///
/// Ordered from most constrained to least constrained. The boundaries are
/// deliberately fuzzy because real-world RAM availability shifts as the
/// kernel, swap, other services etc. consume memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HardwareClass {
    /// <= 1 GB RAM. Pi Zero 2 W. Cannot run any local LLM comfortably.
    /// Must offload LLM to a network host.
    Tiny,
    /// 1.5 – 3 GB RAM. Pi 4 2 GB / low-spec SBCs. Smallest models only
    /// (qwen2.5:0.5b class, ~400 MB quantized).
    Constrained,
    /// 3 – 5 GB RAM. Pi 4 4 GB / low-spec desktops. Small models (1.5 B,
    /// ~1 GB quantized). llama3.2:1b just fits.
    Small,
    /// 5 – 10 GB RAM. Pi 5 4/8 GB, mainstream laptops. llama3.2:1b or
    /// qwen2.5:3b comfortably, with STT + TTS co-resident.
    Medium,
    /// 10+ GB RAM. Desktop / server. Full model selection including
    /// `llama3.2:3b` or larger.
    Large,
}

/// Result of a one-shot system probe.
#[derive(Debug, Clone)]
pub struct SystemProfile {
    pub os: Os,
    pub arch: Arch,
    pub device: Device,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub num_cpus: usize,
    pub class: HardwareClass,
}

impl SystemProfile {
    /// Run all detection probes. Never fails — every missing field falls
    /// back to a safe default.
    pub fn detect() -> Self {
        let os = detect_os();
        let arch = detect_arch();
        let (total_ram_mb, available_ram_mb) = detect_ram();
        let num_cpus = detect_num_cpus();
        let device = detect_device(os, arch);
        let class = classify(total_ram_mb, &device);

        Self {
            os,
            arch,
            device,
            total_ram_mb,
            available_ram_mb,
            num_cpus,
            class,
        }
    }

    /// Recommended default Ollama model tag for this hardware.
    ///
    /// Returns `None` if local Ollama is not viable — caller should either
    /// prompt for a remote host or disable the chat feature entirely.
    pub fn recommended_ollama_model(&self) -> Option<&'static str> {
        match self.class {
            HardwareClass::Tiny => None,
            HardwareClass::Constrained => Some("qwen2.5:0.5b"),
            HardwareClass::Small => Some("qwen2.5:1.5b"),
            HardwareClass::Medium => Some("llama3.2:1b"),
            HardwareClass::Large => Some("llama3.2:3b"),
        }
    }

    /// Returns `true` when local Ollama is not viable on this hardware
    /// (Pi Zero 2 W and similar). The user should set `VOX_OLLAMA_HOST` to
    /// a remote machine running Ollama.
    pub fn requires_remote_llm(&self) -> bool {
        self.class == HardwareClass::Tiny
    }

    /// Recommended whisper.cpp model file name for this hardware. Always
    /// falls back to something the user might reasonably have downloaded,
    /// even if tighter.
    pub fn recommended_whisper_model(&self) -> &'static str {
        match self.class {
            HardwareClass::Tiny => "ggml-tiny.en.bin",
            HardwareClass::Constrained => "ggml-tiny.en.bin",
            HardwareClass::Small => "ggml-base.en.bin",
            HardwareClass::Medium => "ggml-base.en.bin",
            HardwareClass::Large => "ggml-small.en.bin",
        }
    }

    /// Recommended number of CPU threads for native inference on this
    /// system. Leaves at least one core free on small systems so the audio
    /// thread doesn't starve.
    pub fn recommended_inference_threads(&self) -> i32 {
        let n = self.num_cpus as i32;
        if n <= 2 { n } else { n - 1 }
    }

    /// One-line human-readable summary for startup logs.
    pub fn summary(&self) -> String {
        format!(
            "{} {} · {} MB RAM · {} cores · class={}",
            self.device, self.arch, self.total_ram_mb, self.num_cpus, self.class,
        )
    }
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Os::MacOs => write!(f, "macOS"),
            Os::Linux => write!(f, "Linux"),
            Os::Other => write!(f, "other"),
        }
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arch::X86_64 => write!(f, "x86_64"),
            Arch::Aarch64 => write!(f, "aarch64"),
            Arch::Armv7 => write!(f, "armv7"),
            Arch::Other => write!(f, "unknown-arch"),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::RaspberryPi5 => write!(f, "Raspberry Pi 5"),
            Device::RaspberryPi4 => write!(f, "Raspberry Pi 4"),
            Device::RaspberryPi3 => write!(f, "Raspberry Pi 3"),
            Device::RaspberryPiZero2W => write!(f, "Raspberry Pi Zero 2 W"),
            Device::RaspberryPiOther(name) => write!(f, "{name}"),
            Device::MacOs => write!(f, "macOS"),
            Device::LinuxDesktop => write!(f, "Linux"),
            Device::Unknown => write!(f, "unknown"),
        }
    }
}

impl fmt::Display for HardwareClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HardwareClass::Tiny => write!(f, "tiny"),
            HardwareClass::Constrained => write!(f, "constrained"),
            HardwareClass::Small => write!(f, "small"),
            HardwareClass::Medium => write!(f, "medium"),
            HardwareClass::Large => write!(f, "large"),
        }
    }
}

// ============================================================================
// Detection internals
// ============================================================================

fn detect_os() -> Os {
    if cfg!(target_os = "macos") {
        Os::MacOs
    } else if cfg!(target_os = "linux") {
        Os::Linux
    } else {
        Os::Other
    }
}

fn detect_arch() -> Arch {
    if cfg!(target_arch = "x86_64") {
        Arch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Arch::Aarch64
    } else if cfg!(target_arch = "arm") {
        Arch::Armv7
    } else {
        Arch::Other
    }
}

/// Returns `(total_mb, available_mb)`. Zero when we can't tell.
fn detect_ram() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0u64;
            let mut avail_kb = 0u64;
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    total_kb = parse_kb(rest);
                } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    avail_kb = parse_kb(rest);
                }
            }
            return (total_kb / 1024, avail_kb / 1024);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // hw.memsize returns total physical memory in bytes.
        if let Some(total_bytes) = run_sysctl_u64("hw.memsize") {
            let total_mb = total_bytes / (1024 * 1024);
            // Available = total - wired pages is complicated; approximate
            // with "free" bytes via `vm_stat`. For auto-detect purposes we
            // only really care about total.
            return (total_mb, total_mb);
        }
    }

    (0, 0)
}

#[cfg(target_os = "linux")]
fn parse_kb(s: &str) -> u64 {
    // Line looks like: "   16384000 kB"
    s.split_whitespace()
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn run_sysctl_u64(key: &str) -> Option<u64> {
    use std::process::Command;
    let out = Command::new("sysctl").arg("-n").arg(key).output().ok()?;
    if !out.status.success() {
        return None;
    }
    std::str::from_utf8(&out.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

fn detect_num_cpus() -> usize {
    // std::thread::available_parallelism is the stdlib way and it
    // respects cgroup limits on modern Linux.
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn detect_device(os: Os, _arch: Arch) -> Device {
    if os == Os::MacOs {
        return Device::MacOs;
    }

    #[cfg(target_os = "linux")]
    {
        // Raspberry Pi OS + Ubuntu for Pi both expose the model name via
        // device tree. Format: "Raspberry Pi 5 Model B Rev 1.0\u0000"
        if let Ok(raw) = std::fs::read("/proc/device-tree/model") {
            let model = String::from_utf8_lossy(&raw)
                .trim_end_matches('\0')
                .trim()
                .to_string();
            let lower = model.to_lowercase();
            if lower.contains("raspberry pi 5") {
                return Device::RaspberryPi5;
            }
            if lower.contains("raspberry pi 4") {
                return Device::RaspberryPi4;
            }
            if lower.contains("raspberry pi zero 2") {
                return Device::RaspberryPiZero2W;
            }
            if lower.contains("raspberry pi 3") {
                return Device::RaspberryPi3;
            }
            if lower.contains("raspberry") {
                return Device::RaspberryPiOther(model);
            }
        }
        return Device::LinuxDesktop;
    }

    #[allow(unreachable_code)]
    Device::Unknown
}

/// Pick a hardware class from the probed RAM + device identification.
fn classify(total_ram_mb: u64, device: &Device) -> HardwareClass {
    // Hard rule: Pi Zero 2 is Tiny regardless of what the meminfo says.
    // (Some variants report closer to 800 MB due to GPU split.)
    if matches!(device, Device::RaspberryPiZero2W) {
        return HardwareClass::Tiny;
    }

    match total_ram_mb {
        0 => HardwareClass::Small, // unknown — assume modest
        n if n <= 1024 => HardwareClass::Tiny,
        n if n <= 3 * 1024 => HardwareClass::Constrained,
        n if n <= 5 * 1024 => HardwareClass::Small,
        n if n <= 10 * 1024 => HardwareClass::Medium,
        _ => HardwareClass::Large,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs_without_panic() {
        let profile = SystemProfile::detect();
        assert!(profile.num_cpus > 0);
    }

    #[test]
    fn ollama_recommendation_honors_tiny() {
        let profile = SystemProfile {
            os: Os::Linux,
            arch: Arch::Aarch64,
            device: Device::RaspberryPiZero2W,
            total_ram_mb: 512,
            available_ram_mb: 300,
            num_cpus: 4,
            class: HardwareClass::Tiny,
        };
        assert_eq!(profile.recommended_ollama_model(), None);
        assert!(profile.requires_remote_llm());
    }

    #[test]
    fn ollama_recommendation_for_pi4_4gb() {
        let profile = SystemProfile {
            os: Os::Linux,
            arch: Arch::Aarch64,
            device: Device::RaspberryPi4,
            total_ram_mb: 4096,
            available_ram_mb: 3000,
            num_cpus: 4,
            class: classify(4096, &Device::RaspberryPi4),
        };
        // 4 GB == 4096 MB, which falls into Small (<= 5 * 1024)
        assert_eq!(profile.class, HardwareClass::Small);
        assert_eq!(profile.recommended_ollama_model(), Some("qwen2.5:1.5b"));
    }

    #[test]
    fn ollama_recommendation_for_pi5_8gb() {
        let profile = SystemProfile {
            os: Os::Linux,
            arch: Arch::Aarch64,
            device: Device::RaspberryPi5,
            total_ram_mb: 8192,
            available_ram_mb: 7000,
            num_cpus: 4,
            class: classify(8192, &Device::RaspberryPi5),
        };
        assert_eq!(profile.class, HardwareClass::Medium);
        assert_eq!(profile.recommended_ollama_model(), Some("llama3.2:1b"));
    }

    #[test]
    fn desktop_gets_largest_model() {
        let profile = SystemProfile {
            os: Os::MacOs,
            arch: Arch::Aarch64,
            device: Device::MacOs,
            total_ram_mb: 16384,
            available_ram_mb: 8000,
            num_cpus: 10,
            class: classify(16384, &Device::MacOs),
        };
        assert_eq!(profile.class, HardwareClass::Large);
        assert_eq!(profile.recommended_ollama_model(), Some("llama3.2:3b"));
    }

    #[test]
    fn recommend_threads_leaves_headroom() {
        let profile = SystemProfile {
            os: Os::Linux,
            arch: Arch::X86_64,
            device: Device::LinuxDesktop,
            total_ram_mb: 16384,
            available_ram_mb: 8000,
            num_cpus: 8,
            class: HardwareClass::Large,
        };
        assert_eq!(profile.recommended_inference_threads(), 7);

        let tiny = SystemProfile {
            num_cpus: 2,
            ..profile
        };
        assert_eq!(tiny.recommended_inference_threads(), 2);
    }
}
