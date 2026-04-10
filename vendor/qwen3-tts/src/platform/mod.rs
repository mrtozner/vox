//! Platform-specific optimizations and feature detection

pub mod arm;

use candle_core::Device;

/// Detect ARM NEON support at runtime
#[cfg(target_arch = "aarch64")]
pub fn has_neon() -> bool {
    // ARM64 always has NEON
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_neon() -> bool {
    false
}

/// Detect best device for Raspberry Pi
pub fn select_device_for_raspberry_pi() -> Device {
    #[cfg(target_arch = "aarch64")]
    {
        // Raspberry Pi is ARM, use CPU with NEON optimizations
        tracing::info!("detected ARM64 (Raspberry Pi), using CPU with NEON");
        Device::Cpu
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        Device::Cpu
    }
}

/// Get optimal thread count for Raspberry Pi
pub fn optimal_thread_count() -> usize {
    // Raspberry Pi 4/5 have 4 cores
    // Leave 1 core for system, use 3 for inference
    let cpu_count = num_cpus::get();
    std::cmp::min(cpu_count, 4).saturating_sub(1).max(1)
}

/// Configure Rayon thread pool for Raspberry Pi
pub fn configure_threadpool_for_raspberry_pi() {
    let thread_count = optimal_thread_count();

    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .thread_name(|i| format!("qwen3-worker-{}", i))
        .build_global()
        .unwrap_or_else(|e| {
            tracing::warn!("failed to configure rayon threadpool: {}", e);
        });

    tracing::info!(
        threads = thread_count,
        "configured rayon threadpool for raspberry pi"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_thread_count() {
        let count = optimal_thread_count();
        // Should be at least 1, at most 3 for Pi
        assert!(count >= 1);
        assert!(count <= 3);
    }

    #[test]
    fn test_select_device() {
        let device = select_device_for_raspberry_pi();
        // Should always be CPU for Pi
        assert!(matches!(device, Device::Cpu));
    }
}
