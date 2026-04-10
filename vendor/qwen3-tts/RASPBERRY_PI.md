# Raspberry Pi Deployment Guide

## Performance Estimates (Without Actual Hardware)

Based on ARM Cortex-A72/A76 specs and INT8 quantization research:

| Model | Device | RTF (estimated) | Memory | Notes |
|-------|--------|-----------------|--------|-------|
| 0.6B Base | Pi 4 (4GB) | ~3.5 | 1.7GB | BF16, no quantization |
| 0.6B INT8 | Pi 4 (4GB) | **~2.0** | **850MB** | 50% memory, 1.75x faster |
| 0.6B INT8 | Pi 5 (8GB) | **~1.0** | **850MB** | NEON + faster CPU |

### Memory Breakdown (0.6B Model):

- **BF16 (original)**: 600M params × 2 bytes = 1.2GB weights + 500MB runtime = **1.7GB total**
- **INT8 (quantized)**: 600M params × 1 byte = 600MB weights + 250MB runtime = **850MB total**
- **Savings**: 50% memory reduction

### Optimization Impact:

1. **INT8 Quantization**: 50% memory, 10-30% faster inference
2. **ARM NEON**: 15-25% faster matrix operations
3. **Thread Pool Tuning**: 10-15% better CPU utilization
4. **Combined**: ~1.75x speedup, 50% less memory

## Build Instructions

```bash
# On development machine (Mac/Linux)
./scripts/build_for_raspberry_pi.sh

# Transfer to Pi
scp target/aarch64-unknown-linux-gnu/release/vox pi@raspberrypi:~/

# On Raspberry Pi
chmod +x vox
./vox serve --features qwen3
```

## Testing Without Pi Hardware

```bash
# Cross-compile to verify builds
cargo build --target aarch64-unknown-linux-gnu --features qwen3,quantized

# Run quantization tests (cross-platform)
cargo test --features quantized quantization

# Check memory estimates
cargo run --features qwen3,quantized --bin estimate_memory
```

## Usage Examples

### Basic Synthesis with Quantization

```rust
use qwen3_tts::{Qwen3TTS, platform};

// Configure thread pool for Pi
platform::configure_threadpool_for_raspberry_pi();

// Load model with quantization
let device = platform::select_device_for_raspberry_pi();
let model = Qwen3TTS::from_pretrained_quantized("path/to/model", device)?;

// Synthesize
let audio = model.synthesize("Hello from Raspberry Pi!", None)?;
audio.save("output.wav")?;
```

### Estimating Memory Usage

```rust
use qwen3_tts::models::QuantizationConfig;

let config = QuantizationConfig::for_raspberry_pi();
println!("Memory savings: {:.1}%", config.memory_savings_percent());
// Output: Memory savings: 50.0%
```

## Platform Detection

The library automatically detects ARM64 architecture and optimizes accordingly:

- **NEON intrinsics**: Used for vectorized math operations
- **Thread pool**: Configured for 3 threads (leaving 1 core for system)
- **CPU-only inference**: No GPU dependency

## Troubleshooting

### Out of Memory

If you encounter OOM errors on Pi 4 (4GB RAM):

1. Ensure quantization is enabled
2. Close other applications
3. Consider using swap space
4. Use the 0.6B model instead of 1.7B

### Slow Performance

If RTF > 3.0:

1. Verify quantization is enabled: `cargo build --features quantized`
2. Check thread count: `rayon::current_num_threads()`
3. Ensure no other CPU-intensive processes are running
4. Consider overclocking (if safe for your Pi)

### Cross-Compilation Errors

If `build_for_raspberry_pi.sh` fails:

- **macOS**: Install ARM toolchain: `brew install aarch64-unknown-linux-gnu`
- **Linux**: Install cross-compiler: `sudo apt install gcc-aarch64-linux-gnu`
- **Alternative**: Use Docker or `cross` tool

## Performance Benchmarks

Once deployed to actual hardware, run benchmarks:

```bash
# On Raspberry Pi
./vox benchmark --model path/to/model --quantized --iterations 10
```

Expected output (estimated):

```
Model: 0.6B Base (INT8 Quantized)
Device: ARM64 CPU (4 cores)
RTF: 2.1 ± 0.3
Memory: 847 MB
Throughput: 12 seconds of audio/minute
```

## Contributing

If you have access to Raspberry Pi hardware and can provide actual benchmarks, please:

1. Run the benchmark suite
2. Report your results (model, Pi version, RTF, memory)
3. Open a PR to update this document with real data

## References

- [ARM NEON Intrinsics Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [INT8 Quantization Paper](https://arxiv.org/abs/1712.05877)
- [Raspberry Pi 5 Specs](https://www.raspberrypi.com/products/raspberry-pi-5/)
