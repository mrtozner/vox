#!/bin/bash
set -e

echo "=== Building vox for Raspberry Pi ==="

# Install cross-compilation tools
echo "Installing cross-compilation tools..."
rustup target add aarch64-unknown-linux-gnu

# Install ARM GCC (platform-specific)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    if command -v brew &> /dev/null; then
        echo "Installing ARM toolchain via Homebrew..."
        brew install aarch64-unknown-linux-gnu || echo "ARM toolchain may already be installed"
    else
        echo "Warning: Homebrew not found. Please install ARM toolchain manually."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    if command -v apt-get &> /dev/null; then
        echo "Installing ARM toolchain via apt..."
        sudo apt-get update
        sudo apt-get install -y gcc-aarch64-linux-gnu || echo "ARM toolchain may already be installed"
    else
        echo "Warning: apt-get not found. Please install ARM toolchain manually."
    fi
fi

# Build for Raspberry Pi 4/5 (ARM64)
echo ""
echo "Building for Raspberry Pi (aarch64)..."
cargo build --release \
    --target aarch64-unknown-linux-gnu \
    --features server,qwen3,quantized \
    --no-default-features

echo ""
echo "✅ Build complete!"
echo "📦 Binary location: target/aarch64-unknown-linux-gnu/release/vox"
echo ""
echo "To deploy to Raspberry Pi:"
echo "  scp target/aarch64-unknown-linux-gnu/release/vox pi@raspberrypi:~/"
echo ""
echo "On Raspberry Pi:"
echo "  chmod +x vox"
echo "  ./vox serve --help"
