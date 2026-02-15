#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# Silero VAD v5.1 (~2MB)
if [ ! -f "$MODELS_DIR/silero_vad.onnx" ]; then
    echo "  Downloading Silero VAD v5.1..."
    curl -L -o "$MODELS_DIR/silero_vad.onnx" \
        "https://github.com/snakers4/silero-vad/raw/v5.1/src/silero_vad/data/silero_vad.onnx"
fi

# Whisper tiny.en (~75MB)
if [ ! -f "$MODELS_DIR/ggml-tiny.en.bin" ]; then
    echo "  Downloading Whisper tiny.en..."
    curl -L -o "$MODELS_DIR/ggml-tiny.en.bin" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
fi

# Kokoro TTS v1.0 (~310MB)
if [ ! -f "$MODELS_DIR/kokoro-v1.0.onnx" ]; then
    echo "  Downloading Kokoro v1.0..."
    curl -L -o "$MODELS_DIR/kokoro-v1.0.onnx" \
        "https://github.com/hexgrad/kokoro/releases/download/v1.0/kokoro-v1.0.onnx"
fi

# Kokoro voices (~27MB)
if [ ! -f "$MODELS_DIR/voices.bin" ]; then
    echo "  Downloading Kokoro voices..."
    curl -L -o "$MODELS_DIR/voices.bin" \
        "https://github.com/hexgrad/kokoro/releases/download/v1.0/voices-v1.0.bin"
fi

echo "Done. Models in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
