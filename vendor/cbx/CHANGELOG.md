# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-12-18

### Initial Release

First public release of cbx, a single-binary CLI for local text-to-speech using Chatterbox.

#### Features

- **Text-to-speech generation** - Generate speech from text using Resemble AI's Chatterbox ONNX models
- **Built-in model management** - Download, list, and clean up model files with simple commands
- **Voice profile caching** - Encode a reference voice once and reuse it across runs
- **Multiple model variants** - Support for fp16, fp32, quantized, q4, and q4f16 variants
- **Cross-platform support** - macOS, Linux, and Windows binaries
- **Platform acceleration** - Optional CoreML (macOS), DirectML (Windows), and CUDA (Linux) support
- **Default voice** - Optional bundled default voice for quick testing

#### Commands

- `cbx speak` - Generate speech from text
- `cbx download` - Pre-download model files
- `cbx sizes` - Show download sizes for each model variant
- `cbx models` - List cached models
- `cbx voice add` - Create a cached voice profile
- `cbx voice list` - List cached voice profiles
- `cbx voice remove` - Delete a voice profile
- `cbx clean` - Remove cached models
