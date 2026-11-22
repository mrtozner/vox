# Vox on a Raspberry Pi

Vox targets 64-bit ARM Linux. That means a Raspberry Pi 4 or 5 (or a Pi 3 /
Zero 2 W, with much less headroom) running a **64-bit** OS. A 32-bit OS is not
supported by the release binaries.

## Install the release binary

The `aarch64-unknown-linux-gnu` archive on the
[releases page](https://github.com/mrtozner/vox/releases/latest) is built on
Ubuntu 24.04 and needs glibc 2.39 or newer: Raspberry Pi OS Trixie (64-bit),
Debian 13, Ubuntu 24.04 or later. Check with `ldd --version`. On Bookworm,
build from source with your own ONNX Runtime.

```bash
tar xzf vox-v*-aarch64-unknown-linux-gnu.tar.gz
sudo install vox-v*-aarch64-unknown-linux-gnu/vox /usr/local/bin/vox
sudo apt-get install -y libasound2t64  # ALSA runtime, for microphone and playback
vox --version
vox doctor
```

The release binary includes Piper and is distributed under GPL-3.0; see
`NOTICE` in the archive.

## Models for a small machine

Models are the main cost in memory and latency. Start small:

```bash
vox models download silero-vad
vox models download whisper-tiny.en-int8     # 42 MB, English
vox models download piper-en-us              # 60 MB
vox models download piper-en-us-config

vox listen --model tiny.en-int8
vox speak "Hello from the Pi" --backend piper --voice en
```

`whisper-base.en-int8` (78 MB) is more accurate and slower. For other
languages use a multilingual model, for example
`vox listen --model base --language de`. `vox models list` shows everything
that can be downloaded and its size.

For the web UI, `vox serve --download` fetches the default set for the
machine; `vox serve --stt-model tiny.en` forces a smaller Whisper model.
Live Talk and Chat also need an Ollama server, which can run on another
machine: `vox serve --ollama-host other-host:11434`.

To reach the UI from another device on your network the server must bind to
a non-loopback address, which it refuses by default because it has no
authentication. Read the server section of the README before setting
`VOX_ALLOW_REMOTE=1`.

## Build on the Pi

Building needs Rust 1.88+, about 4 GB of RAM plus swap, and patience. The
build links the same prebuilt ONNX Runtime as the release binary, so it needs
glibc 2.39+ as well.

```bash
sudo apt-get install -y build-essential cmake clang libclang-dev pkg-config libasound2-dev
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features dist
```

Use `--features cli` for a transcription-only build, which is much quicker.
Add `-j 2` if the build runs out of memory.

## Sherpa-ONNX backends

The `sherpa` feature links the sherpa-onnx shared libraries. On Linux the
build script downloads the matching sherpa-onnx release for the target into
`~/.cache/vox/sherpa-onnx/` and copies the libraries to
`~/.local/share/vox/lib`, which is on the binary's rpath.

- To use your own sherpa-onnx build, set `SHERPA_ONNX_DIR` to its install
  directory (the one containing `lib/`).
- To forbid the download (offline builds), set `VOX_SHERPA_NO_AUTODOWNLOAD=1`
  and provide `SHERPA_ONNX_DIR`.

The release binaries are built without `sherpa`.
