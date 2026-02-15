#!/usr/bin/env bash
# build_static.sh — Build a fully static Vox binary with musl.
#
# Usage:
#   ./scripts/build_static.sh                    # default: silero feature, x86_64
#   ./scripts/build_static.sh --features silero,piper
#   ./scripts/build_static.sh --target aarch64-unknown-linux-musl
#   ./scripts/build_static.sh --docker            # build inside Docker (recommended)
#   ./scripts/build_static.sh --release           # (default) release build
#   ./scripts/build_static.sh --debug             # debug build
#
# Prerequisites (host build):
#   - rustup target add x86_64-unknown-linux-musl
#   - musl-tools (Debian/Ubuntu) or musl-gcc (Fedora/Arch)
#   - ALSA development headers: libasound2-dev / alsa-lib-dev
#   - For whisper feature: cmake, g++ (C++17 capable)
#   - For ort/silero feature: cmake, g++
#
# ============================================================================
# MUSL COMPATIBILITY MATRIX
# ============================================================================
#
# Feature      | musl static | Notes
# -------------|-------------|-----------------------------------------------
# silero (VAD) | YES         | ONNX Runtime builds statically via ort crate
# piper (TTS)  | YES         | ONNX Runtime, lightweight
# whisper (STT)| PARTIAL     | whisper.cpp C++ build works but needs cmake/g++
# kokoro (TTS) | PARTIAL     | Needs ONNX Runtime + rodio (audio output)
# pocket (TTS) | PARTIAL     | Candle (pure Rust) works; rodio needs ALSA
# chatterbox   | PARTIAL     | Needs ONNX Runtime + rodio
# tts (playback)| NO*        | rodio requires ALSA runtime on Linux
# (audio capture)| NO*       | cpal requires ALSA runtime on Linux
#
# *Audio I/O: cpal and rodio link against ALSA (libasound). The static binary
#  will include the ALSA client library, but ALSA still needs a running sound
#  server at runtime. For truly headless/embedded use, audio features should
#  be disabled. NOTE: cpal is currently a non-optional dependency in Cargo.toml,
#  so the ALSA headers are always needed at build time even when audio capture
#  is not used at runtime.
#
# RECOMMENDED static build profiles:
#   1. Headless VAD:       --features silero        (smallest, ~5-10MB)
#   2. Headless VAD+STT:   --features silero,whisper (needs cmake)
#   3. Headless VAD+TTS:   --features silero,piper   (lightweight TTS)
#
# ============================================================================
set -euo pipefail

# --- Defaults ---------------------------------------------------------------
TARGET="x86_64-unknown-linux-musl"
FEATURES="silero"
PROFILE="release"
USE_DOCKER=false
EXTRA_CARGO_ARGS=()

# --- Colors -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

# --- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        --debug)
            PROFILE="debug"
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            EXTRA_CARGO_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Project root -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

info "Vox static build"
info "  Target:   $TARGET"
info "  Features: $FEATURES"
info "  Profile:  $PROFILE"

# --- Docker build path ------------------------------------------------------
if $USE_DOCKER; then
    info "Building inside Docker (Dockerfile.static)..."

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. Install Docker or build without --docker."
        exit 1
    fi

    DOCKER_IMAGE="vox-static-builder"

    # Build the builder image
    info "Building Docker image '$DOCKER_IMAGE'..."
    docker build \
        -f "$PROJECT_ROOT/Dockerfile.static" \
        -t "$DOCKER_IMAGE" \
        --build-arg "FEATURES=$FEATURES" \
        --build-arg "TARGET=$TARGET" \
        --build-arg "PROFILE=$PROFILE" \
        "$PROJECT_ROOT"

    # Extract the binary from the final stage
    CONTAINER_ID=$(docker create "$DOCKER_IMAGE")
    PROFILE_DIR="$PROFILE"
    if [ "$PROFILE" = "debug" ]; then
        PROFILE_DIR="debug"
    fi

    OUTPUT_DIR="$PROJECT_ROOT/target/$TARGET/$PROFILE_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Try to copy the library build artifact (vox is a library crate)
    # For binary targets, the user would specify --bin in EXTRA_CARGO_ARGS
    docker cp "$CONTAINER_ID:/output/." "$OUTPUT_DIR/" 2>/dev/null || true
    docker rm "$CONTAINER_ID" >/dev/null

    ok "Docker build complete. Artifacts in: target/$TARGET/$PROFILE_DIR/"
    exit 0
fi

# --- Host build path --------------------------------------------------------
info "Building on host..."

# Check prerequisites
MISSING_DEPS=()

if ! rustup target list --installed 2>/dev/null | grep -q "$TARGET"; then
    warn "Rust target '$TARGET' not installed. Installing..."
    rustup target add "$TARGET"
fi

if [[ "$TARGET" == *"x86_64"*"musl"* ]]; then
    if ! command -v musl-gcc &>/dev/null; then
        MISSING_DEPS+=("musl-gcc (install: apt install musl-tools / apk add musl-dev)")
    fi
fi

if [[ "$TARGET" == *"aarch64"*"musl"* ]]; then
    if ! command -v aarch64-linux-musl-gcc &>/dev/null; then
        MISSING_DEPS+=("aarch64-linux-musl-gcc (install cross-compilation toolchain)")
    fi
fi

# Check for C/C++ build tools if whisper or ort-dependent features are used
if [[ "$FEATURES" == *"whisper"* ]] || [[ "$FEATURES" == *"silero"* ]] || \
   [[ "$FEATURES" == *"kokoro"* ]] || [[ "$FEATURES" == *"piper"* ]] || \
   [[ "$FEATURES" == *"chatterbox"* ]]; then
    if ! command -v cmake &>/dev/null; then
        MISSING_DEPS+=("cmake (needed for C/C++ native dependencies)")
    fi
fi

# cpal always needs ALSA headers on Linux
if [[ "$(uname)" == "Linux" ]]; then
    if ! pkg-config --exists alsa 2>/dev/null; then
        MISSING_DEPS+=("libasound2-dev / alsa-lib-dev (needed by cpal)")
    fi
fi

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    error "Missing dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        error "  - $dep"
    done
    error ""
    error "Install them or use --docker for a containerized build."
    exit 1
fi

# --- Build ------------------------------------------------------------------
CARGO_CMD=(
    cargo build
    --target "$TARGET"
    --no-default-features
    --features "$FEATURES"
)

if [ "$PROFILE" = "release" ]; then
    CARGO_CMD+=(--release)
fi

if [[ ${#EXTRA_CARGO_ARGS[@]} -gt 0 ]]; then
    CARGO_CMD+=("${EXTRA_CARGO_ARGS[@]}")
fi

info "Running: ${CARGO_CMD[*]}"
echo ""

# Set RUSTFLAGS for full static linking
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+crt-static"

"${CARGO_CMD[@]}"

echo ""
ok "Build complete!"

# --- Summary ----------------------------------------------------------------
PROFILE_DIR="$PROFILE"
if [ "$PROFILE" = "debug" ]; then
    PROFILE_DIR="debug"
fi

ARTIFACT_DIR="target/$TARGET/$PROFILE_DIR"

info "Artifacts in: $ARTIFACT_DIR/"

# Show binary info if any executables were built
if ls "$ARTIFACT_DIR"/vox* &>/dev/null 2>&1; then
    info "Binary details:"
    for bin in "$ARTIFACT_DIR"/vox*; do
        if [ -f "$bin" ] && [ -x "$bin" ]; then
            SIZE=$(du -h "$bin" | cut -f1)
            echo "  $bin ($SIZE)"
            # Check if truly static
            if command -v file &>/dev/null; then
                FILE_INFO=$(file "$bin")
                if echo "$FILE_INFO" | grep -q "statically linked"; then
                    ok "  Statically linked"
                else
                    warn "  Not fully static: $FILE_INFO"
                fi
            fi
        fi
    done
fi

# Show the .rlib for library builds
if ls "$ARTIFACT_DIR"/libvox* &>/dev/null 2>&1; then
    info "Library artifacts:"
    for lib in "$ARTIFACT_DIR"/libvox*; do
        if [ -f "$lib" ]; then
            SIZE=$(du -h "$lib" | cut -f1)
            echo "  $lib ($SIZE)"
        fi
    done
fi

echo ""
info "To verify static linking on a built binary:"
info "  file target/$TARGET/$PROFILE_DIR/<binary>"
info "  ldd  target/$TARGET/$PROFILE_DIR/<binary>  # should say 'not a dynamic executable'"
