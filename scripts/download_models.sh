#!/usr/bin/env bash
# Download the default vox models with curl (no vox binary needed).
#
# Usage: scripts/download_models.sh [MODELS_DIR] [--lang CODE]
#
#   MODELS_DIR   target directory (default: ./models, which a source checkout
#                picks up automatically; installed binaries look in
#                $VOX_MODELS_DIR or the per-user data dir, see `vox models path`)
#   --lang CODE  also fetch what a non-English setup needs: the multilingual
#                Whisper base model and, when available, the Piper voice
#                (supported: de, es, fr, tr, zh)
#
# Every URL is pinned to an immutable revision and every file is checked
# against its SHA-256. The table below mirrors the registry in
# src/cli/models.rs — keep them in sync. If you already have the binary,
# `vox serve --download` does the same with resume support.
set -euo pipefail

MODELS_DIR="models"
LANG_CODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lang) LANG_CODE="${2:?--lang needs a language code}"; shift 2 ;;
        --lang=*) LANG_CODE="${1#--lang=}"; shift ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) MODELS_DIR="$1"; shift ;;
    esac
done

HF="https://huggingface.co"
WHISPER="$HF/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1"
PIPER="$HF/rhasspy/piper-voices/resolve/2f358b6a560e3a913ebf509b2f3344f20b367f94"
KOKORO="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
SILERO="https://raw.githubusercontent.com/snakers4/silero-vad/6478567951ae5c9979ad7b234185b5515f4be7a1"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# fetch <label> <relative path> <sha256> <url>
fetch() {
    local label="$1" dest="$MODELS_DIR/$2" want="$3" url="$4" got
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ]; then
        got="$(sha256_of "$dest")"
        if [ "$got" = "$want" ]; then
            echo "  ok        $label"
            return 0
        fi
        echo "  replacing $label (checksum mismatch: truncated or different revision)"
        rm -f "$dest"
    fi
    echo "  fetching  $label"
    # -C - resumes an interrupted download of the .part file.
    curl -fL --retry 3 -C - -o "$dest.part" "$url"
    got="$(sha256_of "$dest.part")"
    if [ "$got" != "$want" ]; then
        rm -f "$dest.part"
        echo "ERROR: checksum mismatch for $label" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
}

echo "Downloading models to $MODELS_DIR..."

fetch "Silero VAD v5.1.2 (2 MB)" "silero_vad.onnx" \
    2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f \
    "$SILERO/src/silero_vad/data/silero_vad.onnx"

fetch "Whisper tiny.en (74 MB)" "ggml-tiny.en.bin" \
    921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f \
    "$WHISPER/ggml-tiny.en.bin"

fetch "Kokoro v1.0 (310 MB)" "kokoro-v1.0.onnx" \
    7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 \
    "$KOKORO/kokoro-v1.0.onnx"

fetch "Kokoro voices (27 MB)" "voices.bin" \
    988549ccf1d68a80a31de1b9d00ac59d5dceb69743cbc607d1de17c7c4cdccf4 \
    "https://github.com/mzdk100/kokoro/releases/download/V1.0/voices.bin"

fetch "Piper en_US lessac (60 MB)" "piper/en_US-lessac-medium.onnx" \
    5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f \
    "$PIPER/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
fetch "Piper en_US lessac config" "piper/en_US-lessac-medium.onnx.json" \
    efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0 \
    "$PIPER/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

if [ -n "$LANG_CODE" ] && [ "$LANG_CODE" != "en" ]; then
    fetch "Whisper base, multilingual (141 MB)" "ggml-base.bin" \
        60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe \
        "$WHISPER/ggml-base.bin"

    case "$LANG_CODE" in
        tr)
            fetch "Piper tr_TR dfki (60 MB)" "piper/tr_TR-dfki-medium.onnx" \
                2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb \
                "$PIPER/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx"
            fetch "Piper tr_TR dfki config" "piper/tr_TR-dfki-medium.onnx.json" \
                13ebd7810f1b61b5027583cf3131a0a233b6ea81c38f2200ebc4ff41c3cca039 \
                "$PIPER/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json"
            ;;
        de)
            fetch "Piper de_DE thorsten (60 MB)" "piper/de_DE-thorsten-medium.onnx" \
                fd270eb3d1ebf91d763f8a0f949b1b1bebdfefaea18c3aaf8577dd34536e064f \
                "$PIPER/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
            fetch "Piper de_DE thorsten config" "piper/de_DE-thorsten-medium.onnx.json" \
                cf245c91668e74cb6716f38647ae3d6d9bac42965872596f65f9ab2b7d7221ed \
                "$PIPER/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
            ;;
        es)
            fetch "Piper es_ES davefx (60 MB)" "piper/es_ES-davefx-medium.onnx" \
                d2c9224e122321aac3c33343957fff4d0f465226cb74c66b0f520ca3f8c0dd07 \
                "$PIPER/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
            fetch "Piper es_ES davefx config" "piper/es_ES-davefx-medium.onnx.json" \
                531c4051502e1acbc10b42437f11679bfce9e3919e1796e51fabd73e01e7fd2e \
                "$PIPER/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
            ;;
        fr)
            fetch "Piper fr_FR siwis (60 MB)" "piper/fr_FR-siwis-medium.onnx" \
                b1d946fd68d0347155fa59bce0a018924abbef3c1561181db45ca463d83963ef \
                "$PIPER/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
            fetch "Piper fr_FR siwis config" "piper/fr_FR-siwis-medium.onnx.json" \
                df814aa3dfedd00510a07144442cc0a220ff21e43d03c3b078444fc0be68fc37 \
                "$PIPER/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json"
            ;;
        zh)
            fetch "Piper zh_CN huayan (60 MB)" "piper/zh_CN-huayan-medium.onnx" \
                9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d \
                "$PIPER/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx"
            fetch "Piper zh_CN huayan config" "piper/zh_CN-huayan-medium.onnx.json" \
                d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b \
                "$PIPER/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"
            ;;
        *)
            echo "  note: no Piper voice bundled for '$LANG_CODE'; the English voice will be used"
            ;;
    esac
fi

echo "Done. Models in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
