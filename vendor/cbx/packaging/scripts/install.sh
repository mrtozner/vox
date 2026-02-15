#!/usr/bin/env bash
# Universal installer script for cbx (downloads from GitHub Releases)

set -euo pipefail

# Configuration
REPO="${REPO:-srv1n/cbx}"
BINARY_NAME="cbx"

# Default to ~/.local/bin (no sudo required)
DEFAULT_INSTALL_DIR="${HOME}/.local/bin"
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
TEMP_DIR="${TEMP_DIR:-/tmp/cbx-install}"

print() { printf "%s\n" "$*"; }
err() { printf "error: %s\n" "$*" >&2; }

detect_platform() {
  local os arch

  case "${OSTYPE:-}" in
    linux*) os="unknown-linux-gnu" ;;
    darwin*) os="apple-darwin" ;;
    msys*|cygwin*|win32*) os="pc-windows-msvc" ;;
    *)
      err "unsupported OS: ${OSTYPE:-unknown}"
      return 1
      ;;
  esac

  case "$(uname -m 2>/dev/null || echo unknown)" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
      err "unsupported arch: $(uname -m 2>/dev/null || echo unknown)"
      return 1
      ;;
  esac

  echo "${arch}-${os}"
}

http_get() {
  local url=$1
  local out=$2

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$out" "$url"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$out" "$url"
    return 0
  fi

  err "neither curl nor wget found"
  return 1
}

get_latest_version() {
  local api="https://api.github.com/repos/${REPO}/releases/latest"
  local tmp="${TEMP_DIR}/latest.json"
  mkdir -p "${TEMP_DIR}"
  http_get "$api" "$tmp"

  local tag
  tag="$(grep -m1 '"tag_name"' "$tmp" | cut -d'"' -f4 || true)"
  if [[ -z "${tag}" ]]; then
    err "failed to determine latest version tag from GitHub API"
    return 1
  fi
  echo "$tag"
}

verify_sha256() {
  local file=$1
  local sha_file=$2
  local norm_sha_file="${sha_file}.normalized"

  # Older releases may have produced checksums that include a leading `dist/` path.
  # Normalize the filename column so `shasum -c` / `sha256sum -c` can find the archive
  # in the current working directory.
  awk '{
    f=$2;
    sub(/^dist\//, "", f);
    sub(/^\.\//, "", f);
    print $1 "  " f
  }' "$sha_file" > "$norm_sha_file"

  if command -v shasum >/dev/null 2>&1; then
    (cd "$(dirname "$file")" && shasum -a 256 -c "$(basename "$norm_sha_file")")
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$file")" && sha256sum -c "$(basename "$norm_sha_file")")
    return 0
  fi

  err "no sha256 verifier found (need shasum or sha256sum)"
  return 1
}

main() {
  local target version archive_ext archive_name base_url download_url sha_url

  target="$(detect_platform)"
  version="${VERSION:-$(get_latest_version)}"

  if [[ "$target" == *"windows"* ]]; then
    archive_ext="zip"
  else
    archive_ext="tar.gz"
  fi

  archive_name="${BINARY_NAME}-${version}-${target}.${archive_ext}"
  base_url="https://github.com/${REPO}/releases/download/${version}"
  download_url="${base_url}/${archive_name}"
  sha_url="${download_url}.sha256"

  print "cbx installer"
  print "  repo:       ${REPO}"
  print "  version:    ${version}"
  print "  target:     ${target}"
  print "  install:    ${INSTALL_DIR}/${BINARY_NAME}"
  print "  download:   ${download_url}"

  rm -rf "${TEMP_DIR}"
  mkdir -p "${TEMP_DIR}"
  cd "${TEMP_DIR}"

  http_get "${download_url}" "${archive_name}"
  http_get "${sha_url}" "${archive_name}.sha256"
  verify_sha256 "${TEMP_DIR}/${archive_name}" "${TEMP_DIR}/${archive_name}.sha256"

  if [[ "$archive_ext" == "zip" ]]; then
    command -v unzip >/dev/null 2>&1 || { err "unzip not found"; return 1; }
    unzip -q "${archive_name}"
  else
    tar -xzf "${archive_name}"
  fi

  local extracted="${BINARY_NAME}"
  if [[ "$target" == *"windows"* ]]; then
    extracted="${BINARY_NAME}.exe"
  fi
  if [[ ! -f "${extracted}" ]]; then
    err "binary not found in archive: ${extracted}"
    return 1
  fi

  mkdir -p "${INSTALL_DIR}"
  if [[ -w "${INSTALL_DIR}" ]]; then
    cp "${extracted}" "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}" || true
  else
    print "Installing to ${INSTALL_DIR} (requires sudo)"
    sudo cp "${extracted}" "${INSTALL_DIR}/"
    sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}" || true
  fi

  print "Installed: ${INSTALL_DIR}/${BINARY_NAME}"
  if command -v "${BINARY_NAME}" >/dev/null 2>&1; then
    "${BINARY_NAME}" --help >/dev/null 2>&1 || true
    print "OK: '${BINARY_NAME}' is on PATH"
  else
    print "Note: '${BINARY_NAME}' is not on PATH. Add this to your shell profile:"
    print "  export PATH=\"${INSTALL_DIR}:\\$PATH\""
  fi

  print "First run will download models from Hugging Face Hub:"
  print "  ${BINARY_NAME} speak --dtype fp16 --text \"Hello\" --voice-wav ./voice.wav --out-wav ./out.wav"
}

main "$@"
