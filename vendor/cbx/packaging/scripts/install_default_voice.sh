#!/usr/bin/env bash
# Installs the bundled "default" cbx voice profiles into the local voice cache.
# This downloads from GitHub Releases and verifies SHA256.

set -euo pipefail

REPO="${REPO:-srv1n/cbx}"
VERSION="${VERSION:-}"

print() { printf "%s\n" "$*"; }
err() {
  printf "\n\033[1;31m✗ Error:\033[0m %s\n" "$*" >&2
}
info() { printf "\033[1;34mℹ\033[0m %s\n" "$*"; }
success() { printf "\033[1;32m✓\033[0m %s\n" "$*"; }

detect_hf_home() {
  if [[ -n "${HF_HOME:-}" ]]; then
    print "${HF_HOME}"
    return 0
  fi

  # Common defaults used by Hugging Face tooling:
  # - Linux: ~/.cache/huggingface
  # - macOS: ~/Library/Caches/huggingface
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Darwin) print "${HOME}/Library/Caches/huggingface" ;;
    *) print "${HOME}/.cache/huggingface" ;;
  esac
}

http_get() {
  local url=$1
  local out=$2

  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsSL -o "$out" "$url" 2>/dev/null; then
      err "Failed to download: $url"
      info "This could mean:"
      info "  • The file hasn't been uploaded to GitHub Releases yet"
      info "  • The version tag ($VERSION) is incorrect"
      info "  • Network connectivity issues"
      info ""
      info "Expected URL: $url"
      return 1
    fi
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    if ! wget -q -O "$out" "$url" 2>/dev/null; then
      err "Failed to download: $url"
      info "This could mean:"
      info "  • The file hasn't been uploaded to GitHub Releases yet"
      info "  • The version tag ($VERSION) is incorrect"
      info "  • Network connectivity issues"
      info ""
      info "Expected URL: $url"
      return 1
    fi
    return 0
  fi

  err "Neither curl nor wget found"
  info "Please install curl or wget to continue"
  return 1
}

get_latest_version() {
  local api="https://api.github.com/repos/${REPO}/releases/latest"
  local tmp
  tmp="$(mktemp 2>/dev/null || mktemp -t cbx-latest)"

  if ! http_get "$api" "$tmp"; then
    rm -f "$tmp" || true
    return 1
  fi

  local tag
  # Parse `"tag_name": "vX.Y.Z"` from the GitHub API response.
  # Use `cut -d'"'` (1-byte delimiter); macOS `cut` rejects multi-byte delimiters.
  tag="$(grep -m1 '\"tag_name\"' "$tmp" | cut -d'"' -f4 || true)"
  rm -f "$tmp" || true
  if [[ -z "${tag}" ]]; then
    err "Failed to determine latest version tag from GitHub API"
    info "Could not parse release information from: $api"
    return 1
  fi
  print "$tag"
}

verify_sha256() {
  local file=$1
  local sha_file=$2

  if command -v shasum >/dev/null 2>&1; then
    if ! (cd "$(dirname "$file")" && shasum -a 256 -c "$(basename "$sha_file")" 2>&1 | grep -q "OK"); then
      err "SHA256 checksum verification failed for $(basename "$file")"
      info "The downloaded file may be corrupted or tampered with"
      return 1
    fi
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    if ! (cd "$(dirname "$file")" && sha256sum -c "$(basename "$sha_file")" 2>&1 | grep -q "OK"); then
      err "SHA256 checksum verification failed for $(basename "$file")"
      info "The downloaded file may be corrupted or tampered with"
      return 1
    fi
    return 0
  fi

  err "No SHA256 verifier found"
  info "Please install shasum or sha256sum to verify downloads"
  return 1
}

main() {
  local hf_home voice_dir base_url tmp
  hf_home="$(detect_hf_home)"
  voice_dir="${hf_home}/cbx/voices"
  if [[ -z "${VERSION}" ]]; then
    info "Detecting latest version..."
    VERSION="$(get_latest_version)"
  fi
  base_url="https://github.com/${REPO}/releases/download/${VERSION}"
  tmp="$(mktemp -d 2>/dev/null || mktemp -d -t cbx-voice)"

  print ""
  print "═══════════════════════════════════════════════════"
  print "  cbx default voice installer"
  print "═══════════════════════════════════════════════════"
  print ""
  info "Repository:   ${REPO}"
  info "Version:      ${VERSION}"
  info "HF Home:      ${hf_home}"
  info "Install to:   ${voice_dir}"
  print ""
  info "This installs two profiles:"
  print "  • default.cbxvoice      (dtype=fp16)"
  print "  • default-fp32.cbxvoice (dtype=fp32)"
  print ""

  mkdir -p "${voice_dir}"

  for name in cbx-voice-default-fp16.cbxvoice cbx-voice-default-fp32.cbxvoice; do
    local url="${base_url}/${name}"
    local sha="${url}.sha256"
    local out="${tmp}/${name}"
    local sha_out="${tmp}/${name}.sha256"

    info "Downloading ${name}..."
    if ! http_get "${url}" "${out}"; then
      rm -rf "${tmp}" || true
      exit 1
    fi

    info "Downloading checksum..."
    if ! http_get "${sha}" "${sha_out}"; then
      rm -rf "${tmp}" || true
      exit 1
    fi

    info "Verifying integrity..."
    if ! verify_sha256 "${out}" "${sha_out}"; then
      rm -rf "${tmp}" || true
      exit 1
    fi

    if [[ "${name}" == "cbx-voice-default-fp16.cbxvoice" ]]; then
      cp -f "${out}" "${voice_dir}/default.cbxvoice"
    else
      cp -f "${out}" "${voice_dir}/default-fp32.cbxvoice"
    fi
  done

  rm -rf "${tmp}" || true

  print ""
  print "═══════════════════════════════════════════════════"
  success "Installation complete!"
  print "═══════════════════════════════════════════════════"
  print ""
  info "Installed files:"
  print "  • ${voice_dir}/default.cbxvoice"
  print "  • ${voice_dir}/default-fp32.cbxvoice"
  print ""
  info "Try it out:"
  print "  cbx speak --text \"Hello\" --out-wav out.wav"
  print "  cbx speak --dtype fp32 --voice default-fp32 --text \"Hello\" --out-wav out.wav"
  print ""
}

main "$@"
