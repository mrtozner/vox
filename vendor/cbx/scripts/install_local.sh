#!/usr/bin/env bash
set -euo pipefail

#
# Local installer for a locally-built cbx binary.
#
# Examples:
#   cargo build --release
#   ./scripts/install_local.sh
#
#   PREFIX=$HOME/.local ./scripts/install_local.sh
#

PREFIX="${PREFIX:-/usr/local}"
BIN_NAME="${BIN_NAME:-cbx}"
SRC_BIN="${SRC_BIN:-./target/release/${BIN_NAME}}"
DST_BIN="${DST_BIN:-${PREFIX}/bin/${BIN_NAME}}"

if [[ ! -f "${SRC_BIN}" ]]; then
  echo "error: binary not found at ${SRC_BIN}" >&2
  echo "hint: run: cargo build --release" >&2
  exit 1
fi

echo "Installing ${BIN_NAME}"
echo "  from: ${SRC_BIN}"
echo "  to:   ${DST_BIN}"

mkdir -p "$(dirname "${DST_BIN}")"
cp -f "${SRC_BIN}" "${DST_BIN}"
chmod +x "${DST_BIN}"

echo "Done."
echo "Run: ${BIN_NAME} --help"

