#!/usr/bin/env bash
set -euo pipefail

#
# Local uninstaller for cbx installed via scripts/install_local.sh.
#
# Examples:
#   ./scripts/uninstall_local.sh
#   PREFIX=$HOME/.local ./scripts/uninstall_local.sh
#

PREFIX="${PREFIX:-/usr/local}"
BIN_NAME="${BIN_NAME:-cbx}"
DST_BIN="${DST_BIN:-${PREFIX}/bin/${BIN_NAME}}"

echo "Uninstalling ${BIN_NAME}"
echo "  target: ${DST_BIN}"

if [[ -f "${DST_BIN}" ]]; then
  rm -f "${DST_BIN}"
  echo "Removed ${DST_BIN}"
else
  echo "Not found; nothing to do."
fi

echo "Note: model files are stored in your Hugging Face cache."
