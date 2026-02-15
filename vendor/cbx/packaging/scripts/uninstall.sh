#!/usr/bin/env bash
# Uninstall script for cbx (removes the installed binary; caches are managed separately)

set -euo pipefail

BINARY_NAME="cbx"

# Match install.sh default
DEFAULT_INSTALL_DIR="${HOME}/.local/bin"
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"

print() { printf "%s\n" "$*"; }

main() {
  local bin="${INSTALL_DIR}/${BINARY_NAME}"

  print "cbx uninstaller"
  print "  uninstall:  ${bin}"

  if [[ -f "${bin}" ]]; then
    rm -f "${bin}"
    print "Removed: ${bin}"
  else
    print "Not found: ${bin}"
  fi

  print ""
  print "Note: model + voice caches are NOT removed by this script."
  print "To remove caches, run:"
  print "  cbx clean --all --voices --yes"
}

main "$@"

