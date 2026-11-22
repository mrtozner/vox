#!/usr/bin/env bash
# Smoke test for a built `vox` binary. Used by CI and by the release workflow.
#
#   .github/scripts/smoke-test.sh <path-to-vox> [expected-version]
#
# Checks, without any model files installed:
#   - `vox --version` (and that it matches expected-version when given)
#   - `--help` of every subcommand the README documents
#   - `vox models list`
#   - when the binary has the server feature: `vox serve` answers GET /health
#
# Environment:
#   VOX_SMOKE_PORT   port for `vox serve` (default 38917)
set -euo pipefail

BIN="${1:?usage: smoke-test.sh <path-to-vox> [expected-version]}"
EXPECTED_VERSION="${2:-}"
PORT="${VOX_SMOKE_PORT:-38917}"

WORK="$(mktemp -d)"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# An empty models directory: the binary must behave without any model.
export VOX_MODELS_DIR="$WORK/models"
mkdir -p "$VOX_MODELS_DIR"

echo "== vox --version"
VERSION_OUT="$("$BIN" --version)"
echo "$VERSION_OUT"
if [ -n "$EXPECTED_VERSION" ] && [ "$VERSION_OUT" != "vox $EXPECTED_VERSION" ]; then
    echo "error: expected 'vox $EXPECTED_VERSION', got '$VERSION_OUT'" >&2
    exit 1
fi

HELP_OUT="$("$BIN" --help)"
has_subcommand() {
    printf '%s\n' "$HELP_OUT" | grep -Eq "^  $1( |\$)"
}

# Every subcommand the README documents. `serve` exists only in builds with
# the server feature, so it is checked below.
for sub in listen speak chat test benchmark tune config doctor models; do
    echo "== vox $sub --help"
    if ! has_subcommand "$sub"; then
        echo "error: 'vox --help' does not list '$sub'" >&2
        exit 1
    fi
    "$BIN" "$sub" --help >/dev/null
done
for sub in list download path; do
    echo "== vox models $sub --help"
    "$BIN" models "$sub" --help >/dev/null
done

echo "== vox models list"
"$BIN" models list >/dev/null
echo "== vox models path"
"$BIN" models path

if ! has_subcommand serve; then
    echo "== no 'serve' subcommand in this build, skipping the /health check"
    echo "smoke test passed"
    exit 0
fi

echo "== vox serve --help"
"$BIN" serve --help >/dev/null

echo "== vox serve --port $PORT, GET /health"
"$BIN" serve --host 127.0.0.1 --port "$PORT" >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!

BODY=""
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: 'vox serve' exited before answering /health" >&2
        cat "$WORK/serve.log" >&2
        SERVER_PID=""
        exit 1
    fi
    if BODY="$(curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)"; then
        break
    fi
    BODY=""
    sleep 1
done

if [ -z "$BODY" ]; then
    echo "error: /health did not answer within 60 s" >&2
    cat "$WORK/serve.log" >&2
    exit 1
fi
echo "$BODY"
case "$BODY" in
    *'"status":"ok"'*) ;;
    *)
        echo "error: unexpected /health body" >&2
        exit 1
        ;;
esac
if [ -n "$EXPECTED_VERSION" ]; then
    case "$BODY" in
        *"\"version\":\"$EXPECTED_VERSION\""*) ;;
        *)
            echo "error: /health does not report version $EXPECTED_VERSION" >&2
            exit 1
            ;;
    esac
fi

echo "smoke test passed"
