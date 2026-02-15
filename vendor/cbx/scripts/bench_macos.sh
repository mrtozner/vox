#!/usr/bin/env bash
set -euo pipefail

# macOS benchmarking helper for cbx.
#
# Produces a markdown report with wall-clock timings for a set of variants
# (CPU vs CoreML, different thread counts).
#
# Usage:
#   ./scripts/bench_macos.sh
#
# Optional env vars:
#   CBX=cbx                         # command/binary to run
#   OUT_DIR=./bench                 # where to write report + logs
#   DTYPE=fp16                      # fp16 recommended default
#   TEXT="Hello ..."                # text to synthesize
#   WARMUP=1                        # warmup runs per case (not recorded)
#   REPEATS=5                       # recorded runs per case
#   COREML_CACHE_DIR=...            # persistent CoreML cache dir
#

CBX="${CBX:-cbx}"
OUT_DIR="${OUT_DIR:-./bench}"
DTYPE="${DTYPE:-fp16}"
TEXT="${TEXT:-The quick brown fox jumps over the lazy dog. This is a benchmark run for cbx.}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-5}"
COREML_CACHE_DIR="${COREML_CACHE_DIR:-$HOME/.cache/cbx-coreml}"

mkdir -p "$OUT_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: missing required command: $1" >&2; exit 1; }
}

require_cmd /usr/bin/time
require_cmd awk
require_cmd uname
require_cmd sysctl

if ! command -v "$CBX" >/dev/null 2>&1; then
  echo "error: cbx not found on PATH (CBX=$CBX)" >&2
  echo "hint: run: curl -fsSL https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install.sh | bash" >&2
  exit 1
fi

REPORT_MD="$OUT_DIR/bench-results.md"
CSV="$OUT_DIR/bench-results.csv"
LOG="$OUT_DIR/bench.log"

echo "cbx benchmark starting ($(ts))" | tee "$LOG"
echo "  cbx:      $CBX" | tee -a "$LOG"
echo "  dtype:    $DTYPE" | tee -a "$LOG"
echo "  warmup:   $WARMUP" | tee -a "$LOG"
echo "  repeats:  $REPEATS" | tee -a "$LOG"
echo "  out_dir:  $OUT_DIR" | tee -a "$LOG"
echo "  coreml_cache_dir: $COREML_CACHE_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Preflight: try to ensure we won't accidentally benchmark downloads.
set +e
$CBX models >/dev/null 2>&1
HAS_MODELS=$?
$CBX voice list >/dev/null 2>&1
HAS_VOICES=$?
set -e
if [[ "$HAS_MODELS" -ne 0 ]]; then
  echo "warning: 'cbx models' failed; continuing (but you may end up benchmarking downloads)." | tee -a "$LOG"
fi
if [[ "$HAS_VOICES" -ne 0 ]]; then
  echo "warning: 'cbx voice list' failed; continuing (but you may need a cached voice for fast-path)." | tee -a "$LOG"
fi

echo "case,ep,intra,parallel,inter,avg_s,min_s,avg_inner_total_ms,avg_lm_ms,avg_decoder_ms,runs" > "$CSV"

run_case() {
  local name="$1"
  local ep="$2"
  local intra="$3"
  local parallel="$4"
  local inter="$5"

  local tmp_times="$OUT_DIR/${name}.times.txt"
  local tmp_inner_total="$OUT_DIR/${name}.inner_total_ms.txt"
  local tmp_lm="$OUT_DIR/${name}.lm_ms.txt"
  local tmp_dec="$OUT_DIR/${name}.decoder_ms.txt"
  : > "$tmp_times"
  : > "$tmp_inner_total"
  : > "$tmp_lm"
  : > "$tmp_dec"

  local global_args=()
  global_args+=(--ep "$ep")
  global_args+=(--intra-threads "$intra")
  if [[ "$parallel" == "true" ]]; then
    global_args+=(--parallel-execution --inter-threads "$inter")
  fi
  if [[ "$ep" == "coreml" || "$ep" == "auto" ]]; then
    global_args+=(--coreml-cache-dir "$COREML_CACHE_DIR")
  fi

  echo "== $name ==" | tee -a "$LOG"
  echo "  args: ${global_args[*]} speak --dtype $DTYPE ..." | tee -a "$LOG"

  local i
  for ((i = 1; i <= WARMUP; i++)); do
    /usr/bin/time -p "$CBX" "${global_args[@]}" speak --dtype "$DTYPE" --text "$TEXT" --out-wav /tmp/cbx-bench.wav \
      >/dev/null 2>&1 || true
  done

  for ((i = 1; i <= REPEATS; i++)); do
    local tfile="$OUT_DIR/${name}.time.${i}.txt"
    local ofile="$OUT_DIR/${name}.out.${i}.txt"
    set +e
    /usr/bin/time -p "$CBX" "${global_args[@]}" speak --dtype "$DTYPE" --text "$TEXT" --out-wav /tmp/cbx-bench.wav \
      >"$ofile" 2>"$tfile"
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      echo "  run $i failed (exit=$rc); see $ofile" | tee -a "$LOG"
      continue
    fi

    # Extract `real` seconds.
    awk '/^real /{print $2}' "$tfile" >> "$tmp_times"

    # Extract inner timings (ms) from the `Timings(ms):` line printed by cbx.
    # Example:
    # Timings(ms): ... | tokenize=2 embed=26 lm=12878 decoder=8697 inner_total=21667
    if grep -qF "Timings(ms):" "$ofile"; then
      awk '
        /^Timings\(ms\):/{
          for(i=1;i<=NF;i++){
            if($i ~ /^lm=/){sub(/^lm=/,"",$i); print $i}
          }
        }' "$ofile" >> "$tmp_lm"
      awk '
        /^Timings\(ms\):/{
          for(i=1;i<=NF;i++){
            if($i ~ /^decoder=/){sub(/^decoder=/,"",$i); print $i}
          }
        }' "$ofile" >> "$tmp_dec"
      awk '
        /^Timings\(ms\):/{
          for(i=1;i<=NF;i++){
            if($i ~ /^inner_total=/){sub(/^inner_total=/,"",$i); print $i}
          }
        }' "$ofile" >> "$tmp_inner_total"
    fi
  done

  local avg min
  avg="$(awk '{s+=$1; n+=1} END{if(n>0) printf "%.3f", s/n; else print "nan"}' "$tmp_times")"
  min="$(awk 'NR==1{m=$1} {if($1<m) m=$1} END{if(NR>0) printf "%.3f", m; else print "nan"}' "$tmp_times")"

  local avg_inner avg_lm avg_dec
  avg_inner="$(awk '{s+=$1; n+=1} END{if(n>0) printf "%.0f", s/n; else print "nan"}' "$tmp_inner_total")"
  avg_lm="$(awk '{s+=$1; n+=1} END{if(n>0) printf "%.0f", s/n; else print "nan"}' "$tmp_lm")"
  avg_dec="$(awk '{s+=$1; n+=1} END{if(n>0) printf "%.0f", s/n; else print "nan"}' "$tmp_dec")"

  echo "$name,$ep,$intra,$parallel,$inter,$avg,$min,$avg_inner,$avg_lm,$avg_dec,$REPEATS" >> "$CSV"
  echo "  avg_s=$avg min_s=$min (runs=$REPEATS)" | tee -a "$LOG"
  echo "  avg_inner_total_ms=$avg_inner avg_lm_ms=$avg_lm avg_decoder_ms=$avg_dec" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

# Keep the default matrix small but useful.
# You can edit/extend this list as needed.
run_case "cpu_intra_8" "cpu" "8" "false" "0"
run_case "cpu_intra_4" "cpu" "4" "false" "0"
run_case "cpu_intra_2" "cpu" "2" "false" "0"

run_case "coreml_intra_1" "coreml" "1" "false" "0"
run_case "coreml_intra_2" "coreml" "2" "false" "0"
run_case "coreml_intra_4" "coreml" "4" "false" "0"

run_case "coreml_parallel_2x2" "coreml" "2" "true" "2"
run_case "coreml_parallel_4x2" "coreml" "4" "true" "2"

# Also test auto (should pick coreml on macOS builds that include it).
run_case "auto_intra_2" "auto" "2" "false" "0"
run_case "auto_intra_4" "auto" "4" "false" "0"

{
  echo "# cbx macOS benchmark"
  echo ""
  echo "- Timestamp (UTC): $(ts)"
  echo "- Host: $(uname -a | sed 's/`/\\`/g')"
  echo "- CPU cores (hw.ncpu): $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "- cbx version: $($CBX --version 2>/dev/null || echo unknown)"
  echo "- dtype: \`$DTYPE\`"
  echo "- warmup: \`$WARMUP\`"
  echo "- repeats: \`$REPEATS\`"
  echo "- coreml_cache_dir: \`$COREML_CACHE_DIR\`"
  echo ""
  echo "## Results"
  echo ""
  echo "| case | ep | intra | parallel | inter | avg_s | min_s | avg_inner_total_ms | avg_lm_ms | avg_decoder_ms | runs |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  awk -F',' 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11}' "$CSV"
  echo ""
  echo "## Raw CSV"
  echo ""
  echo "\`$CSV\`"
} > "$REPORT_MD"

echo "Done ($(ts))." | tee -a "$LOG"
echo "Report: $REPORT_MD" | tee -a "$LOG"
echo "CSV:    $CSV" | tee -a "$LOG"
