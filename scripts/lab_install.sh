#!/usr/bin/env bash
# One-command launcher/monitor for the PAC lab setup. CPU/download-only.
set -euo pipefail

ROOT="${PAC_LLM_ROOT:-/egr/research-pac/$USER}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/local_llm/lab_autopilot"
PYTHON="${PYTHON:-python3}"
ACTION="${1:-help}"

case "$ACTION" in
  plan)
    "$PYTHON" "$REPO/scripts/lab_autopilot.py" --plan
    ;;
  start|resume)
    mkdir -p "$BASE"
    chmod 700 "$BASE"
    if [[ -s "$BASE/controller.pid" ]]; then
      OLD="$(cat "$BASE/controller.pid")"
      if [[ "$OLD" =~ ^[0-9]+$ ]] && kill -0 "$OLD" 2>/dev/null; then
        echo "A setup process may already be running (PID $OLD)."
        echo "Inspect: bash scripts/lab_install.sh status"
        exit 1
      fi
    fi
    nohup "$PYTHON" "$REPO/scripts/lab_autopilot.py" --run \
      > "$BASE/controller.log" 2>&1 < /dev/null &
    CHILD=$!
    echo "$CHILD" > "$BASE/controller.pid"
    chmod 600 "$BASE/controller.pid"
    echo "Launched CPU/download-only setup with PID $CHILD"
    echo "Progress: bash scripts/lab_install.sh status"
    echo "Log:      $BASE/controller.log"
    ;;
  status)
    "$PYTHON" "$REPO/scripts/lab_autopilot.py" --status
    if [[ -s "$BASE/controller.pid" ]]; then
      PID="$(cat "$BASE/controller.pid")"
      if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
        echo "Controller PID $PID: process exists (see status and logs)."
      else
        echo "Controller PID $PID: no longer running."
      fi
    fi
    ;;
  logs)
    if [[ -f "$BASE/controller.log" ]]; then
      tail -n 70 "$BASE/controller.log"
    else
      echo "No controller log yet."
    fi
    ;;
  follow)
    if [[ -f "$BASE/controller.log" ]]; then
      tail -n 20 -f "$BASE/controller.log"
    else
      echo "No log yet; start the installer first." >&2
      exit 1
    fi
    ;;
  verify)
    "$PYTHON" "$REPO/scripts/lab_verify.py" --root "$ROOT" --repo "$REPO"
    ;;
  *)
    cat <<'USAGE'
PAC laboratory setup:
  bash scripts/lab_install.sh plan    # preview, no changes
  bash scripts/lab_install.sh start   # install in background, resumable
  bash scripts/lab_install.sh status  # stage status and controller state
  bash scripts/lab_install.sh logs    # most recent setup output
  bash scripts/lab_install.sh follow  # follow controller output, Ctrl+C exits tail
  bash scripts/lab_install.sh resume  # recover an interrupted/failed stage
  bash scripts/lab_install.sh verify  # offline check after completed installation

Only 'start' and 'resume' install packages or download public weights.
This never starts vLLM, launches GPUs, or performs model training.
USAGE
    ;;
esac
