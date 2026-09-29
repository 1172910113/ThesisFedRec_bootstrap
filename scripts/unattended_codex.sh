#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/yuanhaochen/PythonCode/ThesisFedRec_bootstrap"
CONDA_SH="/home/yuanhaochen/miniconda3/etc/profile.d/conda.sh"
CODEX="/home/yuanhaochen/.local/bin/codex"

# 7-day hard stop.
STOP_AT="2026-10-09 13:00:00 +0800"

LOG_DIR="$PROJECT/logs/unattended"
LOCK_FILE="/tmp/thesisfedrec_codex.lock"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"

# If another Codex run is still active, skip this invocation.
if ! flock -n 9; then
    echo "$(date -Is) another unattended run is still active; skipping" \
        >> "$LOG_DIR/scheduler.log"
    exit 0
fi

NOW_EPOCH=$(date +%s)
STOP_EPOCH=$(date -d "$STOP_AT" +%s)

if (( NOW_EPOCH >= STOP_EPOCH )); then
    echo "$(date -Is) unattended period ended; no action taken" \
        >> "$LOG_DIR/scheduler.log"
    exit 0
fi

cd "$PROJECT"

source "$CONDA_SH"
conda activate thesis-fedrec

# Safety checks.
CURRENT_BRANCH=$(git branch --show-current)

if [[ "$CURRENT_BRANCH" != "codex/unattended-7days" ]]; then
    echo "$(date -Is) ERROR: unexpected branch: $CURRENT_BRANCH" \
        >> "$LOG_DIR/scheduler.log"
    exit 1
fi

# Do not let automation start on top of uncommitted work.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "$(date -Is) ERROR: dirty worktree; unattended run skipped" \
        >> "$LOG_DIR/scheduler.log"
    exit 1
fi

# Skip this run if GitHub DNS is temporarily unavailable.
if ! getent hosts github.com >/dev/null 2>&1; then
    echo "$(date -Is) WARNING: github.com DNS resolution failed; skipping run" \
        >> "$LOG_DIR/scheduler.log"
    exit 0
fi

# Skip this run on temporary Git/network failure.
if ! git fetch origin; then
    echo "$(date -Is) WARNING: git fetch failed; skipping run" \
        >> "$LOG_DIR/scheduler.log"
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

echo "$(date -Is) starting unattended Codex run" \
    >> "$LOG_DIR/scheduler.log"

# Limit one invocation so a hung task cannot occupy the server forever.
EXIT_CODE=0
timeout 4h \
    "$CODEX" exec "$(cat docs/UNATTENDED_PROMPT.md)" \
    > "$LOG_FILE" 2>&1 || EXIT_CODE=$?

echo "$(date -Is) Codex run finished with code $EXIT_CODE; log=$LOG_FILE" \
    >> "$LOG_DIR/scheduler.log"

exit "$EXIT_CODE"
