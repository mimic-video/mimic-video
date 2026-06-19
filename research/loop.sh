#!/usr/bin/env bash
# The autoresearch engine: spawn a FRESH opencode/MiniMax agent per iteration, each doing exactly
# one experiment (per research/program.md) then exiting. State lives in git + the research/*.md/tsv
# files, NOT in any single agent's context. Run detached:  nohup bash research/loop.sh &>/dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OPENCODE="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
MODEL="${OPENCODE_MODEL:-microagi-minimax-m3-mxfp4/olka-fi/MiniMax-M3-MXFP4}"
ITER_TIMEOUT="${ITER_TIMEOUT:-45m}"     # hard cap per iteration (15min train + load + agent reasoning)
BASE_BRANCH="${AR_BASE:-autoresearch/setup}"   # stable base: env + harness + this loop. Runs branch off it.
BRANCH="${AR_BRANCH:-autoresearch/munich-$(date +%Y%m%d-%H%M)}"

# Each run gets a fresh branch cut FROM the stable setup base, so experiments never touch
# main or setup and every run starts from the same known-good state.
git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1 || { echo "[loop] base branch '$BASE_BRANCH' not found"; exit 1; }
git checkout -b "$BRANCH" "$BASE_BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "[loop] branch=$BRANCH (from $BASE_BRANCH) model=$MODEL iter_timeout=$ITER_TIMEOUT"

i=0
while true; do
  i=$((i+1))
  echo "[loop] ===== iteration $i @ $(date -u +%FT%TZ) ====="
  # A wedged training run or hung agent can never freeze the loop: the timeout reaps it and we continue.
  # `< /dev/null` is REQUIRED: when launched via `nohup ... &>/dev/null &`, fd 0 is /dev/null opened
  # WRITE-only, and opencode reading stdin on it throws `EBADF: bad file descriptor, read`. A readable
  # /dev/null gives a clean EOF instead.
  timeout "$ITER_TIMEOUT" "$OPENCODE" run --model "$MODEL" \
    "Read research/program.md and execute exactly ONE experiment iteration, then stop. Do not ask me anything." \
    < /dev/null >> research/loop.console.log 2>&1
  echo "[loop] iteration $i exit=$? @ $(date -u +%FT%TZ)"
  sleep 5
done
