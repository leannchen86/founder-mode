#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-zeus}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/fast1/leann/founder-mode}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-12}"
LIMIT="${LIMIT:-0}"

rsync -a --delete \
  --exclude ".git" \
  --exclude ".env" \
  --exclude ".env.local" \
  "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT/repo/"

limit_arg=()
if [[ "$LIMIT" != "0" ]]; then
  limit_arg=(--limit "$LIMIT")
fi

ssh "$REMOTE" "
  set -euo pipefail
  mkdir -p '$REMOTE_ROOT/images' '$REMOTE_ROOT/manifests' '$REMOTE_ROOT/logs'
  cd '$REMOTE_ROOT/repo'
  python3 scripts/download-profile-images.py \
    --input data/raw/diffbot_people_latest.jsonl \
    --out-dir '$REMOTE_ROOT/images' \
    --manifest '$REMOTE_ROOT/manifests/images_manifest_latest.jsonl' \
    --workers '$WORKERS' \
    ${limit_arg[*]}
"

