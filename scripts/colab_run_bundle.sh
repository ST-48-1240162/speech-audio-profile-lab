#!/usr/bin/env bash
# Upload local project tarball to Colab VM and run colab_verify.py (no Drive OAuth).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TGZ="$(mktemp /tmp/profile-lab-XXXXXX.tgz)"
SESSION="profile-lab-$$"
TIMEOUT="${COLAB_RUN_TIMEOUT:-900}"

cleanup() {
  rm -f "$TGZ"
  colab stop -s "$SESSION" 2>/dev/null || true
}
trap cleanup EXIT

tar czf "$TGZ" -C "$ROOT" \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.git' \
  .

echo "[bundle] Creating session $SESSION ..."
colab new -s "$SESSION"

echo "[bundle] Uploading $(du -h "$TGZ" | cut -f1) ..."
colab upload -s "$SESSION" "$TGZ" /content/profile-lab.tgz

echo "[bundle] Running verify (timeout=${TIMEOUT}s) ..."
if ! colab exec -s "$SESSION" -f "$ROOT/scripts/colab_verify.py" --timeout "$TIMEOUT"; then
  echo "[bundle] FAILED — see output above" >&2
  exit 1
fi

mkdir -p "$ROOT/results"
colab download -s "$SESSION" /content/profile-lab/results/ABLATION.md "$ROOT/results/ABLATION.md" 2>/dev/null || true
colab download -s "$SESSION" /content/profile-lab/results/ablation_table.csv "$ROOT/results/ablation_table.csv" 2>/dev/null || true

echo "[bundle] OK"
