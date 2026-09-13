#!/usr/bin/env bash
# Full ablation on Colab (GPU recommended for Whisper + SenseVoice).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TGZ="$(mktemp /tmp/profile-lab-XXXXXX.tgz)"
SESSION="profile-ablation-$$"
TIMEOUT="${COLAB_RUN_TIMEOUT:-3600}"
GPU="${COLAB_GPU:-T4}"
CORPUS="${CORPUS:-synthetic}"
MAX_UTT="${MAX_UTT:-100}"

cleanup() {
  rm -f "$TGZ"
  colab stop -s "$SESSION" 2>/dev/null || true
}
trap cleanup EXIT

tar czf "$TGZ" -C "$ROOT" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.git' .

colab new -s "$SESSION" --gpu "$GPU"
colab upload -s "$SESSION" "$TGZ" /content/profile-lab.tgz

CORPUS="$CORPUS" MAX_UTT="$MAX_UTT" colab exec -s "$SESSION" --timeout "$TIMEOUT" -f "$ROOT/scripts/colab_remote_ablation.py"

colab download -s "$SESSION" /content/profile-lab/results/ABLATION.md "$ROOT/results/ABLATION.md"
colab download -s "$SESSION" /content/profile-lab/results/ablation_table.csv "$ROOT/results/ablation_table.csv"
echo "[ablation] wrote $ROOT/results/ABLATION.md"
