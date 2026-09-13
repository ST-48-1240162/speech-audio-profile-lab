#!/usr/bin/env python3
"""CLI: run sklearn baseline + optional HF embedding on synthetic audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_ml_jd_lab.audio_io import synth_window
from audio_ml_jd_lab.chain import run_chain
from audio_ml_jd_lab.sklearn_baseline import train_eval_baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", action="store_true", help="Run Hugging Face Wav2Vec2 forward pass")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    sklearn_report = train_eval_baseline()
    payload = {
        "sklearn_baseline": {
            "accuracy": sklearn_report.accuracy,
            "f1": sklearn_report.f1,
            "roc_auc": sklearn_report.roc_auc,
        }
    }

    if args.hf:
        audio = synth_window(freq_hz=200.0, seed=11)
        payload["chain"] = run_chain(audio)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("scikit-learn baseline:", payload["sklearn_baseline"])
        if args.hf:
            print("LangChain + HF chain keys:", list(payload["chain"].keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
