#!/usr/bin/env python3
"""Run on Colab VM after `colab upload ... /content/profile-lab.tgz`."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/profile-lab")
TGZ = Path("/content/profile-lab.tgz")


def main() -> int:
    if not (ROOT / "src").is_dir() and TGZ.is_file():
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TGZ) as archive:
            archive.extractall(ROOT, filter="data")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_ablation.py"),
            "--corpus",
            os.environ.get("CORPUS", "synthetic"),
            "--max-utt",
            os.environ.get("MAX_UTT", "100"),
            "--out",
            str(ROOT / "results"),
        ],
        cwd=ROOT,
        env=env,
    )
    print((ROOT / "results" / "ABLATION.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
