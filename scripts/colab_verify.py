#!/usr/bin/env -S colab run
"""Bootstrap + verify speech-audio-profile-lab on a Colab VM.

Run from project root (Colab CLI reads this file locally, executes on Colab):

  colab run scripts/colab_verify.py
  colab run scripts/colab_verify.py -- --source git
  colab run --gpu T4 scripts/colab_verify.py -- --source git --real-hf

First run: `colab auth` (oauth2) if you have not used Colab CLI before.
Prefer `./scripts/colab_run_bundle.sh` from a local clone (no Drive OAuth).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

DEFAULT_DRIVE_SUBPATH = "speech-audio-profile-lab"
GIT_URL = "https://github.com/ST-48-1240162/speech-audio-profile-lab.git"
GIT_DIR = "/content/speech-audio-profile-lab"
BUNDLE_TGZ = "/content/profile-lab.tgz"
BUNDLE_DIR = "/content/profile-lab"


def default_source() -> str:
    if os.environ.get("PROFILE_LAB_SOURCE"):
        return os.environ["PROFILE_LAB_SOURCE"]
    if Path(BUNDLE_TGZ).is_file() or Path(BUNDLE_DIR, "src", "audio_ml_jd_lab").is_dir():
        return "bundle"
    return "git"


def resolve_root(source: str, drive_subpath: str) -> Path:
    if source == "bundle":
        root = Path(BUNDLE_DIR)
        tgz = Path(BUNDLE_TGZ)
        if not (root / "src" / "audio_ml_jd_lab").is_dir() and tgz.is_file():
            import tarfile

            root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz) as archive:
                archive.extractall(root, filter="data")
        if not (root / "src" / "audio_ml_jd_lab").is_dir():
            raise SystemExit(
                f"Bundle missing src/: {root}\n"
                f"Upload project tarball to {BUNDLE_TGZ} first."
            )
        return root

    if source == "git":
        root = Path(GIT_DIR)
        if not (root / "src" / "audio_ml_jd_lab").is_dir():
            subprocess.check_call(["git", "clone", "-q", GIT_URL, str(root)])
        return root

    if source == "drive":
        from google.colab import drive

        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive") / drive_subpath
        if not (root / "src" / "audio_ml_jd_lab").is_dir():
            raise SystemExit(
                f"Drive project not found: {root}\n"
                "Sync the repo to Drive or pass --drive-subpath."
            )
        return root

    raise SystemExit(f"unknown --source {source!r}")


def run_tests(root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.check_call(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"],
        env=env,
        cwd=root,
    )


def print_demo_profile(root: Path, *, real_hf: bool, lane: str) -> None:
    sys.path.insert(0, str(root / "src"))
    from audio_ml_jd_lab.audio_io import synth_window
    from audio_ml_jd_lab.models import HFEmbeddingResult
    from audio_ml_jd_lab.profile import build_profile_from_heads

    audio = synth_window(freq_hz=180.0, seed=7)
    fake = HFEmbeddingResult(
        model_id="facebook/wav2vec2-base-960h",
        embedding_dim=768,
        mean_activation=0.0,
        std_activation=1.0,
    )

    if real_hf:
        profile = build_profile_from_heads(audio, lane=lane)
    else:
        with patch("audio_ml_jd_lab.chain.wav2vec2_embedding", return_value=fake):
            profile = build_profile_from_heads(audio, lane=lane)

    print("\n--- AudioProfile v1 (demo) ---")
    print(json.dumps(profile.to_dict(), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify profile lab on Colab VM")
    parser.add_argument(
        "--source",
        choices=["bundle", "drive", "git"],
        default=default_source(),
    )
    parser.add_argument(
        "--drive-subpath",
        default=os.environ.get("PROFILE_LAB_DRIVE_SUBPATH", DEFAULT_DRIVE_SUBPATH),
    )
    parser.add_argument("--lane", default="clean", choices=["clean", "telephony", "noise_telecom"])
    parser.add_argument(
        "--real-hf",
        action="store_true",
        help="Download Wav2Vec2 (~360 MB); use with colab run --gpu T4",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args, _unknown = parser.parse_known_args()

    root = resolve_root(args.source, args.drive_subpath)
    print(f"PROJECT_ROOT = {root}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")],
        cwd=root,
    )

    if not args.skip_tests:
        run_tests(root)

    print("[verify] mock ablation table ...")
    subprocess.check_call(
        [
            sys.executable,
            str(root / "scripts" / "run_ablation.py"),
            "--mock",
            "--max-utt",
            "8",
            "--out",
            str(root / "results"),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )

    print_demo_profile(root, real_hf=args.real_hf, lane=args.lane)
    print("\nOK — speech-audio-profile-lab v1.0 verified on Colab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
