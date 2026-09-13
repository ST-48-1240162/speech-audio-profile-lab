#!/usr/bin/env python3
"""Lightweight backend × lane ablation → markdown + CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_ml_jd_lab.audio_io import synth_window
from audio_ml_jd_lab.backends import BACKEND_HEADS, BACKEND_SENSEVOICE, BACKEND_WHISPER
from audio_ml_jd_lab.profile import build_profile

DEFAULT_BACKENDS = [BACKEND_HEADS, BACKEND_WHISPER, BACKEND_SENSEVOICE]
DEFAULT_LANES = ["clean", "telephony"]


@dataclass
class Row:
    backend: str
    lane: str
    n: int
    wer: float | None
    lang_acc: float | None
    json_valid: float
    p50_ms: float
    p95_ms: float


def load_corpus(name: str, max_utt: int) -> list[dict]:
    if name == "synthetic":
        rows: list[dict] = []
        for i in range(max_utt):
            rows.append(
                {
                    "id": f"syn-{i:03d}",
                    "ref_text": f"reference transcript number {i}",
                    "freq_hz": 140.0 + (i % 30) * 5.0,
                    "seed": i,
                }
            )
        return rows

    if name == "librispeech":
        from datasets import load_dataset

        ds = load_dataset("librispeech_asr", "clean", split=f"test[:{max_utt}]")
        return [
            {"id": row["id"], "ref_text": row["text"], "audio": row["audio"]["array"], "sr": row["audio"]["sampling_rate"]}
            for row in ds
        ]

    raise ValueError(f"unknown corpus {name!r}")


def utterance_audio(entry: dict) -> tuple[object, int]:
    if "audio" in entry:
        import numpy as np

        return np.asarray(entry["audio"], dtype=np.float32), int(entry["sr"])
    return synth_window(freq_hz=float(entry["freq_hz"]), seed=int(entry["seed"]), duration_s=1.2), 16_000


def word_error_rate(ref: str, hyp: str) -> float:
    from jiwer import wer

    ref = ref.strip().lower()
    hyp = hyp.strip().lower()
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(wer(ref, hyp))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def run_matrix(
    corpus: list[dict],
    backends: list[str],
    lanes: list[str],
) -> list[Row]:
    latencies: dict[tuple[str, str], list[float]] = {}
    wers: dict[tuple[str, str], list[float]] = {}
    lang_hits: dict[tuple[str, str], list[float]] = {}
    valid: dict[tuple[str, str], list[float]] = {}

    for entry in corpus:
        audio, sr = utterance_audio(entry)
        ref = entry["ref_text"]
        for lane in lanes:
            for backend in backends:
                key = (backend, lane)
                t0 = time.perf_counter()
                profile = build_profile(audio, sr=sr, lane=lane, backend=backend)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                payload = profile.to_dict()

                latencies.setdefault(key, []).append(wall_ms)
                valid.setdefault(key, []).append(1.0 if payload.get("schema_version") == "1.0" else 0.0)

                if backend in {BACKEND_WHISPER, BACKEND_SENSEVOICE} and not profile.asr.get("abstain"):
                    hyp = profile.asr.get("text") or ""
                    wers.setdefault(key, []).append(word_error_rate(ref, hyp))
                if backend == BACKEND_SENSEVOICE and profile.language.get("label"):
                    lang_hits.setdefault(key, []).append(1.0)

    rows: list[Row] = []
    for backend in backends:
        for lane in lanes:
            key = (backend, lane)
            lats = latencies.get(key, [])
            rows.append(
                Row(
                    backend=backend,
                    lane=lane,
                    n=len(lats),
                    wer=(statistics.mean(wers[key]) if key in wers and wers[key] else None),
                    lang_acc=(statistics.mean(lang_hits[key]) if key in lang_hits and lang_hits[key] else None),
                    json_valid=statistics.mean(valid.get(key, [0.0])),
                    p50_ms=percentile(lats, 0.50),
                    p95_ms=percentile(lats, 0.95),
                )
            )
    return rows


def write_outputs(rows: list[Row], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "ABLATION.md"
    csv_path = out_dir / "ablation_table.csv"

    lines = [
        "# speech-audio-profile-lab · ablation table",
        "",
        "Backend × lane on a small utterance subset. Telephony lane = 8 kHz μ-law round-trip (+ noise for `noise_telecom`).",
        "",
        "| Backend | Lane | n | WER ↓ | lang acc ↑ | JSON valid | p50 ms | p95 ms |",
        "|---------|------|---|-------|------------|------------|--------|--------|",
    ]
    for row in rows:
        wer = "—" if row.wer is None else f"{row.wer:.3f}"
        lang = "—" if row.lang_acc is None else f"{row.lang_acc:.3f}"
        lines.append(
            f"| {row.backend} | {row.lane} | {row.n} | {wer} | {lang} | {row.json_valid:.3f} | {row.p50_ms:.1f} | {row.p95_ms:.1f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["backend", "lane", "n", "wer", "lang_acc", "json_valid", "p50_ms", "p95_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(json.dumps({"markdown": str(md), "csv": str(csv_path), "rows": len(rows)}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["synthetic", "librispeech"], default="synthetic")
    parser.add_argument("--max-utt", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--backends", nargs="*", default=DEFAULT_BACKENDS)
    parser.add_argument("--lanes", nargs="*", default=DEFAULT_LANES)
    parser.add_argument("--mock", action="store_true", help="Mock heavy backends (unittest / CI)")
    args = parser.parse_args()

    if args.mock:
        from unittest.mock import patch

        from audio_ml_jd_lab.models import HFEmbeddingResult

        fake_emb = HFEmbeddingResult(
            model_id="facebook/wav2vec2-base-960h",
            embedding_dim=768,
            mean_activation=0.0,
            std_activation=1.0,
        )
        from audio_ml_jd_lab.backends import ASRResult

        with (
            patch(
                "audio_ml_jd_lab.backends.whisper_transcribe",
                return_value=(
                    ASRResult("mock/whisper", "mock transcript", None, None),
                    15.0,
                ),
            ),
            patch(
                "audio_ml_jd_lab.backends.sensevoice_transcribe",
                return_value=(
                    ASRResult("mock/sensevoice", "mock transcript", "en", 0.9),
                    18.0,
                ),
            ),
            patch("audio_ml_jd_lab.chain.wav2vec2_embedding", return_value=fake_emb),
        ):
            rows = run_matrix(load_corpus(args.corpus, min(args.max_utt, 8)), args.backends, args.lanes)
    else:
        rows = run_matrix(load_corpus(args.corpus, args.max_utt), args.backends, args.lanes)

    write_outputs(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
