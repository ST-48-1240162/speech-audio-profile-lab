"""Container-friendly FastAPI service for cloud deployment (GCP/AWS/Azure)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .audio_io import load_mono, synth_window
from .chain import run_chain
from .backends import BACKEND_HEADS
from .profile import SUPPORTED_BACKENDS, build_profile
from .sklearn_baseline import train_eval_baseline

app = FastAPI(title="speech-audio-profile-lab", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Legacy flat dict (LangChain + sklearn + HF embedding)."""
    tmp = Path("/tmp") / (file.filename or "upload.wav")
    content = await file.read()
    tmp.write_bytes(content)
    audio = load_mono(tmp)
    result = run_chain(audio)
    return JSONResponse(result)


@app.post("/profile")
async def profile(
    file: UploadFile = File(...),
    lane: str = "clean",
    backend: str = BACKEND_HEADS,
) -> JSONResponse:
    """AudioProfile v1 JSON."""
    if backend not in SUPPORTED_BACKENDS:
        return JSONResponse({"error": f"unsupported backend {backend!r}"}, status_code=400)
    tmp = Path("/tmp") / (file.filename or "upload.wav")
    content = await file.read()
    tmp.write_bytes(content)
    audio = load_mono(tmp)
    result = build_profile(audio, lane=lane, backend=backend).to_dict()
    return JSONResponse(result)


@app.get("/eval/sklearn-baseline")
def eval_sklearn_baseline() -> dict[str, float | int]:
    report = train_eval_baseline()
    return {
        "accuracy": report.accuracy,
        "f1": report.f1,
        "roc_auc": report.roc_auc,
        "n_train": report.n_train,
        "n_test": report.n_test,
    }


@app.get("/demo")
def demo() -> dict:
    audio = synth_window(freq_hz=180.0, seed=7)
    return run_chain(audio)


@app.get("/demo/profile")
def demo_profile(lane: str = "clean", backend: str = BACKEND_HEADS) -> dict:
    """Synthetic audio → AudioProfile v1 (no upload)."""
    if backend not in SUPPORTED_BACKENDS:
        return {"error": f"unsupported backend {backend!r}"}
    audio = synth_window(freq_hz=180.0, seed=7)
    return build_profile(audio, lane=lane, backend=backend).to_dict()
