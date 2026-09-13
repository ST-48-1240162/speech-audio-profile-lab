"""Synthetic and file-based audio helpers for reproducible eval fixtures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def synth_window(
    *,
    sr: int = 16_000,
    duration_s: float = 1.0,
    freq_hz: float = 220.0,
    noise_std: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Generate a mono float32 sine burst with light Gaussian noise."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    wave = 0.35 * np.sin(2 * math.pi * freq_hz * t)
    wave += rng.normal(0.0, noise_std, size=n)
    return wave.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="FLOAT")
    return path


def load_mono(path: Path, *, sr: int = 16_000) -> np.ndarray:
    data, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        # Lightweight resample via librosa when rates differ.
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data.astype(np.float32)
