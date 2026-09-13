"""Classical audio features for sklearn baselines."""

from __future__ import annotations

import numpy as np
import librosa


def mfcc_summary(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Mean/std pooled MFCC vector (26-D for n_mfcc=13)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def low_frequency_ratio(audio: np.ndarray, *, sr: int = 16_000, cutoff_hz: float = 500.0) -> float:
    """Share of magnitude spectrum energy below cutoff — proxy for lowdom labels."""
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    low = spec[freqs <= cutoff_hz].sum()
    total = spec.sum() + 1e-8
    return float(low / total)
