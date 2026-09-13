"""Audio lane transforms (clean vs telephony-style degradation)."""

from __future__ import annotations

import numpy as np

VALID_LANES = frozenset({"clean", "telephony", "noise_telecom"})


def apply_lane(audio: np.ndarray, *, sr: int, lane: str) -> np.ndarray:
    """Return audio prepared for the requested evaluation lane."""
    if lane not in VALID_LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {sorted(VALID_LANES)}")
    if lane == "clean":
        return audio.astype(np.float32, copy=False)

    import librosa

    narrow = librosa.resample(audio, orig_sr=sr, target_sr=8_000)
    narrow = _g711_ulaw_roundtrip(narrow)
    if lane == "noise_telecom":
        rng = np.random.default_rng(42)
        narrow = narrow + rng.normal(0.0, 0.015, size=narrow.shape)
        narrow = np.clip(narrow, -1.0, 1.0)
    wide = librosa.resample(narrow, orig_sr=8_000, target_sr=sr)
    return wide.astype(np.float32)


def _g711_ulaw_roundtrip(audio: np.ndarray) -> np.ndarray:
    """8-bit μ-law companding round-trip (G.711-ish narrowband)."""
    x = np.clip(audio, -1.0, 1.0)
    pcm = (x * 32_767.0).astype(np.int16)
    ulaw = _linear_to_ulaw(pcm)
    pcm_back = _ulaw_to_linear(ulaw)
    return (pcm_back.astype(np.float32) / 32_767.0).clip(-1.0, 1.0)


def _linear_to_ulaw(pcm: np.ndarray) -> np.ndarray:
    mu = 255
    sign = np.sign(pcm)
    pcm = np.abs(pcm.astype(np.int32)) + 132
    pcm = np.clip(pcm, 0, 32_767)
    exponent = np.floor(np.log2(pcm + 1.0)) - 7.0
    exponent = np.clip(exponent, 0, 7)
    mantissa = (pcm / (2.0 ** (exponent + 3.0))) - 16.0
    ulaw = 255 - (sign * (64 * exponent + mantissa)).astype(np.int32)
    return np.clip(ulaw, 0, 255).astype(np.uint8)


def _ulaw_to_linear(ulaw: np.ndarray) -> np.ndarray:
    ulaw = (~ulaw.astype(np.int32)) & 0xFF
    sign = np.where(ulaw & 0x80, -1, 1)
    exponent = (ulaw >> 4) & 0x07
    mantissa = ulaw & 0x0F
    pcm = sign * (((mantissa << 3) + 132) << exponent) - 132
    return pcm.astype(np.int16)
