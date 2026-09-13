"""AudioProfile v1 — structured voice-turn metadata from analysis backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .backends import (
    BACKEND_HEADS,
    BACKEND_SENSEVOICE,
    BACKEND_WHISPER,
    heads_analyze,
    sensevoice_transcribe,
    whisper_transcribe,
)
from .lanes import VALID_LANES, apply_lane

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = frozenset({BACKEND_HEADS, BACKEND_WHISPER, BACKEND_SENSEVOICE})


@dataclass
class AudioProfile:
    schema_version: str = SCHEMA_VERSION
    backend: str = BACKEND_HEADS
    duration_s: float = 0.0
    lane: str = "clean"
    language: dict[str, Any] = field(
        default_factory=lambda: {"label": None, "confidence": None, "abstain": True}
    )
    asr: dict[str, Any] = field(
        default_factory=lambda: {"text": None, "abstain": True}
    )
    quality: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_profile(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    lane: str = "clean",
    backend: str = BACKEND_HEADS,
) -> AudioProfile:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if lane not in VALID_LANES:
        raise ValueError(f"unknown lane {lane!r}")
    processed = apply_lane(audio, sr=sr, lane=lane)
    if backend == BACKEND_HEADS:
        return build_profile_from_heads(processed, sr=sr, lane=lane)
    if backend == BACKEND_WHISPER:
        return build_profile_from_whisper(processed, sr=sr, lane=lane)
    return build_profile_from_sensevoice(processed, sr=sr, lane=lane)


def build_profile_from_heads(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    lane: str = "clean",
) -> AudioProfile:
    """Route A: frozen Wav2Vec2 + sklearn MFCC baseline → AudioProfile v1."""
    chain, _emb, elapsed = heads_analyze(audio, sr=sr)
    sklearn = chain.get("sklearn", {})
    hf = chain.get("huggingface", {})
    duration_s = float(len(audio)) / float(sr)

    return AudioProfile(
        backend=BACKEND_HEADS,
        duration_s=duration_s,
        lane=lane,
        quality={
            "embedding_model_id": hf.get("model_id"),
            "embedding_dim": hf.get("embedding_dim"),
            "embedding_mean": hf.get("mean_activation"),
            "embedding_std": hf.get("std_activation"),
            "mfcc_dim": chain.get("mfcc_dim"),
            "classical_label": sklearn.get("label"),
            "classical_confidence": sklearn.get("confidence"),
        },
        latency_ms={"inference": round(elapsed, 3)},
    )


def build_profile_from_whisper(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    lane: str = "clean",
) -> AudioProfile:
    asr_result, elapsed = whisper_transcribe(audio, sr=sr)
    duration_s = float(len(audio)) / float(sr)
    return AudioProfile(
        backend=BACKEND_WHISPER,
        duration_s=duration_s,
        lane=lane,
        language={"label": asr_result.language, "confidence": asr_result.language_confidence, "abstain": True},
        asr={"text": asr_result.text, "abstain": not bool(asr_result.text)},
        quality={"asr_model_id": asr_result.model_id},
        latency_ms={"inference": round(elapsed, 3)},
    )


def build_profile_from_sensevoice(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    lane: str = "clean",
) -> AudioProfile:
    asr_result, elapsed = sensevoice_transcribe(audio, sr=sr)
    duration_s = float(len(audio)) / float(sr)
    return AudioProfile(
        backend=BACKEND_SENSEVOICE,
        duration_s=duration_s,
        lane=lane,
        language={
            "label": asr_result.language,
            "confidence": asr_result.language_confidence,
            "abstain": asr_result.language is None,
        },
        asr={"text": asr_result.text, "abstain": not bool(asr_result.text)},
        quality={"asr_model_id": asr_result.model_id},
        latency_ms={"inference": round(elapsed, 3)},
    )
