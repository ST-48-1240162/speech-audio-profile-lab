"""Speech understanding backends → shared result types."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import write_wav
from .chain import run_chain
from .hf_inference import wav2vec2_embedding
from .lanes import apply_lane
from .models import HFEmbeddingResult

BACKEND_HEADS = "wav2vec2-heads"
BACKEND_WHISPER = "whisper-base"
BACKEND_SENSEVOICE = "sensevoice-small"

WHISPER_MODEL_ID = "openai/whisper-base"
SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"

_MODEL_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class ASRResult:
    model_id: str
    text: str
    language: str | None
    language_confidence: float | None


def heads_analyze(audio: np.ndarray, *, sr: int = 16_000) -> tuple[dict, HFEmbeddingResult, float]:
    t0 = time.perf_counter()
    chain = run_chain(audio, sr=sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    hf = chain["huggingface"]
    emb = HFEmbeddingResult(
        model_id=hf["model_id"],
        embedding_dim=hf["embedding_dim"],
        mean_activation=hf["mean_activation"],
        std_activation=hf["std_activation"],
    )
    return chain, emb, elapsed_ms


def whisper_transcribe(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    model_id: str = WHISPER_MODEL_ID,
) -> tuple[ASRResult, float]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    t0 = time.perf_counter()
    processor = _get_cached(f"whisper-proc-{model_id}", lambda: AutoProcessor.from_pretrained(model_id))
    model = _get_cached(
        f"whisper-model-{model_id}",
        lambda: AutoModelForSpeechSeq2Seq.from_pretrained(model_id),
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        ASRResult(
            model_id=model_id,
            text=text,
            language=None,
            language_confidence=None,
        ),
        elapsed_ms,
    )


def sensevoice_transcribe(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    model_id: str = SENSEVOICE_MODEL_ID,
) -> tuple[ASRResult, float]:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    t0 = time.perf_counter()
    model = _get_cached(
        f"sensevoice-{model_id}",
        lambda: AutoModel(model=model_id, trust_remote_code=True),
    )
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "utt.wav"
        write_wav(wav_path, audio, sr)
        raw = model.generate(input=str(wav_path), cache={}, language="auto", use_itn=True)
    text = rich_transcription_postprocess(raw[0]["text"]).strip()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    lang = _parse_sensevoice_language(raw[0].get("text", ""))
    return (
        ASRResult(
            model_id=model_id,
            text=text,
            language=lang,
            language_confidence=None,
        ),
        elapsed_ms,
    )


def _parse_sensevoice_language(raw_text: str) -> str | None:
    """SenseVoice prefixes like <|zh|> in raw output."""
    if raw_text.startswith("<|") and "|>" in raw_text[2:]:
        return raw_text[2 : raw_text.index("|>", 2)]
    return None


def _get_cached(key: str, factory):
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = factory()
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
