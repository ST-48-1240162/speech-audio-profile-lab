"""LangChain orchestration over classical + HF audio ML steps."""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.runnables import RunnableLambda, RunnableSequence

from .features import mfcc_summary
from .hf_inference import wav2vec2_embedding
from .sklearn_baseline import predict_lowdom


def _analyze_payload(audio: np.ndarray, *, sr: int = 16_000) -> dict[str, Any]:
    sklearn_out = predict_lowdom(audio, sr=sr)
    hf_out = wav2vec2_embedding(audio, sr=sr)
    mfcc = mfcc_summary(audio, sr=sr)
    return {
        "sklearn": sklearn_out,
        "huggingface": {
            "model_id": hf_out.model_id,
            "embedding_dim": hf_out.embedding_dim,
            "mean_activation": hf_out.mean_activation,
            "std_activation": hf_out.std_activation,
        },
        "mfcc_dim": int(mfcc.shape[0]),
    }


def build_analysis_chain() -> RunnableSequence:
    """LangChain Runnable: mono waveform -> structured eval dict."""
    return RunnableLambda(
        lambda x: _analyze_payload(x["audio"], sr=x.get("sr", 16_000))
    )


def run_chain(audio: np.ndarray, *, sr: int = 16_000) -> dict[str, Any]:
    chain = build_analysis_chain()
    return chain.invoke({"audio": np.asarray(audio, dtype=np.float32), "sr": sr})
