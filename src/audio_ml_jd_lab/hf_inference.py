"""Hugging Face Transformers inference for audio embeddings."""

from __future__ import annotations

import numpy as np

from .models import HFEmbeddingResult

DEFAULT_MODEL = "facebook/wav2vec2-base-960h"


def wav2vec2_embedding(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    model_id: str = DEFAULT_MODEL,
) -> HFEmbeddingResult:
    """Run Wav2Vec2 forward pass; return pooled hidden-state summary stats."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id)
    model.eval()

    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0).numpy()
    pooled = hidden.mean(axis=0)
    return HFEmbeddingResult(
        model_id=model_id,
        embedding_dim=int(pooled.shape[0]),
        mean_activation=float(pooled.mean()),
        std_activation=float(pooled.std()),
    )
