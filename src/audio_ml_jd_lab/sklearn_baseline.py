"""scikit-learn baseline classifier + eval metrics for windowed audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import low_frequency_ratio, mfcc_summary


@dataclass(frozen=True)
class SklearnEvalReport:
    accuracy: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int


def _synth_labeled_corpus(
    *,
    n_per_class: int = 40,
    sr: int = 16_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tiny corpus: label=1 when low-frequency energy dominates."""
    from .audio_io import synth_window

    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, base_freq in ((1, 120.0), (0, 1200.0)):
        for i in range(n_per_class):
            freq = base_freq + rng.uniform(-20, 20)
            audio = synth_window(sr=sr, freq_hz=freq, seed=seed + label * 100 + i)
            xs.append(mfcc_summary(audio, sr=sr))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int32)


def train_eval_baseline(*, seed: int = 42) -> SklearnEvalReport:
    """Train a logistic-regression MFCC baseline; return held-out metrics."""
    x, y = _synth_labeled_corpus(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=seed)),
        ]
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return SklearnEvalReport(
        accuracy=float(accuracy_score(y_test, pred)),
        f1=float(f1_score(y_test, pred)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        n_train=len(y_train),
        n_test=len(y_test),
    )


def predict_lowdom(audio: np.ndarray, *, sr: int = 16_000, seed: int = 42) -> dict[str, float]:
    """Fit on synthetic corpus and score one clip (demo inference path)."""
    x, y = _synth_labeled_corpus(seed=seed)
    clf = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=seed)),
        ]
    )
    clf.fit(x, y)
    feat = mfcc_summary(audio, sr=sr).reshape(1, -1)
    proba = float(clf.predict_proba(feat)[0, 1])
    return {
        "lowdom_probability": proba,
        "low_frequency_ratio": low_frequency_ratio(audio, sr=sr),
    }
