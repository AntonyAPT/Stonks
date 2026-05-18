"""Shared training utilities for both the technical and fundamental pipelines."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred) -> Dict[str, float]:
    """HuggingFace Trainer-compatible compute_metrics for multi-step classification.

    Works for both the technical pipeline (horizon=5, each step 3-class) and the
    fundamental pipeline (horizon=1, 3-class).  Flattens logits and labels across
    the time dimension before computing metrics.
    """
    logits, labels = eval_pred
    # logits: (N, horizon, n_classes) or (N, n_classes) if horizon=1 was squeezed
    if logits.ndim == 3:
        horizon = logits.shape[1]
        preds = np.argmax(logits, axis=-1).reshape(-1)       # (N*horizon,)
        flat_labels = labels.reshape(-1)                      # (N*horizon,)
    else:
        preds = np.argmax(logits, axis=-1)
        flat_labels = labels.reshape(-1)

    acc = float(accuracy_score(flat_labels, preds))
    macro_f1 = float(f1_score(flat_labels, preds, average="macro", zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    # Per-step accuracy when horizon > 1.
    if logits.ndim == 3 and logits.shape[1] > 1:
        horizon = logits.shape[1]
        step_preds = np.argmax(logits, axis=-1)   # (N, horizon)
        for idx in range(horizon):
            metrics[f"day_{idx + 1}_accuracy"] = float(
                accuracy_score(labels[:, idx], step_preds[:, idx])
            )

    return metrics
