"""Paired bootstrap on per-example correctness for comparing classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np


def per_example_correct(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    return np.array([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)], dtype=np.float64)


def bootstrap_accuracy_diff(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Distribution of (acc_A - acc_B) on bootstrap resamples of the same paired indices."""
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")
    n = len(correct_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(correct_a[idx].mean() - correct_b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_bootstrap": n_bootstrap,
        "n_examples": n,
    }


def pairwise_bootstrap_accuracy(
    correct_by_model: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """All unordered pairs of models: bootstrap of mean accuracy difference on paired resamples."""
    names = list(correct_by_model.keys())
    pairs: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_minus_{b}"
            pairs[key] = bootstrap_accuracy_diff(
                correct_by_model[a],
                correct_by_model[b],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
    return {"pairs": pairs, "n_bootstrap": n_bootstrap, "seed": seed}
