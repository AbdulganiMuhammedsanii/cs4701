"""Classification metrics for claim–evidence verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def collapse_to_supported_vs_unsupported(labels: list[str]) -> list[str]:
    """
    Map three-way verification labels to binary supported vs unsupported.

    ``contradicted`` and ``not_supported`` become ``unsupported``; ``supported`` stays.
    """
    out: list[str] = []
    for lab in labels:
        if lab == "supported":
            out.append("supported")
        else:
            out.append("unsupported")
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: tuple[str, ...] = ("supported", "contradicted", "not_supported"),
) -> dict[str, Any]:
    lab = list(labels)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=lab, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=lab, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", labels=lab, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", labels=lab, zero_division=0)
    prec_weighted = precision_score(
        y_true, y_pred, average="weighted", labels=lab, zero_division=0
    )
    rec_weighted = recall_score(
        y_true, y_pred, average="weighted", labels=lab, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=lab,
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "report": report,
    }
