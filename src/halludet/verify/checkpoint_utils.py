"""Helpers for detecting loadable sequence-classification checkpoints."""

from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")


def _directory_has_model_weights(d: Path) -> bool:
    if not d.is_dir():
        return False
    names = {p.name for p in d.iterdir() if p.is_file()}
    if "model.safetensors" in names or "pytorch_model.bin" in names:
        return True
    return any(n.startswith("model-") and n.endswith(".safetensors") for n in names)


def resolve_classifier_checkpoint_dir(ckpt_dir: Path) -> Path | None:
    """Return a directory that contains HF classifier weights, preferring ``ckpt_dir`` itself.

    HuggingFace ``Trainer`` often leaves the latest epoch under ``checkpoint-<step>/`` while the
    run root may only hold ``config.json`` and tokenizer files. This picks the highest-step
    ``checkpoint-*`` subfolder that actually contains weight files.
    """
    if not ckpt_dir.is_dir():
        return None
    if _directory_has_model_weights(ckpt_dir):
        return ckpt_dir

    candidates: list[tuple[int, Path]] = []
    for sub in ckpt_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _CHECKPOINT_DIR.match(sub.name)
        if not m:
            continue
        if _directory_has_model_weights(sub):
            candidates.append((int(m.group(1)), sub))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def checkpoint_has_classifier_weights(ckpt_dir: Path) -> bool:
    """True if a loadable classifier checkpoint exists under ``ckpt_dir`` (root or nested)."""
    return resolve_classifier_checkpoint_dir(ckpt_dir) is not None
