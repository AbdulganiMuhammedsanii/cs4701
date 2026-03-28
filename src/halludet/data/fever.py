"""Load FEVER (v1.0) via HuggingFace `datasets` 2.x and label helpers."""

from __future__ import annotations

from typing import Literal

from datasets import Dataset, load_dataset

FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

InternalLabel = Literal["supported", "contradicted", "not_supported"]


def fever_row_to_internal_label(raw: str) -> InternalLabel:
    if raw == "SUPPORTS":
        return "supported"
    if raw == "REFUTES":
        return "contradicted"
    return "not_supported"


def internal_label_to_fever(internal: InternalLabel) -> str:
    return {
        "supported": "SUPPORTS",
        "contradicted": "REFUTES",
        "not_supported": "NOT ENOUGH INFO",
    }[internal]


def internal_to_int_three_way(internal: InternalLabel) -> int:
    return {"supported": 0, "contradicted": 1, "not_supported": 2}[internal]


def internal_to_int_binary_supported(internal: InternalLabel) -> int:
    """1 = supported, 0 = unsupported (REFUTES or NOT ENOUGH INFO)."""
    return 1 if internal == "supported" else 0


def load_fever_split(split: str) -> Dataset:
    """Load a FEVER v1.0 split. Requires `datasets>=2.14,<3` and internet on first run."""
    ds_dict = load_dataset("fever", "v1.0", trust_remote_code=True)
    if split not in ds_dict:
        available = ", ".join(sorted(ds_dict.keys()))
        raise KeyError(f"Unknown split {split!r}. Available: {available}")
    return ds_dict[split]
